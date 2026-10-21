"""Fine-tuning, prediction and chunk aggregation."""
import gc
import shutil

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from .config import RunConfig, Runtime


def _f1_macro(output: EvalPrediction) -> dict:
    logits = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
    return {"f1_macro": f1_score(output.label_ids, logits.argmax(axis=1), average="macro", zero_division=0)}


class _KeepBestEpoch(TrainerCallback):
    """Keep the weights of the best epoch (eval macro-F1) in memory, and stop training after
    `patience` epochs without improvement (None: never stop early).

    Replaces the Trainer's load_best_model_at_end, which reloads the best checkpoint from disk:
    in transformers 5.x, BERT checkpoints store LayerNorm weights under their legacy names
    (gamma / beta) and that reload silently skips them."""

    def __init__(self, patience: int | None):
        self.patience = patience
        self.best_score, self.best_weights, self.epochs_without_improvement = None, None, 0

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        score = metrics["eval_f1_macro"]
        if self.best_score is None or score > self.best_score:
            self.best_score, self.epochs_without_improvement = score, 0
            self.best_weights = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            self.epochs_without_improvement += 1
            if self.patience is not None and self.epochs_without_improvement >= self.patience:
                control.should_training_stop = True


def train(config: RunConfig, runtime: Runtime, train_ds: Dataset, eval_ds: Dataset,
          tokenizer, label2id: dict[str, int]) -> tuple[torch.nn.Module, list[dict]]:
    """Fine-tune and return the best model (on eval macro-F1) and the trainer's log history."""
    output_dir = f"{runtime.checkpoints_dir}/{config.run_id}"  # per run: jobs may run concurrently
    # Seed before instantiating the model so that the classification head init is reproducible
    set_seed(config.seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label={i: name for name, i in label2id.items()},
    )
    shutil.rmtree(output_dir, ignore_errors=True)
    device_batch_size = min(config.batch_size, runtime.device_batch_size)
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.n_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        per_device_train_batch_size=device_batch_size,
        per_device_eval_batch_size=device_batch_size,
        gradient_accumulation_steps=max(config.batch_size // device_batch_size, 1),
        bf16=torch.cuda.is_available(),
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",  # the best epoch is kept in memory by _KeepBestEpoch
        report_to="none",
        seed=config.seed,
    )
    trainer = Trainer(
        model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_f1_macro,
        callbacks=[keep_best := _KeepBestEpoch(config.early_stopping_patience)],
    )
    trainer.train()
    shutil.rmtree(output_dir, ignore_errors=True)
    model.load_state_dict(keep_best.best_weights)
    return model, trainer.state.log_history


@torch.no_grad()
def predict(model, ds: Dataset, tokenizer, batch_size: int) -> pd.DataFrame:
    """Chunk-level predictions: ID, ID_CHUNK, GS-LABEL, P-POSITIVE (probability of label id 1),
    in the order of `ds`."""
    model.eval()
    if model.device.type == "cuda":
        model = model.to(torch.bfloat16)
    # Batch chunks of similar length together to minimize padding; longest first, so that
    # an out-of-memory error happens on the first batch
    order = np.argsort([-len(ids) for ids in ds["input_ids"]], kind="stable")
    loader = DataLoader(ds.select(order).select_columns(["input_ids", "attention_mask"]),
                        batch_size=batch_size, collate_fn=DataCollatorWithPadding(tokenizer))
    p_positive = np.empty(len(ds))
    start = 0
    for batch in loader:
        logits = model(**batch.to(model.device)).logits
        p = logits.float().softmax(dim=-1)[:, 1].cpu().numpy()
        p_positive[order[start:start + len(p)]] = p
        start += len(p)
    return pd.DataFrame({
        "ID": ds["ID"],
        "ID_CHUNK": ds["ID_CHUNK"],
        "GS-LABEL": ds["LABEL"],
        "P-POSITIVE": p_positive,
    })


def aggregate(chunk_predictions: pd.DataFrame, label2id: dict[str, int], policies: tuple[str, ...]) -> pd.DataFrame:
    """Document-level predictions, sorted by ID: ID, GS-LABEL, and one predicted-label
    column per policy (see config.POLICIES). Chunks must be in order within a document."""
    id2label = {i: name for name, i in label2id.items()}
    by_doc = chunk_predictions.groupby("ID")
    p_positive = by_doc["P-POSITIVE"]
    scores = {"any": p_positive.max(), "mean": p_positive.mean(), "trunc": p_positive.first()}
    return pd.DataFrame({
        "GS-LABEL": by_doc["GS-LABEL"].first(),
        **{policy: (scores[policy] >= 0.5).astype(int).map(id2label) for policy in policies},
    }).reset_index()


def macro_f1(predictions: pd.DataFrame, policy: str) -> float:
    return f1_score(predictions["GS-LABEL"], predictions[policy], average="macro", zero_division=np.nan)


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
