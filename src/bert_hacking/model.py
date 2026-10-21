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
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .config import RunConfig, Runtime


def _f1_macro(output: EvalPrediction) -> dict:
    logits = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
    return {"f1_macro": f1_score(output.label_ids, logits.argmax(axis=1), average="macro", zero_division=0)}


def train(config: RunConfig, runtime: Runtime, train_ds: Dataset, eval_ds: Dataset,
          tokenizer, label2id: dict[str, int]) -> tuple[torch.nn.Module, list[dict]]:
    """Fine-tune and return the best model (on eval macro-F1) and the trainer's log history."""
    # Seed before instantiating the model so that the classification head init is reproducible
    set_seed(config.seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label={i: name for name, i in label2id.items()},
    )
    shutil.rmtree(runtime.output_dir, ignore_errors=True)
    device_batch_size = min(config.batch_size, runtime.device_batch_size)
    args = TrainingArguments(
        output_dir=runtime.output_dir,
        num_train_epochs=config.n_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        per_device_train_batch_size=device_batch_size,
        per_device_eval_batch_size=device_batch_size,
        gradient_accumulation_steps=max(config.batch_size // device_batch_size, 1),
        bf16=torch.cuda.is_available(),
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_only_model=True,  # checkpoints are only used to reload the best epoch, never to resume
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
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
        callbacks=([EarlyStoppingCallback(config.early_stopping_patience)]
                   if config.early_stopping_patience is not None else []),
    )
    trainer.train()
    shutil.rmtree(runtime.output_dir, ignore_errors=True)  # best model is loaded in memory
    return trainer.model, trainer.state.log_history


@torch.no_grad()
def predict(model, ds: Dataset, tokenizer, batch_size: int) -> pd.DataFrame:
    """Chunk-level predictions: ID, ID_CHUNK, GS-LABEL, P-POSITIVE (probability of label id 1)."""
    model.eval()
    if model.device.type == "cuda":
        model = model.to(torch.bfloat16)
    loader = DataLoader(ds.select_columns(["input_ids", "attention_mask"]), batch_size=batch_size,
                        collate_fn=DataCollatorWithPadding(tokenizer))
    p_positive = []
    for batch in loader:
        logits = model(**batch.to(model.device)).logits
        p_positive += logits.float().softmax(dim=-1)[:, 1].tolist()
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
