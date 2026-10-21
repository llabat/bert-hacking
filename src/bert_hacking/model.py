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
    TrainingArguments,
    set_seed,
)

from .config import AT_LEAST, THRESHOLD, RunConfig, Runtime


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
    )
    trainer.train()
    return trainer.model, trainer.state.log_history


@torch.no_grad()
def predict(model, ds: Dataset, tokenizer, batch_size: int) -> pd.DataFrame:
    """Chunk-level predictions: ID, ID_CHUNK, GS-LABEL, PRED-LABEL."""
    model.eval()
    if model.device.type == "cuda":
        model = model.to(torch.bfloat16)
    loader = DataLoader(ds.select_columns(["input_ids", "attention_mask"]), batch_size=batch_size,
                        collate_fn=DataCollatorWithPadding(tokenizer))
    predicted = []
    for batch in loader:
        logits = model(**batch.to(model.device)).logits
        predicted += logits.argmax(dim=-1).tolist()
    return pd.DataFrame({
        "ID": ds["ID"],
        "ID_CHUNK": ds["ID_CHUNK"],
        "GS-LABEL": ds["LABEL"],
        "PRED-LABEL": [model.config.id2label[i] for i in predicted],
    })


def aggregate(chunk_predictions: pd.DataFrame, label2id: dict[str, int]) -> pd.DataFrame:
    """Document-level predictions (ID, GS-LABEL, PRED-LABEL), sorted by ID.

    A document is positive if at least AT_LEAST of its chunks are, or, when THRESHOLD is
    set, if the share of its positive chunks is >= THRESHOLD.
    """
    id2label = {i: name for name, i in label2id.items()}
    by_doc = (chunk_predictions[["GS-LABEL", "PRED-LABEL"]].apply(lambda col: col.map(label2id))
              .groupby(chunk_predictions["ID"]))
    positive = by_doc.mean() >= THRESHOLD if THRESHOLD is not None else by_doc.sum() >= AT_LEAST
    return positive.astype(int).apply(lambda col: col.map(id2label)).reset_index()


def macro_f1(predictions: pd.DataFrame) -> float:
    return f1_score(predictions["GS-LABEL"], predictions["PRED-LABEL"], average="macro", zero_division=np.nan)


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
