import numpy as np
import pandas as pd
import torch

from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def tokenize_dataset(dataset, tokenizer, text_column="TEXT", label_column="LABEL", max_length=None):
    """
    Tokenize a Hugging Face Dataset.
    """
    def tokenize_batch(batch):
        tokenized = tokenizer(
            batch[text_column],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        
        tokenized["labels"] = batch[label_column]

        return tokenized

    return dataset.map(tokenize_batch, batched=True)


def train(
    train_df,
    validation_df,
    model_name,
    output_dir,
    compute_metrics, # force
    text_column="TEXT", # delete
    label_column="LABEL", # delete
    num_labels=2, # delete
    max_length=None,
    learning_rate=2e-5,
    per_device_train_batch_size=8, # same
    per_device_eval_batch_size=8, # same
    num_train_epochs=3,
    weight_decay=0.0,
    evaluation_strategy="epoch", # force
    save_strategy="epoch", # force
    logging_strategy="epoch", # force
    load_best_model_at_end=True, # force
    metric_for_best_model="f1", # force
    greater_is_better=True,  # force
    save_total_limit=2, # force 2
    seed=42,
):
    """
    Train a sequence classification model with Hugging Face Trainer.
    """
    if num_labels is None:
        num_labels = train_df[label_column].nunique()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        # add id2label / label2id
    )

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    validation_dataset = Dataset.from_pandas(validation_df.reset_index(drop=True))

    train_dataset = tokenize_dataset(
        train_dataset,
        tokenizer=tokenizer,
        text_column=text_column,
        label_column=label_column,
        max_length=max_length,
    )

    validation_dataset = tokenize_dataset(
        validation_dataset,
        tokenizer=tokenizer,
        text_column=text_column,
        label_column=label_column,
        max_length=max_length,
    )

    columns_to_keep = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in train_dataset.column_names:
        columns_to_keep.append("token_type_ids")

    train_dataset = train_dataset.remove_columns(
        [col for col in train_dataset.column_names if col not in columns_to_keep]
    )
    validation_dataset = validation_dataset.remove_columns(
        [col for col in validation_dataset.column_names if col not in columns_to_keep]
    )

    train_dataset.set_format("torch")
    validation_dataset.set_format("torch")

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        eval_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        logging_strategy=logging_strategy,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        save_total_limit=save_total_limit,
        seed=seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return {
        "trainer": trainer,
        "model": trainer.model,
        "tokenizer": tokenizer,
    }


def predict(
    df,
    model,
    tokenizer,
    text_column="TEXT",
    id_column=None,
    label_column=None,
    max_length=None,
    batch_size=32,
    device=None,
):
    """
    Run prediction on a pandas dataframe and return a dataframe of outputs.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in dataframe")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    texts = df[text_column].tolist()

    all_predictions = []
    all_scores = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )

            inputs = {key: value.to(device) for key, value in inputs.items()}

            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

            batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            batch_scores = probabilities.cpu().numpy()

            all_predictions.extend(batch_predictions.tolist())
            all_scores.extend(batch_scores.tolist())

    result = pd.DataFrame({
        "prediction": all_predictions,
        "scores": all_scores,
    })

    if id_column is not None:
        if id_column not in df.columns:
            raise ValueError(f"Column '{id_column}' not found in dataframe")
        result.insert(0, id_column, df[id_column].values)

    if label_column is not None:
        if label_column not in df.columns:
            raise ValueError(f"Column '{label_column}' not found in dataframe")
        result["true_label"] = df[label_column].values

    return result
