# evaluate.py

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics_multiclass(eval_pred, average="macro"):
    """
    Compute standard multiclass classification metrics from Hugging Face Trainer output.

    Parameters
    ----------
    eval_pred : tuple
        Usually a pair (logits, labels), or an EvalPrediction object unpacked by Trainer.
    average : str
        Averaging method for precision / recall / f1.
        Common choices: "macro", "weighted", "micro".

    Returns
    -------
    dict
        Dictionary of metric values.
    """
    logits, labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average=average, zero_division=0),
        "recall": recall_score(labels, predictions, average=average, zero_division=0),
        "f1": f1_score(labels, predictions, average=average, zero_division=0),
    }