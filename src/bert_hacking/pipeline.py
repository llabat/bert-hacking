"""One run: sample, fine-tune, evaluate on the test split, predict on the inference set, save."""
import logging
import time
from contextlib import contextmanager
from dataclasses import asdict

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

from . import data, model, storage
from .config import CHUNK_OVERLAP, POLICIES, RunConfig, Runtime

log = logging.getLogger(__name__)


@contextmanager
def _timed(timings: dict, name: str):
    start = time.perf_counter()
    yield
    timings[name] = time.perf_counter() - start


def run(config: RunConfig, runtime: Runtime, train_df: pd.DataFrame, predict_df: pd.DataFrame) -> dict:
    """`train_df` and `predict_df` come from data.load_texts. Returns the run record."""
    label2id, _ = data.label_maps(config.dichotomization_label)
    policies = POLICIES[config.long_texts]
    timings = {}
    trained = None
    try:
        with _timed(timings, "preprocess_data"):
            sample = data.sample(data.dichotomize(train_df, config.dichotomization_label), config)
            distribution = sample["LABEL"].value_counts(normalize=True).sort_index().to_dict()
            log.info("Sampled %d rows, label distribution %s (requested %s)",
                     len(sample), distribution, config.sampling_method)
            tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
            records, train_length = data.encode(sample, tokenizer, config.model_name, label2id, config.long_texts)
            splits = data.split_ids(sample["ID"], config.splits_ratio, config.seed)
            datasets = {name: _dataset([r for r in records if r["ID"] in ids], runtime)
                        for name, ids in splits.items()}
            log.info("Chunks per split: %s", {name: len(ds) for name, ds in datasets.items()})

        with _timed(timings, "training"):
            trained, trainer_logs = model.train(config, runtime, datasets["train"], datasets["eval"],
                                                tokenizer, label2id)

        with _timed(timings, "evaluation"):
            test_chunks = model.predict(trained, datasets["test"], tokenizer, runtime.prediction_batch_size)
            test = model.aggregate(test_chunks, label2id, policies)
            score = {policy: model.macro_f1(test, policy) for policy in policies}
            log.info("Macro-F1 on test: %s", score)

        with _timed(timings, "prediction"):
            records, predict_length = data.encode(data.dichotomize(predict_df, config.dichotomization_label),
                                                  tokenizer, config.model_name, label2id, config.long_texts)
            chunks = model.predict(trained, _dataset(records, runtime), tokenizer, runtime.prediction_batch_size)
            predictions = model.aggregate(chunks, label2id, policies)
    finally:
        del trained
        model.free_memory()

    record = {
        "run_id": config.run_id,
        **asdict(config),
        "score_on_test": score,
        "effective_distrib": distribution,
        "effective_context_window_for_training": train_length,
        "effective_context_window_for_inference": predict_length,
        "chunk_overlap": CHUNK_OVERLAP,
        "policies": list(policies),
        "trainer_logs": trainer_logs,
        "run_timer": timings,
        "time": str(pd.Timestamp.now()),
    }
    if not runtime.test_mode:
        storage.save_run(record, {
            "predictions": predictions,
            "predictions-chunks": chunks,
            "test": test,
            "test-chunks": test_chunks,
        })
    return record


def _dataset(records: list[dict], runtime: Runtime) -> Dataset:
    ds = Dataset.from_list(records)
    return ds.select(range(min(20, len(ds)))) if runtime.test_mode else ds
