"""Result files. One folder per run, named after the config: a run is done when its run.json exists.

results/
  runs/<run_id>/run.json                 config, scores per policy, timings, trainer logs
  runs/<run_id>/predictions.csv          inference set, per document: ID, GS-LABEL, ANNOTATED, one column
                                         per policy (documents annotated for the run keep their gold label)
  runs/<run_id>/predictions-chunks.csv   predicted documents, per chunk: ID, ID_CHUNK, GS-LABEL, P-POSITIVE
  runs/<run_id>/test.csv                 same two tables for the test split
  runs/<run_id>/test-chunks.csv
  regressions/<run_id>.json              regressions on that run's predictions, for each policy
  regressions/gold-<dataset>-<label>.json
"""
import json
import math
import os
from dataclasses import fields
from pathlib import Path

import pandas as pd

from .config import RunConfig

RUNS = Path("results/runs")
REGRESSIONS = Path("results/regressions")


def run_dir(run_id: str) -> Path:
    return RUNS / run_id


def is_done(run_id: str) -> bool:
    return (run_dir(run_id) / "run.json").exists()


def save_run(record: dict, tables: dict[str, pd.DataFrame]) -> None:
    directory = run_dir(record["run_id"])
    directory.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(directory / f"{name}.csv", index=False)
    write_json(directory / "run.json", record)  # last: marks the run as done


def load_runs() -> list[dict]:
    return [read_json(path) for path in sorted(RUNS.glob("*/run.json"))]


def load_regressions() -> pd.DataFrame:
    """One row per (run, policy, hypothesis): the run's config, its test macro-F1 for that
    policy, the error flags (h_t, error_type_*) and the predicted-label regression
    statistics, plus "task" (dataset-label) and "hypothesis" (covariate-value)."""
    runs = {run["run_id"]: run for run in load_runs()}
    rows = []
    for path in sorted(REGRESSIONS.glob("*.json")):
        if path.name.startswith("gold-"):
            continue
        for record in read_json(path):
            run = runs[record["run_id"]]
            rows.append({
                **{field.name: run[field.name] for field in fields(RunConfig)},
                **{k: v for k, v in record.items() if k not in ("errors", "pred_reg_results")},
                **record.get("errors", {}),
                **record.get("pred_reg_results", {}),
                "score_on_test": run["score_on_test"][record["policy"]],
                "time_run": sum(run["run_timer"].values()),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["task"] = df["dataset_name"] + "-" + df["dichotomization_label"]
        df["hypothesis"] = df["regression_col"] + "-" + df["regression_unique_value"].astype(str)
    return df


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def write_json(path: Path, obj) -> None:
    """Write atomically (no half-written file on crash), with NaN -> null and numpy scalars as numbers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_nan_to_none(obj), default=_to_builtin, ensure_ascii=True))
    os.replace(tmp, path)


def _to_builtin(o):
    if hasattr(o, "item"):  # numpy scalar
        return o.item()
    raise TypeError(f"Not JSON serializable: {type(o)}")


def _nan_to_none(o):
    if isinstance(o, dict):
        return {k: _nan_to_none(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_nan_to_none(v) for v in o]
    if isinstance(o, float) and math.isnan(o):
        return None
    return o
