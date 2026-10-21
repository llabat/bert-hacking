"""Result files. One folder per run, named after the config: a run is done when its run.json exists.

results/
  runs/<run_id>/run.json                 config, scores, timings, trainer logs
  runs/<run_id>/predictions.csv          document-level predictions on the inference set
  runs/<run_id>/predictions-chunks.csv   chunk-level predictions on the inference set
  runs/<run_id>/test.csv                 document-level predictions on the test split
  runs/<run_id>/test-chunks.csv
  regressions/<run_id>.json              regressions on that run's predictions
  regressions/gold-<dataset>-<label>.json
"""
import json
import math
import os
from pathlib import Path

import pandas as pd

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
