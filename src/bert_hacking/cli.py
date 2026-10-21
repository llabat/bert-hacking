"""Command line entry point: python -m bert_hacking {train,regress,status,sample-configs} --help"""
import argparse
import json
import logging
import os
import smtplib
import ssl
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from email.message import EmailMessage
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from . import data, pipeline, storage  # noqa: E402
from .config import Experiment, Runtime, sample_configs  # noqa: E402
from .regression import gold_regressions, regress_run  # noqa: E402

log = logging.getLogger("bert_hacking")


def train(args) -> str:
    experiment = Experiment.load(args.config)
    subsample = json.loads(Path(args.subsample).read_text()) if args.subsample else None
    runtime = Runtime(args.device_batch_size, args.prediction_batch_size, test_mode=args.test_mode)
    for spec in experiment.datasets:
        configs = [c for c in experiment.run_configs(spec, subsample) if not storage.is_done(c.run_id)]
        log.info("%s: %d runs to do", spec.name, len(configs))
        if not configs:
            continue
        train_df = data.load_texts(spec.train_path, spec)
        data.check_labels(train_df, spec)
        predict_df = data.load_texts(spec.predict_path, spec)
        for config in configs:
            log.info("Run %s: %s", config.run_id, config)
            try:
                pipeline.run(config, runtime, train_df, predict_df)
            except Exception:
                log.error("Run %s failed\n%s", config.run_id, traceback.format_exc())
    return status_report(experiment, subsample)


def regress(args) -> str:
    experiment = Experiment.load(args.config)
    runs = [r for r in storage.load_runs() if not (storage.REGRESSIONS / f"{r['run_id']}.json").exists()]
    log.info("%d runs to regress", len(runs))
    if not runs:
        return "Nothing to regress"
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for (dataset_name, label), group in pd.DataFrame(runs).groupby(["dataset_name", "dichotomization_label"]):
            spec = experiment.dataset(dataset_name)
            gold_df = _gold_table(spec, label)
            gold = _gold(gold_df, spec, label)
            futures += [executor.submit(_regress_run, run_id, gold_df, spec, label, gold)
                        for run_id in group["run_id"]]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Runs"):
            future.result()
    return f"{len(futures)} runs regressed"


def _gold_table(spec, label: str) -> pd.DataFrame:
    """Inference set with gold labels (GS-LABEL) and covariates."""
    texts = data.dichotomize(data.load_texts(spec.predict_path, spec), label)
    metadata = pd.read_csv(spec.metadata_path, dtype={"ID": str})
    if not metadata["ID"].is_unique:
        raise ValueError(f"{spec.metadata_path}: IDs are not unique")
    df = texts[["ID", "LABEL"]].rename(columns={"LABEL": "GS-LABEL"}).merge(
        metadata[["ID", *spec.covariates]], on="ID", how="left", validate="one_to_one", indicator=True)
    if (df.pop("_merge") != "both").any():
        raise ValueError(f"Some inference IDs of {spec.name} are missing from {spec.metadata_path}")
    return df


def _gold(gold_df: pd.DataFrame, spec, label: str) -> dict:
    gold = gold_regressions(gold_df, label, spec.covariates)
    storage.write_json(storage.REGRESSIONS / f"gold-{spec.name}-{label}.json",
                       [{"regression_col": c, "regression_unique_value": v, **res} for (c, v), res in gold.items()])
    return gold


def _regress_run(run_id: str, gold_df: pd.DataFrame, spec, label: str, gold: dict) -> None:
    warnings.filterwarnings("ignore")  # statsmodels convergence warnings, one per failed fit
    predictions = pd.read_csv(storage.run_dir(run_id) / "predictions.csv", dtype={"ID": str})
    if set(predictions["ID"]) != set(gold_df["ID"]):
        raise ValueError(f"Run {run_id}: predicted IDs differ from the inference set")
    records = []
    for policy in predictions.columns.drop(["ID", "GS-LABEL"]):  # one column per policy
        df = gold_df.merge(predictions[["ID", policy]].rename(columns={policy: "PRED-LABEL"}), on="ID")
        records += [{"run_id": run_id, "dataset_name": spec.name, "dichotomization_label": label,
                     "policy": policy, **r}
                    for r in regress_run(df, label, spec.covariates, gold)]
    storage.write_json(storage.REGRESSIONS / f"{run_id}.json", records)


def status(args) -> str:
    experiment = Experiment.load(args.config)
    subsample = json.loads(Path(args.subsample).read_text()) if args.subsample else None
    return status_report(experiment, subsample)


def status_report(experiment: Experiment, subsample: list[dict] | None) -> str:
    lines = []
    for spec in experiment.datasets:
        for label in spec.labels:
            configs = [c for c in experiment.run_configs(spec, subsample) if c.dichotomization_label == label]
            done = sum(storage.is_done(c.run_id) for c in configs)
            lines.append(f"{spec.name} x {label}: {done} / {len(configs)}")
    return "\n".join(lines)


def sample(args) -> str:
    space = json.loads(Path(args.space).read_text())
    configs = sample_configs(space, args.per_cell, args.seed)
    Path(args.out).write_text(json.dumps(configs, indent=4, ensure_ascii=True))
    return f"{len(configs)} configurations written to {args.out}"


def notify(message: str) -> None:
    """Email `message` if EMAIL_FROM, EMAIL_TO and EMAIL_FROM_PWD are set (env or .env)."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    sender, to, password = (os.environ.get(k) for k in ("EMAIL_FROM", "EMAIL_TO", "EMAIL_FROM_PWD"))
    if not (sender and to and password):
        return
    email = EmailMessage()
    email["From"], email["To"], email["Subject"] = sender, to, "bert-hacking run stopped"
    email.set_content(message)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as smtp:
        smtp.login(sender, password)
        smtp.send_message(email)


def main() -> None:
    parser = argparse.ArgumentParser(prog="bert_hacking")
    commands = parser.add_subparsers(required=True)

    p = commands.add_parser("train", help="fine-tune and predict for every config not done yet")
    p.add_argument("--config", required=True, help="experiment JSON (datasets + parameter grid)")
    p.add_argument("--subsample", help="JSON list of configs: only run grid configs in this list")
    p.add_argument("--device-batch-size", type=int, default=4)
    p.add_argument("--prediction-batch-size", type=int, default=256)
    p.add_argument("--test-mode", action="store_true", help="20 rows per split, save nothing")
    p.set_defaults(command=train, notify=True)

    p = commands.add_parser("regress", help="run the regressions for every run not regressed yet")
    p.add_argument("--config", required=True)
    p.add_argument("--workers", type=int, default=5)
    p.set_defaults(command=regress, notify=True)

    p = commands.add_parser("status", help="count done runs per dataset x label")
    p.add_argument("--config", required=True)
    p.add_argument("--subsample")
    p.set_defaults(command=status, notify=False)

    p = commands.add_parser("sample-configs", help="randomly sample configs from an exploration space")
    p.add_argument("--space", required=True, help="JSON dict: parameter -> list of values")
    p.add_argument("--per-cell", type=int, default=60, help="configs per (N_annotated, model_name)")
    p.add_argument("--seed", type=int, default=31851920)
    p.add_argument("--out", required=True)
    p.set_defaults(command=sample, notify=False)

    args = parser.parse_args()
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler("logs/bert-hacking.log")])
    try:
        message = args.command(args)
        print(message)
    except BaseException as e:
        message = f"Stopped by {e!r}\n{traceback.format_exc()}"
        raise
    finally:
        if args.notify:
            notify(f"{args.command.__name__} stopped\n\n{message}")
