"""Data preparation: loading, dichotomization, sampling, splitting, tokenization and chunking."""
import hashlib
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoConfig

from .config import CHUNK_OVERLAP, DatasetSpec, RunConfig

SAMPLE_CACHE = Path(".cache/samples")
TOKENS_CACHE = Path(".cache/tokens")


def load_corpus(spec: DatasetSpec) -> pd.DataFrame:
    """Read the corpus with its id / text / label columns renamed ID (str, unique), TEXT, LABEL.
    Other columns are kept as they are: covariates may have missing values (never tested),
    stratification columns may not (checked when sampling)."""
    df = pd.read_csv(spec.path, sep=spec.sep)
    renaming = {spec.id_col: "ID", spec.text_col: "TEXT", spec.label_col: "LABEL"}
    missing = (set(renaming) | set(spec.covariates)) - set(df.columns)
    if missing:
        raise KeyError(f"{spec.path}: missing columns {sorted(missing)}")
    clashing = {"ID", "TEXT", "LABEL"} & (set(df.columns) - set(renaming))
    if clashing:
        raise ValueError(f"{spec.path}: columns {sorted(clashing)} are reserved, rename them")
    df = df.rename(columns=renaming)
    if df[["ID", "TEXT", "LABEL"]].isna().any().any():
        raise ValueError(f"{spec.path}: missing values\n{df[['ID', 'TEXT', 'LABEL']].isna().sum()}")
    df["ID"] = df["ID"].astype(str)
    if not df["ID"].is_unique:
        raise ValueError(f"{spec.path}: IDs are not unique")
    return df


def check_labels(df: pd.DataFrame, spec: DatasetSpec) -> None:
    """Fail early on a dataset's label list: every listed label must exist, and a binary
    dataset must list only one of its two labels."""
    values = set(df["LABEL"])
    missing = set(spec.labels) - values
    if missing:
        raise ValueError(f"{spec.name}: labels {sorted(missing)} not found, available: {sorted(values)}")
    if len(values) == 2 and len(spec.labels) == 2:
        raise ValueError(
            f"{spec.name} is binary {sorted(values)}: list only one of its labels. Both tasks "
            f"would be the same task with the positive class swapped: twice the compute, "
            f"twice the weight in the risks, and the 'any' policy would differ between them "
            f"only because of which label is called positive.")


def label_maps(label: str) -> tuple[dict[str, int], dict[int, str]]:
    label2id = {label: 1, f"not-{label}": 0}
    return label2id, {i: name for name, i in label2id.items()}


def dichotomize(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if label not in df["LABEL"].values:
        raise ValueError(f"Label {label!r} not found, available: {df['LABEL'].unique()}")
    return df.assign(LABEL=np.where(df["LABEL"] == label, label, f"not-{label}"))


def _draw_ids(df: pd.DataFrame, config: RunConfig) -> list[str]:
    """Draw N_annotated IDs without replacement, spread evenly over the strata.

    With a numeric `balance`, each draw is positive with probability `balance`:
    the number of positives is Binomial(N, balance), not exactly balance * N.
    """
    balance = config.sampling_method["balance"]
    strat_col = config.sampling_method["stratified"]
    if strat_col and (strat_col not in df or df[strat_col].isna().any()):
        raise ValueError(f"Stratification column {strat_col!r} is missing or has missing values")
    strata = df.groupby(strat_col) if strat_col else [(None, df)]
    n_base, n_remainder = divmod(config.N_annotated, len(strata))
    rng = np.random.default_rng(seed=config.seed)

    sampled = []
    for i, (stratum, stratum_df) in enumerate(strata):
        n = n_base + int(i < n_remainder)
        if len(stratum_df) < n:
            raise ValueError(f"Stratum {strat_col}={stratum} has {len(stratum_df)} rows, cannot sample {n}")
        ids = stratum_df["ID"].to_numpy()
        positive = (stratum_df["LABEL"] == config.dichotomization_label).to_numpy()
        for _ in range(n):
            p = None
            if balance != "random":
                share = positive.mean()
                if share in (0, 1):
                    raise ValueError(f"Cannot sample with balance={balance}: stratum "
                                     f"{strat_col}={stratum} has only one class left")
                weights = np.where(positive, balance / share, (1 - balance) / (1 - share))
                p = weights / sum(weights)
            pick = rng.choice(len(ids), p=p)
            sampled.append(str(ids[pick]))
            ids, positive = np.delete(ids, pick), np.delete(positive, pick)
    return sampled


def sample(df: pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    """Rows of `df` drawn for this config (cached), in their original order."""
    s = config.sampling_method
    cache = SAMPLE_CACHE / (f"{config.dataset_name}-{config.dichotomization_label}-{config.N_annotated}-"
                            f"{s['stratified']}-{s['balance']}-{config.seed}.csv")
    if cache.exists():
        ids = pd.read_csv(cache, dtype=str)["ID"].tolist()
    else:
        ids = _draw_ids(df, config)
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.Series(ids, name="ID").to_csv(cache, index=False)
    return df[df["ID"].isin(ids)]


def split_ids(ids, splits_ratio: tuple[int, int, int], seed: int) -> dict[str, set[str]]:
    """Split unique IDs into train / eval / test according to splits_ratio (in %)."""
    # sorted: the iteration order of a set of str depends on PYTHONHASHSEED
    shuffled = pd.Series(sorted(set(ids))).sample(frac=1, random_state=seed).tolist()
    n_train = splits_ratio[0] * len(shuffled) // 100
    n_test = splits_ratio[2] * len(shuffled) // 100
    return {
        "train": set(shuffled[:n_train]),
        "eval": set(shuffled[n_train:len(shuffled) - n_test]),
        "test": set(shuffled[len(shuffled) - n_test:]),
    }


def max_length(tokenizer, model_name: str) -> int:
    # max_position_embeddings alone is wrong for RoBERTa-like models (514 positions but
    # 512 usable tokens), and some tokenizers leave model_max_length unset (huge value)
    return min(tokenizer.model_max_length, AutoConfig.from_pretrained(model_name).max_position_embeddings)


def chunk(input_ids: list[int], length: int, overlap: int) -> list[list[int]]:
    """Split [CLS] t1 ... tn [SEP] into windows of at most `length` tokens, each wrapped in
    [CLS]/[SEP], overlapping by `overlap` tokens. A trailing window shorter than 10% of
    `length` is dropped."""
    if len(input_ids) <= length:
        return [input_ids]
    cls, body, sep = input_ids[0], input_ids[1:-1], input_ids[-1]
    size = length - 2
    chunks = []
    for start in range(0, len(body), size - overlap):
        window = body[start:start + size]
        if len(window) < 0.1 * length:
            break
        chunks.append([cls, *window, sep])
    return chunks


def tokenize(texts: list[str], tokenizer, model_name: str, cache: bool = False) -> list[list[int]]:
    """Token ids of each text, untruncated. With cache=True, the result is stored under a
    hash of the model name and the texts, so any change to the texts invalidates it."""
    if not cache:
        return tokenizer(texts)["input_ids"]
    digest = hashlib.sha256("\0".join([model_name, *texts]).encode()).hexdigest()[:16]
    path = TOKENS_CACHE / f"{model_name.replace('/', '--')}-{digest}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    encoded = tokenizer(texts)["input_ids"]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")  # atomic: concurrent runs may write the same file
    tmp.write_bytes(pickle.dumps(encoded))
    os.replace(tmp, path)
    return encoded


def encode(df: pd.DataFrame, tokenizer, model_name: str, label2id: dict[str, int],
           long_texts: str, cache: bool = False) -> tuple[list[dict], int]:
    """Tokenize texts; texts longer than the model's window are chunked, or truncated
    to their first window if long_texts == "truncate".

    Returns one record per chunk (ID, ID_CHUNK, LABEL, input_ids, attention_mask, labels)
    and the chunk length, i.e. min(longest text, model window).
    """
    encoded = tokenize(df["TEXT"].tolist(), tokenizer, model_name, cache)
    length = min(max(len(ids) for ids in encoded), max_length(tokenizer, model_name))
    records = []
    for doc_id, label, ids in zip(df["ID"], df["LABEL"], encoded):
        chunks = chunk(ids, length, CHUNK_OVERLAP)
        for i, chunk_ids in enumerate(chunks[:1] if long_texts == "truncate" else chunks):
            records.append({
                "ID": doc_id,
                "ID_CHUNK": f"{doc_id}-{i}",
                "LABEL": label,
                "input_ids": chunk_ids,
                "attention_mask": [1] * len(chunk_ids),
                "labels": label2id[label],
            })
    return records, length
