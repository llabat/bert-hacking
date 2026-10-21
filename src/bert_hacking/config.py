"""Experiment configuration: datasets, run hyperparameters and execution settings."""
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# Fixed for the whole study, recorded in every run
CHUNK_OVERLAP = 50  # tokens shared by consecutive chunks of a long document

# How a document's prediction is derived from its chunks, depending on RunConfig.long_texts.
# Both "chunk" rules come from the same trained model, so they are not RunConfig fields:
#   any:   positive if any chunk has P(positive) >= 0.5
#   mean:  positive if the mean P(positive) over its chunks is >= 0.5
#   trunc: the model is trained and applied on the first window only
POLICIES = {"chunk": ("any", "mean"), "truncate": ("trunc",)}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    train_path: str
    predict_path: str
    metadata_path: str
    text_col: str
    label_col: str
    id_col: str
    labels: list[str]
    covariates: list[str]
    extra_cols: list[str] = field(default_factory=list)
    sep: str = ","

    @classmethod
    def from_json(cls, d: dict) -> "DatasetSpec":
        return cls(
            name=d["name"],
            train_path=d["filepath-train"],
            predict_path=d["filepath-predict"],
            metadata_path=d["filepath-metadata"],
            text_col=d["text_col"],
            label_col=d["label_col"],
            id_col=d["id_col"],
            labels=list(d["labels"]),
            covariates=list(d["columns-for-independant-variables"]),
            extra_cols=list(d.get("extra_cols_to_keep", [])),
            sep=d.get("csv-sep", ","),
        )


@dataclass(frozen=True)
class RunConfig:
    """Everything that determines a run's results: equal configs are the same run."""
    dataset_name: str
    dichotomization_label: str
    N_annotated: int = 500
    sampling_method: dict = field(default_factory=dict)  # {"balance": "random" | float, "stratified": column | None}
    splits_ratio: tuple[int, int, int] = (80, 10, 10)    # train / eval / test, in %
    model_name: str = "google-bert/bert-base-uncased"
    n_epochs: int = 4
    learning_rate: float = 3e-5
    weight_decay: float = 0.0
    batch_size: int = 8
    seed: int = 42
    long_texts: str = "chunk"  # texts longer than the model's window: "chunk" or "truncate"
    # Stop after this many epochs without eval macro-F1 improvement (None: always run n_epochs).
    # The best epoch is kept either way, so this only matters if F1 would improve again later.
    early_stopping_patience: int | None = 1

    def __post_init__(self):
        # Normalize JSON values (0 vs 0.0, list vs tuple, missing keys) so that equal
        # configs get equal run_ids
        unknown = set(self.sampling_method) - {"balance", "stratified"}
        if unknown:
            raise ValueError(f"Unknown sampling_method keys: {unknown}")
        normalized = {
            "N_annotated": int(self.N_annotated),
            "sampling_method": {"balance": "random", "stratified": None, **self.sampling_method},
            "splits_ratio": tuple(int(r) for r in self.splits_ratio),
            "n_epochs": int(self.n_epochs),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "batch_size": int(self.batch_size),
            "seed": int(self.seed),
            "early_stopping_patience": (None if self.early_stopping_patience is None
                                        else int(self.early_stopping_patience)),
        }
        for key, value in normalized.items():
            object.__setattr__(self, key, value)
        if self.long_texts not in POLICIES:
            raise ValueError(f"long_texts must be one of {list(POLICIES)}, got {self.long_texts!r}")
        if len(self.splits_ratio) != 3 or sum(self.splits_ratio) != 100:
            raise ValueError(f"splits_ratio must be 3 ints summing to 100, got {self.splits_ratio}")

    @property
    def run_id(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Runtime:
    """Execution settings: they change speed and memory use, not results."""
    device_batch_size: int = 4
    prediction_batch_size: int = 256
    output_dir: str = "./models/current"
    test_mode: bool = False  # train and predict on 20 rows, save nothing


@dataclass(frozen=True)
class Experiment:
    datasets: list[DatasetSpec]
    grid: dict[str, list]

    @classmethod
    def load(cls, path: str | Path) -> "Experiment":
        raw = json.loads(Path(path).read_text())
        return cls([DatasetSpec.from_json(d) for d in raw["datasets"]], raw["parameters"])

    def dataset(self, name: str) -> DatasetSpec:
        return {d.name: d for d in self.datasets}[name]

    def run_configs(self, dataset: DatasetSpec, subsample: list[dict] | None = None) -> list[RunConfig]:
        """Grid configs for every label of `dataset`, restricted to `subsample` if given."""
        configs = [
            RunConfig(dataset.name, label, **dict(zip(self.grid, values)))
            for label in dataset.labels
            for values in itertools.product(*self.grid.values())
        ]
        if subsample is None:
            return configs
        wanted = {RunConfig(dataset.name, label, **c).run_id for label in dataset.labels for c in subsample}
        return [c for c in configs if c.run_id in wanted]


def sample_configs(space: dict[str, list], per_cell: int, seed: int,
                   cell_keys: tuple[str, ...] = ("N_annotated", "model_name")) -> list[dict]:
    """Randomly pick `per_cell` hyperparameter combinations for each value of `cell_keys`."""
    other_keys = [k for k in space if k not in cell_keys]
    sampled = []
    for i, cell in enumerate(itertools.product(*(space[k] for k in cell_keys))):
        combos = list(itertools.product(*(space[k] for k in other_keys)))
        np.random.default_rng(seed=seed * i).shuffle(combos)
        sampled += [
            {**dict(zip(cell_keys, cell)), **dict(zip(other_keys, combo))}
            for combo in combos[:per_cell]
        ]
    return sampled
