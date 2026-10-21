# BERT Hacking

## Setup

Python 3.11. The code does not support multi-GPU.

```bash
conda create -n bert-hacking python=3.11 -y
conda activate bert-hacking
pip install -e .
```

## Datasets

Any dataset fits if it is **one CSV file** (the corpus) with:

- an ID column (unique), a text column and a label column, without missing values;
  the names `ID`, `TEXT` and `LABEL` are reserved for these three;
- the metadata columns used as covariates, which may have missing values (never tested);
- the columns used for stratified sampling (`"stratified"` in the grid), without missing
  values, each stratum holding at least `N_annotated / number of strata` documents.

Every run samples its `N_annotated` documents from the corpus (they keep their gold label),
fine-tunes on them and predicts the rest of the corpus; the regressions use the whole corpus.
Preparing a dataset (merging files, deriving or cleaning columns) happens outside the pipeline.

A dataset entry of the experiment file:

```json
{
    "name": "misinfo",
    "filepath": "./data/misinfo-full.csv",
    "csv-sep": ",",
    "id_col": "ID",
    "text_col": "headline",
    "label_col": "gold_label",
    "labels": ["misinfo"],
    "columns-for-independant-variables": ["source"],
    "min-hypothesis-size": 100
}
```

- `labels`: one binary task per label (label vs rest). For a binary dataset list only one of
  its two labels (`train` refuses both: same task with the classes swapped), chosen
  deliberately, as the `any` policy is not symmetric.
- `columns-for-independant-variables`: each value with at least `min-hypothesis-size`
  documents (default 0) is a hypothesis, `logit(1[column == value] ~ 1 + label)`.

## Usage

Run from the repo root (data paths in the config files are relative to it).

```bash
# Fine-tune + predict every config of the grid that is not done yet
# (--subsample: only the grid configs listed in that file)
bert-hacking train --config config_files/config-loop.json --subsample config_files/sample-configurations.json
bert-hacking train --config config_files/config-loop.json --test-mode   # quick check, saves nothing

# Regressions on every finished run not regressed yet
bert-hacking regress --config config_files/config-loop.json --workers 5

# Progress
bert-hacking status --config config_files/config-loop.json --subsample config_files/sample-configurations.json

# Regenerate the random subsample of the exploration space
bert-hacking sample-configs --space config_files/exploration-space-english.json --out config_files/sample-configurations.json
```

`python -m bert_hacking ...` works too without installing (`PYTHONPATH=src`).
If `EMAIL_FROM`, `EMAIL_TO` and `EMAIL_FROM_PWD` are set (environment or `.env`),
`train` and `regress` send an email when they stop.

Texts longer than a model's window are chunked (`"long_texts": "chunk"`, default) or
truncated (`"truncate"`); add `"long_texts": ["chunk", "truncate"]` to the grid to compare
both. Chunk runs produce two document-level predictions, `any` (any chunk positive) and
`mean` (mean probability ≥ 0.5); truncate runs produce `trunc`. Scores and regressions
are reported per policy.

Training stops early after `early_stopping_patience` epochs (default 1, `null` to disable)
without eval macro-F1 improvement; the best epoch is kept.

A run is identified by a hash of its config (`RunConfig.run_id`), so it is done
when `results/runs/<run_id>/run.json` exists; see `src/bert_hacking/storage.py`
for the layout of `results/`. Logs go to `logs/bert-hacking.log`.

| Module | Role |
|---|---|
| `config.py` | experiment file, `RunConfig` (hyperparameters), `Runtime` (batch sizes, test mode), fixed chunking constants |
| `data.py` | loading, dichotomization, sampling, train/eval/test split, tokenization and chunking |
| `model.py` | fine-tuning, prediction, chunk aggregation |
| `pipeline.py` | one run, end to end |
| `regression.py` | logit regressions and type I / II / S / M errors |
| `storage.py` | result files; `load_regressions()` gives one table of all regressions for analysis |
| `cli.py` | commands above |

## Protocole expérimental

Le but est de reproduire l'expérience d'annotation et croisement annotations / métadonnées à l'aide de regressions linéaires OLS. Pour ce faire on procède en 2 étapes: 

- Fine-tuning de BERT-models et génération des labels sur le corpus entier
- Regression linéaire $\text{label}\approx \text{metadonnée}$ et analyse des résultats

### 1. Fine-tuning de BERT-models

1. Choix d'un jeu de données:
  - doit contenir des métadonnées intéressantes
  - pas d'ensemble d'inférence mis de côté : chaque run tire ses `N_annotated` documents (annotés) dans le corpus entier, entraîne dessus et prédit le reste du corpus. Les régressions portent sur le corpus entier, avec le label gold pour les documents annotés et le label prédit pour les autres, comme le ferait un·e chercheur·se. Les régressions gold sont donc les mêmes pour tous les runs.
  - Pour du multiclasse, le jeux de données doit être binarisé.
  - <span style="background-color:orange;font-weight:bold;">Pour le moment nous avons choisi 3 jeux de données: ideology news, manifestos et misinfo</span>
  - <span style="background-color:orange;font-weight:bold;">Pour des limites en terme de temps de calcul, nous nous autorisons à abaisser le nombre d'annotations à 3000?? 5000??</span>
2. Choix d'hyperparamètres et entraînement
  - Pour explorer l'espace des hyperparamètres et leur impact sur les résultats, on fait varier les hyperparamètres sur les critères suivants: `N_annotated`[^n-annotated-values] (nombre d'annotation utilisées pour l'entraînement, tous splits confondus), `splits_ratio`[^splits-ratio-values] (train: mise à jour des poids; eval: eval perf interne; test: evaluation finale), `sampling_method`[^sampling-method-values] (aléatoire, stratifié, ou forcer une distribution de positifs/negatifs), `model_name`[^model-name-values], `learning_rate`[^learning-rate-values], `weight_decay`[^weight-decay-values], `batch_size`[^bach-size-values].
  - toute la procédure est seedée pour la reproductibilité
  - Pour les textes dépassant la fenêtre de contexte on chunk les entrées avec des chunks de la taille de la fenêtre de contexte et un overlap de 50 tokens. **Les chunks de taille inférieure à 10% de la taille d'un chunk (typiquement la fin d'une séquence) sont ignorés**
  - <span style="background-color:orange;font-weight:bold;">en l'état l'espace d'exploration est constitué de 11,520 combinaisons par tâche (dataset x label binarisé). Pour limiter le temps de calcul, nous procédons à un tirage aléatoire de 60 configurations par valeur de `N_annotated` et `model_name` = 60 x 4 x 4 = 960 configurations par tâches</span>
3. Prédiction sur le reste du corpus (les documents annotés gardent leur label gold) et enregistrement des prédictions 

[^n-annotated-values]: `N_annotated` values: 500, 1000, 1500, 2000
[^splits-ratio-values]: `splits_ratio` values: [80-10-10], [70,15,15], [50,10,40]
[^sampling-method-values]: `sampling_method` values: random, label 25%, label 50%, label 75%, label 25% strat par année, label 50% strat par année, label 75% strat par année <span style="background-color:orange;font-weight:bold;">à rediscuter</span> 
[^model-name-values]: `model_name` values: (jeux de données anglophones) BERT-base, modernBERT deberta V2, roberta (jeux de données multilingues) MBERT, xlm-robeta, multilingual E5, MMBERT
[^learning-rate-values]: `learning_rate` values: 5e-4, 1e-4, 1e-5, 2e-5, 5e-5 
[^weight-decay-values]: `weight_decay` values: 0, 0.01, 0.03, 0.1
[^bach-size-values]: `batch_size` values: 8, 16, 32

### Regressions

- Regression d'une métadonnée du jeux origine (binarisée) sur les labels (prédits / gold). (ex: `sm.Logit(y = df["label-centre], X = df["topic-economy"]) 
- Sauvegarde des données de regression:
  - `Pseudo R-squared`
  - `Coef`
  - `Std err`
  - `pvalues`
  - `Conf Int`
  - `Log-Likelihood`
  - `LL-Null`
  - `LLR p-value`
  - `AIC`
  - `BIC`
  - `N iterations`
- Analyse des résultats:
  - Filtrer les regressions qui n'ont pas fonctionné (`res_success = res.loc['FAILED' != res['Coef']]`)
  - Créer des paires de regressions
    - grouper par task (dataset x label)
    - grouper par hypotèse (covariate explique label)
    - grouper par configuration (modele, learning rate etc..)
    - Chaque groupe devrait contenir 2 regressions, une où le label est gold-standard et un ou le label est prédit
  - Ne conserver que les regressions faisant partie d'un couple `valid_for_comparison = res_success.groupby([ ... ]).size() == 2`
  - Pour chaque groupe de regression évaluer la présence d'erreur
    - `error_type_1 : bool = pred_significant and not GS_significant`
    - `error_type_2 : bool = GS_significant and not pred_significant`
    - `error_type_S : bool = pred_significant and GS_significant and (GS_coef * pred_coef < 0)`
    - `error_type_M : float = pred_significant and GS_significant and (GS_coef * pred_coef < 0) * magnitude_coef`
    - _voir `analyse-regression-results.py` pour les détails_
  - Évaluer les risques d'après la définition du papier