# BERT Hacking

## Setup

Python version : 3.11

Code does not support multi-gpu.

```bash 
conda create -n bert-hacking python=3.11 -y
conda activate bert-hacking
conda install pytorch 'transformers>=4.52.4' datasets pandas scikit-learn pyyaml 'accelerate>=1.1.0' tiktoken sentencepiece protobuf -y
```

A minimal, configurable pipeline for fine-tuning BERT (and other Hugging Face transformers) on text classification tasks.

This project focuses on **simplicity, reproducibility, and fast experimentation** and is based on Hugging Face classes.

## Protocole expérimental

Le but est de reproduire l'expérience d'annotation et croisement annotations / métadonnées à l'aide de regressions linéaires OLS. Pour ce faire on procède en 2 étapes: 

- Fine-tuning de BERT-models et génération des labels sur un ensemble d'inférence
- Regression linéaire $\text{label}\approx \text{metadonnée}$ et analyse des résultats

### 1. Fine-tuning de BERT-models

1. Choix d'un jeu de données:
  - doit contenir au moins 2000 éléments pour l'entraînement + le nombre d'annotations utilisées pour l'inférence (cf p.9)<br/>_ex: pour un nombre d'annotation p9 de 3000, il faut que le jeux de données contienne au moins 3000 + 2000 textes annotés._
  - doit contenir des métadonnées intéressantes
  - le tirage des lignes utilisées pour l'inférence doit être tiré aléatoirement à partir du jeux de donnée entier, les lignes non sélectionnées peuvent être utilisées pour le fine-tuning. 
  - Pour du multiclasse, le jeux de données doit être binarisé.
  - <span style="background-color:orange;font-weight:bold;">Pour le moment nous avons choisi 3 jeux de données: ideology news, manifestos et misinfo</span>
  - <span style="background-color:orange;font-weight:bold;">Pour des limites en terme de temps de calcul, nous nous autorisons à abaisser le nombre d'annotations à 3000?? 5000??</span>
2. Choix d'hyperparamètres et entraînement
  - Pour explorer l'espace des hyperparamètres et leur impact sur les résultats, on fait varier les hyperparamètres sur les critères suivants: `N_annotated`[^n-annotated-values] (nombre d'annotation utilisées pour l'entraînement, tous splits confondus), `splits_ratio`[^splits-ratio-values] (train: mise à jour des poids; eval: eval perf interne; test: evaluation finale), `sampling_method`[^sampling-method-values] (aléatoire, stratifié, ou forcer une distribution de positifs/negatifs), `model_name`[^model-name-values], `learning_rate`[^learning-rate-values], `weight_decay`[^weight-decay-values], `batch_size`[^bach-size-values].
  - toute la procédure est seedée pour la reproductibilité
  - Pour les textes dépassant la fenêtre de contexte on chunk les entrées avec des chunks de la taille de la fenêtre de contexte et un overlap de 50 tokens. **Les chunks de taille inférieure à 10% de la taille d'un chunk (typiquement la fin d'une séquence) sont ignorés**
  - <span style="background-color:orange;font-weight:bold;">en l'état l'espace d'exploration est constitué de 11,520 combinaisons par tâche (dataset x label binarisé). Pour limiter le temps de calcul, nous procédons à un tirage aléatoire de 60 configurations par valeur de `N_annotated` et `model_name` = 60 x 4 x 4 = 960 configurations par tâches</span>
3. Prédiction sur le jeu d'inférence et enregistrement des prédictions 

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