## Objectif: 

**Adapter [LLM Hacking](https://arxiv.org/pdf/2509.08825) au cas des classifieurs encodeurs**

Ça signifie:
    
- Prendre des jeux de données dans leur banque (p.9)
- Produire des labels grâce à un classifieur basé sur l'architecture encodeur
- Comparer les résultats de régressions linéaires $label_{true} \sim y$ et $label_{pred} \sim y$ (Erreur I, II, S et M)

## Protocole expérimental: 

### Selection du jeu de données et prétraitement

- Choix d'un jeu de données avec **au moins 2000 éléments** et qui **contient des métadonnées intéressantes**. _Jeux de données considérés: manifestos (Léo), ideology news (Axel), misinfo (Alexandre)._
- Pas d'ensemble d'inférence mis de côté. Chaque run tire `N_annotated` documents (hyperparamètre) dans le corpus entier, les découpe en Train / Eval / Test pour le fine-tuning, puis prédit le reste du corpus. Les régressions portent sur le corpus entier : label gold pour les documents annotés, label prédit pour les autres.
    - _À discuter : taille du corpus (corpus complet, ou sous-échantillon de la taille utilisée dans LLM Hacking p.9, ex. misinfo 9 991 utilisés / 25 164)._
- Les splits doivent être créés par tirage aléatoire _(vérifier qu'il n'y a pas de doublon)_. Le tirage peut être stratifié (à documenter).
    - _Possibilité d'étendre à un tirage non aléatoire similaire à une situation d'active learning._
 
> 🚨 **Documenter le jeux de données ainsi que les métadonnées existantes.** 🚨

### Génération des labels

- **Pour chaque label, entraînement d'un classifieur binaire**
- Ensemble d'hyperparamètres pouvant être explorés:
    - `N_annotated` ⚠️ est ce qu'il y a suffisamment de données pour toutes les tâches?
        - 500
        - 1000
        - 1500
        - 2000
    - `train_ratio` `train-eval-ratio`(eval dans training loop) `test-ratio` (evalue la F1)
        - 80-10-10
        - 70-15-15
        - 50-10-40 ⚠️ à rediscuter
    - `balanced_dataset` (l'ensemble d'entraînement contient 50% de labels True, 50% de labels false)
        - 25% de valeurs positives
        - 50% de valeurs positives
        - 75% de valeurs positives
    - `model`: contraintes: "multilingue" + fenetre de contexte max > 3,000 + fenetre roulantes ? 
        - [MBERT](https://huggingface.co/google-bert/bert-base-multilingual-cased) ⚠️ rolling window
        - [mmnert - multilingual modern bert](https://huggingface.co/blog/mmbert) 
        - [xlm-roberta](https://huggingface.co/docs/transformers/model_doc/xlm-roberta) ⚠️ rolling window
        - [multilingual E5](https://huggingface.co/intfloat/multilingual-e5-large) ⚠️ rolling window
    - `context_window_size`:
        - 100% max
        - 75% max
        - 50% max
    - `learning_rate` ⚠️ qu'est ce que ça représente à la fin? Est ce que ça rentre dans les configuration non-raisonnables? **A mettre en avant**
        - 5e-4 
        - 1e-4
        - 1e-5 
        - 2e-5 
        - 5e-5
    - `weight_decay`: 
        - 0.0
        - 0.01
        - 0.001
        - 0.05
    - `dropout`:
        - 0.1
        - 0.2
    - `warmup_ratio` 
        - 0.05
        - 0.1
        - 0.15
    - `pooling_method`
        - first (CLS)
        - mean
        - max
    - *Hyperparamètres supplémentaires: `sampling_method` (active ou random)*
- On génère les labels sur le reste du corpus ; les documents annotés gardent leur label gold.

Retravailler leur risque? comment on fait pour pondérer correctement enlever des configs absured

_Exemple de pipeline:_ `src/single_run.py`

### Regression

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

**Type I Risk**

$$
= \frac1{|T|}\sum_{t\in T}\frac1{|H_t^0|}\sum_{h\in H_t^0}\frac1{|\Phi|}\sum_{\phi \in \Phi}\mathbb 1\left[S_{h,\phi}^{LLM} = 1\right]
$$

code demo: 

```python
risk = 0 
T = 0
PHI = len(unique_configs) # configs independantes de la tâche (dataset + labels) et des regressions
for dataset in unique_datasets:
    T += len(unique_labels[dataset])
    for label in unique_labels[dataset]:
        H_t_0_counter = 0
        hypothesis_risk_counter = 0
        for covariate in unique_covariates[dataset]:
            if GS_significant[dataset, label, covariate] == 0: # i.e. non signifiant
                H_t_0_counter += 1
                config_risk_counter = 0
                for config in unique_configs:
                    if (pred_significant[dataset, label, covariate, config] == 1):  # i.e. signifiant
                        config_risk_counter += 1
                hypothesis_risk_counter += config_risk_counter / PHI
        risk += hypothesis_risk_counter / H_t_0_counter
risk_I = risk / T
```
**Type II Risk**

$$
= \frac1{|T|}\sum_{t\in T}\frac1{|H_t^1|}\sum_{h\in H_t^1}\frac1{|\Phi|}\sum_{\phi \in \Phi}\mathbb 1\left[S_{h,\phi}^{LLM} = 0\right]
$$

code demo:

```python
risk = 0 
T = 0
PHI = len(unique_configs) # configs independantes de la tâche (dataset + labels) et des regressions
for dataset in unique_datasets:
    T += len(unique_labels[dataset])
    for label in unique_labels[dataset]:
        H_t_1_counter = 0
        hypothesis_risk_counter = 0
        for covariate in unique_covariates[dataset]:
            if GS_significant[dataset, label, covariate] == 1: # i.e. signifiant
                H_t_1_counter += 1
                config_risk_counter = 0
                for config in unique_configs:
                    if (pred_significant[dataset, label, covariate, config] == 0):  # i.e. non signifiant
                        config_risk_counter += 1
                hypothesis_risk_counter += config_risk_counter / PHI
        risk += hypothesis_risk_counter / H_t_1_counter
risk_II = risk / T
```
**Type S Risk**

$$
= \frac1{|T|}\sum_{t\in T}\frac1{|H_t^1|}\sum_{h\in H_t^1}\frac1{|\Phi|}\sum_{\phi \in \Phi}\mathbb 1\left[S_{h,\phi}^{LLM} = 1, sgn(\beta^{GT}_h) \neq sgn(\beta^{LLM}_{h,\phi}\right]
$$

code demo:

```python
risk = 0 
T = 0
PHI = len(unique_configs) # configs independantes de la tâche (dataset + labels) et des regressions
for dataset in unique_datasets:
    T += len(unique_labels[dataset])
    for label in unique_labels[dataset]:
        H_t_1_counter = 0
        hypothesis_risk_counter = 0
        for covariate in unique_covariates[dataset]:
            if GS_significant[dataset, label, covariate] == 1: # i.e. signifiant
                H_t_1_counter += 1
                config_risk_counter = 0
                for config in unique_configs:
                    if (
                        pred_significant[dataset, label, covariate, config] == 1 # i.e. signifiant
                        and
                        coef_GT[dataset, label, covariate] * coef_pred[dataset, label, covariate, config] < 0 # i.e. pas meme signe
                    ): 
                        config_risk_counter += 1
                hypothesis_risk_counter += config_risk_counter / PHI
        risk += hypothesis_risk_counter / H_t_1_counter
risk_S = risk / T
```

**Type M Risk**

$$
= \frac1{|T|}\sum_{t\in T}\frac1{|H_t^1|}\sum_{h\in H_t^1}\frac1{|\Phi|}\sum_{\phi \in \Phi}\mathbb 1\left[S_{h,\phi}^{LLM} = 1, sgn(\beta^{GT}_h) \neq sgn(\beta^{LLM}_{h,\phi}\right]
$$

code demo :
 
```python
risk = 0 
T = 0
PHI = len(unique_configs) # configs independantes de la tâche (dataset + labels) et des regressions
for dataset in unique_datasets:
    T += len(unique_labels[dataset])
    for label in unique_labels[dataset]:
        H_t_1_counter = 0
        hypothesis_risk_counter = 0
        for covariate in unique_covariates[dataset]:
            if GS_significant[dataset, label, covariate] == 1: # i.e. signifiant
                H_t_1_counter += 1
                config_risk_counter = 0
                for config in unique_configs:
                    if (
                        pred_significant[dataset, label, covariate, config] == 1 # i.e. signifiant
                        and
                        coef_GT[dataset, label, covariate] * coef_pred[dataset, label, covariate, config] > 0 # i.e. meme signe
                    ): 
                        delta_p_pred = abs(
                            (labels_pred[dataset, label, covariate, config] == 1).mean()
                            - 
                            (labels_pred[dataset, label, covariate, config] == 0).mean()
                        )
                        delta_p_GS = abs(
                            (labels_GS[dataset, label, covariate] == 1).mean()
                            - 
                            (labels_GS[dataset, label, covariate] == 0).mean()
                        )
                        config_risk_counter += abs(
                            (delta_p_pred - delta_p_GS)
                            / 
                            delta_p_GS
                        )
                hypothesis_risk_counter += config_risk_counter / PHI
        risk += hypothesis_risk_counter / H_t_1_counter
risk_M = risk / T
```

- Discussion à avoir:
    - T représente l'ensemble des tâches, tandis que $H_t$ l'ensemble des hypothèses. D'après notre lecture, (Alexandre et Axel), on comprend que T revient à être la somme du nombre de tâches de classification à travers les jeux de données (i.e. $\sum_{d\in datasets}N^{labels}_d$ ) tandis que $H_t$ l'ensemble des regressions réalisé par label et par dataset (i.e. le nombre de covariates par dataset $\sum_{d\in datasets}N^{cov}_d$).
    - Les quantités "Risk" sont des moyennes, de moyennes, de moyennes, ... est-ce bien serieux?
    - aussi, il ne semble pas y avoir de contrôle sur la qualité des regressions (pas de filtre sur le F-score, ni le respect des hypothèses sur les erreurs). Est ce que le risque n'englobe pas tout un tas de regression qui seraient recallées en faisant les choses correctement? 
    - Réduction d'hyperparamètres ? <br/>Avec misfinfo, entre 3 et 5min pour entrainement de modèles : 86400 runs de prévus (cf commande dans debbug_onyxia.txt). Ça fait donc au minimum 3*86400 = /60 = 259500min 4325heures /24 = 180jours. Avec ce simple jeu de donnée, et une seule carte GPU de 15Go, je mets donc minimum 180 jours, soit 6mois, pour tout faire tourner. C'est beaucoup trop, il faut donc qu'on réduire le nombre d'hyperparamètres.
    - Est ce qu'on garde la fenêtre de contexte? 
    - Est ce qu'on garde le dropout ? Quelle expertise avons nous? -> Uniquement dans la couche de classification
    - Est ce qu'on garde le pooling ?
    - Est ce qu'on évalue tous les hyperparamètres? random search vs grid search<br/>**Dépendant du test statistique réalisé à la fin** Combien de configs devrions nous tester? Si oui, il faut documenter ça correctement + laisser la possibilité dans le pipeline de rajouter ces hyperparmètres si besoin dans le futur<br/>+ enjeux ethique sur le cout de calcul, notre argument contre les LLMs<br/>Note: 708 régressions par config d'hyperparamètres = + $60.10^6$ couples de regressions (Axel)
    - Comment qu'on gère les regressions qui convergent pas ? on traite comme "non significative" ? comment on les comptes au niveau du type de l'erreur? _doit y avoir un moyen que ça converge tout le temps_
    - Au lieu d'une grid on pourrait avoir des valeurs qui dépendent du modèle (ex lr grand pour petit modele et lr plus petits pour grands modèles) ⚠️ génère des correlations entre variables independantes pour possible regression à posteriori)

Sugestions d'amélioration du code :
- Mettre enregistrement de modèles sur Onyxia en facultatif (si on veut dans le ca soù les hyper paramètres sont réduits)
- Renommer les folders voire certains scripts
- Fusion scripts Axel et Léo
- Logs enregistrés suffisants ?
