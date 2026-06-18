import json
from itertools import product
import numpy as np 
CHOICE_SEED = 31851920
N_sample_per_n_annotated_and_model_name = 60
from toolbox import LoopConfig


GOLD_EXPLORATION_SPACE_ENGLISH = {
    "N_annotated": [500, 1000, 1500, 2000],
    "splits_ratio": [
        [80, 10, 10],
        [70, 15, 15],
        [50, 10, 40]
    ],
    "sampling_method" : [
        {"balance":"random"},
        {"balance":0.25},
        {"balance":0.50}
    ],
    "model_name": [
        "google-bert/bert-base-uncased",
        "FacebookAI/roberta-base",
        "microsoft/deberta-v2-xlarge",
        "answerdotai/ModernBERT-base"
    ],
    "learning_rate":[5e-4,1e-4,1e-5, 2e-5, 5e-5],
    "weight_decay":[0.0,0.01,0.03,0.1], 
    "batch_size":[8, 16, 32], 
}

configs_to_do = []
i = 0
for N_annotated, model_name in product(GOLD_EXPLORATION_SPACE_ENGLISH["N_annotated"], GOLD_EXPLORATION_SPACE_ENGLISH["model_name"]):
    all_configs = [
        config for config in product(*[
            GOLD_EXPLORATION_SPACE_ENGLISH["splits_ratio"],
            GOLD_EXPLORATION_SPACE_ENGLISH["sampling_method"],
            GOLD_EXPLORATION_SPACE_ENGLISH["learning_rate"],
            GOLD_EXPLORATION_SPACE_ENGLISH["weight_decay"],
            GOLD_EXPLORATION_SPACE_ENGLISH["batch_size"],
        ])
    ]

    rng = np.random.default_rng(seed=CHOICE_SEED * i )
    rng.shuffle(all_configs)
    
    chosen = all_configs[:N_sample_per_n_annotated_and_model_name]

    configs_to_do += [
        {
            "N_annotated": N_annotated,
            "model_name": model_name,
            "splits_ratio" :    c[0],
            "sampling_method" : c[1],
            "learning_rate" :   c[2],
            "weight_decay" :    c[3],
            "batch_size" :      c[4],
        }
        for c in chosen
    ]
    i+=1
    
with open("./config_files/sample-configurations.json","w") as file:
    json.dump(configs_to_do, file, ensure_ascii=True)