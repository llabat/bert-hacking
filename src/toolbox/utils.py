import os
import hashlib 
import json 
from gc import collect as gc_collect
from time import time

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from torch import device
from torch.cuda import is_available as cuda_available
from torch.cuda import empty_cache, synchronize, ipc_collect
from torch.backends.mps import is_available as mps_available

from . import LoopConfig

def extract_hyperparameters(config_json: dict):
    """
    extract the names and values of hyperparameters
    """
    parameter_names = [
        *[name for name in config_json["data-hyperparameters"].keys()],
        *[name for name in config_json["model-hyperparameters"].keys()],
    ]
    parameters_values = [
        *[values for values in config_json["data-hyperparameters"].values()],
        *[values for values in config_json["model-hyperparameters"].values()],
    ]
    return parameter_names, parameters_values

def create_hash(loop_config:LoopConfig)->str:
    s = str(time()).replace(".","") + f"-{loop_config.task_name}"
    h = hashlib.new('sha256')
    h.update(s.encode())
    return h.hexdigest()

def already_done(loop_config:LoopConfig):
    """check if the config exists in the saving logs."""
    with open("./results/saving_logs.json", "r") as file :
        saving_logs = json.load(file)
    check_list = [
        loop_config == LoopConfig(**v)
        for v in saving_logs.values()
    ]
    return np.array(check_list).any()

def load_tokenizer(loop_config: LoopConfig):
    try: 
        return AutoTokenizer.from_pretrained(loop_config.model_name, trust_remote_code = True)
    except Exception as e:
        raise ValueError(f"Could not load the Tokenizer.\nErreur:{e}")
    
def get_device() -> device:
    if cuda_available():
        empty_cache()
        return device("cuda")
    if mps_available():
        return device("mps")
    return device("cpu")

def clean():
    """
    """
    empty_cache()
    if cuda_available():
        synchronize()
        ipc_collect()
    gc_collect()
    print("Memory flushed")

def to_saving_logs(hash_: str, to_save: dict|None):
    if to_save is None : return
    with open("./results/saving_logs.json", "r") as file :
        saving_logs = json.load(file)

    # Overwrite 
    saving_logs[hash_] = to_save
    
    with open("./results/saving_logs.json", "w") as file:
        json.dump(saving_logs, file, ensure_ascii=True, indent=4)

def aggregate_predictions(
    df : pd.DataFrame, 
    label2id: dict, 
    id2label:dict, 
    threshold : float|None = None, 
    at_least : int|None = None 
) -> str:
    """"""
    df = df.copy().reset_index()
    df = df[["ID", "GS-LABEL", "PRED-LABEL"]].set_index("ID").replace(label2id).reset_index()
    if isinstance(threshold, float): 
        df_aggregated = (
            df
            .groupby("ID")
            .agg("mean")
        )
        df_aggregated = df_aggregated >= threshold
    elif isinstance(at_least, int):
        df_aggregated = (
            df
            .groupby("ID")
            .agg("sum")
        )
        df_aggregated = df_aggregated >= at_least
    else:
        raise ValueError(f"criterion not provided. Received threshold: {threshold}; at_least: {at_least}")
    df_aggregated = df_aggregated.astype(int).replace(id2label).reset_index()
    return df_aggregated.set_index("ID")

def retrieve_trainer_logs(directory: str) -> dict:
    """"""
    sorted_checkpoints = sorted(
        os.listdir(directory),
        key = lambda checkpoint : int(checkpoint.removeprefix("checkpoint-"))
    )
    last_checkpoint = sorted_checkpoints[-1]
    with open(f"{directory}/{last_checkpoint}/trainer_state.json", "r") as file:
        content = json.load(file)
    return content.get("log_history", "failed retrieving the logs")
