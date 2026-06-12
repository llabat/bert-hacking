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

from email.message import EmailMessage
import ssl
import smtplib

from . import LoopConfig

def get_config(configuration_file: str) -> tuple[list[dict], list[str], list]:
    """"""
    if not configuration_file in os.listdir("./config_files"):
        raise FileExistsError((f"File {configuration_file} does not exist in ./config_files\n"
            f"Found:\n{os.listdir('./config_files')}"))
    
    with open(f"./config_files/{configuration_file}") as file:
        config_json = json.load(file)
    
    if not isinstance(config_json, dict):
        raise TypeError((f"The config_json should be a dictionary.\n"
            f"Found ({type(config_json)}):\n{config_json}"))
    if "datasets" not in config_json:
        raise KeyError((f"The configuration file must contain an object 'datasets'\n"
            f"Only found: {list(config_json.keys())}"))
    if not isinstance(config_json["datasets"], list):
        raise TypeError((f"The object 'datasets' should be a list.\n"
            f"Got ({type(config_json['datasets'])}):\n{config_json['datasets']}"))
    if  not np.array([isinstance(d, dict) for d in config_json["datasets"]]).all():
        raise TypeError((f"The object 'datasets', must be a list of dictionaries."
            f"At least one object within this list is not a dictionary"))
    if not np.array([
        np.isin(["name", "filepath-train", "filepath-predict", "text_col", "label_col", "id_col"], list(d.keys())).all()
        for d in config_json["datasets"]
    ]).all():
        raise KeyError((f"All dictionaries in the object 'datasets' should contain "
            "at least the following keys: 'name', 'filepath-train', 'filepath-predict'"
            " 'text_col', 'label_col', 'id_col'. Some were not found."))
    if "parameters" not in config_json:
        raise KeyError((f"The configuration file must contain an object 'parameters'\n"
            f"Only found: {list(config_json.keys())}"))
    if not isinstance(config_json["parameters"], dict):
        raise TypeError(("The object 'parameters' should be a dictionary.\n"
            f"Got: ({type(config_json['parameters'])})\n{config_json['parameters']}"))
    return (
        config_json["datasets"],
        list(config_json["parameters"].keys()),
        list(config_json["parameters"].values()),
    )

def in_subsample(
    loop_config: LoopConfig, 
    dataset_name:str, 
    dichotomization_label:str, 
    subsample_file: str|None
)->bool:
    if not subsample_file:
        return True
    if not subsample_file in os.listdir("./config_files"):
        raise FileExistsError((f"File {subsample_file} does not exist in ./config_files\n"
            f"Found:\n{os.listdir('./config_files')}"))
    
    with open(f"./config_files/{subsample_file}") as file:
        subsample = json.load(file)

    if not isinstance(subsample, list):
        raise TypeError((f"The subsample should be a list of configurations.\n"
            f"Found ({type(subsample)}):\n{subsample}"))
    
    ds_info={"dataset_name":dataset_name, "dichotomization_label":dichotomization_label}
    for config in subsample:
        try: 
            if LoopConfig(**ds_info,**config) == loop_config:
                return True 
        except Exception as e:
            print(config)
            print(e)
            raise ValueError((f"A configuration in the subsample file {subsample_file}"
                f" was invalid."))
    return False

def create_hash(loop_config:LoopConfig)->str:
    s = str(time()).replace(".","") + f"-{loop_config.dataset_name}-{loop_config.dichotomization_label}"
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
    loop_config: LoopConfig 
) -> str:
    """"""
    df = df.copy().reset_index()
    df[["GS-LABEL", "PRED-LABEL"]] = df[["GS-LABEL", "PRED-LABEL"]].replace(loop_config.label2id)
    if isinstance(loop_config.THRESHOLD, float): 
        df_aggregated = (
            df
            [["ID", "GS-LABEL", "PRED-LABEL"]] # Exclude ID_CHUNK
            .groupby("ID")
            .agg("mean")
        )
        df_aggregated = df_aggregated >= loop_config.THRESHOLD
    elif isinstance(loop_config.AT_LEAST, int):
        df_aggregated = (
            df
            [["ID", "GS-LABEL", "PRED-LABEL"]] # Exclude ID_CHUNK
            .groupby("ID")
            .agg("sum")
        )
        df_aggregated = df_aggregated >= loop_config.AT_LEAST
    else:
        raise ValueError(f"criterion not provided. Received threshold: {loop_config.THRESHOLD}; at_least: {loop_config.AT_LEAST}")
    df_aggregated = df_aggregated.reset_index()
    df_aggregated[["GS-LABEL", "PRED-LABEL"]] = df_aggregated[["GS-LABEL", "PRED-LABEL"]].astype(int).replace(loop_config.id2label)
    return df_aggregated
    
def retrieve_checkpoint_number(s: str)->int:
    """"""
    if isinstance(s, str):
        try: 
            output = int(s.removeprefix("checkpoint-"))
            return output
        except: pass 
    return -1

def retrieve_trainer_logs(directory: str) -> dict:
    """"""
    sorted_checkpoints = sorted(
        os.listdir(directory),
        key = retrieve_checkpoint_number
    )
    last_checkpoint = sorted_checkpoints[-1]
    with open(f"{directory}/{last_checkpoint}/trainer_state.json", "r") as file:
        content = json.load(file)
    return content.get("log_history", "failed retrieving the logs")

def send_notification(message : str = '') : 
    """send an email when finished"""
    try: 
        from dotenv import load_dotenv
        load_dotenv()
    except: pass 
    EMAIL_FROM = os.environ.get("EMAIL_FROM")
    EMAIL_TO = os.environ.get("EMAIL_TO")
    EMAIL_FROM_PWD= os.environ.get("EMAIL_FROM_PWD")

    if not(EMAIL_FROM and EMAIL_TO and EMAIL_FROM_PWD): print("Mail not sent"); return 

    subj = "Onyxia run — stopped"
    body = (f"Loop Stopped\n{message}")
    em = EmailMessage()
    em["From"] = EMAIL_FROM
    em["To"] = EMAIL_TO
    em["Subject"] = subj
    em.set_content(body)

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp : 
        print(smtp.login(EMAIL_FROM,EMAIL_FROM_PWD))
        print(smtp.sendmail(EMAIL_FROM,EMAIL_TO, em.as_string()))
