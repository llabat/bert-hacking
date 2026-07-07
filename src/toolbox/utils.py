"""
File with many miscellaneous functions 
"""
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
    """Read the configuration file (that should live in the folder "config_files"), 
    load it and check its format for the run to ... well ... run"""

    if configuration_file not in os.listdir("./config_files"):
        raise FileExistsError((f"File {configuration_file} does not exist in ./config_files\n"
            f"Found:\n{os.listdir('./config_files')}"))
    
    with open(f"./config_files/{configuration_file}") as file:
        config_json = json.load(file)
    if not isinstance(config_json, dict):
        raise TypeError((f"The config_json should be a dictionary.\n"
            f"Found ({type(config_json)}):\n{config_json}"))
    
    # Datasets checkup 
    if "datasets" not in config_json:
        raise KeyError((f"The configuration file must contain an object 'datasets'\n"
            f"Only found: {list(config_json.keys())}"))
    if not isinstance(config_json["datasets"], list):
        raise TypeError((f"The object 'datasets' should be a list.\n"
            f"Got ({type(config_json['datasets'])}):\n{config_json['datasets']}"))
    if  not np.array([isinstance(d, dict) for d in config_json["datasets"]]).all():
        raise TypeError(("The object 'datasets', must be a list of dictionaries."
            "At least one object within this list is not a dictionary"))
    columns_to_find_in_dict = [
        "name", 
        "filepath-train", 
        "filepath-predict", 
        "text_col", 
        "label_col", 
        "id_col", 
        "labels", 
        "filepath-metadata", 
        "columns-for-independant-variables"
    ]
    if not np.array([np.isin(columns_to_find_in_dict, list(d.keys())).all() 
        for d in config_json["datasets"]]).all():
        raise KeyError((f"All dictionaries in the object 'datasets' should contain "
            f"at least the following keys: {', '.join(columns_to_find_in_dict)}"
            "Some were not found."))
    
    # Parameters checkup
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
    """If provided, read the subsample file and check if the current loop config
    appears in the subsample file."""

    if not subsample_file:
        return True
    
    if subsample_file not in os.listdir("./config_files"):
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

def create_hash_from_config_loop(loop_config:LoopConfig)->str:
    """Takes a loop config and generate a hash. 
    Hash string: time + dataset_name + dichotomization_label"""
    s = str(time()) + f"-{loop_config.dataset_name}-{loop_config.dichotomization_label}"
    return create_hash_from_string(s)

def create_hash_from_string(s:str) -> str:
    """Generate a hash from a string"""
    h = hashlib.new('sha256')
    h.update(s.encode())
    return h.hexdigest()

def already_done(loop_config:LoopConfig):
    """Check if the config appears in the "saving_logs.json"."""
    with open("./results/saving_logs.json", "r") as file :
        saving_logs = json.load(file)
    
    for v in saving_logs.values(): 
        if loop_config == LoopConfig(**v):
            return True
    return False

def load_tokenizer(loop_config: LoopConfig):
    """Load tokenizer using the loop config's "model_name" attribute."""
    try: 
        return AutoTokenizer.from_pretrained(loop_config.model_name, trust_remote_code = True)
    except Exception as e:
        raise ValueError(f"Could not load the Tokenizer.\nError:{e}")
    
def get_device() -> device:
    """Select the device (cuda, mps or cpu) depending on what's available."""
    if cuda_available():
        empty_cache()
        return device("cuda")
    if mps_available():
        return device("mps")
    return device("cpu")

def clean():
    """Flush the device memory (cuda, mps or cpu)"""
    empty_cache()
    if cuda_available():
        synchronize()
        ipc_collect()
    gc_collect()
    print("Memory flushed")

def to_saving_logs(hash_: str, to_save: dict|None):
    """Save the output from the function single_run to the saving logs 
    ( located at ./results/saving_logs.json). If the hash already exists, the 
    content is overwritten.
    
    Potentially suboptimal."""
    if to_save is None : return
    with open("./results/saving_logs.json", "r") as file :
        saving_logs = json.load(file)

    # Overwrite 
    saving_logs[hash_] = to_save
    
    with open("./results/saving_logs.json", "w") as file:
        json.dump(saving_logs, file, ensure_ascii=True)

def aggregate_predictions(
    df : pd.DataFrame, 
    loop_config: LoopConfig 
) -> pd.DataFrame:
    """Takes in a dataframe of prediction (must contain "ID", "GS-LABEL", 
    "PRED-LABEL") using the provided aggregation strategy.
    Available strategies: 
    - THRESHOLD ([0.,1.]) : mean per article >= THRESHOLD
    - AT_LEAST (> 0): number of positive cases >= AT_LEAST
    return a dataframe with the following columns: "ID", "GS-LABEL", "PRED-LABEL"
    """
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
        raise ValueError(("criterion not provided. Received threshold: "
            f"{loop_config.THRESHOLD}; at_least: {loop_config.AT_LEAST}"))
    
    df_aggregated = df_aggregated.reset_index()
    df_aggregated[["GS-LABEL", "PRED-LABEL"]] = df_aggregated[["GS-LABEL", "PRED-LABEL"]].astype(int).replace(loop_config.id2label)
    return df_aggregated
    
def retrieve_checkpoint_number(s: str)->int:
    """Given a string that should look like "checkpoint-XXX", return (int(XXX))
    If the string pattern does not match, return -1"""
    if isinstance(s, str):
        try: 
            output = int(s.removeprefix("checkpoint-"))
            return output
        except: pass 
    return -1

def retrieve_trainer_logs(directory: str) -> dict:
    """Given a directory, find the latest checkpoint and reads the trainer state
    (if exists) and return the "log_history". """
    sorted_checkpoints = sorted(
        os.listdir(directory),
        key = retrieve_checkpoint_number
    )
    last_checkpoint = sorted_checkpoints[-1]
    try: 
        with open(f"{directory}/{last_checkpoint}/trainer_state.json", "r") as file:
            content = json.load(file)
    except: 
        content = {}
    return content.get("log_history", "failed retrieving the logs")

def send_notification(message : str = '') : 
    """Send an email when finished. Read the ./.env file that should contain the 
    following keys:
    - EMAIL_FROM
    - EMAIL_FROM_PWD
    - EMAIL_TO"""
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

def get_run_info_for_regression(saving_logs_filename: str) -> dict[str:dict]:
    """Read the saving logs (must live in the ./results directory) and retrieve 
    information to run the regresions loops"""

    if saving_logs_filename not in os.listdir("./results"):
        raise FileExistsError((f"File {saving_logs_filename} does not exist in ./results\n"
            f"Found:\n{os.listdir('./results')}"))
    
    with open(f"./results/{saving_logs_filename}") as file:
        saving_logs : dict[str:dict] = json.load(file)
    
    if not isinstance(saving_logs, dict):
        raise TypeError((f"The saving_logs should be a dictionary.\n"
            f"Found ({type(saving_logs)}):\n{saving_logs}"))
    if  not np.array([isinstance(d, dict) for d in saving_logs.values()]).all():
        raise TypeError(("The saving_logs, must be a dictionary of dictionaries."
            "At least one object within this dictionary is not a dictionary"))
    
    columns_to_find_in_dict = ["dataset_name","prediction-csv"]

    if not np.array([np.isin(columns_to_find_in_dict, list(d.keys())).all() 
        for d in saving_logs.values()]).all():
        raise KeyError((f"All dictionaries should contain "
            f"at least the following keys: {', '.join(columns_to_find_in_dict)}"
            "Some were not found."))
    
    output = {}
    for key_run in saving_logs:
        keys = saving_logs[key_run].keys()
        if "dataset_name" not in keys:
            raise KeyError(f"For run {key_run}, could not find 'dataset_name'")
        if "dichotomization_label" not in keys:
            raise KeyError(f"For run {key_run}, could not find 'dichotomization_label'")

        if "prediction-aggregated-csv" in keys:
            prediction_key = "prediction-aggregated-csv"
        elif "prediction-csv" in keys:
            prediction_key = "prediction-csv"
        else:
            raise KeyError(f"For run {key_run}, could not find 'prediction-aggregated-csv'"
                " nor 'prediction-csv'")
        
        output[key_run] =  {
            "dataset_name" : saving_logs[key_run]["dataset_name"],
            "dichotomization_label" : saving_logs[key_run]["dichotomization_label"],
            "prediction-filepath": saving_logs[key_run][prediction_key]
        }

    return output

def get_df_with_metadata(run_info: dict, datasets_config: list[dict]) -> pd.DataFrame:
    """Retrieve the predictions and metadata (from label ~ metadata) associated.
    Metadata are stored in a csv file, this csv must at least have matching 
    identifiers and contain the metadata columns as found in the config-loop file.
    
    Return a dataframe containing the following columns: ID, GS-LABEL, PRED-LABEL, 
    and all metadata columns."""
    predictions = pd.read_csv(run_info["prediction-filepath"])

    if not np.isin(["ID", "GS-LABEL", "PRED-LABEL"], predictions.columns).all():
        raise KeyError(f"The prediction file misses some necessary columns ('ID',"
            f" 'GS-LABEL', 'PRED-LABEL'). Found: {predictions.columns}")
    if not predictions["ID"].is_unique:
        raise ValueError("(predictions) The ID column is not unique, please "
            "aggregate the results before running the regression.")
    
    # Find the appropriate metadata file: 
    candidates = [d for d in datasets_config if d["name"] == run_info["dataset_name"]]

    if len(candidates) != 1:
        raise ValueError(f"Should find one candidate in datasets_config; Found "
            f"{len(candidates)} candidates.")
    
    metadata = pd.read_csv(candidates[0]["filepath-metadata"])
    metadata_columns = candidates[0]["columns-for-independant-variables"]
    if not np.isin(["ID",*metadata_columns], metadata.columns).all():
        raise ValueError(f"The metadata file should contain the following columns:"
            f"{', '.join(['ID', *metadata_columns])}. Found: {metadata.columns}")
    if not metadata["ID"].is_unique:
        raise ValueError("The metadata ID column is not unique.")
    if not np.isin(predictions["ID"], metadata["ID"]).all():
        raise ValueError("Some ID from predictions were not found in the metadata"
            " file. Cannot join the two dataframes.")
    output = (
        predictions
        .set_index("ID")
        .join(metadata.set_index("ID")[metadata_columns])
        .reset_index()
    )
    return output, metadata_columns

def save_errors(batch_run_hash: list[str], batch_to_save: list[str]) -> None:
    """Save a batch of outputs from the run_regression_and_assess_errors to 
    ./results/errors_save.json . Save will overwrite existing content with the 
    same run_hash"""
    with open("./results/errors_save.json") as file:
        all_ = json.load(file)
    for hash_, to_save in zip(batch_run_hash, batch_to_save):
        all_[hash_] = to_save 
    with open("./results/errors_save.json", "w") as file:
        json.dump(all_, file, ensure_ascii=True)

def ensure_no_na(o: list|dict) -> list[dict]:
    """Given a list or a dict, ensure there are no np.nan and replace items with
    None instead. This is used before saving results in a json file because np.nan
    are badly delt with."""
    if isinstance(o, list):
        out = []
        for el in o:
            if isinstance(el, list) or isinstance(el, dict):
                out += [ensure_no_na(el)]
            else: 
                try:    out += [None if np.isnan(el) else el]
                except: out += [el]
    elif isinstance(o, dict):
        out = {}
        for k,v in o.items():
            if isinstance(v, list) or isinstance(v, dict):
                out[k] = ensure_no_na(v)
            else: 
                try:    out[k] = None if np.isnan(v) else v
                except: out[k] = v

    else:
        out = o 
    return out


def regression_already_done(hash_: str) -> bool:
    """Check if a hash exists in the ./results/errors_save.json file."""
    with open("./results/errors_save.json") as file:
        keys = list(json.load(file).keys())
    return hash_ in keys
