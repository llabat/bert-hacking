import hashlib 
import json 
from gc import collect as gc_collect

from transformers import AutoTokenizer
from torch import device
from torch.cuda import is_available as cuda_available
from torch.cuda import empty_cache, synchronize, ipc_collect
from torch.backends.mps import is_available as mps_available

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

def create_hash(**kwargs)->str:
    s = "|".join([f"{k}:{v}" for k,v in kwargs.items()])
    h = hashlib.new('sha256')
    h.update(s.encode())
    return h.hexdigest()

def already_done(hash_:str):
    """check if the hash exists in the saving logs."""
    with open("./saving_logs.json", "r") as file :
        saving_logs = json.load(file)
    return hash_ in saving_logs

def load_tokenizer(model_name: str, **kwargs):
    try: 
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
    except Exception as e:
        raise ValueError("Could not load the Tokenizer.\nErreur:{e}")
    
def pick_seed(**kwargs)->int:
    if "SEED" in kwargs:
        return kwargs["SEED"]
    else:
        return 42
    
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
    with open("./saving_logs.json", "r") as file :
        saving_logs = json.load(file)

    # Overwrite 
    saving_logs[hash_] = to_save
    
    with open("./saving_logs.json", "w") as file:
        json.dump(saving_logs, file, ensure_ascii=True, indent=4)