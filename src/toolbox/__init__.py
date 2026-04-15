import os 
from pathlib import Path 
import json


os.environ["TOKENIZERS_PARALLELISM"] = "false"

if not Path("./data").is_dir(): os.mkdir("./data")
if not Path("./models").is_dir(): os.mkdir("./models")
if not Path("./config_files").is_dir(): os.mkdir("./config_files")
if not Path("./results").is_dir(): os.mkdir("./results")
if not Path("./custom_logs").is_dir(): os.mkdir("./custom_logs")
if not Path("./predictions_save").is_dir(): os.mkdir("./predictions_save")
if not Path("./results/saving_logs.json").exists():
    with open("./results/saving_logs.json", "w") as file:
        json.dump({}, file)

from .CustomLogger import *
from .utils import *
from .preprocess import *
from .model import *
