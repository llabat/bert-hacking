import json 

from itertools import product
import pandas as pd 

from toolbox import extract_hyperparameters, sanitize_df, to_saving_logs
from single_run import single_run

TEST_MODE = True

def loop():
    with open("./config_files/config-loop.json") as file:
        config_json = json.load(file)

    parameter_names, parameters_values = extract_hyperparameters(config_json)
    for dataset_info in config_json["datasets"]:
        df = pd.read_csv(dataset_info["filepath-train"])
        df = sanitize_df(df, **dataset_info)
        labels = list(df["LABEL"].unique())
        
        df_prediction = pd.read_csv(dataset_info["filepath-predict"])
        df_prediction = sanitize_df(df_prediction, **dataset_info)
        for label in labels: 
            task_name = f"{dataset_info['name']}-{label}"
            for local_config in product(*parameters_values):
                loop_config = {n: v for n,v in zip(parameter_names,local_config)}
                hash_, to_save = single_run(
                    df, 
                    df_prediction, 
                    label, 
                    task_name, 
                    dataset_info["filepath-train"], 
                    dataset_info["filepath-predict"], 
                    loop_config, 
                    TEST_MODE = TEST_MODE
                )
                to_saving_logs(hash_, to_save)

                break
            break
        break 

if __name__ == "__main__":
    loop()
