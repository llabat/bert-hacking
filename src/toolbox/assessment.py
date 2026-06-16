import json 
import os

from itertools import product
import numpy as np 

from . import LoopConfig, get_config

def assess(
    dataset_name: str, 
    dichotomization_label: str,
    configuration_filename: str, 
    result_filename: str,
    subsample_filename: str|None=None, 
) -> list[int]:
    """Given a config file, and possibly a subsample file, assess the share of the 
    configuration found in the results"""

    # Create configurations_to_assess
    if subsample_filename: 
        # Subsample is a list of configurations 
        # {
        #     "N_annotated": 500,
        #     "model_name": "google-bert/bert-base-uncased",
        #     ...
        # },
        # See sample-configurations.py
        if subsample_filename not in os.listdir("./config_files"):
            raise FileExistsError((f"Subsample file {subsample_filename} not found"
                f"in ./config_files"))
        with open(f"./config_files/{subsample_filename}") as file:
            configurations_to_assess = [
                LoopConfig(dataset_name, dichotomization_label, **config)
                for config in json.load(file)
            ]
    else: 
        if configuration_filename not in os.listdir("./config_files"):
            raise FileExistsError((f"Configuration file {configuration_filename} not " 
                f"found in ./config_files"))
        # Read the configuration file and create the object configurations_to_assess
        _, parameter_names,parameters_values = get_config(configuration_filename)
        configurations_to_assess = [
            LoopConfig(dataset_name,dichotomization_label,
            **{n: v for n,v in zip(parameter_names,local_config)}) 
            for local_config in product(*parameters_values)
        ]
    
    # Create configurations_computed
    if result_filename not in os.listdir("./results"):
        raise FileExistsError((f"Configuration file {result_filename} not " 
            f"found in ./results"))
    with open(f"./results/{result_filename}") as file:
        configurations_computed = [
            LoopConfig(**config)
            for config in json.load(file).values()
        ]
    
    # compare the two 
    output = []
    for config in configurations_to_assess:
        output += [int(config in configurations_computed)]
    return output

def get_report(
    configuration_filename: str, 
    result_filename: str,
    subsample_filename: str|None=None, 
) -> str:
    report =  ""
    datasets_config, _, _ = get_config(configuration_filename)
    for dataset_info in datasets_config:
        for label in dataset_info["labels"]:           
            output = assess(
                dataset_name= dataset_info['name'], 
                dichotomization_label = label,
                configuration_filename = configuration_filename,
                result_filename = result_filename,
                subsample_filename = subsample_filename,   
            )
            report += f"{dataset_info['name']} x {label} : {np.sum(output)}  / {len(output)} ({np.mean(output) * 100:.0f} %)\n"
    return report

if __name__ == "__main__":
    get_report("config-loop.json", "saving_logs.json", "sample-configurations.json")