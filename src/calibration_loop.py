"""
Run calibration loop to ensure that the configuration does not alter the results
"""
import os
import json
import pandas as pd 

from single_run import single_run
from toolbox import sanitize_df, LoopConfig

datafile = "./data/ideology_news-dataset_for_training.csv"
loop_config = LoopConfig(
    dataset_name="calibration-loop",
    dichotomization_label="left",

    N_annotated = 1000, 
    sampling_method = {"balance": 0.5},
    splits_ratio = [70, 15, 15],

    model_name = "google-bert/bert-base-uncased", 
    n_epochs = 2, 
    learning_rate = 1e-5,
    weight_decay = .3,
    batch_size = 16, 

    output_dir = "./models/calibration", 
    seed = 31851920,
    device_batch_size = 4, 
    device_batch_size_for_prediction = 8,
)

df = pd.read_csv(datafile)
df = sanitize_df(df, text_col = "content", label_col = "bias_text", id_col="ID")
df_prediction = df.iloc[:2].copy()

_, logs_to_save = single_run(df, df_prediction, loop_config)

os.system(f"mv {logs_to_save['prediction-csv']} ./calibration/predictions.csv")
logs_to_save["prediction-csv"] = "./calibration/predictions.csv"
with open("./calibration/logs.json", 'w') as file:
    json.dump(logs_to_save, file, indent=4, ensure_ascii=True)