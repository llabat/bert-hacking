from time import time
import json
from typing import Any 

from datasets import Dataset, DatasetDict
import numpy as np 
import pandas as pd 
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification

from toolbox import (
    CustomLogger, 
    LoopConfig,
    create_hash,
    dichotomize,
    load_tokenizer,
    get_max_tokens, 
    sample_N_elements,
    split_ds,
    tokenize_dataset_dict,
    load_training_arguments,
    train_model,
    predict,
    clean, 
    sanitize_df,
    cap_max_length,
    aggregate_predictions
)

AT_LEAST = 1
THRESHOLD = None

def single_run(
        df : pd.DataFrame,
        df_prediction: pd.DataFrame,
        loop_config : LoopConfig,
    ) -> tuple[str, dict | None]: 
    """
    Input: 
        df: df for training
        df_prediction: df for inference
        loop_config: LoopConfig object containing all necessary information to train the model
    
    Data saved:
        predictions: save the predictions as ./predictions_save/HASH.csv
    Output:
        hash_
        logs_to_save: the necessary information to reproduce the loop and the 
            output (F1, hash and path to predictions csv)
    """

    logger = CustomLogger("./custom_logs")

    # Use time as hash
    hash_, logs_to_save = create_hash(loop_config), None
    tokenizer, ds_loop, dsd_loop, ds_pred, predictions, model = (None,) * 6
    try: 
        # Dichotomization: dichotomization_label
        dichotomized_df, label2id, id2label = dichotomize(df, loop_config)
        dichotomized_df_prediction, _, _ = dichotomize(df_prediction, loop_config)
        
        # Prepare tokenizer: model_name
        tokenizer = load_tokenizer(loop_config)

        max_n_tokens = get_max_tokens(dichotomized_df["TEXT"], tokenizer)
        max_length_capped = cap_max_length(max_n_tokens, loop_config)
        tokenization_parameters = {
            'padding' : 'max_length',
            'truncation' : True,
            'max_length' : max_length_capped
        }

        # Prepare dataset: N_annotated, splits_ratio, seed
        ds_loop, effective_distrib = sample_N_elements(dichotomized_df, label2id, loop_config)
        logger(f"Sample {ds_loop['ID'].nunique()} documents; corresponds to {len(ds_loop)} rows")
        logger(f"Effective distribution: {effective_distrib} — requested : {loop_config.sampling_method}")
        dsd_loop : DatasetDict = split_ds(ds_loop, loop_config)
        dsd_loop = dsd_loop.map(lambda row: tokenize_dataset_dict(row,label2id, tokenizer,tokenization_parameters))
        
        # Prepare model: model_name
        model = AutoModelForSequenceClassification.from_pretrained(
            loop_config.model_name,
            num_labels = len(label2id),
            id2label   = id2label,
            label2id   = label2id,
        )

        # Prepare trainer: n_epochs, learning_rate, weight_decay, batch_size, device_batch_size, output_dir, seed
        training_args = load_training_arguments(loop_config)

        logger("Everything loaded — Start training")

        # Launch training: test_mode
        tstart = time()
        best_model_checkpoint = train_model(model, training_args,dsd_loop,loop_config)
        logger(f"Training done in {time() - tstart:.0f}s - best model checkpoint: {best_model_checkpoint}")
        
        # Reload model from checkpoint: test_mode, device_batch_size
        model = AutoModelForSequenceClassification.from_pretrained(best_model_checkpoint)
        predictions : pd.DataFrame = predict(model, dsd_loop["test"], loop_config, id2label=id2label)
        predictions_aggregated : pd.DataFrame = aggregate_predictions(predictions, label2id, id2label, THRESHOLD, AT_LEAST)
        score_on_test = f1_score(
            y_true = predictions_aggregated["GS-LABEL"], 
            y_pred = predictions_aggregated["PRED-LABEL"], 
            average="macro",
            zero_division=np.nan
        )
        logger(f"Evaluate best model. Score: {score_on_test}")

        # Predict on full data
        ds_pred = Dataset.from_pandas(dichotomized_df_prediction)
        ds_pred = ds_pred.map(lambda row: tokenize_dataset_dict(row,label2id, tokenizer,tokenization_parameters))

        logger("Start Inference")
        tstart = time()
        predictions : pd.DataFrame = predict(model, ds_pred, loop_config, id2label=id2label)
        logger(f"Inference done in {time() - tstart:.0f} s")

        if not loop_config.test_mode:
            predictions.to_csv(f"./predictions_save/{hash_}.csv")
            logs_to_save = {
                **loop_config.to_dict(),
                "effective_context_window": max_length_capped,
                "score_on_test": score_on_test,
                "prediction-csv": f"./predictions_save/{hash_}.csv",
                "effective_distrib": effective_distrib
            }                
            if "ID_CHUNK" in predictions.columns:
                (
                    aggregate_predictions(predictions, label2id, id2label, THRESHOLD, AT_LEAST)
                    .to_csv(f"./predictions_save/{hash_}-aggregated.csv")
                )
                logs_to_save["prediction-aggregated-csv"] = f"./predictions_save/{hash_}-aggregated.csv"

            logger(f"Information saved with hash {hash_}")
            
    except Exception as e: 
        logger("Loop failed")
        logger(f"Error during loop {hash_}\n{loop_config}\n{e}\n\n", type="ERRORS")
    finally: 
            del tokenizer, ds_loop, dsd_loop, ds_pred, predictions, model
            clean() 

    return hash_, logs_to_save

if __name__=="__main__":
    # Implement the python -u single_run.py XXX

    df = pd.read_csv("./data/ideology_news-stratified_year_balanced.csv")
    df = sanitize_df(df, text_col = "content", label_col = "bias_text", id_col="ID")
    df_prediction = df.copy()
    loop_config = LoopConfig(task_name = "TASK-left", dichotomization_label="left", test_mode=True)

    print(single_run(df, df_prediction, loop_config))