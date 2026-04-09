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
    create_hash,
    already_done,
    dichotomize,
    load_tokenizer,
    get_max_tokens, 
    cap_max_length,
    sample_N_elements,
    split_ds,
    tokenize_dataset_dict,
    load_training_arguments,
    train_model,
    predict,
    to_saving_logs,
    clean
)

def single_run(
        df : pd.DataFrame,
        df_prediction: pd.DataFrame,
        label: str,
        task_name : str,
        path_train_dataset : str, 
        path_inference_dataset : str, 
        loop_config : dict[str:Any],
        **kwargs
    ) -> tuple[str, dict | None]: 
    """
    Input: 
        df : df for training
        df_prediction: df for inference
        label : label used for dichotomization (must be in df)

        task_name: string, doesn't matter, just need to be consistent throughout the loop, used for creating the hash
        path_train_dataset: string, doesn't need to exist per se, used for creating the hash
        path_inference_dataset: string, doesn't need to exist per se, used for creating the hash
    Data saved: 
        logs: save information in the loop_info log
        predictions: save the predictions as ./predictions_save/HASH.csv
    Output:
        hash_
        loop results: save everything in ./saving_logs.json
    """

    logger = CustomLogger("./custom_logs")

    TEST_MODE = kwargs.get("TEST_MODE", False)
    BATCH_SIZE = kwargs.get("BATCH_SIZE", 4)
    TOTAL_BATCH_SIZE = kwargs.get("TOTAL_BATCH_SIZE", 16)

    logger("START LOOP" + "#" * 91, skip_line="before")
    logger(f"Starting Loop on task {task_name} {'(TEST_MODE)' if TEST_MODE else ''} and config {loop_config}")

    loop_ID = {
        **loop_config, 
        "task_name": task_name,
        "dataset_train": path_train_dataset,
        "dataset_predict": path_inference_dataset,
    }
    # Initialise outputs
    hash_, to_save = create_hash(**loop_ID), None

    if already_done(hash_): 
        logger("Loop was already completed")
    else: 
        tokenizer, ds_loop, dsd_loop, ds_pred, predictions, model = (None,) * 6
        try: 
            dichotomized_df, label2id, id2label = dichotomize(df, label)
            dichotomized_df_prediction, _, _ = dichotomize(df_prediction, label)
            
            # Prepare tokenizer: model_name, context_window_rel_to_max
            tokenizer = load_tokenizer(**loop_config)

            max_n_tokens = get_max_tokens(dichotomized_df["TEXT"], tokenizer)
            # ⚠️ How do we deal with entries longer than the model's context window
            max_length_capped = cap_max_length(max_n_tokens=max_n_tokens, **loop_config)
            tokenization_parameters = {
                'padding' : 'max_length',
                'truncation' : True,
                'max_length' : max_length_capped
            }

            # Prepare dataset: N_train, train_eval_test_ratios
            ds_loop: Dataset = sample_N_elements(dichotomized_df, SEED = 0, **loop_config)
            dsd_loop : DatasetDict = split_ds(ds_loop, SEED = 0, **loop_config)
            dsd_loop = dsd_loop.map(lambda row: tokenize_dataset_dict(row,label2id, tokenizer,tokenization_parameters))

            # Prepare model: model_name
            model = AutoModelForSequenceClassification.from_pretrained(
                loop_config["model_name"],
                num_labels = len(label2id),
                id2label   = id2label,
                label2id   = label2id,
            )

            # Prepare trainer: learning_rate, weight_decay, warmup_ratio, dropout
            output_dir = kwargs.get("output_dir","./models/current")
            training_args = load_training_arguments(
                output_dir=output_dir, 
                batch_size_device=BATCH_SIZE, 
                total_batch_size=TOTAL_BATCH_SIZE, 
                **loop_config
                #TODO check for dropout
            )

            logger("Everything loaded — Start training")

            tstart = time()
            best_model_checkpoint = train_model(
                model, 
                training_args,
                dsd_loop,
                TEST_MODE
            )
            logger(f"Training done in {time() - tstart:.0f}s - best model checkpoint: {best_model_checkpoint}")

            # Reload model from checkpoint
            model = AutoModelForSequenceClassification.from_pretrained(best_model_checkpoint)
            predictions : pd.DataFrame = predict(model, dsd_loop["test"], batch_size=BATCH_SIZE, id2label=id2label)
            score_on_test = f1_score(y_true = predictions["GS-LABEL"], y_pred = predictions["PRED-LABEL"], average="macro",zero_division=np.nan)
            logger(f"Evaluate best model. Score: {score_on_test}")

            # Predict on full data
            ds_pred = Dataset.from_pandas(dichotomized_df_prediction)
            
            if TEST_MODE : ds_pred = ds_pred.select(range(50))

            ds_pred = ds_pred.map(lambda row: tokenize_dataset_dict(
                row,
                label2id, 
                tokenizer,
                tokenization_parameters
            ))

            logger("Start Inference")
            tstart = time()
            predictions : pd.DataFrame = predict(model, ds_pred, batch_size=BATCH_SIZE, id2label=id2label)
            logger(f"Inference done in {time() - tstart:.0f} s")

            if not TEST_MODE:
                predictions.to_csv(f"./predictions_save/{hash_}.csv")
                to_save = {
                    **loop_ID,
                    "effective_context_window": max_length_capped,
                    "score_on_test": score_on_test,
                    "prediction-csv": f"./predictions_save/{hash_}.csv"
                }                

                logger(f"Information saved with hash {hash_}")
                
        except Exception as e: 
            logger("Loop failed")
            logger(f"Error during loop {hash_}\n\n{e}\n\n", type="ERRORS")
        finally: 
                del tokenizer, ds_loop, dsd_loop, ds_pred, predictions, model
                clean() 
    
    logger("END LOOP" + "#" * 92)
    return hash_, to_save

if __name__=="__main__":
    single_run(**{}) # Implement the python -u single_run.py XXX