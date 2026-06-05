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
    sample_N_documents,
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

OVERLAP = 50
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
    loop_config.set_fixed_parameters(OVERLAP, AT_LEAST, THRESHOLD)
    run_timer = {}

    # Use time as hash
    hash_, logs_to_save = create_hash(loop_config), None
    tokenizer, ds_loop, dsd_loop, ds_pred, predictions, model = (None,) * 6
    try: 
        # Dichotomization: dichotomization_label
        run_timer["preprocess_data"] = time()
        dichotomized_df, label2id, id2label = dichotomize(df, loop_config)
        loop_config.set_label_id_mapper(label2id, id2label)
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
        ds_loop, effective_distrib = sample_N_documents(dichotomized_df, label2id, loop_config)
        logger(f"Sample {ds_loop['ID'].nunique()} documents; corresponds to {len(ds_loop)} rows")
        logger(f"Effective distribution: {effective_distrib} — requested : {loop_config.sampling_method}")
        dsd_loop : DatasetDict = split_ds(ds_loop, loop_config)
        dsd_loop = dsd_loop.map(lambda row: tokenize_dataset_dict(row,label2id, tokenizer,tokenization_parameters))
        
        run_timer["preprocess_data"] = time() - run_timer["preprocess_data"] 
        
        # Prepare model: model_name
        run_timer["training"] = time()
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
        best_model_checkpoint, trainer_logs = train_model(model, training_args,dsd_loop,loop_config)
        logger(f"Training done in {time() - tstart:.0f}s - best model checkpoint: {best_model_checkpoint}")
        run_timer["training"] = time() - run_timer["training"] 
        
        # Reload model from checkpoint: test_mode, device_batch_size
        run_timer["evaluation"] = time()
        model = AutoModelForSequenceClassification.from_pretrained(best_model_checkpoint)
        predictions_on_test : pd.DataFrame = predict(model, dsd_loop["test"], loop_config, id2label=id2label)
        predictions_on_test_aggregated : pd.DataFrame = aggregate_predictions(predictions_on_test, loop_config)
        score_on_test = f1_score(
            y_true = predictions_on_test_aggregated["GS-LABEL"], 
            y_pred = predictions_on_test_aggregated["PRED-LABEL"], 
            average="macro",
            zero_division=np.nan
        )
        logger(f"Evaluate best model. Score: {score_on_test}")
        run_timer["evaluation"] = time() - run_timer["evaluation"]

        # Predict on full data
        run_timer["prediction"] = time() 
        ds_pred = Dataset.from_pandas(dichotomized_df_prediction)
        ds_pred = ds_pred.map(lambda row: tokenize_dataset_dict(row,label2id, tokenizer,tokenization_parameters))

        logger("Start Inference")
        tstart = time()
        predictions : pd.DataFrame = predict(model, ds_pred, loop_config, id2label=id2label)
        logger(f"Inference done in {time() - tstart:.0f} s")
        run_timer["prediction"] = time() - run_timer["prediction"]

        if not loop_config.test_mode:
            run_timer["saving_predictions"] = time()
            predictions_on_test.to_csv(f"./predictions_save/{hash_}-on-test.csv")
            predictions.to_csv(f"./predictions_save/{hash_}.csv")
            run_timer["saving_predictions"] = time() - run_timer["saving_predictions"]
            logs_to_save = {
                **loop_config.to_dict(),
                "effective_context_window": max_length_capped,
                "score_on_test": score_on_test,
                "prediction-on-test-csv": f"./predictions_save/{hash_}-on-test.csv",
                "prediction-csv": f"./predictions_save/{hash_}.csv",
                "effective_distrib": effective_distrib,
                "trainer-logs": trainer_logs,
            }                
            if "ID_CHUNK" in predictions.columns:
                run_timer["saving_predictions_aggregated"] = time()
                predictions_on_test_aggregated.to_csv(f"./predictions_save/{hash_}-on-test-aggregated.csv")
                (
                    aggregate_predictions(predictions, loop_config)
                    .to_csv(f"./predictions_save/{hash_}-aggregated.csv")
                )
                logs_to_save["prediction-on-test-aggregated-csv"] = f"./predictions_save/{hash_}-on-test-aggregated.csv"
                logs_to_save["prediction-aggregated-csv"] = f"./predictions_save/{hash_}-aggregated.csv"
                logs_to_save["aggregation-strategy"] = {"at_least": loop_config.AT_LEAST, "threshold":loop_config.THRESHOLD}
                run_timer["saving_predictions_aggregated"] = time() - run_timer["saving_predictions_aggregated"] 
            logs_to_save["run_timer"] = run_timer
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