"""
This file defines 2 functions as well as a main loop for launching a single run.

# single_run
    Function used for formatting data, fine-tuning a model, evaluate the best checkpoint
    and use this best checkpoint for prediction on an inference dataframe. 
    It returns a dictionary with many metadata.

# single_run_dummy
    A function creating a dummy and return a dictionnary with the same format as 
    single run for debugging.

# Main loop
    TODO: implement
    should retrieve some parameters and launch a single run.
"""
from time import time

from datasets import Dataset
import numpy as np 
import pandas as pd 
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification

from toolbox import (
    CustomLogger, 
    LoopConfig,
    create_hash_from_config_loop,
    dichotomize,
    sample_N_documents,
    tokenize_chunk_pad,
    split_ds,
    load_training_arguments,
    train_model,
    predict,
    clean, 
    sanitize_df,
    aggregate_predictions, 
)

def single_run(
        df_training  : pd.DataFrame,
        df_prediction: pd.DataFrame,
        loop_config : LoopConfig,
    ) -> tuple[str, dict | None]: 
    """Main function to perform the following tasks:
    - Format data: Dichotomization, sample the required number of documents,
        tokenized, chunk and pad texts and split the dataframe in a train, eval 
        and test splits
    - Fine tune a model: Load a base model, the training arguments and start the 
        finetuning process.
    - Evaluate the best checkpoint: Load the best model checkpoint, predict the 
    labels for the test split and evaluate the macro F1 score
    - Use the best checkpoint for prediction: format the dataframe used for 
        inference (dichotomize, tokenize, chunk and pad texts), predict on this 
        dataset.
    - Save results and return metadata:save the predictions on the test set and 
        inference set (aggregated and non-aggregated when available). 
        Metadata: 
            - unique run hash                                           | STR
            - Configuration (defined in LoopConfig.to_dict method)         | Any
            - effective_context_window_for_training                     | int
            - effective_context_window_for_inference                    | int
            - score_on_test                                             | float
            - prediction-on-test-csv      + aggregated when necessary   | str
            - prediction-csv (inference)  + aggregated when necessary   | str
            - aggregation-strategy                    (when necessary)  | dict[str:int]
            - effective_distrib                                         | dict[str:float]
            - trainer-logs                                              | dict
            - time                                                      | str 
            - run_timer                                                 | dict[str:float]

    All throughout, we use a logger to provide insights on the run status and 
    possible errors

    Input: 
        df_training: full dataset used for training prior to sampling and splitting. 
            The dataframe should be sanitized as defined in the sanitize_df. Notably, 
            the text column should be "TEXT", label column "LABEL" and id column 
            "ID". No cell should be a nan.
        df_prediction: full dataset used for inference 
            The dataframe should be sanitized as defined in the sanitize_df. Notably, 
            the text column should be "TEXT", label column "LABEL" and id column 
            "ID". No cell should be a nan.
        loop_config: LoopConfig object containing all necessary information. 
            See LoopConfig docstring.
    
    Data saved:
        predictions on test:            saved as ./predictions_save/{hash_}-on-test.csv
        predictions on inference set:   saved as ./predictions_save/{hash_}.csv

        When texts are chunked: 
            predictions aggregated on test:            saved as ./predictions_save/{hash_}-on-test-aggregated.csv
            predictions aggregated on inference set:   saved as ./predictions_save/{hash_}-aggregated.csv

    Output:
        hash_ : run identifier
        logs_to_save: the necessary information as described above
    """

    logger = CustomLogger("./custom_logs")
    run_timer = {}

    # Use time as hash
    hash_, logs_to_save = create_hash_from_config_loop(loop_config), None
    logger(hash_)
    dichotomized_df_training, dichotomized_df_prediction, dsd_loop, model, ds_pred = (None,)*5
    try: 
        # Dichotomization: dichotomization_label
        run_timer["preprocess_data"] = time()
        dichotomized_df_training, label2id, id2label = dichotomize(df_training, loop_config)
        loop_config.set_label_id_mapper(label2id, id2label)
        dichotomized_df_prediction, _, _ = dichotomize(df_prediction, loop_config)
        
        # Prepare dataset: N_annotated, splits_ratio, seed
        # N_documents is a dictionary of dictionaries
        df_training_sample, effective_distrib = sample_N_documents(dichotomized_df_training, loop_config)
        logger(f"Sample {len(df_training_sample)} rows")
        logger(f"Effective distribution: {effective_distrib} — requested : {loop_config.sampling_method}")
        # Prepare tokenize texts: model_name
        N_documents, max_length_capped = tokenize_chunk_pad(
            df_full = dichotomized_df_training, 
            df_sample = df_training_sample, 
            df_name = "training", 
            loop_config = loop_config
        )
        dsd_loop = split_ds(N_documents, loop_config)
        del df_training_sample, N_documents
        logger(dsd_loop)

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
        predictions_on_test : pd.DataFrame = predict(model, dsd_loop["test"], loop_config)
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
        N_documents, max_length_capped_inference = tokenize_chunk_pad(
            df_full = dichotomized_df_prediction,
            df_sample = dichotomized_df_prediction, 
            df_name = "inference", 
            loop_config = loop_config
        )
        ds_pred = Dataset.from_list([d for d in N_documents.values()])
        del N_documents
        logger("Start Inference")
        tstart = time()
        predictions : pd.DataFrame = predict(model, ds_pred, loop_config)
        logger(f"Inference done in {time() - tstart:.0f} s")
        run_timer["prediction"] = time() - run_timer["prediction"]

        if not loop_config.test_mode:
            run_timer["saving_predictions"] = time()
            predictions_on_test.to_csv(f"./predictions_save/{hash_}-on-test.csv")
            predictions.to_csv(f"./predictions_save/{hash_}.csv")
            run_timer["saving_predictions"] = time() - run_timer["saving_predictions"]
            logs_to_save = {
                **loop_config.to_dict(),
                "effective_context_window_for_training": max_length_capped,
                "effective_context_window_for_inference": max_length_capped_inference,
                "score_on_test": score_on_test,
                "prediction-on-test-csv": f"./predictions_save/{hash_}-on-test.csv",
                "prediction-csv": f"./predictions_save/{hash_}.csv",
                "effective_distrib": effective_distrib,
                "trainer-logs": trainer_logs,
                "time": str(pd.Timestamp.now())
            }                
            if "ID_CHUNK" in predictions.columns:
                run_timer["saving_predictions_aggregated"] = time()
                predictions_on_test_aggregated.to_csv(f"./predictions_save/{hash_}-on-test-aggregated.csv", index=False)
                (
                    aggregate_predictions(predictions, loop_config)
                    .to_csv(f"./predictions_save/{hash_}-aggregated.csv", index=False)
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
            del dichotomized_df_training, dichotomized_df_prediction, dsd_loop, model, ds_pred
            clean() 

    return hash_, logs_to_save

def single_run_dummy(
    df_training : pd.DataFrame,
    df_prediction: pd.DataFrame,
    loop_config : LoopConfig,
) -> tuple[str, dict | None]: 
    """Dummy function to test the loop"""
    hash_, logs_to_save = create_hash_from_config_loop(loop_config), None
    logs_to_save = {
        "THIS IS DUMMY": "it is",
        **loop_config.to_dict(),
        "time": str(pd.Timestamp.now())
    }
    return hash_, logs_to_save

if __name__=="__main__":
    # Implement the python -u single_run.py XXX

    df = pd.read_csv("./data/ideology_news-stratified_year_balanced.csv")
    df = sanitize_df(df, text_col = "content", label_col = "bias_text", id_col="ID")
    df_prediction = df.copy()
    loop_config = LoopConfig(dataset_name = "TASK-left", dichotomization_label="left", test_mode=True)

    print(single_run(df, df_prediction, loop_config))
