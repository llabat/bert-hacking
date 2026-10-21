import os
import getopt
import sys 
import warnings
import json 

from tqdm import tqdm

from toolbox import (
    get_config, 
    get_run_info_for_regression, 
    get_df_with_metadata, 
    create_hash_from_string,
    run_regression_and_assess_errors,
    send_notification
)

def regression_loop(configuration_filename: str, saving_logs_filename: str) -> None: 
    """Run all the regressions and assess the errors for a given a configuration 
    file (for metadatacolumns) and saving logs (for the predictions).
    Save the results by batch, one file = one fine-tuning (= run, i.e. one set of predictions)
    Each regression and error assessment has a unique hash depending on the run, 
    metadata column and metadata value.
    """

    datasets_config, _, _ = get_config(configuration_filename)
    all_run_info_for_regression = get_run_info_for_regression(saving_logs_filename)

    for run_hash, run_info in tqdm(all_run_info_for_regression.items(), position=0):
        if f"{run_hash}.json" in os.listdir("./results/regressions"): # already done
            continue
        df_regression, metadata_columns = get_df_with_metadata(run_info, datasets_config)
        batch_regressions = {}
        for regression_col in metadata_columns:
            for regression_unique_value in df_regression[regression_col].unique():
                regression_hash = create_hash_from_string(f"{run_hash}-{regression_col}-{regression_unique_value}")
                
                output = run_regression_and_assess_errors(
                    df_regression, 
                    run_info, 
                    regression_col, 
                    regression_unique_value
                )
                batch_regressions[regression_hash] = {
                    **run_info,
                    "regression_col": regression_col, 
                    "regression_unique_value": regression_unique_value, 
                    **output,   
                }
        with open(f"./results/regressions/{run_hash}.json", "w") as file:
            json.dump(batch_regressions, file, ensure_ascii=True)
    
if __name__ == "__main__":
    reason = []
    warnings.filterwarnings("ignore")
    try: 
        argv = sys.argv[1:] 
        opts, _ = getopt.getopt(argv, "", ["config-file=", "saving-logs-file="]) 
        opts = {k:v for k,v in opts}
        configuration_file = opts.get("--config-file", "config-loop.json")
        saving_logs_filename = opts.get("--saving-logs-file", "saving_logs.json")
        regression_loop(configuration_file, saving_logs_filename)
    except Exception as e:
        print(e)
        reason.append(str(e))
    finally: 
        message = "Loop regression"
        send_notification(message)


