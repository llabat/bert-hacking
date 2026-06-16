import sys 
import getopt 
import json 

from itertools import product
import pandas as pd 

from toolbox import (
    LoopConfig, 
    CustomLogger,
    get_config,
    in_subsample,
    sanitize_df, 
    to_saving_logs, 
    already_done,
    get_report,
    send_notification
)
from single_run import single_run_dummy

TEST_MODE = False
DEVICE_BATCH_SIZE = 4
DEVICE_BATCH_SIZE_FOR_PREDICTION = 256
logger = CustomLogger("./custom_logs")

def loop(configuration_file : str, subsample_file: str|None = None):

    datasets_config, parameter_names,parameters_values = get_config(configuration_file)
    for dataset_info in datasets_config:
        df = pd.read_csv(dataset_info["filepath-train"], sep=dataset_info.get("csv-sep", ","))
        df = sanitize_df(df, **dataset_info)

        df_prediction = pd.read_csv(dataset_info["filepath-predict"], sep=dataset_info.get("csv-sep", ","))
        df_prediction = sanitize_df(df_prediction, **dataset_info)
        for label in dataset_info["labels"]: 
            for local_config in product(*parameters_values):
                loop_config = LoopConfig(
                    dataset_name=dataset_info['name'],
                    dichotomization_label = label,
                    **{n: v for n,v in zip(parameter_names,local_config)},
                    test_mode = TEST_MODE, 
                    device_batch_size = DEVICE_BATCH_SIZE,
                    device_batch_size_for_prediction = DEVICE_BATCH_SIZE_FOR_PREDICTION,
                )
                if not already_done(loop_config) and in_subsample(loop_config,dataset_info['name'], label, subsample_file):
                    logger.start_loop_log(loop_config)
                    hash_, to_save = single_run_dummy(df, df_prediction, loop_config)
                    to_saving_logs(hash_, to_save)
                    logger("END LOOP" + "#" * 92)

if __name__ == "__main__":

    reason = []
    try: 
        argv = sys.argv[1:] 
        opts, _ = getopt.getopt(argv, "", ["config-file=", "subsample-file="]) 
        opts = {k:v for k,v in opts}
        configuration_file = opts.get("--config-file", "config-loop.json")
        subsample_file = opts.get("--subsample-file")
        loop(configuration_file,subsample_file)
    except Exception as e:
        print(e)
        reason.append(str(e))
    finally: 
        report = get_report(configuration_file, "saving_logs.json", subsample_file)
        message = f"Report:\n{report}\n\nReason it stopped:{'  '.join(reason)}"
        send_notification(message)

