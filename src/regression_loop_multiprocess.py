import getopt
import sys 
import warnings

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from toolbox import (
    get_config, 
    get_run_info_for_regression, 
    get_df_with_metadata, 
    create_hash_from_string,
    regression_already_done, 
    run_regression_and_assess_errors,
    save_errors, 
    send_notification
)

def regression_task(task):
    (
        run_hash,
        run_info,
        regression_col,
        regression_unique_value,
        df_regression,
    ) = task

    regression_hash = create_hash_from_string(
        f"{run_hash}-{regression_col}-{regression_unique_value}"
    )

    output = run_regression_and_assess_errors(
        df_regression,
        run_info,
        regression_col,
        regression_unique_value,
    )

    return (
        regression_hash,
        run_info,
        regression_col,
        regression_unique_value,
        output,
    )


def main(configuration_filename, saving_logs_filename):
    datasets_config, _, _ = get_config(configuration_filename)
    all_run_info_for_regression = get_run_info_for_regression(
        saving_logs_filename
    )

    # Build all tasks first
    for run_hash, run_info in tqdm(all_run_info_for_regression.items(), position=0, desc="Runs"):
        df_regression, metadata_columns = get_df_with_metadata(
            run_info,
            datasets_config,
        )

        tasks = []

        for regression_col in metadata_columns:
            for regression_unique_value in df_regression[
                regression_col
            ].unique():

                regression_hash = create_hash_from_string(
                    f"{run_hash}-{regression_col}-{regression_unique_value}"
                )

                if regression_already_done(regression_hash):
                    continue

                tasks.append(
                    (
                        run_hash,
                        run_info,
                        regression_col,
                        regression_unique_value,
                        df_regression,
                    )
                )

        with ProcessPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(regression_task, task)
                for task in tasks
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Regressions",
                position=1
            ):
                (
                    regression_hash,
                    run_info,
                    regression_col,
                    regression_unique_value,
                    output,
                ) = future.result()

                # Single process writes => no file corruption
                save_errors(
                    regression_hash,
                    {
                        **run_info,
                        "regression_col": regression_col,
                        "regression_unique_value": regression_unique_value,
                        **output,
                    },
                )

if __name__ == "__main__":
    reason = []
    warnings.filterwarnings("ignore")
    try: 
        argv = sys.argv[1:] 
        opts, _ = getopt.getopt(argv, "", ["config-file=", "saving-logs-file="]) 
        opts = {k:v for k,v in opts}
        configuration_file = opts.get("--config-file", "config-loop.json")
        saving_logs_filename = opts.get("--saving-logs-file", "saving_logs.json")
        main(configuration_file, saving_logs_filename)
    except Exception as e:
        print(e)
        reason.append(str(e))
    finally: 
        message = f"Loop regression"
        send_notification(message)


