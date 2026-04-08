import sys
import pandas as pd

from config import load_config
from environment import prepare_env
from evaluate import compute_metrics_multiclass
from experiment import create_hash
from model import train, predict
from preprocess import sanitize_df, make_ovr_dataset
from splits import split_ds


def main():
    config_path = sys.argv[1]
    config = load_config(config_path)

    data_config = config["data"]
    split_config = config["split"]
    model_config = config["model"]
    training_config = config["training"]

    prepare_env([training_config["output_dir"]])

    df = pd.read_csv(data_config["input_path"], data_config.get("sep", ";"))

    df = sanitize_df(
        df,
        text_column=data_config["text_column"],
        label_column=data_config["label_column"],
        id_column=data_config.get("id_column"),
        drop_na_text=data_config.get("drop_na_text", True),
        drop_na_label=data_config.get("drop_na_label", True),
        enforce_unique_id=data_config.get("enforce_unique_id", False),
    )

    classes = sorted(df["LABEL"].unique())
    num_classes = len(classes)

    print(f"Detected {num_classes} classes: {classes}")

    # =========================
    # 🔹 STANDARD (BINARY)
    # =========================
    if num_classes <= 2:
        print("Binary classification → standard training")

        splits = split_ds(
            df,
            train_size=split_config.get("train_size", 0.8),
            validation_size=split_config.get("validation_size", 0.1),
            test_size=split_config.get("test_size", 0.1),
            random_state=split_config.get("random_state", 42),
            stratify=split_config.get("stratify", False),
            label_column="LABEL",
        )

        train_df = splits["train"]
        validation_df = splits["validation"]
        test_df = splits["test"]

        run_hash = create_hash({
            "mode": "binary",
            "model_name": model_config["model_name"],
            "epochs": training_config.get("num_train_epochs", 3),
            # ajouter les parametres manquants
        })

        output_dir = training_config.get("output_dir", "model.current")
        # add an overwrite parameter
        # make sure models aren't oversaved

        training_output = train(
            train_df = train_df,
            validation_df = validation_df,
            model_name = model_config["model_name"],
            output_dir = output_dir,
            compute_metrics = compute_metrics_multiclass,
            text_column = "TEXT", # force
            label_column = "LABEL", # always the same
            num_labels = num_classes, # always binary
            max_length = model_config.get("max_length"),
            learning_rate = float(training_config.get("learning_rate", 2e-5)),
            per_device_train_batch_size = training_config.get("per_device_train_batch_size", 8),
            per_device_eval_batch_size = training_config.get("per_device_eval_batch_size", 8),
            num_train_epochs = training_config.get("num_train_epochs", 3),
            weight_decay = training_config.get("weight_decay", 0.0),
            seed = training_config.get("seed", 42),
        )

        # Predict
        if test_df is not None and len(test_df) > 0:
            predictions = predict(
                df=test_df,
                model=training_output["model"],
                tokenizer=training_output["tokenizer"],
                text_column="TEXT",
                id_column="ID" if "ID" in test_df.columns else None,
                label_column="LABEL",
                max_length=model_config.get("max_length"),
                batch_size=training_config.get("per_device_eval_batch_size", 8),
            )

            predictions_path = f"{output_dir}/predictions_{run_hash}.csv"
            predictions.to_csv(predictions_path, index=False)

    # =========================
    # 🔹 OVR (MULTICLASS)
    # =========================
    else:
        print("Multiclass → One-vs-Rest training")

        for cls in classes:
            print(f"\n===== Class {cls} vs Rest =====")

            df_ovr = make_ovr_dataset(df, target_class=cls)

            splits = split_ds(
                df_ovr,
                train_size=split_config.get("train_size", 0.8),
                validation_size=split_config.get("validation_size", 0.1),
                test_size=split_config.get("test_size", 0.1),
                random_state=split_config.get("random_state", 42),
                stratify=split_config.get("stratify", False),
                label_column="LABEL",
            )

            train_df = splits["train"]
            validation_df = splits["validation"]
            test_df = splits["test"]

            run_hash = create_hash({
                "mode": "ovr",
                "class": cls,
                "model_name": model_config["model_name"],
                "epochs": training_config.get("num_train_epochs", 3),
            })

            class_output_dir = f"{training_config['output_dir']}/class_{cls}"
            prepare_env([class_output_dir])

            training_output = train(
                train_df=train_df,
                validation_df=validation_df,
                model_name=model_config["model_name"],
                output_dir=class_output_dir,
                compute_metrics=compute_metrics_multiclass,
                text_column="TEXT",
                label_column="LABEL",
                num_labels=2,
                max_length=model_config.get("max_length"),
                learning_rate=float(training_config.get("learning_rate", 2e-5)),
                per_device_train_batch_size=training_config.get("per_device_train_batch_size", 8),
                per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 8),
                num_train_epochs=training_config.get("num_train_epochs", 3),
                weight_decay=training_config.get("weight_decay", 0.0),
                seed=training_config.get("seed", 42),
            )

            if test_df is not None and len(test_df) > 0:
                predictions = predict(
                    df=test_df,
                    model=training_output["model"],
                    tokenizer=training_output["tokenizer"],
                    text_column="TEXT",
                    id_column="ID" if "ID" in test_df.columns else None,
                    label_column="LABEL",
                    max_length=model_config.get("max_length"),
                    batch_size=training_config.get("per_device_eval_batch_size", 8),
                )

                predictions_path = f"{class_output_dir}/predictions_{run_hash}.csv"
                predictions.to_csv(predictions_path, index=False)

            # Log
            with open(f"{class_output_dir}/run_{run_hash}.log", "w") as f:
                f.write(f"class: {cls}\n")
                f.write(f"train_size: {len(train_df)}\n")
                f.write(f"val_size: {len(validation_df)}\n")
                f.write(f"test_size: {len(test_df)}\n")

if __name__ == "__main__":
    main()
