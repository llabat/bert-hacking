import os 
import shutil

from datasets import DatasetDict, Dataset
import numpy as np
import pandas as pd 
from sklearn.metrics import f1_score
from torch import Tensor
from transformers import TrainingArguments, Trainer, EvalPrediction
from tqdm import tqdm

from . import LoopConfig
from .utils import get_device, clean

def load_training_arguments(loop_config: LoopConfig) -> TrainingArguments:
    device = get_device()

    # Overwrite output dir
    if os.path.isdir(loop_config.output_dir):
        shutil.rmtree(loop_config.output_dir)

    return TrainingArguments(
        bf16= str(device) == "cuda", # Faster training
        # Hyperparameters
        num_train_epochs = loop_config.n_epochs,
        learning_rate = loop_config.learning_rate,
        weight_decay  = loop_config.weight_decay,
        # Second order hyperparameters
        per_device_train_batch_size = loop_config.device_batch_size,
        per_device_eval_batch_size = loop_config.device_batch_size,
        gradient_accumulation_steps = max(loop_config.batch_size // loop_config.device_batch_size, 1),
        # Metrics
        metric_for_best_model="f1_macro",
        # Pipe
        output_dir = loop_config.output_dir,
        eval_strategy = "epoch",
        logging_strategy = "epoch",
        save_strategy = "epoch",
        load_best_model_at_end = True,
        save_total_limit =  2,
        disable_tqdm = False, 
        dataloader_pin_memory = str(device) != "cuda",
        # SEED
        seed = loop_config.seed
    )

def compute_metrics_multiclass(model_output: EvalPrediction):
    if isinstance(model_output.predictions,tuple):
        results_matrix = model_output.predictions[0]
    else:
        results_matrix = model_output.predictions
    y_true : list[int] = model_output.label_ids
    y_pred_probs = Tensor(results_matrix).softmax(1).numpy()
    y_pred = np.argmax(y_pred_probs, axis = 1).reshape(-1)

    return {
        "f1_macro": f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
    }

def train_model(
    model, 
    training_args : TrainingArguments,
    dsd : DatasetDict,
    loop_config: LoopConfig,
) -> str :
    """
    """
    output = None
    try: 
        device = get_device()
        for split in dsd:
            dsd[split] = dsd[split].with_format("torch", device=device)
            if loop_config.test_mode:
                dsd[split] = dsd[split].select(range(20))
        
        model = model.to(device=device)
        trainer = Trainer(
            model, 
            args = training_args,
            train_dataset=dsd["train"],
            eval_dataset=dsd["train-eval"], 
            compute_metrics = compute_metrics_multiclass,
        )
        print(f"Begin training on {device}")
        trainer.train()
        output = trainer.state.best_model_checkpoint
    except Exception as e:
        print(f"ERROR in train_model: \n{e}")
    finally:
        del trainer
        clean()
    return output

def predict(model, ds : Dataset, loop_config: LoopConfig, id2label: dict[int:str])->pd.DataFrame:
    if "input_ids" not in ds.features:
        raise ValueError("Please tokenize texts first")
    if "attention_mask" not in ds.features:
        raise ValueError("Please tokenize texts first")
    if not np.isin(["ID", "LABEL"], list(ds.features.keys())).all():
        raise ValueError("Please sanitize you Dataset first")
    
    device = get_device()
    print(f"Predict on {device}")
    
    if loop_config.test_mode: ds = ds.select(range(20))

    ds = ds.with_format("torch", device=device)
    model = model.to(device=device)
    model.eval()

    output_df = []
    for batch in tqdm(ds.batch(loop_config.device_batch_size_for_prediction), desc="Prediction"):
        probs = (
            model(input_ids = batch["input_ids"], attention_mask= batch["attention_mask"])
            .logits
            .detach().cpu()
            .softmax(1)
            .numpy()
        )
        y_pred = np.argmax(probs, axis = 1).reshape(-1)
        
        if "ID_CHUNK" in batch:
            output_df += [
                {
                    "ID": id,
                    "ID_CHUNK": id_chunk,
                    "GS-LABEL": label,
                    "PRED-LABEL": id2label[int(pred)],
                }
                for id, id_chunk, label, pred in zip(
                    batch["ID"], 
                    batch["ID_CHUNK"], 
                    batch["LABEL"], 
                    y_pred
                )
            ]
        else:
            output_df += [
                {
                    "ID": id,
                    "GS-LABEL": label,
                    "PRED-LABEL": id2label[int(pred)],
                }
                for id, label, pred in zip(batch["ID"], batch["LABEL"], y_pred)
            ]
    return pd.DataFrame(output_df).set_index("ID")