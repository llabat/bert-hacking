import os 
import shutil

from datasets import DatasetDict, Dataset
import numpy as np
import pandas as pd 
from sklearn.metrics import f1_score
from torch import Tensor, no_grad
from transformers import TrainingArguments, Trainer, EvalPrediction
from tqdm import tqdm

from . import LoopConfig
from .utils import get_device, clean, retrieve_trainer_logs

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
        dataloader_pin_memory = str(device) == "cpu",
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
) -> tuple[str, dict] :
    """
    """
    output, trainer_logs = None, None
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
            eval_dataset=dsd["eval"], 
            compute_metrics = compute_metrics_multiclass,
        )
        print(f"Begin training on {device}")
        trainer.train()
        output = trainer.state.best_model_checkpoint
        trainer_logs = retrieve_trainer_logs(training_args.output_dir)
    except Exception as e:
        print(f"ERROR in train_model: \n{e}")
    finally:
        del trainer
        clean()
    return output, trainer_logs

def predict(model, ds : Dataset, loop_config: LoopConfig)->pd.DataFrame:
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
    if str(device)=="cuda": model = model.bfloat16()

    ID_= []
    ID_chunk_ = []
    GS_ = []
    PRED_ = []
    for batch in tqdm(ds.batch(loop_config.device_batch_size_for_prediction), desc="Prediction"):
        with no_grad():
            probs = (
                model(input_ids = batch["input_ids"], attention_mask= batch["attention_mask"])
                .logits.cpu().softmax(1).float().numpy()
            )
        y_pred = np.argmax(probs, axis = 1).reshape(-1)
        
        ID_ += batch["ID"]
        GS_ += batch["LABEL"]
        PRED_ += [loop_config.id2label[int(y)] for y in y_pred]
        if "ID_CHUNK" in batch:
            ID_chunk_ += batch["ID"]
    if len(ID_chunk_)>0:
        return pd.DataFrame({
            "ID": ID_, 
            "ID_CHUNK":ID_chunk_, 
            "GS-LABEL":GS_, 
            "PRED-LABEL":PRED_
        }).set_index("ID")
    
    return pd.DataFrame({
        "ID": ID_, 
        "GS-LABEL": GS_, 
        "PRED-LABEL": PRED_
    }).set_index("ID")