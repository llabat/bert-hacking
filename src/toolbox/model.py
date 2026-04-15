from datasets import DatasetDict, Dataset
import numpy as np
import pandas as pd 
from sklearn.metrics import f1_score
from torch import Tensor
from transformers import TrainingArguments, Trainer, EvalPrediction

from .utils import get_device, clean, pick_seed

def load_training_arguments(
        output_dir :str, 
        batch_size_device: int, 
        total_batch_size : int = 16, 
        **kwargs
    ) -> TrainingArguments:
    device = get_device()
    return TrainingArguments(
        bf16=True, # Faster training
        # Hyperparameters
        num_train_epochs = kwargs.get("n_epochs", 4),
        learning_rate = kwargs.get("learning_rate", 1e-5),
        weight_decay  = kwargs.get("weight_decay", 0.0),
        warmup_ratio  = kwargs.get("warmup_ratio", 0.05),
        # dropout = kwargs.get("dropout", 0.1), #TODO check for dropout
        # Second order hyperparameters
        per_device_train_batch_size = batch_size_device,
        per_device_eval_batch_size = batch_size_device,
        gradient_accumulation_steps = total_batch_size // batch_size_device,
        # Metrics
        metric_for_best_model="f1_macro",
        # Pipe
        output_dir = output_dir,
        overwrite_output_dir=True,
        eval_strategy = "epoch",
        logging_strategy = "epoch",
        save_strategy = "epoch",
        load_best_model_at_end = True,
        save_total_limit =  2,
        disable_tqdm = kwargs.get("disable_tqdm", False), 
        dataloader_pin_memory = False if str(device) == "cuda" else True,
        # SEED
        seed = pick_seed(**kwargs)
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
    test_mode: bool = False, 
) -> str :
    """
    """
    try: 
        device = get_device()
        for split in dsd:
            dsd[split] = dsd[split].with_format("torch", device=device)
            if test_mode:
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

def predict(model, ds : Dataset, batch_size: int, id2label: dict[int:str])->pd.DataFrame:
    if "input_ids" not in ds.features:
        raise ValueError("Please tokenize texts first")
    if not np.isin(["ID", "LABEL"], list(ds.features.keys())).all():
        raise ValueError("Please sanitize you Dataset first")
    
    device = get_device()
    print(f"Predict on {device}")
    
    ds = ds.with_format("torch", device=device)
    model = model.to(device=device)# .eval() ??

    output_df = []
    for batch in ds.batch(batch_size):
        probs = model(input_ids = batch["input_ids"]).logits.detach().cpu().softmax(1).numpy()
        y_pred = np.argmax(probs, axis = 1).reshape(-1)

        output_df += [
            {
                "ID": id,
                "GS-LABEL": label,
                "PRED-LABEL": id2label[int(pred)],
            }
            for id, label, pred in zip(batch["ID"], batch["LABEL"], y_pred)
        ]
    return pd.DataFrame(output_df).set_index("ID")