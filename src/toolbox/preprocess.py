from datasets import Dataset, DatasetDict
import numpy as np
import pandas as pd 
from transformers import AutoConfig

from .utils import pick_seed

def sanitize_df(df: pd.DataFrame, text_col: str, label_col:str, id_col:str, **kwargs)->pd.DataFrame:
    if not np.isin([text_col, label_col, id_col], df.columns).all():
        raise ValueError(
            f"The columns you provided cannot be found in the dataframe. "
            f"You provided: {[text_col, label_col, id_col]}. "
            f"The dataframe contains: {df.columns}"
        )
    df = df.rename(columns={
        text_col: "TEXT",
        label_col: "LABEL",
        id_col: "ID",
    })
    if np.array([
        df["ID"].isna().sum() > 0,
        df["TEXT"].isna().sum() > 0,
        df["LABEL"].isna().sum() > 0,
    ]).any():
        raise ValueError(
            f"Missing values: "
            f"\t ID: {df['ID'].isna().sum()}"
            f"\t TEXT: {df['TEXT'].isna().sum()}"
            f"\t LABEL: {df['LABEL'].isna().sum()}"
        )
    if df["ID"].is_unique:
        return df[["ID", "TEXT", "LABEL"]].set_index("ID")
    else:
        raise ValueError("ID column contains non-unique values.")

def dichotomize(df: pd.DataFrame, label:str) -> tuple[pd.DataFrame, dict[str:int], dict[int:str]]:
    if label not in df["LABEL"].values:
        raise ValueError(f"Label ({label}) not in df[\"LABEL\"]. "
                         f"Available labels: {df['LABEL'].unique()}")
    df["LABEL"] = (df["LABEL"] == label).replace({True:label, False:f"not-{label}"})
    label2id = {label:1, f"not-{label}": 0}
    id2label = {1:label, 0: f"not-{label}"}
    return df, label2id, id2label

def get_max_tokens(texts: pd.Series, tokenizer, top_n : int = 15)->int:
    """
    Tokenize the top_n longest entries (in term of characters), tokenize them and 
    return the maximum length of the encoded sentences
    """
    longests_as_index = texts.apply(len).sort_values(ascending=False).head(top_n).index
    max_tokenizing_len = (
        texts[longests_as_index]
        .apply(lambda txt: tokenizer(txt)["input_ids"])
        .apply(len)
        .max()
    )
    return max_tokenizing_len

def cap_max_length(max_n_tokens : int, context_window_rel_to_max : int, model_name : str, **kwargs) -> int:
    requested = context_window_rel_to_max * max_n_tokens / 100
    model_max = AutoConfig.from_pretrained(model_name).max_position_embeddings
    return int(min(requested, model_max))

def sample_N_elements(df: pd.DataFrame, N_train: int, **kwargs)->pd.DataFrame:
    """
    Sample N_elements
    """
    return Dataset.from_pandas(df.sample(N_train, random_state=pick_seed(**kwargs)))

def split_ds(ds : Dataset, train_eval_test_ratios : list[int], **kwargs)-> DatasetDict:
    """
    takes the train_eval_test_ratios (ex: [80, 10, 10]) and return a DatasetDict
    """
    if len(train_eval_test_ratios) != 3:
        raise ValueError(
            f"There should be three ints in train_eval_test_ratios. Found: " 
            f"{train_eval_test_ratios}"
        )
    if sum(train_eval_test_ratios) != 100:
        raise ValueError(
            f"The sum of train_eval_test_ratios shoul be 100. Found: "
            f"{train_eval_test_ratios}"
        )
    out_dsd = ds.train_test_split(
        train_size= train_eval_test_ratios[0] / 100, # Train proportion 
        shuffle=True,
        seed=pick_seed(**kwargs)
    )
    resplit_ratio = 100 * train_eval_test_ratios[1] / (train_eval_test_ratios[1] + train_eval_test_ratios[2])
    temp_dsd = out_dsd["test"].train_test_split(
        train_size = resplit_ratio / 100, 
        shuffle=True, 
        seed=pick_seed(**kwargs)
    )
    out_dsd["train-eval"] = temp_dsd["train"]
    out_dsd["test"] = temp_dsd["test"]
    return out_dsd

def tokenize_dataset_dict(
        row: dict, 
        label2id: dict[str:int], 
        tokenizer,  
        tokenization_parameters: dict
    ) -> dict:
    """"""
    row = row.copy()
    tokenized_entry = tokenizer(row["TEXT"], **tokenization_parameters)
    return {
        **row,
        **tokenized_entry,
        "labels": label2id[row["LABEL"]]
    }