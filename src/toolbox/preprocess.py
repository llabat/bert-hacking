import os 

from datasets import Dataset, DatasetDict
import numpy as np
import pandas as pd 
from transformers import AutoConfig

from . import LoopConfig

def sanitize_df(
    df: pd.DataFrame, 
    text_col: str, 
    label_col:str, 
    id_col:str, 
    id_chunk_col: str|None = None, 
    extra_cols_to_keep : list[str]|None=None, 
    **kwargs
)->pd.DataFrame:
    """"""
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
    main_columns = ["TEXT","LABEL", "ID"]
    column_for_index = "ID"
    if id_chunk_col: 
        df = df.rename(columns={id_chunk_col: "ID_CHUNK"})
        main_columns += ["ID_CHUNK"]
        column_for_index = "ID_CHUNK"
    if extra_cols_to_keep: 
        main_columns += extra_cols_to_keep

    if np.array([df[col].isna().sum() > 0 for col in main_columns]).any():
        raise ValueError(
            f"Missing values: "
            f"\t ID: {df['ID'].isna().sum()}"
            f"\t TEXT: {df['TEXT'].isna().sum()}"
            f"\t LABEL: {df['LABEL'].isna().sum()}"
        )
    if df[column_for_index].is_unique:
        return df[main_columns]
    else:
        raise ValueError("ID column contains non-unique values.")

def dichotomize(df: pd.DataFrame, loop_config: LoopConfig) -> tuple[pd.DataFrame, dict[str:int], dict[int:str]]:
    """
    Dichotomize dataframe given a label
    """
    label = loop_config.dichotomization_label
    
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

def cap_max_length(max_n_tokens : int, loop_config: LoopConfig) -> int:
    model_max = AutoConfig.from_pretrained(loop_config.model_name).max_position_embeddings - 1
    return int(min(max_n_tokens, model_max))

def _sample_N_documents(df: pd.DataFrame, label2id : dict, loop_config: LoopConfig)->pd.DataFrame:
    """
    Sample N elements
    """
    stratification_col = loop_config.sampling_method["stratified"]
    balance = loop_config.sampling_method["balance"]
    df = df.copy()
    if stratification_col is None:
        # Create dummy stratification column
        df["stratification_col"] = 0
        stratification_col = "stratification_col"

    df_for_ID_sampling = df.groupby("ID").sample(1).set_index("ID")
    df_for_ID_sampling["LABEL"] = df_for_ID_sampling["LABEL"].map(label2id)
    rng = np.random.default_rng(seed=loop_config.seed)
    N_per_strata = int(loop_config.N_annotated / 
                       df_for_ID_sampling[stratification_col].nunique())

    id_samples = []
    for _, subdf in df_for_ID_sampling.groupby(stratification_col):
        batch_indexes = []
        for _ in range(N_per_strata):
            local_distrib = subdf.drop(index=batch_indexes)["LABEL"].mean()
            if balance == "random":
                local_weights = None
            else: 
                local_weights = (
                    subdf.drop(index=batch_indexes)
                    ["LABEL"]
                    .map({
                        1: balance / local_distrib, 
                        0 : (1 - balance) / (1 - local_distrib)
                    })
                )
                local_weights = local_weights / sum(local_weights)
            batch_indexes += [rng.choice(
                list(subdf.drop(index=batch_indexes).index), 
                p = local_weights
            )]
        id_samples += batch_indexes
    return df.loc[np.isin(df["ID"], id_samples)]

def sample_N_documents(df: pd.DataFrame, label2id : dict, loop_config: LoopConfig)->tuple[pd.DataFrame, dict]:
    """
    Sample N elements with cache 
    """
    stratification_col = loop_config.sampling_method["stratified"]
    balance = loop_config.sampling_method["balance"]
    cache_file = (f"{loop_config.task_name}-{loop_config.N_annotated}-"
        f"{stratification_col}-{balance}.csv")
    if cache_file in os.listdir("./.cache"):
        out_df = pd.read_csv(f"./.cache/{cache_file}")
    else: 
        print("Start sampling, might take a while") #TODELETE
        out_df = _sample_N_documents(df, label2id, loop_config)
        print("Done sampling") #TODELETE
        out_df.to_csv(f"./.cache/{cache_file}", index=False)
    df_for_effective_distrib_calc = out_df.groupby("ID").sample(n=1,random_state=0)
    label, count = np.unique_counts(df_for_effective_distrib_calc['LABEL'])
    effective_distrib = {l:float(c / sum(count)) for l,c in zip(label, count)}
    return out_df, effective_distrib

def split_ds(df : pd.DataFrame, loop_config: LoopConfig)-> DatasetDict:
    """
    takes the splits_ratio (ex: [80, 10, 10]) and return a DatasetDict
    """
    splits_ratio = loop_config.splits_ratio
    if len(splits_ratio) != 3:
        raise ValueError(
            f"There should be three ints in splits_ratio. Found: " 
            f"{splits_ratio}"
        )
    if sum(splits_ratio) != 100:
        raise ValueError(
            f"The sum of splits_ratio shoul be 100. Found: "
            f"{splits_ratio}"
        )
    ids = pd.Series(df["ID"].unique()).sample(frac=1, random_state=loop_config.seed)
    N_ids = len(ids)
    ids_train = ids.head(splits_ratio[0] * N_ids // 100)
    ids_test = ids.tail(splits_ratio[2] * N_ids // 100)
    ids_train_eval = ids.drop(index=[*ids_train.index.to_list(), *ids_test.index.to_list()])

    out_dsd = DatasetDict({
        "train": Dataset.from_pandas(df.loc[np.isin(df["ID"], ids_train)]),
        "eval": Dataset.from_pandas(df.loc[np.isin(df["ID"], ids_train_eval)]),
        "test": Dataset.from_pandas(df.loc[np.isin(df["ID"], ids_test)]),
    })
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