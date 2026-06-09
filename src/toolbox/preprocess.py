import os 
import json 

from datasets import Dataset, DatasetDict
import numpy as np
import pandas as pd 
from transformers import AutoConfig
from tqdm import tqdm

from . import LoopConfig, load_tokenizer

def sanitize_df(
    df: pd.DataFrame, 
    text_col: str, 
    label_col:str, 
    id_col:str, 
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
    if extra_cols_to_keep: 
        main_columns += extra_cols_to_keep

    if np.array([df[col].isna().sum() > 0 for col in main_columns]).any():
        raise ValueError(
            f"Missing values: "
            f"\t ID: {df['ID'].isna().sum()}"
            f"\t TEXT: {df['TEXT'].isna().sum()}"
            f"\t LABEL: {df['LABEL'].isna().sum()}"
        )
    # Force ID to be strings
    df["ID"] = df["ID"].astype(str)
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

def get_max_tokens(N_documents:dict[str:dict])->int:
    """"""
    return max([d["N_tokens"] for d in N_documents.values()])

def cap_max_length(max_n_tokens : int, loop_config: LoopConfig) -> int:
    model_max = AutoConfig.from_pretrained(loop_config.model_name).max_position_embeddings - 1
    return int(min(max_n_tokens, model_max))

def _sample_N_documents_by_their_ID(df: pd.DataFrame, loop_config: LoopConfig)->list:
    """
    Sample N elements, return the sampled IDs
    At this stage the texts are not chunked, therefore one row = one ID 
    """
    if not df["ID"].is_unique:
        raise ValueError(f"Can't sample if IDs are not unique\nfrom: _sample_N_documents_by_their_ID")

    stratification_col = loop_config.sampling_method["stratified"]
    balance = loop_config.sampling_method["balance"]
    df_for_ID_sampling = df.copy()
    if stratification_col is None:
        # Create dummy stratification column
        df_for_ID_sampling["stratification_col"] = 0
        stratification_col = "stratification_col"

    # Switch from LABEL/not-LABEL to 1/0 for easier distribution calculation
    df_for_ID_sampling["LABEL"] = df_for_ID_sampling["LABEL"].map(loop_config.label2id)
    rng = np.random.default_rng(seed=loop_config.seed)
    N_per_strata = int(loop_config.N_annotated / 
                       df_for_ID_sampling[stratification_col].nunique())

    id_samples = []
    for _, strata_df in df_for_ID_sampling.groupby(stratification_col):
        batch_indexes = []
        available_rows = strata_df.copy().set_index("ID")
        available_rows_indexes = list(available_rows.index)
        for _ in range(N_per_strata):
            local_distrib = available_rows["LABEL"].mean()
            if balance == "random":
                local_weights = None
            else: 
                local_weights = (
                    available_rows
                    ["LABEL"]
                    .map({
                        1: balance / local_distrib, 
                        0 : (1 - balance) / (1 - local_distrib)
                    })
                )
                local_weights = local_weights / sum(local_weights)
            choice = str(rng.choice(available_rows_indexes, p = local_weights))
            # Update for next pick
            batch_indexes += [choice]
            available_rows = available_rows.drop(index=[choice])
            available_rows_indexes.remove(choice)
        id_samples += batch_indexes
    return id_samples

def sample_N_documents(df: pd.DataFrame, loop_config: LoopConfig)->tuple[pd.DataFrame, dict]:
    """
    Sample N elements with cache 
    """
    stratification_col = loop_config.sampling_method["stratified"]
    balance = loop_config.sampling_method["balance"]
    cache_file = (f"{loop_config.dataset_name}-{loop_config.dichotomization_label}-"
        f"{loop_config.N_annotated}-{stratification_col}-{balance}-{loop_config.seed}.csv")
    
    if cache_file in os.listdir("./.cache"):
        id_samples = pd.read_csv(f"./.cache/{cache_file}")["id_samples"].tolist()
    else: 
        id_samples = _sample_N_documents_by_their_ID(df, loop_config)
        pd.Series(id_samples, name="id_samples").to_csv(f"./.cache/{cache_file}", index=False)

    out_df = df.loc[np.isin(df["ID"], id_samples)]
    label, count = np.unique_counts(out_df['LABEL'])
    effective_distrib = {l:float(c / sum(count)) for l,c in zip(label, count)}
    return out_df, effective_distrib

def split_ds(N_documents: dict[str:dict], loop_config: LoopConfig)-> DatasetDict:
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
    unique_IDs = (
        pd.Series(list(set([d["ID"] for d in N_documents.values()]))) # Unique
        .sample(frac=1, random_state=loop_config.seed) # Shuffle
    )
    N_ids = len(unique_IDs)
    ids_train = unique_IDs.head(splits_ratio[0] * N_ids // 100)
    ids_test = unique_IDs.tail(splits_ratio[2] * N_ids // 100)
    ids_eval = unique_IDs.drop(index=[*ids_train.index.to_list(), *ids_test.index.to_list()])

    out_dsd = DatasetDict({
        "train": Dataset.from_list([d for d in N_documents.values() if d["ID"] in ids_train.values]),
        "eval": Dataset.from_list([d for d in N_documents.values() if d["ID"] in ids_eval.values]),
        "test": Dataset.from_list([d for d in N_documents.values() if d["ID"] in ids_test.values]),
    })

    columns_to_keep = ["ID", "TEXT", "LABEL", "input_ids", "attention_mask", "labels"]
    if "ID_CHUNK" in N_documents.popitem()[1]: 
        columns_to_keep += ["ID_CHUNK"]
    return out_dsd.select_columns(columns_to_keep)

def get_tokenized_texts(
    texts : pd.DataFrame, 
    df_name: str, 
    tokenizer,
    loop_config: LoopConfig
) -> dict[str:dict]:
    """"""
    cache_file = (f"full-tokenized-{df_name}-{loop_config.dataset_name}-"
        f"{loop_config.model_name.replace('/','-')}.json")
    
    if cache_file in os.listdir("./.cache"):
        with open(f"./.cache/{cache_file}", "r") as file:
            output = json.load(file)
    else:
        output = {}
        for batch in tqdm(Dataset.from_pandas(texts).batch(32), desc="Tokenizing texts"):
            tokenized_entry = tokenizer(batch["TEXT"]) # (32, ???) Not padded
            output.update({
                id: {
                    key: tokenized_entry[key][i]
                    for key in tokenized_entry
                }
                for i, id in enumerate(batch["ID"])
            })
        with open(f"./.cache/{cache_file}", "w") as file:
            json.dump(output, file, ensure_ascii=True)
    return output

def join_tokenized_texts(N_documents: dict[str:dict], tokenized_texts:dict[str:dict])->dict:
    N_documents = N_documents.copy()
    for id in N_documents:
        N_documents[id].update({
            **tokenized_texts[id], 
            "ID": id,
            "N_tokens": len(tokenized_texts[id]["input_ids"])
        })
    return N_documents

def chunk_texts(N_documents: dict[str:dict], chunk_length: int, overlap: int) -> DatasetDict:
    """"""
    effective_chunk_length = chunk_length - 2 # to account for "CLS" and "SEP"
    output = {}
    for id_doc, row in N_documents.items():
        if row["N_tokens"] > chunk_length:
            n_indices = row["N_tokens"] - 2 # To account for "CLS" and "SEP"
            s, e, i_chunk = 0, effective_chunk_length, 0
            while s < n_indices:
                # skip chunks that are too small
                if min(e, n_indices) - s < 0.1 * chunk_length: break

                output[f"{id_doc}-{i_chunk}"] = {
                    **{k:v for k,v in row.items() if k not in ["input_ids", "attention_mask"]}, 
                    "input_ids": [
                        row["input_ids"][0], 
                        *row["input_ids"][s:min(e, n_indices)], 
                        row["input_ids"][-1]
                    ], 
                    "attention_mask": [
                        row["attention_mask"][0], 
                        *row["attention_mask"][s:min(e, n_indices)], 
                        row["attention_mask"][-1]
                    ],
                    "ID" : id_doc,
                    "ID_CHUNK" : f"{id_doc}-{i_chunk}"
                }
                s += effective_chunk_length - overlap
                e += effective_chunk_length - overlap
                i_chunk += 1
        else: 
            output[f"{id_doc}-0"] = {
                **row, 
                "ID": id_doc,
                "ID_CHUNK": f"{id_doc}-0"
            }
    return output

def pad_texts(N_documents: dict[str:dict], chunk_length: int, pad_token_id:int)-> dict[str:dict]:
    """"""
    for id_doc in tqdm(N_documents, desc="Padding texts"):
        _n_tok = len(N_documents[id_doc]["input_ids"])
        N_documents[id_doc]["input_ids"] += [pad_token_id] * (chunk_length - _n_tok)
        N_documents[id_doc]["attention_mask"] += [0] * (chunk_length - _n_tok)
    return N_documents

def format_labels(N_documents: dict[str:dict], loop_config: LoopConfig) -> dict[str:dict]:
    """"""
    for id_doc in N_documents:
        N_documents[id_doc]["labels"] = loop_config.label2id[N_documents[id_doc]["LABEL"]]
    return N_documents

def tokenize_chunk_pad(
    df_full : pd.DataFrame, 
    df_sample: pd.DataFrame, 
    df_name: str, 
    loop_config: LoopConfig, 
    force_max_length_capped: int|None=None
) -> tuple[DatasetDict, int]:
    """"""
    tokenizer = load_tokenizer(loop_config)
    tokenized_texts = get_tokenized_texts(df_full[["ID", "TEXT"]], df_name, tokenizer, loop_config) # TODO: implement partial json loader
    
    # swith to dict[ID:row] format for easier and faster formatting
    N_documents = df_sample.set_index("ID").T.to_dict()
    N_documents = join_tokenized_texts(N_documents, tokenized_texts)
    N_documents = format_labels(N_documents, loop_config)
    
    max_n_tokens = get_max_tokens(N_documents)
    max_length_capped = cap_max_length(max_n_tokens, loop_config)
    if force_max_length_capped: 
        if force_max_length_capped > max_length_capped: 
            raise ValueError(f"Requested force_max_length_capped = {force_max_length_capped} "
                f"but model ({loop_config.model_name}) caps inputs at {max_length_capped}")
        max_length_capped = force_max_length_capped
    if max_n_tokens > max_length_capped: 
        N_documents = chunk_texts(N_documents, max_length_capped, loop_config.OVERLAP)
    N_documents = pad_texts(N_documents,max_length_capped,tokenizer.pad_token_id)

    return N_documents, max_length_capped