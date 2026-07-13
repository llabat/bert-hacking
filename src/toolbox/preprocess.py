"""
File with data management, checking and cleaning functions
"""
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
    """Sanitize the dataframe to match the expected keys in single_run.
    For a given dataframe, check that the text col, label col and id col exist 
    then rename them as "TEXT", "LABEL" and "ID".
    Ensures the ID column is indeed unique, and format the column as strings for 
    easier downstream management.
    If extra_cols_to_keep is provided, ensure they exist in the dataframe and 
    include them in the output dataframe"""
    if not np.isin([text_col, label_col, id_col], df.columns).all():
        raise KeyError(
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
    if extra_cols_to_keep: 
        main_columns += extra_cols_to_keep
        if not np.isin(extra_cols_to_keep, df.columns).all():
            raise KeyError(
                f"The columns you provided asked to include (extra_cols_to_keep) "
                "cannot be found in the dataframe. "
                f"You provided extra_cols_to_keep: {extra_cols_to_keep}. "
                f"The dataframe contains: {df.columns}"
            )

    if np.array([df[col].isna().sum() > 0 for col in main_columns]).any():
        raise ValueError(
            f"Missing values: "
            f"\t ID: {df['ID'].isna().sum()}"
            f"\t TEXT: {df['TEXT'].isna().sum()}"
            f"\t LABEL: {df['LABEL'].isna().sum()}"
        )
    # Force ID to be strings
    df["ID"] = df["ID"].astype(str)
    if df["ID"].is_unique:
        return df[main_columns]
    else:
        raise ValueError("ID column contains non-unique values.")

def dichotomize(df: pd.DataFrame, loop_config: LoopConfig) -> tuple[pd.DataFrame, dict[str:int], dict[int:str]]:
    """
    Dichotomize dataframe as ("X"; "not-X") given a label. 
    Create a label2id and id2label as {"X":1,"not-X":0} and vice-versa.
    """
    df = df.copy()
    label = loop_config.dichotomization_label
    if label not in df["LABEL"].values:
        raise ValueError(f"Label ({label}) not in df[\"LABEL\"]. "
                         f"Available labels: {df['LABEL'].unique()}")
    df["LABEL"] = (df["LABEL"] == label).replace({True:label, False:f"not-{label}"})
    label2id = {label:1, f"not-{label}": 0}
    id2label = {1:label, 0: f"not-{label}"}
    return df, label2id, id2label

def cap_max_length(max_n_tokens : int, loop_config: LoopConfig) -> int:
    """Provided a model name, ensures that the required text length can be handled
    by the model and cap it if necessary."""
    model_max = AutoConfig.from_pretrained(loop_config.model_name).max_position_embeddings - 1
    return int(min(max_n_tokens, model_max))

def _sample_N_documents_by_their_ID(df: pd.DataFrame, loop_config: LoopConfig)->list:
    """
    Sample N documents, return the sampled IDs
    At this stage the texts are not chunked, therefore one row = one ID. 

    Allows for: 
    - balance ([0.,1.]): force the balance of True/False labels 
    - stratification (str): ensures the same number of element per strata given a 
        column of the dataframe

    To sample the documents, we pick them one by one, each time updating the 
    probability for picking an ID. This peculiar implementation is due to a numpy 
    bug where the weights are ignored when "replace=False".
    
    We enforce the balance distribution for each strata.
    
    This implementation asserts that there are enough documents in each strata.
    No edgecase catch. Error raised."""

    if not df["ID"].is_unique:
        raise ValueError(("Can't sample if IDs are not unique\n"
            "from: _sample_N_documents_by_their_ID"))

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
    for strata, strata_df in df_for_ID_sampling.groupby(stratification_col):
        # Fill id_sampes one strata at a time
        batch_indexes = []
        available_rows = strata_df.copy().set_index("ID")
        available_rows_indexes = list(available_rows.index)

        if N_per_strata < len(available_rows):
            raise OverflowError((f"Trying to sample {N_per_strata} documents "
                f"but strata {strata} only contains {len(available_rows)} "
                "available rows"))
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
    Sample N documents from the dataframe and return the effective distribution.

    Use cache (as csv in ./.cache).
    One cache file per dataset_name x dichotomization_label x N_annotated x 
        sampling_method x seed
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
    takes the splits_ratio (ex: [80, 10, 10]) and return a DatasetDict containing 
    all N_documents and with the splits_ratio distribution provided.
    
    N_documents is a dictionary of dictionaries. Each dictionary must the following keys: 
    "ID", "TEXT", "LABEL", "input_ids", "attention_mask", "labels"

    Return a dataset dict with the following keys provided before
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
    """Tokenize texts. 
    
    - texts (DataFrame):  must contain a "TEXT" column and an "ID" column.
    - df_name (str): name for caching purposes

    Use cache (json file).
    Output is a dictionary binding the text id (document ID because not yet chunked),
    to the tokenizer output for the TEXT"""
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
    """Join the tokenizer output from tokenized texts (dict[ID:tokenizer output])
    to the N_documents dictionary matching the "ID" """
    N_documents = N_documents.copy()
    for id in N_documents:
        N_documents[id].update({
            **tokenized_texts[id], 
            "ID": id,
            "N_tokens": len(tokenized_texts[id]["input_ids"])
        })
    return N_documents

def chunk_texts(N_documents: dict[str:dict], chunk_length: int, overlap: int) -> DatasetDict:
    """Given a chunk length and overlap, chunk all documents using the provided 
    parameters. Reconstruct the chunks by extracting the tokens (minus CLS and SEP)
    and add the CLS and SEP manually.
    Chunk too small (< 0.1 * chunk length) are ignored
    """
    effective_chunk_length = chunk_length - 2 # to account for "CLS" and "SEP"
    output = {}
    for id_doc, row in N_documents.items():
        if row["N_tokens"] > chunk_length:
            index_max = row["N_tokens"] - 2 # length - 1, -1 to not sample SEP twice
            s, e, i_chunk = 1, effective_chunk_length + 1, 0 # Start at 1 not to  sample CLS twice
            while s < index_max:
                # Ignore chunks that are too small
                if min(e, index_max) - s < 0.1 * chunk_length: break

                output[f"{id_doc}-{i_chunk}"] = {
                    **{k:v for k,v in row.items() if k not in ["input_ids", "attention_mask"]}, 
                    "input_ids": [
                        row["input_ids"][0], # CLS
                        *row["input_ids"][s:min(e, index_max)], 
                        row["input_ids"][-1] # SEP
                    ], 
                    "attention_mask": [
                        row["attention_mask"][0], # CLS
                        *row["attention_mask"][s:min(e, index_max)], 
                        row["attention_mask"][-1] # SEP
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
    """Manually pad the tokenized entries by adding [PAD]s and 0s to the "input_ids"
    and "attention_mask" respectivelly"""
    for id_doc in tqdm(N_documents, desc="Padding texts"):
        _n_tok = len(N_documents[id_doc]["input_ids"])
        N_documents[id_doc]["input_ids"] += [pad_token_id] * (chunk_length - _n_tok)
        N_documents[id_doc]["attention_mask"] += [0] * (chunk_length - _n_tok)
    return N_documents

def format_labels(N_documents: dict[str:dict], loop_config: LoopConfig) -> dict[str:dict]:
    """Format labels as expected by the transformers framework """
    for id_doc in N_documents:
        N_documents[id_doc]["labels"] = loop_config.label2id[N_documents[id_doc]["LABEL"]]
    return N_documents

def tokenize_chunk_pad(
    df_full : pd.DataFrame, 
    df_sample: pd.DataFrame, 
    df_name: str, 
    loop_config: LoopConfig, 
) -> tuple[DatasetDict, int]:
    """Tokenize, chunk and pad a dataframe.
    
    The pipeline works as followed:
    - get the tokenizer output as a dictionary binding text ids with the output
    - bind each text with the tokenizer output
    - format labels
    - if necessary, chunk each text

    df_full: dataframe containing all texts to tokenize the full dataset regardless 
        of the loop config
    df_sample: dataframe containing the texts used for this run, loop config dependent
    df_name: name of the dataframe used for caching
    """
    tokenizer = load_tokenizer(loop_config)
    tokenized_texts = get_tokenized_texts(df_full[["ID", "TEXT"]], df_name, tokenizer, loop_config) # TODO: implement partial json loader
    
    # swith to dict[ID:row] format for easier and faster formatting
    N_documents = df_sample.set_index("ID").T.to_dict()
    N_documents = join_tokenized_texts(N_documents, tokenized_texts)
    N_documents = format_labels(N_documents, loop_config)
    
    max_n_tokens = max([d["N_tokens"] for d in N_documents.values()])
    max_length_capped = cap_max_length(max_n_tokens, loop_config)
    
    if max_n_tokens > max_length_capped: 
        N_documents = chunk_texts(N_documents, max_length_capped, loop_config.OVERLAP)
    N_documents = pad_texts(N_documents,max_length_capped,tokenizer.pad_token_id)

    return N_documents, max_length_capped
