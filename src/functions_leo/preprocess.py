import pandas as pd

def sanitize_df(
    df,
    text_column,
    label_column,
    id_column=None,
    drop_na_text=True,
    drop_na_label=True,
    enforce_unique_id=False,
    copy=True,
):
    """
    Standardize a dataframe by renaming the main columns to:
    - TEXT
    - LABEL
    - ID (optional)

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    text_column : str
        Name of the text column in the original dataframe.
    label_column : str
        Name of the label column in the original dataframe.
    id_column : str or None
        Name of the id column in the original dataframe.
    drop_na_text : bool
        Whether to drop rows with missing text.
    drop_na_label : bool
        Whether to drop rows with missing labels.
    enforce_unique_id : bool
        Whether to require IDs to be unique.
    copy : bool
        Whether to work on a copy of the dataframe.

    Returns
    -------
    pandas.DataFrame
        Sanitized dataframe.
    """
    if copy:
        df = df.copy()

    required_columns = [text_column, label_column]
    if id_column is not None:
        required_columns.append(id_column)

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    rename_map = {
        text_column: "TEXT",
        label_column: "LABEL",
    }

    if id_column is not None:
        rename_map[id_column] = "ID"

    df = df.rename(columns=rename_map)

    columns_to_keep = ["TEXT", "LABEL"]
    if id_column is not None:
        columns_to_keep.append("ID")
    df = df[columns_to_keep]

    if drop_na_text:
        df = df.dropna(subset=["TEXT"])

    if drop_na_label:
        df = df.dropna(subset=["LABEL"])

    df["TEXT"] = df["TEXT"].astype(str).str.strip()

    if drop_na_text:
        df = df[df["TEXT"] != ""]

    if id_column is not None and enforce_unique_id:
        if df["ID"].duplicated().any():
            duplicated_ids = df.loc[df["ID"].duplicated(), "ID"].tolist()
            raise ValueError(f"Duplicate IDs found: {duplicated_ids[:10]}")

    return df.reset_index(drop=True)

def make_ovr_dataset(df, target_class, label_column="LABEL"):
    """
    Extract a binary dataset (One-versus-Rest) from a dataset given a label
    """
    df = df.copy()
    df[label_column] = (df[label_column] == target_class).astype(int)
    return df
