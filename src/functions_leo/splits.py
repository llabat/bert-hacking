from sklearn.model_selection import train_test_split

def split_ds(
    df,
    train_size=0.8,
    validation_size=0.1,
    test_size=0.1,
    random_state=42,
    stratify=False,
    label_column="LABEL",
):
    """
    Split a dataframe into train / validation / test sets.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    train_size : float
        Proportion of data to use for training.
    validation_size : float
        Proportion of data to use for validation.
    test_size : float
        Proportion of data to use for test.
    random_state : int
        Random seed for reproducibility.
    stratify : bool
        Whether to stratify splits based on the label column.
    label_column : str
        Column used for stratification if stratify=True.

    Returns
    -------
    dict
        Dictionary with keys:
        - "train"
        - "validation"
        - "test"
    """
    total = train_size + validation_size + test_size
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            f"train_size + validation_size + test_size must sum to 1.0, got {total}"
        )

    if len(df) == 0:
        raise ValueError("Cannot split an empty dataframe")

    if stratify:
        if label_column not in df.columns:
            raise ValueError(
                f"Column '{label_column}' not found, cannot stratify"
            )
        stratify_values = df[label_column]
    else:
        stratify_values = None

    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        stratify=stratify_values,
    )

    temp_size = validation_size + test_size
    if temp_size == 0:
        return {
            "train": train_df.reset_index(drop=True),
            "validation": None,
            "test": None,
        }

    if stratify:
        temp_stratify = temp_df[label_column]
    else:
        temp_stratify = None

    relative_validation_size = validation_size / temp_size

    validation_df, test_df = train_test_split(
        temp_df,
        train_size=relative_validation_size,
        random_state=random_state,
        stratify=temp_stratify,
    )

    return {
        "train": train_df.reset_index(drop=True),
        "validation": validation_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }