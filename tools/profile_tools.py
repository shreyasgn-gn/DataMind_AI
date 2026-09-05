import pandas as pd


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Create a high-level profile of a dataset.
    """

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    datetime_columns = df.select_dtypes(
        include=["datetime"]
    ).columns.tolist()

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "missing_values": {
            column: int(value)
            for column, value in df.isna().sum().items()
            if value > 0
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_values": {
            column: int(df[column].nunique())
            for column in df.columns
        },
    }