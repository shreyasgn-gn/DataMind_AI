import pandas as pd


def analyze_data_quality(df: pd.DataFrame) -> dict:
    """
    Analyze common data-quality issues without modifying the dataset.
    """

    missing_values = {
        column: int(value)
        for column, value in df.isna().sum().items()
        if value > 0
    }

    duplicate_rows = int(df.duplicated().sum())

    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

    return {
        "missing_values": missing_values,
        "total_missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "has_quality_issues": bool(
            missing_values
            or duplicate_rows
            or constant_columns
        ),
    }