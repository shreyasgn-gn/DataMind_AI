from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a CSV or XLSX dataset into a Pandas DataFrame.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported file type. Please use a CSV or XLSX file."
        )

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    return pd.read_excel(path)


def inspect_dataset(file_path: str) -> dict:
    """
    Inspect the structure and basic quality of a dataset.
    """

    df = load_dataset(file_path)

    return {
        "file_name": Path(file_path).name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "data_types": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "missing_values": {
            column: int(value)
            for column, value in df.isna().sum().items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_summary": df.describe(
            include="number"
        ).to_dict(),
    }