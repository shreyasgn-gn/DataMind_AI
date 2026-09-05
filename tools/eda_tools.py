import pandas as pd


def perform_eda(df: pd.DataFrame) -> dict:
    """
    Perform basic exploratory data analysis.
    """

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    result = {
        "numeric_summary": {},
        "categorical_summary": {},
        "correlations": {},
    }

    if numeric_columns:
        result["numeric_summary"] = (
            df[numeric_columns]
            .describe()
            .round(2)
            .to_dict()
        )

        correlation_matrix = (
            df[numeric_columns]
            .corr()
            .round(3)
        )

        result["correlations"] = correlation_matrix.to_dict()

    for column in categorical_columns:
        result["categorical_summary"][column] = (
            df[column]
            .value_counts(dropna=False)
            .head(10)
            .to_dict()
        )

    return result