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


def detect_outliers(
    df: pd.DataFrame,
    multiplier: float = 1.5
) -> dict:
    """
    Detect numerical outliers using the IQR method.

    Returns the number of outliers and the calculated bounds
    for each numerical column.
    """

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    outlier_report = {}

    for column in numeric_columns:

        series = df[column].dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        outlier_mask = (
            (df[column] < lower_bound)
            | (df[column] > upper_bound)
        )

        outlier_count = int(outlier_mask.sum())

        outlier_report[column] = {
            "q1": round(float(q1), 4),
            "q3": round(float(q3), 4),
            "iqr": round(float(iqr), 4),
            "lower_bound": round(float(lower_bound), 4),
            "upper_bound": round(float(upper_bound), 4),
            "outlier_count": outlier_count,
        }

    return outlier_report


def suggest_cleaning_actions(
    df: pd.DataFrame
) -> list:
    """
    Suggest appropriate cleaning actions based on detected
    data-quality issues.

    This function does not modify the dataset.
    """

    actions = []

    # Missing values
    for column, count in df.isna().sum().items():

        if count > 0:

            if pd.api.types.is_numeric_dtype(df[column]):
                strategy = "median"

            else:
                strategy = "mode"

            actions.append({
                "issue": "missing_values",
                "column": column,
                "count": int(count),
                "recommended_action": (
                    f"Fill missing values using {strategy}"
                ),
            })

    # Duplicate rows
    duplicate_count = int(df.duplicated().sum())

    if duplicate_count > 0:
        actions.append({
            "issue": "duplicate_rows",
            "column": None,
            "count": duplicate_count,
            "recommended_action": (
                "Remove duplicate rows"
            ),
        })

    # Constant columns
    for column in df.columns:

        if df[column].nunique(dropna=False) <= 1:

            actions.append({
                "issue": "constant_column",
                "column": column,
                "count": int(len(df)),
                "recommended_action": (
                    "Review and consider removing the column"
                ),
            })

    # Outliers
    outlier_report = detect_outliers(df)

    for column, details in outlier_report.items():

        count = details["outlier_count"]

        if count > 0:

            actions.append({
                "issue": "outliers",
                "column": column,
                "count": count,
                "recommended_action": (
                    "Review extreme values before deciding "
                    "whether to cap, transform, or retain them"
                ),
            })

    return actions


def clean_dataset(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """
    Clean common data-quality issues using conservative strategies.

    Cleaning rules:
    - Numeric missing values → median
    - Categorical missing values → mode
    - Duplicate rows → removed
    - Constant columns → retained but reported
    - Outliers → retained and reported

    Returns:
        cleaned_dataframe
        cleaning_report
    """

    cleaned_df = df.copy()

    original_rows = len(cleaned_df)
    original_columns = len(cleaned_df.columns)

    report = {
        "original_rows": original_rows,
        "original_columns": original_columns,
        "missing_values_filled": {},
        "duplicate_rows_removed": 0,
        "constant_columns": [],
        "outliers_detected": {},
    }

    # ---------------------------------------------------------
    # 1. Handle missing values
    # ---------------------------------------------------------

    for column in cleaned_df.columns:

        missing_count = int(
            cleaned_df[column].isna().sum()
        )

        if missing_count == 0:
            continue

        if pd.api.types.is_numeric_dtype(
            cleaned_df[column]
        ):

            fill_value = cleaned_df[column].median()
            strategy = "median"

        else:

            mode = cleaned_df[column].mode(dropna=True)

            if mode.empty:
                continue

            fill_value = mode.iloc[0]
            strategy = "mode"

        cleaned_df[column] = cleaned_df[column].fillna(
            fill_value
        )

        report["missing_values_filled"][column] = {
            "count": missing_count,
            "strategy": strategy,
            "fill_value": (
                float(fill_value)
                if pd.api.types.is_numeric_dtype(
                    cleaned_df[column]
                )
                else str(fill_value)
            ),
        }

    # ---------------------------------------------------------
    # 2. Remove duplicate rows
    # ---------------------------------------------------------

    duplicate_count = int(
        cleaned_df.duplicated().sum()
    )

    if duplicate_count > 0:

        cleaned_df = cleaned_df.drop_duplicates()

        report["duplicate_rows_removed"] = duplicate_count

    # ---------------------------------------------------------
    # 3. Identify constant columns
    # ---------------------------------------------------------

    report["constant_columns"] = [
        column
        for column in cleaned_df.columns
        if cleaned_df[column].nunique(dropna=False) <= 1
    ]

    # ---------------------------------------------------------
    # 4. Detect outliers
    # ---------------------------------------------------------

    outlier_report = detect_outliers(cleaned_df)

    report["outliers_detected"] = {
        column: details["outlier_count"]
        for column, details in outlier_report.items()
        if details["outlier_count"] > 0
    }

    # ---------------------------------------------------------
    # 5. Final validation
    # ---------------------------------------------------------

    report["final_rows"] = len(cleaned_df)
    report["final_columns"] = len(cleaned_df.columns)

    report["remaining_missing_values"] = int(
        cleaned_df.isna().sum().sum()
    )

    report["remaining_duplicate_rows"] = int(
        cleaned_df.duplicated().sum()
    )

    report["rows_removed"] = (
        original_rows - len(cleaned_df)
    )

    return cleaned_df, report