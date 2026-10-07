"""
DataMind AI - Feature Engineering Tools

Provides:
- feature type detection
- identifier detection/removal
- robust datetime parsing
- datetime feature creation
- numerical feature creation
- categorical encoding
- final ML feature preparation
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tools.profile_tools import detect_id_columns


# ============================================================
# FEATURE TYPE DETECTION
# ============================================================

def detect_feature_types(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Detect numerical, categorical, and datetime columns.
    """

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    datetime_columns = df.select_dtypes(
        include=["datetime"]
    ).columns.tolist()

    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
    }


# ============================================================
# DATETIME FEATURES
# ============================================================

def create_datetime_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Extract useful calendar features from datetime columns.

    Important:
    Raw datetime columns are NOT converted to categorical values.
    Only meaningful calendar features are created.
    """

    result = df.copy()
    created_features: list[str] = []

    datetime_columns = result.select_dtypes(
        include=["datetime"]
    ).columns.tolist()

    for column in datetime_columns:
        prefix = column.lower()

        features = {
            f"{prefix}_year": result[column].dt.year,
            f"{prefix}_month": result[column].dt.month,
            f"{prefix}_day": result[column].dt.day,
            f"{prefix}_day_of_week": result[column].dt.dayofweek,
            f"{prefix}_quarter": result[column].dt.quarter,
            f"{prefix}_is_weekend": (
                result[column].dt.dayofweek >= 5
            ).astype(int),
        }

        for feature_name, values in features.items():
            result[feature_name] = values
            created_features.append(feature_name)

    return result, created_features


# ============================================================
# NUMERICAL FEATURES
# ============================================================

def create_numerical_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Create safe mathematical features only when the source
    columns required for the calculation are available.

    Derived features are not created when the corresponding
    source/result column already exists.
    """

    result = df.copy()
    created_features: list[str] = []

    # --------------------------------------------------------
    # Profit
    # --------------------------------------------------------

    if (
        "Sales" in result.columns
        and "Cost" in result.columns
        and "Profit" not in result.columns
    ):
        result["calculated_profit"] = (
            result["Sales"] - result["Cost"]
        )
        created_features.append("calculated_profit")

    # --------------------------------------------------------
    # Profit Margin
    # --------------------------------------------------------

    if (
        "Profit" in result.columns
        and "Sales" in result.columns
        and "Profit_Margin" not in result.columns
    ):
        sales_nonzero = result["Sales"].replace(0, pd.NA)

        result["calculated_profit_margin"] = (
            result["Profit"] / sales_nonzero
        ) * 100

        created_features.append(
            "calculated_profit_margin"
        )

    # --------------------------------------------------------
    # Sales from Quantity * Unit Price
    #
    # Only create when Sales does not already exist.
    # This prevents the feature from being recreated after
    # leakage removal.
    # --------------------------------------------------------

    if (
        "Quantity" in result.columns
        and "Unit_Price" in result.columns
        and "Sales" not in result.columns
    ):
        result["calculated_sales"] = (
            result["Quantity"] * result["Unit_Price"]
        )
        created_features.append("calculated_sales")

    return result, created_features


# ============================================================
# IDENTIFIER REMOVAL
# ============================================================

def remove_identifier_columns(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove ID-like columns before ML processing.
    """

    result = df.copy()

    try:
        identifier_columns = detect_id_columns(result)
    except Exception:
        identifier_columns = []

    if not identifier_columns:
        return result, []

    identifier_columns = [
        column
        for column in identifier_columns
        if column in result.columns
    ]

    if identifier_columns:
        result = result.drop(
            columns=identifier_columns
        )

    return result, identifier_columns


# ============================================================
# CATEGORICAL ENCODING
# ============================================================

def encode_categorical_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    One-hot encode categorical columns.

    Datetime columns are deliberately excluded.
    """

    result = df.copy()

    categorical_columns = result.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    if not categorical_columns:
        return result, []

    encoded = pd.get_dummies(
        result,
        columns=categorical_columns,
        drop_first=False,
        dtype=int,
    )

    created_features = [
        column
        for column in encoded.columns
        if column not in df.columns
    ]

    return encoded, created_features


# ============================================================
# DATE DETECTION / PARSING
# ============================================================

def _is_date_like_column(column: str) -> bool:
    """
    Identify columns whose names strongly indicate date/time data.
    """

    column_lower = column.lower()

    date_keywords = (
        "date",
        "datetime",
        "timestamp",
        "time",
    )

    return any(
        keyword in column_lower
        for keyword in date_keywords
    )


def _convert_date_column(
    series: pd.Series,
) -> tuple[pd.Series, float]:
    """
    Robustly convert a column to datetime.

    Uses format='mixed' so values such as:
    19-05-2025
    2025-05-19
    19/05/2025

    can be handled without forcing the date itself into
    one-hot encoded categorical features.
    """

    # First attempt: mixed-format parsing with day-first support.
    try:
        converted = pd.to_datetime(
            series,
            errors="coerce",
            format="mixed",
            dayfirst=True,
        )

        valid_ratio = float(converted.notna().mean())

        if valid_ratio >= 0.8:
            return converted, valid_ratio

    except Exception:
        pass

    # Fallback: normal pandas datetime parsing.
    try:
        converted = pd.to_datetime(
            series,
            errors="coerce",
        )

        valid_ratio = float(converted.notna().mean())

        return converted, valid_ratio

    except Exception:
        return series, 0.0


# ============================================================
# MAIN FEATURE PREPARATION
# ============================================================

def prepare_ml_features(
    df: pd.DataFrame,
    encode_categories: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Prepare a dataset for machine learning.

    Steps:
    1. Copy input data.
    2. Remove identifier columns.
    3. Robustly convert date/time columns.
    4. Create calendar features.
    5. Create safe numerical features.
    6. Remove raw datetime columns.
    7. One-hot encode categorical columns.
    8. Validate the final result.
    """

    result = df.copy()

    report: dict[str, Any] = {
        "original_rows": len(result),
        "original_columns": len(result.columns),
        "identifier_columns_removed": [],
        "datetime_features_created": [],
        "numerical_features_created": [],
        "categorical_features_created": [],
        "datetime_columns_removed": [],
        "final_rows": 0,
        "final_columns": 0,
        "remaining_missing_values": 0,
        "feature_types": {},
    }

    # --------------------------------------------------------
    # 1. Remove identifier columns first
    # --------------------------------------------------------

    (
        result,
        identifier_columns,
    ) = remove_identifier_columns(result)

    report[
        "identifier_columns_removed"
    ] = identifier_columns

    # --------------------------------------------------------
    # 2. Convert recognized date columns
    # --------------------------------------------------------

    for column in list(result.columns):

        # Already datetime
        if pd.api.types.is_datetime64_any_dtype(
            result[column]
        ):
            continue

        if not _is_date_like_column(column):
            continue

        converted, valid_ratio = _convert_date_column(
            result[column]
        )

        if valid_ratio >= 0.8:
            result[column] = converted

    # --------------------------------------------------------
    # 3. Create datetime features
    # --------------------------------------------------------

    (
        result,
        date_features,
    ) = create_datetime_features(result)

    report[
        "datetime_features_created"
    ] = date_features

    # --------------------------------------------------------
    # 4. Create numerical features
    # --------------------------------------------------------

    (
        result,
        numerical_features,
    ) = create_numerical_features(result)

    report[
        "numerical_features_created"
    ] = numerical_features

    # --------------------------------------------------------
    # 5. Remove raw datetime columns
    # --------------------------------------------------------

    datetime_columns = result.select_dtypes(
        include=["datetime"]
    ).columns.tolist()

    if datetime_columns:
        result = result.drop(
            columns=datetime_columns
        )

        report[
            "datetime_columns_removed"
        ] = datetime_columns

    # --------------------------------------------------------
    # 6. Encode categorical features
    # --------------------------------------------------------

    if encode_categories:
        (
            result,
            categorical_features,
        ) = encode_categorical_features(result)

        report[
            "categorical_features_created"
        ] = categorical_features

    # --------------------------------------------------------
    # 7. Final validation
    # --------------------------------------------------------

    report["final_rows"] = len(result)

    report["final_columns"] = len(
        result.columns
    )

    report[
        "remaining_missing_values"
    ] = int(result.isna().sum().sum())

    report["feature_types"] = detect_feature_types(
        result
    )

    return result, report