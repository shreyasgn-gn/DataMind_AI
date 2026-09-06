import re

import numpy as np
import pandas as pd


def normalize_column_name(column: str) -> str:
    """
    Normalize a column name for relationship detection.
    """

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(column).lower(),
    )


def detect_algebraic_leakage(
    df: pd.DataFrame,
    target_column: str,
    tolerance: float = 1e-6,
) -> dict:
    """
    Detect simple arithmetic relationships between numeric
    features and the target.

    Currently checks whether the target can be reproduced by:

        feature_a + feature_b
        feature_a - feature_b
        feature_a * feature_b
        feature_a / feature_b

    This is intentionally conservative and is meant to catch
    obvious target leakage rather than discover every possible
    mathematical relationship.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "does not exist."
        )

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    numeric_columns = [
        column
        for column in numeric_columns
        if column != target_column
    ]

    if not numeric_columns:
        return {
            "target_column": target_column,
            "leakage_detected": False,
            "relationships": [],
        }

    target = pd.to_numeric(
        df[target_column],
        errors="coerce",
    )

    relationships = []

    for i, left_column in enumerate(
        numeric_columns
    ):

        left = pd.to_numeric(
            df[left_column],
            errors="coerce",
        )

        for right_column in numeric_columns[
            i + 1:
        ]:

            right = pd.to_numeric(
                df[right_column],
                errors="coerce",
            )

            comparisons = {
                "addition": left + right,
                "subtraction_left_right": (
                    left - right
                ),
                "subtraction_right_left": (
                    right - left
                ),
                "multiplication": (
                    left * right
                ),
                "division_left_right": (
                    left.div(
                        right.replace(
                            0,
                            np.nan,
                        )
                    )
                ),
                "division_right_left": (
                    right.div(
                        left.replace(
                            0,
                            np.nan,
                        )
                    )
                ),
            }

            for operation, calculated in (
                comparisons.items()
            ):

                valid_mask = (
                    target.notna()
                    & calculated.notna()
                )

                if valid_mask.sum() == 0:
                    continue

                actual = target[
                    valid_mask
                ].to_numpy()

                predicted = calculated[
                    valid_mask
                ].to_numpy()

                difference = np.abs(
                    actual - predicted
                )

                # Scale-aware tolerance
                scale = np.maximum(
                    np.abs(actual),
                    1.0,
                )

                relative_error = (
                    difference / scale
                )

                match_ratio = float(
                    (
                        relative_error
                        <= tolerance
                    ).mean()
                )

                if match_ratio >= 0.99:

                    relationships.append({
                        "features": [
                            left_column,
                            right_column,
                        ],
                        "operation": operation,
                        "match_ratio": round(
                            match_ratio,
                            4,
                        ),
                        "description": (
                            f"{target_column} appears to be "
                            f"reproducible from "
                            f"{left_column} and "
                            f"{right_column} using "
                            f"{operation}."
                        ),
                    })

    return {
        "target_column": target_column,
        "leakage_detected": bool(
            relationships
        ),
        "relationships": relationships,
    }


def detect_name_based_leakage(
    df: pd.DataFrame,
    target_column: str,
) -> dict:
    """
    Detect columns whose names strongly suggest they are
    derived versions of the target.

    Examples:
        Profit -> Profit_Margin
        Revenue -> Revenue_Percentage
        Sales -> Sales_Ratio
    """

    target_normalized = normalize_column_name(
        target_column
    )

    derived_terms = (
        "margin",
        "ratio",
        "rate",
        "percentage",
        "percent",
        "share",
        "proportion",
        "normalized",
        "derived",
        "calculated",
        "prediction",
    )

    suspicious_columns = []

    for column in df.columns:

        if column == target_column:
            continue

        normalized = normalize_column_name(
            column
        )

        if (
            target_normalized in normalized
            and any(
                term in normalized
                for term in derived_terms
            )
        ):

            suspicious_columns.append({
                "column": column,
                "reason": (
                    f"Column name suggests that "
                    f"'{column}' may be derived from "
                    f"target '{target_column}'."
                ),
            })

    return {
        "target_column": target_column,
        "suspicious_columns": suspicious_columns,
    }


def detect_target_leakage(
    df: pd.DataFrame,
    target_column: str,
) -> dict:
    """
    Run conservative target-leakage checks.

    The tool distinguishes between:
    - confirmed mathematical leakage
    - suspicious name-based leakage

    It does not automatically delete any columns.
    """

    algebraic = detect_algebraic_leakage(
        df,
        target_column,
    )

    name_based = detect_name_based_leakage(
        df,
        target_column,
    )

    confirmed_columns = set()

    for relationship in algebraic[
        "relationships"
    ]:

        for column in relationship[
            "features"
        ]:
            confirmed_columns.add(column)

    suspicious_columns = [
        item["column"]
        for item in name_based[
            "suspicious_columns"
        ]
    ]

    return {
        "target_column": target_column,
        "leakage_detected": (
            algebraic["leakage_detected"]
            or bool(suspicious_columns)
        ),
        "confirmed_leakage_columns": sorted(
            confirmed_columns
        ),
        "suspicious_leakage_columns": (
            suspicious_columns
        ),
        "algebraic_relationships": (
            algebraic["relationships"]
        ),
        "name_based_findings": (
            name_based["suspicious_columns"]
        ),
    }


def remove_leakage_features(
    df: pd.DataFrame,
    leakage_report: dict,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove only confirmed leakage columns.

    Suspicious name-based columns are retained because they
    require additional evidence before automatic removal.
    """

    columns_to_remove = leakage_report.get(
        "confirmed_leakage_columns",
        [],
    )

    columns_to_remove = [
        column
        for column in columns_to_remove
        if column in df.columns
    ]

    cleaned_df = df.drop(
        columns=columns_to_remove
    )

    return (
        cleaned_df,
        columns_to_remove,
    )