import re

import numpy as np
import pandas as pd


def normalize_column_name(
    column: str,
) -> str:
    """
    Normalize a column name for relationship detection.
    """

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(column).lower(),
    )


def _safe_match_ratio(
    actual: pd.Series,
    calculated: pd.Series,
    relative_tolerance: float = 1e-6,
    absolute_tolerance: float = 0.01,
) -> float:
    """
    Calculate how many valid rows match within a combined
    absolute + relative tolerance.

    This supports derived columns that were rounded before
    being saved to CSV/XLSX.

    Example:
        Profit_Margin = round(Profit / Sales * 100, 2)
    """

    valid_mask = (
        actual.notna()
        & calculated.notna()
        & np.isfinite(
            calculated
        )
        & np.isfinite(
            actual
        )
    )

    if valid_mask.sum() == 0:
        return 0.0

    actual_values = actual[
        valid_mask
    ].to_numpy(
        dtype=float
    )

    calculated_values = calculated[
        valid_mask
    ].to_numpy(
        dtype=float
    )

    difference = np.abs(
        actual_values
        - calculated_values
    )

    scale = np.maximum(
        np.abs(actual_values),
        1.0,
    )

    allowed_error = np.maximum(
        absolute_tolerance,
        relative_tolerance * scale,
    )

    matches = (
        difference
        <= allowed_error
    )

    return float(
        matches.mean()
    )


def detect_algebraic_leakage(
    df: pd.DataFrame,
    target_column: str,
    relative_tolerance: float = 1e-6,
    absolute_tolerance: float = 0.01,
) -> dict:
    """
    Detect simple arithmetic relationships between numeric
    features and the target.

    Supported relationships:
    - addition
    - subtraction
    - multiplication
    - division
    - percentage ratios

    The detector allows small rounding differences common
    in CSV/XLSX derived columns.
    """

    if target_column not in df.columns:

        raise ValueError(
            f"Target column '{target_column}' "
            "does not exist."
        )

    numeric_columns = (
        df.select_dtypes(
            include="number"
        ).columns.tolist()
    )

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

            calculations = {
                "addition": (
                    left + right
                ),

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

                "percentage_left_right": (
                    left.div(
                        right.replace(
                            0,
                            np.nan,
                        )
                    ) * 100
                ),

                "percentage_right_left": (
                    right.div(
                        left.replace(
                            0,
                            np.nan,
                        )
                    ) * 100
                ),
            }

            for (
                operation,
                calculated,
            ) in calculations.items():

                match_ratio = (
                    _safe_match_ratio(
                        target,
                        calculated,
                        relative_tolerance=(
                            relative_tolerance
                        ),
                        absolute_tolerance=(
                            absolute_tolerance
                        ),
                    )
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
                            f"{target_column} appears "
                            "to be reproducible from "
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
    Detect columns whose names strongly suggest they
    are derived versions of the target.
    """

    target_normalized = (
        normalize_column_name(
            target_column
        )
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
        "predicted",
        "forecast",
    )

    suspicious_columns = []

    for column in df.columns:

        if column == target_column:
            continue

        normalized = normalize_column_name(
            column
        )

        if (
            target_normalized
            in normalized
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
        "suspicious_columns": (
            suspicious_columns
        ),
    }


def detect_target_leakage(
    df: pd.DataFrame,
    target_column: str,
) -> dict:
    """
    Run conservative target-leakage checks.
    """

    algebraic = (
        detect_algebraic_leakage(
            df,
            target_column,
        )
    )

    name_based = (
        detect_name_based_leakage(
            df,
            target_column,
        )
    )

    confirmed_columns = set()

    for relationship in (
        algebraic[
            "relationships"
        ]
    ):

        for column in relationship[
            "features"
        ]:

            confirmed_columns.add(
                column
            )

    suspicious_columns = [
        item["column"]
        for item in name_based[
            "suspicious_columns"
        ]
    ]

    return {
        "target_column": target_column,

        "leakage_detected": (
            algebraic[
                "leakage_detected"
            ]
            or bool(
                suspicious_columns
            )
        ),

        "confirmed_leakage_columns": (
            sorted(
                confirmed_columns
            )
        ),

        "suspicious_leakage_columns": (
            suspicious_columns
        ),

        "algebraic_relationships": (
            algebraic[
                "relationships"
            ]
        ),

        "name_based_findings": (
            name_based[
                "suspicious_columns"
            ]
        ),
    }


def remove_leakage_features(
    df: pd.DataFrame,
    leakage_report: dict,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove confirmed mathematical leakage columns.

    Suspicious name-based columns are not removed here.
    The orchestrator decides how to handle them.
    """

    columns_to_remove = (
        leakage_report.get(
            "confirmed_leakage_columns",
            [],
        )
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