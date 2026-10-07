import re

import pandas as pd


def _parse_datetime_series(
    series: pd.Series,
) -> tuple[pd.Series, float]:
    """
    Robustly parse a possible datetime series.

    Supports mixed formats such as:
    - 19-05-2025
    - 19/05/2025
    - 2025-05-19
    - 2025/05/19
    """

    non_null = series.dropna()

    if non_null.empty:
        return (
            pd.Series(
                index=series.index,
                dtype="datetime64[ns]",
            ),
            0.0,
        )

    try:
        converted = pd.to_datetime(
            non_null,
            errors="coerce",
            format="mixed",
            dayfirst=True,
        )

        valid_ratio = float(
            converted.notna().mean()
        )

        if valid_ratio >= 0.8:
            return converted, valid_ratio

    except Exception:
        pass

    try:
        converted = pd.to_datetime(
            non_null,
            errors="coerce",
        )

        valid_ratio = float(
            converted.notna().mean()
        )

        return converted, valid_ratio

    except Exception:
        return (
            pd.Series(
                index=non_null.index,
                dtype="datetime64[ns]",
            ),
            0.0,
        )


def detect_datetime_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Detect columns that contain genuine date/time information.

    Uses:
    - actual pandas datetime dtype
    - column-name signals
    - recognizable date patterns
    - robust mixed-format parsing

    Avoids incorrectly classifying:
    - January
    - February
    - Q1
    - Q2
    as dates.
    """

    datetime_columns: list[str] = []

    date_name_indicators = (
        "date",
        "datetime",
        "timestamp",
        "time",
    )

    date_pattern = re.compile(
        r"""
        ^
        (
            \d{4}[-/]\d{1,2}[-/]\d{1,2}
            |
            \d{1,2}[-/]\d{1,2}[-/]\d{4}
        )
        $
        """,
        re.VERBOSE,
    )

    for column in df.columns:

        series = df[column]

        # --------------------------------------------------------
        # Already a real datetime column
        # --------------------------------------------------------

        if pd.api.types.is_datetime64_any_dtype(
            series
        ):
            datetime_columns.append(
                column
            )
            continue

        # --------------------------------------------------------
        # Only inspect text-like columns
        # --------------------------------------------------------

        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            continue

        column_lower = column.lower()

        has_date_name = any(
            indicator in column_lower
            for indicator in date_name_indicators
        )

        non_null = (
            series
            .dropna()
            .astype(str)
            .str.strip()
        )

        if non_null.empty:
            continue

        # --------------------------------------------------------
        # Strong column-name signal
        # --------------------------------------------------------

        if has_date_name:

            _, valid_ratio = (
                _parse_datetime_series(
                    series
                )
            )

            if valid_ratio >= 0.8:
                datetime_columns.append(
                    column
                )

            continue

        # --------------------------------------------------------
        # Generic date-pattern detection
        # --------------------------------------------------------

        date_pattern_ratio = (
            non_null
            .apply(
                lambda value: bool(
                    date_pattern.match(
                        value
                    )
                )
            )
            .mean()
        )

        if date_pattern_ratio >= 0.8:

            _, valid_ratio = (
                _parse_datetime_series(
                    series
                )
            )

            if valid_ratio >= 0.8:
                datetime_columns.append(
                    column
                )

    return datetime_columns


def detect_id_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Detect columns that appear to be identifiers.
    """

    id_columns: list[str] = []

    for column in df.columns:

        column_lower = column.lower()

        unique_ratio = (
            df[column].nunique(
                dropna=False
            )
            / len(df)
            if len(df) > 0
            else 0
        )

        # Explicit identifier naming
        if (
            column_lower.endswith("_id")
            or column_lower == "id"
            or "identifier" in column_lower
        ):
            id_columns.append(
                column
            )

        # Very high-cardinality text
        elif (
            unique_ratio >= 0.95
            and (
                pd.api.types.is_object_dtype(
                    df[column]
                )
                or pd.api.types.is_string_dtype(
                    df[column]
                )
            )
        ):
            id_columns.append(
                column
            )

    return id_columns


def profile_dataset(
    df: pd.DataFrame,
) -> dict:
    """
    Create an intelligent high-level profile.

    Date-like columns are explicitly removed from the
    categorical list and reported as datetime columns.
    Identifier columns are also excluded from categorical data.
    """

    numeric_columns = (
        df.select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    datetime_columns = (
        detect_datetime_columns(
            df
        )
    )

    id_columns = (
        detect_id_columns(
            df
        )
    )

    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "string",
                "category",
            ]
        )
        .columns
        .tolist()
    )

    categorical_columns = [
        column
        for column in categorical_columns
        if column not in datetime_columns
        and column not in id_columns
    ]

    return {
        "rows": int(
            df.shape[0]
        ),

        "columns": int(
            df.shape[1]
        ),

        "numeric_columns": (
            numeric_columns
        ),

        "categorical_columns": (
            categorical_columns
        ),

        "datetime_columns": (
            datetime_columns
        ),

        "id_columns": (
            id_columns
        ),

        "missing_values": {
            column: int(value)
            for column, value
            in df.isna().sum().items()
            if value > 0
        },

        "duplicate_rows": int(
            df.duplicated().sum()
        ),

        "unique_values": {
            column: int(
                df[column].nunique()
            )
            for column in df.columns
        },
    }