import pandas as pd

from tools.profile_tools import detect_id_columns


def detect_feature_types(
    df: pd.DataFrame,
) -> dict:
    """
    Identify columns by their role for feature engineering.
    """

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=[
            "object",
            "string",
            "category",
        ]
    ).columns.tolist()

    datetime_columns = df.select_dtypes(
        include=["datetime"]
    ).columns.tolist()

    id_columns = detect_id_columns(
        df
    )

    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "id_columns": id_columns,
    }


def create_datetime_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list]:
    """
    Extract useful calendar features from datetime columns.

    Original datetime columns are retained temporarily.
    """

    result = df.copy()

    created_features = []

    datetime_columns = result.select_dtypes(
        include=["datetime"]
    ).columns.tolist()

    for column in datetime_columns:

        prefix = column.lower()

        features = {
            f"{prefix}_year": (
                result[column].dt.year
            ),
            f"{prefix}_month": (
                result[column].dt.month
            ),
            f"{prefix}_day": (
                result[column].dt.day
            ),
            f"{prefix}_day_of_week": (
                result[column].dt.dayofweek
            ),
            f"{prefix}_quarter": (
                result[column].dt.quarter
            ),
            f"{prefix}_is_weekend": (
                result[column].dt.dayofweek >= 5
            ),
        }

        for feature_name, values in (
            features.items()
        ):

            if feature_name not in result.columns:

                result[feature_name] = values

                created_features.append(
                    feature_name
                )

    return (
        result,
        created_features,
    )


def create_numerical_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list]:
    """
    Create conservative numerical features when
    an obvious relationship exists.
    """

    result = df.copy()

    created_features = []

    columns_lower = {
        column.lower(): column
        for column in result.columns
    }

    if (
        "sales" in columns_lower
        and "cost" in columns_lower
        and "profit" not in columns_lower
    ):

        sales = columns_lower["sales"]
        cost = columns_lower["cost"]

        result["calculated_profit"] = (
            result[sales] - result[cost]
        )

        created_features.append(
            "calculated_profit"
        )

    if (
        "profit" in columns_lower
        and "sales" in columns_lower
        and "profit_margin" not in columns_lower
    ):

        profit = columns_lower["profit"]
        sales = columns_lower["sales"]

        result["calculated_profit_margin"] = (
            result[profit]
            .div(
                result[sales].replace(
                    0,
                    pd.NA,
                )
            )
            .mul(100)
        )

        created_features.append(
            "calculated_profit_margin"
        )

    if (
        "quantity" in columns_lower
        and "unit_price" in columns_lower
        and "sales" not in columns_lower
    ):

        quantity = columns_lower[
            "quantity"
        ]

        unit_price = columns_lower[
            "unit_price"
        ]

        result["calculated_sales"] = (
            result[quantity]
            * result[unit_price]
        )

        created_features.append(
            "calculated_sales"
        )

    return (
        result,
        created_features,
    )


def remove_identifier_columns(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list]:
    """
    Remove columns that appear to be identifiers.

    Identifiers are excluded from ML features because
    arbitrary IDs should not be treated as meaningful
    predictive variables.

    Examples:
    - Order_ID
    - Customer_ID
    - ID
    - identifier-like columns
    """

    result = df.copy()

    id_columns = detect_id_columns(
        result
    )

    removable_columns = [
        column
        for column in id_columns
        if column in result.columns
    ]

    if removable_columns:

        result = result.drop(
            columns=removable_columns
        )

    return (
        result,
        removable_columns,
    )


def encode_categorical_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list]:
    """
    One-hot encode categorical columns.

    Original categorical columns are replaced by
    one-hot encoded features.
    """

    result = df.copy()

    categorical_columns = (
        result.select_dtypes(
            include=[
                "object",
                "string",
                "category",
            ]
        ).columns.tolist()
    )

    if not categorical_columns:

        return (
            result,
            [],
        )

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

    return (
        encoded,
        created_features,
    )


def prepare_ml_features(
    df: pd.DataFrame,
    encode_categories: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Prepare a dataset for machine-learning workflows.

    Steps:
    1. Detect and remove identifier columns.
    2. Convert recognized dates to datetime.
    3. Create date-based features.
    4. Create safe numerical features.
    5. Remove raw datetime columns.
    6. Optionally one-hot encode categorical columns.
    7. Validate the final dataset.
    """

    result = df.copy()

    report = {
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

        "data_types": {},
    }

    # ---------------------------------------------
    # 1. Remove identifier columns
    # ---------------------------------------------

    (
        result,
        identifier_columns,
    ) = remove_identifier_columns(
        result
    )

    report[
        "identifier_columns_removed"
    ] = identifier_columns

    # ---------------------------------------------
    # 2. Convert recognized date columns
    # ---------------------------------------------

    for column in result.columns:

        if pd.api.types.is_datetime64_any_dtype(
            result[column]
        ):
            continue

        column_lower = column.lower()

        if (
            "date" in column_lower
            or "datetime" in column_lower
            or "timestamp" in column_lower
        ):

            converted = pd.to_datetime(
                result[column],
                errors="coerce",
            )

            valid_ratio = (
                converted.notna().mean()
            )

            if valid_ratio >= 0.8:

                result[column] = converted

    # ---------------------------------------------
    # 3. Create datetime features
    # ---------------------------------------------

    (
        result,
        date_features,
    ) = create_datetime_features(
        result
    )

    report[
        "datetime_features_created"
    ] = date_features

    # ---------------------------------------------
    # 4. Create numerical features
    # ---------------------------------------------

    (
        result,
        numerical_features,
    ) = create_numerical_features(
        result
    )

    report[
        "numerical_features_created"
    ] = numerical_features

    # ---------------------------------------------
    # 5. Remove raw datetime columns
    # ---------------------------------------------

    datetime_columns = (
        result.select_dtypes(
            include=["datetime"]
        ).columns.tolist()
    )

    if datetime_columns:

        result = result.drop(
            columns=datetime_columns
        )

        report[
            "datetime_columns_removed"
        ] = datetime_columns

    # ---------------------------------------------
    # 6. Encode categorical features
    # ---------------------------------------------

    if encode_categories:

        (
            result,
            categorical_features,
        ) = encode_categorical_features(
            result
        )

        report[
            "categorical_features_created"
        ] = categorical_features

    # ---------------------------------------------
    # 7. Final validation
    # ---------------------------------------------

    report["final_rows"] = len(
        result
    )

    report["final_columns"] = len(
        result.columns
    )

    report["remaining_missing_values"] = int(
        result.isna().sum().sum()
    )

    report["data_types"] = {
        column: str(dtype)
        for column, dtype in result.dtypes.items()
    }

    return (
        result,
        report,
    )