import pandas as pd

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def prepare_training_data(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix X and target vector y.

    The target column is removed from X.
    Categorical features are one-hot encoded.
    Datetime features are converted into useful numeric
    components.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "does not exist in the dataset."
        )

    if df[target_column].nunique(dropna=True) <= 1:
        raise ValueError(
            f"Target column '{target_column}' "
            "contains one or fewer unique values."
        )

    data = df.copy()

    y = data[target_column]
    X = data.drop(columns=[target_column])

    # Convert datetime columns into numeric features
    datetime_columns = X.select_dtypes(
        include=["datetime"]
    ).columns.tolist()

    for column in datetime_columns:

        X[f"{column}_year"] = X[column].dt.year
        X[f"{column}_month"] = X[column].dt.month
        X[f"{column}_day"] = X[column].dt.day
        X[f"{column}_day_of_week"] = (
            X[column].dt.dayofweek
        )

    if datetime_columns:
        X = X.drop(columns=datetime_columns)

    # One-hot encode categorical columns
    categorical_columns = X.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    if categorical_columns:

        X = pd.get_dummies(
            X,
            columns=categorical_columns,
            drop_first=False,
            dtype=int,
        )

    # Convert remaining values to numeric where possible
    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Fill numerical missing values
    X = X.fillna(X.median(numeric_only=True))

    # Remove columns that are still completely unusable
    X = X.dropna(axis=1, how="all")

    if X.empty:
        raise ValueError(
            "No usable feature columns remain after "
            "preprocessing."
        )

    return X, y


def split_training_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    classification: bool = False,
):
    """
    Split the dataset into training and testing sets.

    Stratification is used for classification when possible.
    """

    stratify = None

    if classification:

        class_counts = y.value_counts()

        if (
            len(class_counts) > 1
            and class_counts.min() >= 2
        ):
            stratify = y

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def train_classification_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 42,
) -> dict:
    """
    Train multiple classification models and evaluate them.
    """

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=random_state,
        ),
    }

    results = {}

    for name, model in models.items():

        try:

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            results[name] = {
                "model": model,
                "accuracy": round(
                    float(
                        accuracy_score(
                            y_test,
                            predictions,
                        )
                    ),
                    4,
                ),
                "precision": round(
                    float(
                        precision_score(
                            y_test,
                            predictions,
                            average="weighted",
                            zero_division=0,
                        )
                    ),
                    4,
                ),
                "recall": round(
                    float(
                        recall_score(
                            y_test,
                            predictions,
                            average="weighted",
                            zero_division=0,
                        )
                    ),
                    4,
                ),
                "f1": round(
                    float(
                        f1_score(
                            y_test,
                            predictions,
                            average="weighted",
                            zero_division=0,
                        )
                    ),
                    4,
                ),
            }

        except Exception as error:

            results[name] = {
                "model": None,
                "error": str(error),
            }

    return results


def train_regression_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 42,
) -> dict:
    """
    Train multiple regression models and evaluate them.
    """

    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            random_state=random_state,
        ),
    }

    results = {}

    for name, model in models.items():

        try:

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            mse = mean_squared_error(
                y_test,
                predictions,
            )

            rmse = mse ** 0.5

            results[name] = {
                "model": model,
                "mae": round(
                    float(
                        mean_absolute_error(
                            y_test,
                            predictions,
                        )
                    ),
                    4,
                ),
                "rmse": round(
                    float(rmse),
                    4,
                ),
                "r2": round(
                    float(
                        r2_score(
                            y_test,
                            predictions,
                        )
                    ),
                    4,
                ),
            }

        except Exception as error:

            results[name] = {
                "model": None,
                "error": str(error),
            }

    return results


def select_best_classification_model(
    results: dict,
) -> tuple[str | None, dict | None]:
    """
    Select the classification model with the highest F1 score.
    """

    valid_results = {
        name: result
        for name, result in results.items()
        if "f1" in result
    }

    if not valid_results:
        return None, None

    best_name = max(
        valid_results,
        key=lambda name: valid_results[name]["f1"],
    )

    return best_name, valid_results[best_name]


def select_best_regression_model(
    results: dict,
) -> tuple[str | None, dict | None]:
    """
    Select the regression model with the highest R² score.
    """

    valid_results = {
        name: result
        for name, result in results.items()
        if "r2" in result
    }

    if not valid_results:
        return None, None

    best_name = max(
        valid_results,
        key=lambda name: valid_results[name]["r2"],
    )

    return best_name, valid_results[best_name]


def train_models(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Main ML training entry point.

    Supports:
    - classification
    - regression

    Returns model comparison results and the selected
    best model.
    """

    if problem_type not in {
        "classification",
        "regression",
    }:
        raise ValueError(
            "ML training currently supports only "
            "classification and regression."
        )

    X, y = prepare_training_data(
        df,
        target_column,
    )

    is_classification = (
        problem_type == "classification"
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_training_data(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        classification=is_classification,
    )

    if is_classification:

        results = train_classification_models(
            X_train,
            X_test,
            y_train,
            y_test,
            random_state,
        )

        best_name, best_result = (
            select_best_classification_model(
                results
            )
        )

    else:

        results = train_regression_models(
            X_train,
            X_test,
            y_train,
            y_test,
            random_state,
        )

        best_name, best_result = (
            select_best_regression_model(
                results
            )
        )

    # Remove model objects from the serializable summary
    comparison = {}

    for name, result in results.items():

        comparison[name] = {
            key: value
            for key, value in result.items()
            if key != "model"
        }

    return {
        "problem_type": problem_type,
        "target_column": target_column,
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "feature_count": X.shape[1],
        "models": comparison,
        "best_model": best_name,
        "best_model_metrics": (
            {
                key: value
                for key, value in best_result.items()
                if key != "model"
            }
            if best_result
            else None
        ),
        "best_model_object": (
            best_result["model"]
            if best_result
            else None
        ),
    }