import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
)


def evaluate_classification(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate a classification model on test data.
    """

    predictions = model.predict(X_test)

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    return {
        "accuracy": round(
            float(
                accuracy_score(
                    y_test,
                    predictions
                )
            ),
            4
        ),
        "precision": round(
            float(
                precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )
            ),
            4
        ),
        "recall": round(
            float(
                recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )
            ),
            4
        ),
        "f1": round(
            float(
                f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )
            ),
            4
        ),
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_regression(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate a regression model on test data.
    """

    predictions = model.predict(X_test)

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = mse ** 0.5

    return {
        "mae": round(
            float(
                mean_absolute_error(
                    y_test,
                    predictions
                )
            ),
            4
        ),
        "rmse": round(
            float(rmse),
            4
        ),
        "r2": round(
            float(
                r2_score(
                    y_test,
                    predictions
                )
            ),
            4
        ),
    }


def get_feature_importance(
    model,
    feature_names: list[str]
) -> dict:
    """
    Extract feature importance from models that expose
    feature_importances_ or coefficients_.
    """

    if hasattr(model, "feature_importances_"):

        values = model.feature_importances_

    elif hasattr(model, "coef_"):

        values = model.coef_

        # Multiclass logistic regression can have
        # one coefficient array per class.
        if len(values.shape) > 1:
            values = abs(values).mean(axis=0)
        else:
            values = abs(values)

    else:

        return {
            "available": False,
            "features": {},
            "reason": (
                "This model does not expose a standard "
                "feature-importance or coefficient attribute."
            ),
        }

    if len(values) != len(feature_names):

        return {
            "available": False,
            "features": {},
            "reason": (
                "Number of feature-importance values does "
                "not match the number of feature names."
            ),
        }

    importance = {
        feature: round(
            float(value),
            6
        )
        for feature, value in zip(
            feature_names,
            values
        )
    }

    importance = dict(
        sorted(
            importance.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )

    return {
        "available": True,
        "features": importance,
        "reason": (
            "Feature importance extracted successfully."
        ),
    }


def generate_predictions(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Create a prediction table containing actual and predicted
    values.
    """

    predictions = model.predict(X_test)

    result = pd.DataFrame({
        "actual": y_test.reset_index(drop=True),
        "predicted": predictions,
    })

    return result


def summarize_evaluation(
    problem_type: str,
    evaluation: dict,
    feature_importance: dict,
) -> dict:
    """
    Create a compact explainability summary.
    """

    summary = {
        "problem_type": problem_type,
        "evaluation": evaluation,
        "feature_importance_available": feature_importance.get(
            "available",
            False
        ),
    }

    if feature_importance.get("available"):

        ranked_features = list(
            feature_importance["features"].items()
        )

        summary["top_features"] = [
            {
                "feature": feature,
                "importance": importance,
            }
            for feature, importance in ranked_features[:10]
        ]

    else:

        summary["top_features"] = []

    return summary