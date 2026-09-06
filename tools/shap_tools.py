import numpy as np
import pandas as pd
import shap


def _normalize_shap_values(
    shap_values,
    model,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize SHAP output.

    Returns:
        importance_values:
            Absolute SHAP values with shape
            (samples, features).

        signed_values:
            Signed SHAP values for the class predicted
            for each row when classification is used.
    """

    # --------------------------------------------------
    # Older SHAP versions:
    # list of arrays, one array per class
    # --------------------------------------------------
    if isinstance(shap_values, list):

        arrays = [
            np.asarray(values)
            for values in shap_values
            if np.asarray(values).size > 0
        ]

        if not arrays:
            raise ValueError(
                "SHAP returned an empty list of values."
            )

        # Multiple classes
        if len(arrays) > 1:

            stacked = np.stack(
                arrays,
                axis=-1
            )

            # Global importance across all classes
            importance_values = np.mean(
                np.abs(stacked),
                axis=-1
            )

            # Use the predicted class for each row
            if hasattr(model, "predict"):

                predictions = model.predict(X)

                if hasattr(model, "classes_"):

                    classes = list(model.classes_)

                    class_indices = [
                        classes.index(prediction)
                        for prediction in predictions
                    ]

                    signed_values = np.vstack([
                        stacked[
                            row_index,
                            :,
                            class_indices[row_index]
                        ]
                        for row_index in range(
                            len(X)
                        )
                    ])

                else:

                    signed_values = stacked[:, :, 0]

            else:

                signed_values = stacked[:, :, 0]

            return (
                np.asarray(importance_values),
                np.asarray(signed_values),
            )

        # Single-output model
        signed_values = arrays[0]

        if signed_values.ndim == 1:

            signed_values = signed_values.reshape(
                1,
                -1
            )

        return (
            np.abs(signed_values),
            signed_values,
        )

    values = np.asarray(shap_values)

    # --------------------------------------------------
    # Newer SHAP classification output:
    # (samples, features, classes)
    # --------------------------------------------------
    if values.ndim == 3:

        importance_values = np.mean(
            np.abs(values),
            axis=-1
        )

        if hasattr(model, "predict"):

            predictions = model.predict(X)

            if hasattr(model, "classes_"):

                classes = list(model.classes_)

                class_indices = [
                    classes.index(prediction)
                    for prediction in predictions
                ]

                signed_values = np.vstack([
                    values[
                        row_index,
                        :,
                        class_indices[row_index]
                    ]
                    for row_index in range(
                        len(X)
                    )
                ])

            else:

                signed_values = values[:, :, 0]

        else:

            signed_values = values[:, :, 0]

        return (
            np.asarray(importance_values),
            np.asarray(signed_values),
        )

    # --------------------------------------------------
    # Standard regression / single-output classification
    # --------------------------------------------------
    if values.ndim == 2:

        return (
            np.abs(values),
            values,
        )

    # --------------------------------------------------
    # Single row
    # --------------------------------------------------
    if values.ndim == 1:

        values_2d = values.reshape(
            1,
            -1
        )

        return (
            np.abs(values_2d),
            values_2d,
        )

    raise ValueError(
        f"Unsupported SHAP output shape: {values.shape}"
    )


def explain_model(
    model,
    X: pd.DataFrame,
    max_samples: int = 100,
) -> dict:
    """
    Generate SHAP-based global feature importance
    and row-level explanations.
    """

    if X.empty:

        raise ValueError(
            "Cannot explain a model with an empty "
            "feature dataset."
        )

    X_sample = X.head(
        max_samples
    ).copy()

    model_name = type(model).__name__.lower()

    try:

        # --------------------------------------------------
        # Choose SHAP explainer
        # --------------------------------------------------

        if (
            "forest" in model_name
            or "gradientboosting" in model_name
            or "xgb" in model_name
            or "tree" in model_name
        ):

            explainer = shap.TreeExplainer(
                model
            )

            shap_output = explainer.shap_values(
                X_sample
            )

        else:

            background = shap.sample(
                X_sample,
                min(
                    50,
                    len(X_sample)
                ),
                random_state=42,
            )

            explainer = shap.Explainer(
                model,
                background,
            )

            explanation = explainer(
                X_sample
            )

            shap_output = explanation.values

        # --------------------------------------------------
        # Normalize output
        # --------------------------------------------------

        (
            importance_values,
            signed_values,
        ) = _normalize_shap_values(
            shap_output,
            model,
            X_sample,
        )

        feature_names = X_sample.columns.tolist()

        if importance_values.ndim != 2:

            raise ValueError(
                "Normalized SHAP importance must be 2D."
            )

        if signed_values.ndim != 2:

            raise ValueError(
                "Normalized SHAP contributions must be 2D."
            )

        if (
            importance_values.shape[0]
            != len(X_sample)
        ):

            raise ValueError(
                "SHAP sample count does not match "
                "the input dataset."
            )

        if (
            importance_values.shape[1]
            != len(feature_names)
        ):

            raise ValueError(
                "SHAP feature count does not match "
                "the input feature columns."
            )

        # --------------------------------------------------
        # Global feature importance
        # --------------------------------------------------

        mean_abs_shap = np.mean(
            importance_values,
            axis=0
        )

        global_importance = {
            feature: round(
                float(value),
                6,
            )
            for feature, value in zip(
                feature_names,
                mean_abs_shap,
            )
        }

        global_importance = dict(
            sorted(
                global_importance.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        # --------------------------------------------------
        # Row-level explanations
        # --------------------------------------------------

        sample_explanations = []

        rows_to_explain = min(
            len(X_sample),
            10
        )

        for row_index in range(
            rows_to_explain
        ):

            row_values = signed_values[
                row_index
            ]

            contributions = {
                feature: round(
                    float(value),
                    6,
                )
                for feature, value in zip(
                    feature_names,
                    row_values,
                )
            }

            ranked_contributions = dict(
                sorted(
                    contributions.items(),
                    key=lambda item: abs(
                        item[1]
                    ),
                    reverse=True,
                )
            )

            sample_explanations.append({
                "row_index": row_index,
                "feature_contributions": (
                    ranked_contributions
                ),
                "top_contributors": [
                    {
                        "feature": feature,
                        "shap_value": value,
                    }
                    for feature, value in list(
                        ranked_contributions.items()
                    )[:5]
                ],
            })

        return {
            "success": True,
            "explainer": "SHAP",
            "model": type(model).__name__,
            "samples_explained": len(X_sample),
            "global_feature_importance": (
                global_importance
            ),
            "sample_explanations": (
                sample_explanations
            ),
        }

    except Exception as error:

        return {
            "success": False,
            "explainer": "SHAP",
            "model": type(model).__name__,
            "samples_explained": 0,
            "global_feature_importance": {},
            "sample_explanations": [],
            "error": str(error),
        }