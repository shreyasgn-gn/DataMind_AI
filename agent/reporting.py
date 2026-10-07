import json

from agent.llm import ask_llm


def _top_correlations(
    correlations: dict,
    limit: int = 8,
) -> list[dict]:
    """
    Extract the strongest unique correlation pairs.
    """

    pairs = []

    columns = list(
        correlations.keys()
    )

    for index, left in enumerate(columns):

        left_values = correlations.get(
            left,
            {},
        )

        for right in columns[index + 1:]:

            value = left_values.get(
                right
            )

            if value is None:
                continue

            try:
                numeric_value = float(value)
            except (
                TypeError,
                ValueError,
            ):
                continue

            pairs.append({
                "feature_1": left,
                "feature_2": right,
                "correlation": round(
                    numeric_value,
                    4,
                ),
            })

    pairs.sort(
        key=lambda item: abs(
            item["correlation"]
        ),
        reverse=True,
    )

    return pairs[:limit]


def _get_verified_categorical_columns(
    state,
) -> list[str]:
    """
    Return categorical columns from the intelligent
    dataset profile.

    Identifier and datetime columns are excluded.
    """

    return state.profile.get(
        "categorical_columns",
        [],
    )


def _get_top_categorical_values(
    state,
    limit_columns: int = 5,
    limit_values: int = 3,
) -> dict:
    """
    Return actual top categorical values from the dataset.

    This uses the full dataset rather than the truncated
    EDA summary, so it never confuses "top 10 values"
    with "10 categories".
    """

    result = {}

    columns = (
        _get_verified_categorical_columns(
            state
        )
    )

    for column in columns[:limit_columns]:

        if column not in state.dataset.columns:
            continue

        counts = (
            state.dataset[column]
            .value_counts(
                dropna=False
            )
            .head(limit_values)
        )

        result[column] = {
            str(value): int(count)
            for value, count in counts.items()
        }

    return result


def build_report_context(
    state,
) -> dict:
    """
    Build a compact verified context for reporting.

    Only facts produced by DataMind tools are included.
    """

    profile = state.profile

    context = {
        "user_question": state.question,
        "task_type": state.task_type,
        "problem_type": state.problem_type,
        "target_column": state.target_column,
    }

    # ---------------------------------------------
    # Dataset facts
    # ---------------------------------------------

    context["dataset"] = {
        "rows": profile.get(
            "rows",
            0,
        ),
        "columns": profile.get(
            "columns",
            0,
        ),
        "numeric_columns": profile.get(
            "numeric_columns",
            [],
        ),
        "categorical_columns": (
            profile.get(
                "categorical_columns",
                [],
            )
        ),
        "datetime_columns": profile.get(
            "datetime_columns",
            [],
        ),
        "missing_values": profile.get(
            "missing_values",
            {},
        ),
        "duplicate_rows": profile.get(
            "duplicate_rows",
            0,
        ),
    }

    # ---------------------------------------------
    # Data quality
    # ---------------------------------------------

    context["data_quality"] = {
        "quality_check_performed": True,
        "has_quality_issues": (
            state.data_quality.get(
                "has_quality_issues",
                False,
            )
        ),
        "total_missing_values": (
            state.data_quality.get(
                "total_missing_values",
                0,
            )
        ),
        "duplicate_rows": (
            state.data_quality.get(
                "duplicate_rows",
                0,
            )
        ),
        "constant_columns": (
            state.data_quality.get(
                "constant_columns",
                [],
            )
        ),
    }

    # ---------------------------------------------
    # EDA facts
    # ---------------------------------------------

    context["eda"] = {
        "analysis_performed": True,
        "top_correlations": _top_correlations(
            state.eda.get(
                "correlations",
                {},
            )
        ),
        "top_categorical_values": (
            _get_top_categorical_values(
                state
            )
        ),
    }

    # ---------------------------------------------
    # Leakage facts
    # ---------------------------------------------

    leakage = state.leakage_report

    if leakage:

        context["leakage"] = {
            "check_performed": True,
            "leakage_detected": leakage.get(
                "leakage_detected",
                False,
            ),
            "confirmed_leakage_columns": (
                leakage.get(
                    "confirmed_leakage_columns",
                    [],
                )
            ),
            "suspicious_leakage_columns": (
                leakage.get(
                    "suspicious_leakage_columns",
                    [],
                )
            ),
            "removed_columns": (
                leakage.get(
                    "removed_columns",
                    [],
                )
            ),
            "excluded_suspicious_columns": (
                leakage.get(
                    "excluded_suspicious_columns",
                    [],
                )
            ),
        }

    else:

        context["leakage"] = {
            "check_performed": False,
            "note": (
                "Target leakage detection was not "
                "performed because this request did "
                "not require supervised ML."
            ),
        }

    # ---------------------------------------------
    # ML facts
    # ---------------------------------------------

    if state.problem_type in {
        "classification",
        "regression",
    }:

        training = state.training_result

        context["ml"] = {
            "results_available": True,
            "target_column": (
                state.target_column
            ),
            "best_model": training.get(
                "best_model"
            ),
            "model_metrics": training.get(
                "models",
                {},
            ),
            "evaluation": (
                state.evaluation_result
            ),
        }

        explanation = (
            state.explainability_result
        )

        # ONLY actual feature importance.
        importance = explanation.get(
            "feature_importance",
            {},
        )

        if importance.get(
            "available",
            False,
        ):

            context["ml"][
                "top_feature_importance"
            ] = [
                {
                    "feature": feature,
                    "importance": value,
                }
                for feature, value in list(
                    importance.get(
                        "features",
                        {},
                    ).items()
                )[:10]
            ]

        # ONLY actual SHAP importance.
        shap_result = explanation.get(
            "shap",
            {},
        )

        if shap_result.get(
            "success",
            False,
        ):

            context["ml"][
                "top_shap_features"
            ] = [
                {
                    "feature": feature,
                    "importance": value,
                }
                for feature, value in list(
                    shap_result.get(
                        "global_feature_importance",
                        {},
                    ).items()
                )[:10]
            ]

    else:

        context["ml"] = {
            "results_available": False,
            "note": (
                "No supervised machine-learning "
                "workflow was executed."
            ),
        }

    return context


def _generate_deterministic_eda_report(
    state,
) -> str:
    """
    Generate an EDA report directly from verified
    Python results.
    """

    profile = state.profile
    quality = state.data_quality
    eda = state.eda

    rows = profile.get(
        "rows",
        0,
    )

    columns = profile.get(
        "columns",
        0,
    )

    numeric_columns = profile.get(
        "numeric_columns",
        [],
    )

    categorical_columns = profile.get(
        "categorical_columns",
        [],
    )

    datetime_columns = profile.get(
        "datetime_columns",
        [],
    )

    missing = quality.get(
        "total_missing_values",
        0,
    )

    duplicates = quality.get(
        "duplicate_rows",
        0,
    )

    lines = [
        "## Dataset Overview",
        (
            f"The dataset contains {rows} rows "
            f"and {columns} columns."
        ),
        (
            f"Numeric columns: {len(numeric_columns)}."
        ),
        (
            f"Categorical columns: "
            f"{len(categorical_columns)}."
        ),
    ]

    if datetime_columns:

        lines.append(
            "Datetime columns: "
            + ", ".join(
                datetime_columns
            )
            + "."
        )

    lines.extend([
        "",
        "## Verified Findings",
    ])

    correlations = _top_correlations(
        eda.get(
            "correlations",
            {},
        ),
        limit=5,
    )

    if correlations:

        for item in correlations:

            lines.append(
                f"- {item['feature_1']} and "
                f"{item['feature_2']} have a "
                f"correlation of "
                f"{item['correlation']}."
            )

    else:

        lines.append(
            "- No numeric correlations were available."
        )

    categorical_values = (
        _get_top_categorical_values(
            state
        )
    )

    if categorical_values:

        lines.extend([
            "",
            "## Categorical Data",
        ])

        for column, values in (
            categorical_values.items()
        ):

            formatted_values = ", ".join(
                f"{value} ({count})"
                for value, count in values.items()
            )

            lines.append(
                f"- {column}: "
                f"{formatted_values}."
            )

    lines.extend([
        "",
        "## Data Quality",
        f"Missing values: {missing}.",
        f"Duplicate rows: {duplicates}.",
    ])

    constant_columns = quality.get(
        "constant_columns",
        [],
    )

    if constant_columns:

        lines.append(
            "Constant columns: "
            + ", ".join(
                constant_columns
            )
            + "."
        )

    else:

        lines.append(
            "No constant columns were detected."
        )

    return "\n".join(lines)


def _generate_deterministic_ml_report(
    state,
) -> str:
    """
    Generate an ML report directly from verified
    Python model results.

    This is the safety fallback when the LLM produces
    unsupported claims.
    """

    evaluation = (
        state.evaluation_result
    )

    training = state.training_result

    importance = (
        state.explainability_result.get(
            "feature_importance",
            {},
        )
    )

    lines = [
        "## Task",
        (
            f"Target: {state.target_column}."
        ),
        (
            f"Problem type: "
            f"{state.problem_type}."
        ),
        "",
        "## Model Performance",
        (
            f"Best model: "
            f"{training.get('best_model', 'Unknown')}."
        ),
    ]

    if state.problem_type == "classification":

        lines.extend([
            (
                f"Accuracy: "
                f"{evaluation.get('accuracy')}."
            ),
            (
                f"Precision: "
                f"{evaluation.get('precision')}."
            ),
            (
                f"Recall: "
                f"{evaluation.get('recall')}."
            ),
            (
                f"F1: "
                f"{evaluation.get('f1')}."
            ),
        ])

    else:

        lines.extend([
            (
                f"MAE: "
                f"{evaluation.get('mae')}."
            ),
            (
                f"RMSE: "
                f"{evaluation.get('rmse')}."
            ),
            (
                f"R²: "
                f"{evaluation.get('r2')}."
            ),
        ])

    lines.extend([
        "",
        "## Important Features",
    ])

    if importance.get(
        "available",
        False,
    ):

        for feature, value in list(
            importance.get(
                "features",
                {},
            ).items()
        )[:5]:

            lines.append(
                f"- {feature}: "
                f"{value}."
            )

    else:

        lines.append(
            "- Standard feature importance "
            "is not available for this model."
        )

    lines.extend([
        "",
        "## Data Leakage",
    ])

    leakage = state.leakage_report

    removed = leakage.get(
        "removed_columns",
        [],
    )

    excluded = leakage.get(
        "excluded_suspicious_columns",
        [],
    )

    if removed:

        lines.append(
            "Confirmed leakage columns removed: "
            + ", ".join(removed)
            + "."
        )

    else:

        lines.append(
            "No confirmed leakage columns were removed."
        )

    if excluded:

        lines.append(
            "Suspicious derived columns excluded: "
            + ", ".join(excluded)
            + "."
        )

    return "\n".join(lines)


def _validate_llm_ml_report(
    report: str,
    state,
) -> bool:
    """
    Perform lightweight safety validation on an LLM
    generated ML report.

    Rejects reports that incorrectly mention removed
    leakage features as current model features.
    """

    leakage = state.leakage_report

    forbidden = set(
        leakage.get(
            "removed_columns",
            [],
        )
    )

    forbidden.update(
        leakage.get(
            "excluded_suspicious_columns",
            [],
        )
    )

    report_lower = report.lower()

    for column in forbidden:

        if str(column).lower() in report_lower:

            return False

    target = state.target_column

    if target:

        if str(target).lower() not in report_lower:

            return False

    best_model = (
        state.training_result.get(
            "best_model"
        )
    )

    if best_model:

        if (
            str(best_model).lower()
            not in report_lower
        ):

            return False

    return True


def _generate_llm_ml_report(
    state,
) -> str:
    """
    Use Ollama to phrase verified ML facts.

    If the LLM introduces a forbidden leakage feature,
    use the deterministic verified report instead.
    """

    context = build_report_context(
        state
    )

    prompt = f"""
You are the final reporting assistant for DataMind AI.

The Python system has already completed the ML analysis.

Your only job is to turn the VERIFIED ML RESULTS below
into clear professional prose.

VERIFIED ML RESULTS:
{json.dumps(context["ml"], indent=2, default=str)}

VERIFIED LEAKAGE RESULTS:
{json.dumps(context["leakage"], indent=2, default=str)}

Rules:
1. Use only facts in the supplied results.
2. Do not perform calculations.
3. Do not invent statistics.
4. Do not invent features.
5. Do not use correlation values as feature importance.
6. Only call something an important feature when it appears
   in top_feature_importance or top_shap_features.
7. Do not mention removed leakage columns as model features.
8. The target is exactly:
{state.target_column}
9. The best model is exactly:
{state.training_result.get("best_model")}
10. Do not reveal internal reasoning.
11. Keep the report below 180 words.

Use these sections:

Task
Model Performance
Important Features
Data Leakage
"""

    try:

        response = ask_llm(
            prompt
        )

        if not response:

            raise ValueError(
                "Ollama returned an empty ML report."
            )

        if _validate_llm_ml_report(
            response,
            state,
        ):

            return response.strip()

    except Exception:
        pass

    return _generate_deterministic_ml_report(
        state
    )


def generate_llm_report(
    state,
) -> str:
    """
    Generate the final report.

    EDA uses deterministic reporting because exact
    dataset facts are more important than free-form
    generation.

    ML uses Ollama for wording with a deterministic
    safety fallback.
    """

    if state.task_type == "exploratory_analysis":

        return _generate_deterministic_eda_report(
            state
        )

    if state.problem_type in {
        "classification",
        "regression",
    }:

        return _generate_llm_ml_report(
            state
        )

    return (
        "The requested task was identified, but "
        "no completed reporting results are available."
    )