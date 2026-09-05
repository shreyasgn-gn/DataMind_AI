import re

import pandas as pd


TASK_KEYWORDS = {
    "classification": [
        "classify",
        "classification",
        "predict class",
        "predict category",
        "churn",
        "will",
        "yes or no",
        "yes/no",
        "approve",
        "default",
        "fraud",
        "spam",
        "purchased",
        "survive",
        "survived",
    ],
    "regression": [
        "regression",
        "predict price",
        "predict sales",
        "predict revenue",
        "predict profit",
        "predict income",
        "predict salary",
        "forecast",
        "estimate",
        "how much",
        "value of",
    ],
    "clustering": [
        "cluster",
        "clustering",
        "segment",
        "segmentation",
        "group customers",
        "groups of customers",
        "similar groups",
        "similar customers",
    ],
    "anomaly_detection": [
        "anomaly",
        "anomalies",
        "outlier",
        "outliers",
        "unusual",
        "unusual transactions",
        "fraud detection",
        "detect fraud",
    ],
    "exploratory_analysis": [
        "analyze",
        "analysis",
        "explore",
        "exploratory",
        "eda",
        "insights",
        "trends",
        "summary",
        "summarize",
        "describe",
    ],
}


def normalize_text(text: str) -> str:
    """
    Normalize a user's natural-language request.
    """

    return re.sub(
        r"\s+",
        " ",
        text.lower().strip()
    )


def find_column_reference(
    question: str,
    columns: list[str]
) -> str | None:
    """
    Find a dataset column explicitly mentioned in the
    user's question.

    Matching is case-insensitive.
    """

    normalized_question = normalize_text(question)

    # Prefer longer column names first so that a column
    # such as Customer_ID is checked before Customer.
    sorted_columns = sorted(
        columns,
        key=lambda column: len(str(column)),
        reverse=True
    )

    for column in sorted_columns:

        column_text = normalize_text(str(column))

        if column_text in normalized_question:
            return column

    return None


def detect_task_type(
    question: str
) -> dict:
    """
    Detect the user's requested analytical task.
    """

    normalized_question = normalize_text(question)

    scores = {
        task: 0
        for task in TASK_KEYWORDS
    }

    matched_keywords = {
        task: []
        for task in TASK_KEYWORDS
    }

    for task, keywords in TASK_KEYWORDS.items():

        for keyword in keywords:

            if keyword in normalized_question:

                scores[task] += 1
                matched_keywords[task].append(
                    keyword
                )

    best_task = max(
        scores,
        key=scores.get
    )

    best_score = scores[best_task]

    # No task-specific language → general analysis
    if best_score == 0:

        return {
            "task_type": "exploratory_analysis",
            "confidence": "medium",
            "matched_keywords": [],
            "reason": (
                "No specific machine-learning objective "
                "was detected. Defaulting to exploratory "
                "analysis."
            ),
        }

    return {
        "task_type": best_task,
        "confidence": (
            "high"
            if best_score >= 2
            else "medium"
        ),
        "matched_keywords": matched_keywords[best_task],
        "reason": (
            f"Detected {best_task.replace('_', ' ')} "
            "from the user's request."
        ),
    }


def validate_target(
    df: pd.DataFrame,
    target_column: str | None
) -> dict:
    """
    Validate a requested target column against the dataset.
    """

    if target_column is None:

        return {
            "valid": False,
            "target_column": None,
            "reason": (
                "No target column was specified."
            ),
        }

    if target_column not in df.columns:

        return {
            "valid": False,
            "target_column": target_column,
            "reason": (
                f"Target column '{target_column}' "
                "does not exist in the dataset."
            ),
        }

    if df[target_column].nunique(
        dropna=True
    ) <= 1:

        return {
            "valid": False,
            "target_column": target_column,
            "reason": (
                f"Target column '{target_column}' "
                "contains one or fewer unique values."
            ),
        }

    return {
        "valid": True,
        "target_column": target_column,
        "reason": (
            f"Target column '{target_column}' "
            "exists and contains usable values."
        ),
    }


def understand_task(
    question: str,
    df: pd.DataFrame
) -> dict:
    """
    Understand a user's analytical request using both
    natural language and the uploaded dataset.

    The user's question determines the requested task.
    The dataset is used to validate referenced columns.
    """

    if not question or not question.strip():

        return {
            "success": False,
            "task_type": None,
            "problem_type": None,
            "target_column": None,
            "confidence": "low",
            "reason": (
                "A user question is required."
            ),
        }

    task = detect_task_type(question)

    target_column = find_column_reference(
        question,
        df.columns.tolist()
    )

    validation = validate_target(
        df,
        target_column
    )

    task_type = task["task_type"]

    # ---------------------------------------------------------
    # Map task type to ML problem type
    # ---------------------------------------------------------

    problem_type = None

    if task_type == "classification":
        problem_type = "classification"

    elif task_type == "regression":
        problem_type = "regression"

    elif task_type == "clustering":
        problem_type = "clustering"

    elif task_type == "anomaly_detection":
        problem_type = "anomaly_detection"

    elif task_type == "exploratory_analysis":
        problem_type = None

    # ---------------------------------------------------------
    # Supervised learning requires a valid target
    # ---------------------------------------------------------

    if task_type in (
        "classification",
        "regression"
    ):

        if not validation["valid"]:

            return {
                "success": False,
                "task_type": task_type,
                "problem_type": problem_type,
                "target_column": target_column,
                "confidence": task["confidence"],
                "reason": (
                    f"{task['reason']} "
                    f"{validation['reason']}"
                ),
                "requires_target": True,
            }

    return {
        "success": True,
        "task_type": task_type,
        "problem_type": problem_type,
        "target_column": target_column,
        "confidence": task["confidence"],
        "reason": task["reason"],
        "requires_target": task_type in (
            "classification",
            "regression"
        ),
        "target_validation": validation,
    }