import json
import re

from ollama import chat

from agent.llm import MODEL_NAME, SYSTEM_PROMPT


ALLOWED_TASKS = {
    "classification",
    "regression",
    "clustering",
    "anomaly_detection",
    "exploratory_analysis",
}


def normalize_text(text: str) -> str:
    """
    Normalize text for matching.
    """

    text = str(text).lower().strip()

    text = re.sub(
        r"[^a-z0-9\s_]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalize_token(token: str) -> str:
    """
    Normalize a token and handle common word endings.
    """

    token = normalize_text(token)
    token = token.replace("_", "")

    if token.endswith("ies"):
        token = token[:-3] + "y"

    elif token.endswith("ing"):
        token = token[:-3]

    elif token.endswith("ed"):
        token = token[:-2]

    elif token.endswith("es"):
        token = token[:-2]

    elif token.endswith("s"):
        token = token[:-1]

    return token


def _column_tokens(
    column: str,
) -> list[str]:
    """
    Convert a column name into normalized semantic tokens.
    """

    normalized = normalize_text(
        column
    )

    return [
        normalize_token(token)
        for token in re.split(
            r"[\s_\-/]+",
            normalized,
        )
        if len(token) > 2
    ]


def _target_match_score(
    question: str,
    column: str,
) -> float:
    """
    Calculate how strongly the question matches a column.

    A score of 1.0 means every meaningful column token
    appears in the question.

    Example:
        "predict profit" + Profit
            -> 1.0

        "predict profit" + Profit_Margin
            -> 0.5
    """

    question_tokens = {
        normalize_token(token)
        for token in normalize_text(
            question
        ).split()
        if len(token) > 2
    }

    column_tokens = _column_tokens(
        column
    )

    if not column_tokens:
        return 0.0

    matching_tokens = (
        set(column_tokens)
        & question_tokens
    )

    return (
        len(matching_tokens)
        / len(set(column_tokens))
    )


def resolve_target_column(
    question: str,
    columns: list[str],
) -> str | None:
    """
    Resolve the strongest target using deterministic
    semantic matching.

    This is intentionally allowed to override an LLM target
    when the user's wording clearly matches another column.
    """

    candidates = []

    for column in columns:

        score = _target_match_score(
            question,
            column,
        )

        if score > 0:

            candidates.append({
                "column": column,
                "score": score,
                "token_count": len(
                    _column_tokens(column)
                ),
            })

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item["score"],
            -item["token_count"],
        ),
        reverse=True,
    )

    best = candidates[0]

    # Strong enough match to safely use.
    if best["score"] >= 1.0:
        return best["column"]

    return None


def _extract_json(
    text: str,
) -> dict:
    """
    Extract a JSON object from the model response.
    """

    cleaned = text.strip()

    cleaned = re.sub(
        r"^```json\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    try:

        return json.loads(
            cleaned
        )

    except json.JSONDecodeError:

        match = re.search(
            r"\{.*\}",
            cleaned,
            flags=re.DOTALL,
        )

        if not match:

            raise ValueError(
                "Ollama did not return valid JSON."
            )

        return json.loads(
            match.group(0)
        )


def understand_task_with_llm(
    question: str,
    columns: list[str],
) -> dict:
    """
    Use Ollama to identify the task.

    Python deterministically validates and resolves targets
    when the user's wording provides a strong exact match.
    """

    if not question or not question.strip():

        raise ValueError(
            "A user question is required."
        )

    if not columns:

        raise ValueError(
            "At least one dataset column is required."
        )

    column_text = ", ".join(
        str(column)
        for column in columns
    )

    prompt = f"""
You are analyzing a user request for a data-science system.

DATASET COLUMNS:
{column_text}

USER REQUEST:
{question}

Return ONLY this JSON:

{{
  "task_type": "classification|regression|clustering|anomaly_detection|exploratory_analysis",
  "target_column": "exact dataset column name or null"
}}

Rules:
- Choose exactly one task_type.
- classification means predicting a class/category/yes-no outcome.
- regression means predicting a numeric value.
- clustering means grouping similar records.
- anomaly_detection means finding unusual records.
- exploratory_analysis means analysis, trends, summaries,
  relationships, or insights.
- Clustering MUST use null for target_column.
- anomaly_detection MUST use null for target_column.
- exploratory_analysis MUST use null for target_column.
- Never invent a dataset column.
- Copy column names exactly.
- Return JSON only.
"""

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        options={
            "temperature": 0,
        },
        think=False,
    )

    content = response.message.content

    if not content:

        raise ValueError(
            "Ollama returned an empty response."
        )

    result = _extract_json(
        content
    )

    task_type = result.get(
        "task_type"
    )

    llm_target = result.get(
        "target_column"
    )

    if task_type not in ALLOWED_TASKS:

        raise ValueError(
            f"Invalid task type returned by Ollama: "
            f"{task_type}"
        )

    # --------------------------------------------------
    # Target-free tasks
    # --------------------------------------------------

    if task_type in {
        "clustering",
        "anomaly_detection",
        "exploratory_analysis",
    }:

        return {
            "success": True,
            "task_type": task_type,
            "target_column": None,
            "model": MODEL_NAME,
            "source": "ollama",
            "target_resolution": "not_required",
            "llm_target": None,
        }

    # --------------------------------------------------
    # Deterministic target resolution
    # --------------------------------------------------

    deterministic_target = (
        resolve_target_column(
            question,
            columns,
        )
    )

    if deterministic_target:

        target_column = deterministic_target

        if (
            llm_target
            and llm_target
            != deterministic_target
        ):

            target_resolution = (
                "deterministic_override"
            )

        elif llm_target:

            target_resolution = "llm"

        else:

            target_resolution = (
                "deterministic_fallback"
            )

    else:

        target_column = llm_target

        target_resolution = (
            "llm"
            if llm_target
            else None
        )

    # --------------------------------------------------
    # Validate target
    # --------------------------------------------------

    if target_column is not None:

        if target_column not in columns:

            raise ValueError(
                f"Target '{target_column}' does not "
                "exist in the dataset."
            )

    if target_column is None:

        raise ValueError(
            f"{task_type} requires a target column. "
            "The LLM and deterministic target resolver "
            "could not identify one."
        )

    return {
        "success": True,
        "task_type": task_type,
        "target_column": target_column,
        "model": MODEL_NAME,
        "source": "ollama",
        "target_resolution": target_resolution,
        "llm_target": llm_target,
    }