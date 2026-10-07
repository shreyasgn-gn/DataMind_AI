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

    token = token.replace(
        "_",
        "",
    )

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


def resolve_target_column(
    question: str,
    columns: list[str],
) -> str | None:
    """
    Resolve a likely target column from the user's question
    using conservative token matching.
    """

    question_normalized = normalize_text(
        question
    )

    question_tokens = {
        normalize_token(token)
        for token in question_normalized.split()
        if len(token) > 2
    }

    candidates = []

    for column in columns:

        column_normalized = normalize_text(
            column
        )

        column_tokens = [
            normalize_token(token)
            for token in re.split(
                r"[\s_\-/]+",
                column_normalized,
            )
            if len(token) > 2
        ]

        if not column_tokens:
            continue

        matching_tokens = (
            set(column_tokens)
            & question_tokens
        )

        if matching_tokens:

            candidates.append({
                "column": column,
                "score": len(
                    matching_tokens
                ),
            })

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return candidates[0]["column"]


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
    Use Ollama to identify the requested analytical task
    and target column.

    The LLM interprets intent.
    Python performs deterministic target validation.
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
You are analyzing a user request for a data science system.

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
- exploratory_analysis means analysis, trends, summaries, relationships, or insights.
- Classification and regression may have a target.
- Clustering MUST use target_column = null.
- Anomaly detection MUST use target_column = null.
- Exploratory analysis MUST use target_column = null.
- Never invent a dataset column.
- If the target is unclear for classification or regression, use null.
- Copy dataset column names exactly when selecting a target.
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

    target_column = result.get(
        "target_column"
    )

    if task_type not in ALLOWED_TASKS:

        raise ValueError(
            f"Invalid task type returned by Ollama: "
            f"{task_type}"
        )

    # --------------------------------------------------
    # Target-free tasks NEVER use a target.
    # Python overrides any hallucinated target returned
    # by the small LLM.
    # --------------------------------------------------

    if task_type in {
        "clustering",
        "anomaly_detection",
        "exploratory_analysis",
    }:

        target_column = None

        target_resolution = "not_required"

    else:

        # --------------------------------------------------
        # Resolve missing target deterministically
        # --------------------------------------------------

        if target_column is None:

            target_column = resolve_target_column(
                question,
                columns,
            )

            target_resolution = (
                "deterministic_fallback"
                if target_column
                else None
            )

        else:

            target_resolution = "llm"

        # --------------------------------------------------
        # Validate supervised-learning target
        # --------------------------------------------------

        if target_column is not None:

            if target_column not in columns:

                raise ValueError(
                    f"Ollama selected target "
                    f"'{target_column}', but that column "
                    "does not exist in the dataset."
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
    }