from typing import Optional

from ollama import chat


MODEL_NAME = "qwen3.5:0.8b"


SYSTEM_PROMPT = """
You are the local language model used by DataMind AI.

Your job is to understand user requests about datasets.

Rules:
1. Be concise.
2. Do not invent dataset facts.
3. Do not perform calculations.
4. Do not reveal internal reasoning.
5. Return only the requested answer.
"""


def ask_llm(
    user_prompt: str,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Send a prompt to the local Ollama model.
    """

    if not user_prompt or not user_prompt.strip():
        raise ValueError(
            "user_prompt cannot be empty."
        )

    prompt = (
        system_prompt
        if system_prompt is not None
        else SYSTEM_PROMPT
    )

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
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

    return content.strip()


def test_connection() -> dict:
    """
    Test communication between Python and Ollama.
    """

    try:

        response = ask_llm(
            "Reply with only: DataMind AI is ready."
        )

        return {
            "success": True,
            "model": MODEL_NAME,
            "response": response,
        }

    except Exception as error:

        return {
            "success": False,
            "model": MODEL_NAME,
            "response": "",
            "error": str(error),
        }