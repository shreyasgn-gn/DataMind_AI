from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    """
    Shared state passed between DataMind agent steps.
    """

    question: str = ""
    file_path: str = ""

    dataset: Any = None

    task_type: str | None = None
    problem_type: str | None = None
    target_column: str | None = None

    task_understanding: dict = field(
        default_factory=dict
    )

    inspection: dict = field(
        default_factory=dict
    )

    profile: dict = field(
        default_factory=dict
    )

    data_quality: dict = field(
        default_factory=dict
    )

    cleaning_report: dict = field(
        default_factory=dict
    )

    eda: dict = field(
        default_factory=dict
    )

    leakage_report: dict = field(
        default_factory=dict
    )

    feature_report: dict = field(
        default_factory=dict
    )

    training_result: dict = field(
        default_factory=dict
    )

    evaluation_result: dict = field(
        default_factory=dict
    )

    explainability_result: dict = field(
        default_factory=dict
    )

    unsupervised_result: dict = field(
        default_factory=dict
    )

    recovery_history: list[dict] = field(
        default_factory=list
    )

    errors: list[str] = field(
        default_factory=list
    )

    completed_steps: list[str] = field(
        default_factory=list
    )

    status: str = "initialized"

    final_answer: str = ""

    final_answer_source: str = "deterministic"