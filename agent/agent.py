from pathlib import Path
from typing import Any, Callable

import pandas as pd

from agent.state import AgentState
from agent.task_understanding import (
    understand_task,
    validate_target,
)
from agent.llm_task import (
    understand_task_with_llm,
)
from agent.recovery import execute_with_recovery

from tools.data_tools import (
    inspect_dataset,
    load_dataset,
)

from tools.profile_tools import profile_dataset

from tools.cleaning_tools import (
    analyze_data_quality,
    clean_dataset,
)

from tools.eda_tools import perform_eda

from tools.feature_tools import (
    prepare_ml_features,
)

from tools.leakage_tools import (
    detect_target_leakage,
    remove_leakage_features,
)

from tools.ml_tools import (
    prepare_training_data,
    split_training_data,
    train_models,
)

from tools.evaluation_tools import (
    evaluate_classification,
    evaluate_regression,
    get_feature_importance,
    generate_predictions,
    summarize_evaluation,
)

from tools.shap_tools import explain_model


class DataMindAgent:
    """
    Autonomous data-science orchestration engine.

    Coordinates:
    - dataset loading
    - LLM task understanding
    - deterministic task fallback
    - target-type validation
    - dataset inspection
    - profiling
    - data-quality analysis
    - cleaning
    - EDA
    - target-leakage detection
    - feature engineering
    - model training
    - evaluation
    - feature importance
    - SHAP explainability
    - error recovery
    """

    def __init__(self):
        self.state = AgentState()

    def _record_step(
        self,
        step_name: str,
    ) -> None:
        """
        Record a successfully completed workflow step.
        """

        if step_name not in self.state.completed_steps:
            self.state.completed_steps.append(
                step_name
            )

    def _record_error(
        self,
        error: Exception,
    ) -> None:
        """
        Record an unrecovered execution error.
        """

        self.state.errors.append(
            str(error)
        )

        self.state.status = "failed"

    def _run_with_recovery(
        self,
        operation_name: str,
        operation: Callable[[], Any],
        max_retries: int = 1,
    ) -> Any:
        """
        Execute an operation through the recovery engine.
        """

        result = execute_with_recovery(
            operation,
            max_retries=max_retries,
        )

        if (
            result["attempts"] > 1
            or not result["success"]
        ):

            self.state.recovery_history.append({
                "operation": operation_name,
                "success": result["success"],
                "attempts": result["attempts"],
                "errors": result["errors"],
                "recovery_actions": (
                    result["recovery_actions"]
                ),
            })

        if not result["success"]:

            errors = result.get(
                "errors",
                [],
            )

            if errors:

                last_error = errors[-1]["error"]

            else:

                last_error = (
                    "Unknown execution failure."
                )

            raise RuntimeError(
                f"{operation_name} failed after "
                f"{result['attempts']} attempt(s). "
                f"Last error: {last_error}"
            )

        return result["result"]

    def _infer_problem_type_from_target(
        self,
        target_column: str,
    ) -> str:
        """
        Determine whether a target is suitable for
        classification or regression.

        This deterministic validation protects the system
        when the local LLM misclassifies a numeric target.
        """

        if target_column not in self.state.dataset.columns:

            raise ValueError(
                f"Target column '{target_column}' "
                "does not exist."
            )

        target = self.state.dataset[
            target_column
        ]

        unique_count = int(
            target.nunique(
                dropna=True
            )
        )

        if unique_count <= 1:

            raise ValueError(
                f"Target column '{target_column}' "
                "contains one or fewer unique values."
            )

        if pd.api.types.is_bool_dtype(
            target
        ):

            return "classification"

        if (
            pd.api.types.is_object_dtype(target)
            or pd.api.types.is_string_dtype(target)
            or isinstance(
                target.dtype,
                pd.CategoricalDtype,
            )
        ):

            if unique_count <= 20:

                return "classification"

            raise ValueError(
                f"Target column '{target_column}' "
                "has too many categorical values for "
                "automatic classification."
            )

        if pd.api.types.is_numeric_dtype(
            target
        ):

            unique_ratio = (
                unique_count / len(target)
                if len(target) > 0
                else 0
            )

            if unique_count == 2:

                return "classification"

            if (
                unique_count <= 10
                and unique_ratio <= 0.10
                and pd.api.types.is_integer_dtype(
                    target
                )
            ):

                return "classification"

            return "regression"

        raise ValueError(
            f"Target column '{target_column}' "
            "has an unsupported data type."
        )

    def _validate_llm_decision(
        self,
        llm_result: dict,
    ) -> dict:
        """
        Validate the LLM task decision against the
        actual dataset.

        Target-free tasks remain unchanged.

        For supervised tasks, the target datatype determines
        whether classification or regression is valid.
        """

        task_type = llm_result[
            "task_type"
        ]

        target_column = llm_result[
            "target_column"
        ]

        if task_type in {
            "clustering",
            "anomaly_detection",
            "exploratory_analysis",
        }:

            return {
                **llm_result,
                "validation": (
                    "Target-free task validated."
                ),
            }

        if target_column is None:

            raise ValueError(
                f"{task_type} requires a target column."
            )

        validation = validate_target(
            self.state.dataset,
            target_column,
        )

        if not validation["valid"]:

            raise ValueError(
                validation["reason"]
            )

        validated_problem_type = (
            self._infer_problem_type_from_target(
                target_column
            )
        )

        original_task_type = task_type

        if task_type != validated_problem_type:

            task_type = validated_problem_type

        if task_type == "classification":

            problem_type = "classification"

        else:

            problem_type = "regression"

        result = {
            **llm_result,
            "task_type": task_type,
            "problem_type": problem_type,
            "target_column": target_column,
            "target_validation": validation,
        }

        if original_task_type != task_type:

            result["llm_task_correction"] = {
                "original_task_type": (
                    original_task_type
                ),
                "corrected_task_type": task_type,
                "reason": (
                    f"The target column "
                    f"'{target_column}' has a data "
                    f"type and value distribution that "
                    f"supports {task_type}, not "
                    f"{original_task_type}."
                ),
            }

        else:

            result["llm_task_correction"] = None

        return result

    def load_data(self) -> None:
        """
        Load the uploaded dataset.
        """

        self.state.dataset = load_dataset(
            self.state.file_path
        )

        self._record_step(
            "load_dataset"
        )

    def understand_request(self) -> None:
        """
        Understand the user's request using Ollama.

        If Ollama fails or returns an unusable result,
        the deterministic task-understanding engine is used.

        The LLM result is also validated against the actual
        target datatype before the workflow continues.
        """

        columns = (
            self.state.dataset.columns.tolist()
        )

        llm_error = None

        try:

            llm_result = (
                understand_task_with_llm(
                    self.state.question,
                    columns,
                )
            )

            result = (
                self._validate_llm_decision(
                    llm_result
                )
            )

            task_type = result[
                "task_type"
            ]

            problem_type = result.get(
                "problem_type"
            )

            if problem_type is None:

                if task_type == "classification":

                    problem_type = (
                        "classification"
                    )

                elif task_type == "regression":

                    problem_type = (
                        "regression"
                    )

                elif task_type == "clustering":

                    problem_type = (
                        "clustering"
                    )

                elif task_type == "anomaly_detection":

                    problem_type = (
                        "anomaly_detection"
                    )

            result["problem_type"] = (
                problem_type
            )

            result["source"] = "ollama"
            result["model"] = (
                llm_result["model"]
            )
            result["confidence"] = "high"

            if result.get(
                "llm_task_correction"
            ):

                result["reason"] = (
                    "Ollama interpreted the request, "
                    "then Python corrected the task type "
                    "using the actual target datatype."
                )

                self._record_step(
                    "llm_task_validation"
                )

            else:

                result["reason"] = (
                    "User request interpreted by "
                    "the local Ollama model."
                )

            self.state.task_understanding = (
                result
            )

            self.state.task_type = (
                task_type
            )

            self.state.problem_type = (
                problem_type
            )

            self.state.target_column = (
                result["target_column"]
            )

            self._record_step(
                "understand_task"
            )

            self._record_step(
                "llm_task_understanding"
            )

            return

        except Exception as error:

            llm_error = str(error)

        # -------------------------------------------------
        # Deterministic fallback
        # -------------------------------------------------

        fallback_result = understand_task(
            self.state.question,
            self.state.dataset,
        )

        if not fallback_result["success"]:

            raise ValueError(
                "LLM task understanding failed and "
                "deterministic fallback also failed. "
                f"LLM error: {llm_error}. "
                f"Fallback error: "
                f"{fallback_result['reason']}"
            )

        fallback_result["source"] = (
            "deterministic_fallback"
        )

        fallback_result["llm_error"] = (
            llm_error
        )

        self.state.task_understanding = (
            fallback_result
        )

        self.state.task_type = (
            fallback_result["task_type"]
        )

        if self.state.task_type == "classification":

            self.state.problem_type = (
                "classification"
            )

        elif self.state.task_type == "regression":

            self.state.problem_type = (
                "regression"
            )

        elif self.state.task_type == "clustering":

            self.state.problem_type = (
                "clustering"
            )

        elif self.state.task_type == "anomaly_detection":

            self.state.problem_type = (
                "anomaly_detection"
            )

        else:

            self.state.problem_type = None

        self.state.target_column = (
            fallback_result[
                "target_column"
            ]
        )

        self._record_step(
            "understand_task"
        )

        self._record_step(
            "deterministic_task_fallback"
        )

    def inspect_and_profile(self) -> None:
        """
        Inspect and profile the dataset.
        """

        self.state.inspection = (
            inspect_dataset(
                self.state.file_path
            )
        )

        self.state.profile = (
            profile_dataset(
                self.state.dataset
            )
        )

        self._record_step(
            "inspect_dataset"
        )

        self._record_step(
            "profile_dataset"
        )

    def check_data_quality(self) -> None:
        """
        Analyze data quality and clean when needed.
        """

        self.state.data_quality = (
            analyze_data_quality(
                self.state.dataset
            )
        )

        self._record_step(
            "data_quality"
        )

        if self.state.data_quality.get(
            "has_quality_issues",
            False,
        ):

            (
                self.state.dataset,
                self.state.cleaning_report,
            ) = clean_dataset(
                self.state.dataset
            )

            self._record_step(
                "clean_dataset"
            )

    def run_eda(self) -> None:
        """
        Perform exploratory data analysis.
        """

        self.state.eda = perform_eda(
            self.state.dataset
        )

        self._record_step(
            "eda"
        )

    def run_ml_workflow(self) -> None:
        """
        Execute the complete supervised ML workflow.

        Steps:
        1. Target leakage detection.
        2. Confirmed leakage removal.
        3. Suspicious derived-feature exclusion.
        4. Feature engineering.
        5. Model training with recovery.
        6. Model evaluation.
        7. Feature importance.
        8. Predictions.
        9. SHAP explainability.
        """

        if self.state.problem_type not in {
            "classification",
            "regression",
        }:

            return

        if not self.state.target_column:

            raise ValueError(
                "A target column is required for ML."
            )

        target = self.state.target_column

        # -------------------------------------------------
        # Target leakage detection
        # -------------------------------------------------

        leakage_report = (
            detect_target_leakage(
                self.state.dataset,
                target,
            )
        )

        self.state.leakage_report = (
            leakage_report
        )

        self._record_step(
            "target_leakage_detection"
        )

        # -------------------------------------------------
        # Remove confirmed leakage
        # -------------------------------------------------

        (
            ml_source_data,
            removed_columns,
        ) = remove_leakage_features(
            self.state.dataset,
            leakage_report,
        )

        self.state.leakage_report[
            "removed_columns"
        ] = removed_columns

        # -------------------------------------------------
        # Feature engineering
        # -------------------------------------------------

        (
            ml_data,
            self.state.feature_report,
        ) = prepare_ml_features(
            ml_source_data
        )

        if target not in ml_data.columns:

            raise ValueError(
                f"Target column '{target}' was not "
                "available after feature preparation."
            )

        # -------------------------------------------------
        # Remove suspicious derived target features
        # -------------------------------------------------

        suspicious_columns = (
            leakage_report.get(
                "suspicious_leakage_columns",
                [],
            )
        )

        suspicious_columns = [
            column
            for column in suspicious_columns
            if (
                column in ml_data.columns
                and column != target
            )
        ]

        if suspicious_columns:

            ml_data = ml_data.drop(
                columns=suspicious_columns
            )

        self.state.leakage_report[
            "excluded_suspicious_columns"
        ] = suspicious_columns

        if ml_data.empty:

            raise ValueError(
                "No usable columns remain after "
                "leakage removal and feature engineering."
            )

        self._record_step(
            "feature_engineering"
        )

        # -------------------------------------------------
        # Model training with recovery
        # -------------------------------------------------

        self.state.training_result = (
            self._run_with_recovery(
                operation_name="model_training",
                operation=lambda: train_models(
                    ml_data,
                    target,
                    self.state.problem_type,
                ),
                max_retries=1,
            )
        )

        self._record_step(
            "model_training"
        )

        best_model_name = (
            self.state.training_result.get(
                "best_model"
            )
        )

        best_model = (
            self.state.training_result.get(
                "best_model_object"
            )
        )

        if best_model is None:

            raise ValueError(
                "No valid model was successfully trained."
            )

        # -------------------------------------------------
        # Prepare evaluation data
        # -------------------------------------------------

        (
            X,
            y,
        ) = prepare_training_data(
            ml_data,
            target,
        )

        is_classification = (
            self.state.problem_type
            == "classification"
        )

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = split_training_data(
            X,
            y,
            classification=is_classification,
        )

        # -------------------------------------------------
        # Evaluation
        # -------------------------------------------------

        if is_classification:

            self.state.evaluation_result = (
                evaluate_classification(
                    best_model,
                    X_test,
                    y_test,
                )
            )

        else:

            self.state.evaluation_result = (
                evaluate_regression(
                    best_model,
                    X_test,
                    y_test,
                )
            )

        self._record_step(
            "model_evaluation"
        )

        # -------------------------------------------------
        # Feature importance
        # -------------------------------------------------

        importance = get_feature_importance(
            best_model,
            X.columns.tolist(),
        )

        self.state.explainability_result = {
            "best_model": best_model_name,
            "feature_importance": importance,
        }

        self._record_step(
            "feature_importance"
        )

        # -------------------------------------------------
        # Predictions
        # -------------------------------------------------

        predictions = generate_predictions(
            best_model,
            X_test,
            y_test,
        )

        self.state.explainability_result[
            "prediction_sample"
        ] = (
            predictions
            .head(10)
            .to_dict(
                orient="records"
            )
        )

        # -------------------------------------------------
        # SHAP
        # -------------------------------------------------

        shap_result = explain_model(
            best_model,
            X_train,
        )

        self.state.explainability_result[
            "shap"
        ] = shap_result

        self._record_step(
            "shap_explainability"
        )

        # -------------------------------------------------
        # Evaluation summary
        # -------------------------------------------------

        self.state.explainability_result[
            "summary"
        ] = summarize_evaluation(
            self.state.problem_type,
            self.state.evaluation_result,
            importance,
        )

    def generate_final_answer(self) -> str:
        """
        Generate a concise user-facing summary.
        """

        if (
            self.state.task_type
            == "exploratory_analysis"
        ):

            rows = self.state.profile.get(
                "rows",
                0,
            )

            columns = self.state.profile.get(
                "columns",
                0,
            )

            self.state.final_answer = (
                "Dataset analysis completed successfully. "
                f"The dataset contains {rows} rows and "
                f"{columns} columns."
            )

        elif self.state.problem_type in {
            "classification",
            "regression",
        }:

            model_name = (
                self.state.training_result.get(
                    "best_model",
                    "Unknown",
                )
            )

            retry_count = sum(
                max(
                    item.get(
                        "attempts",
                        1,
                    ) - 1,
                    0,
                )
                for item in (
                    self.state.recovery_history
                )
            )

            correction = (
                self.state.task_understanding.get(
                    "llm_task_correction"
                )
            )

            self.state.final_answer = (
                f"{self.state.problem_type.title()} "
                "workflow completed successfully. "
                f"Best model: {model_name}."
            )

            if correction:

                self.state.final_answer += (
                    " The initial LLM task interpretation "
                    "was validated and corrected using "
                    "the target data type."
                )

            if retry_count:

                self.state.final_answer += (
                    f" Recovery succeeded after "
                    f"{retry_count} retry attempt(s)."
                )

        elif (
            self.state.task_type
            == "clustering"
        ):

            self.state.final_answer = (
                "Clustering was identified as the requested "
                "task, but the clustering engine has not "
                "been implemented yet."
            )

        elif (
            self.state.task_type
            == "anomaly_detection"
        ):

            self.state.final_answer = (
                "Anomaly detection was identified as the "
                "requested task, but the anomaly detection "
                "engine has not been implemented yet."
            )

        else:

            self.state.final_answer = (
                "The requested analysis completed."
            )

        return self.state.final_answer

    def run(
        self,
        question: str,
        file_path: str,
    ) -> AgentState:
        """
        Execute the complete DataMind workflow.
        """

        self.state = AgentState(
            question=question,
            file_path=file_path,
            status="running",
        )

        try:

            if not Path(file_path).exists():

                raise FileNotFoundError(
                    f"Dataset not found: {file_path}"
                )

            self.load_data()

            self.understand_request()

            self.inspect_and_profile()

            self.check_data_quality()

            self.run_eda()

            if self.state.problem_type in {
                "classification",
                "regression",
            }:

                self.run_ml_workflow()

            self.generate_final_answer()

            self.state.status = "completed"

            return self.state

        except Exception as error:

            self._record_error(
                error
            )

            return self.state