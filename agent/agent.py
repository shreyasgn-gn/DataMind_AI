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

from agent.reporting import (
    generate_llm_report,
)

from agent.recovery import (
    execute_with_recovery,
)

from tools.data_tools import (
    inspect_dataset,
    load_dataset,
)

from tools.profile_tools import (
    profile_dataset,
)

from tools.cleaning_tools import (
    analyze_data_quality,
    clean_dataset,
)

from tools.eda_tools import (
    perform_eda,
)

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

from tools.shap_tools import (
    explain_model,
)

from tools.unsupervised_tools import (
    run_clustering,
    run_anomaly_detection,
)


class DataMindAgent:
    """
    Autonomous data-science orchestration engine.

    Coordinates:
    - LLM task understanding
    - deterministic validation
    - dataset inspection
    - profiling
    - data-quality analysis
    - cleaning
    - EDA
    - feature engineering
    - target-leakage detection
    - leakage removal
    - model training
    - evaluation
    - feature importance
    - SHAP explainability
    - error recovery
    - grounded final reporting
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
        Determine whether a target supports
        classification or regression.
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
            pd.api.types.is_object_dtype(
                target
            )
            or pd.api.types.is_string_dtype(
                target
            )
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
        Validate the LLM task against the actual dataset.
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
                "problem_type": (
                    task_type
                    if task_type
                    != "exploratory_analysis"
                    else None
                ),
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

        corrected_task_type = (
            validated_problem_type
        )

        result = {
            **llm_result,
            "task_type": corrected_task_type,
            "problem_type": corrected_task_type,
            "target_column": target_column,
            "target_validation": validation,
        }

        if (
            original_task_type
            != corrected_task_type
        ):

            result["llm_task_correction"] = {
                "original_task_type": (
                    original_task_type
                ),
                "corrected_task_type": (
                    corrected_task_type
                ),
                "reason": (
                    f"The target column "
                    f"'{target_column}' has a data "
                    f"type and value distribution that "
                    f"supports {corrected_task_type}, "
                    f"not {original_task_type}."
                ),
            }

        else:

            result[
                "llm_task_correction"
            ] = None

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
        use the deterministic task-understanding engine.
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

            self.state.task_understanding = {
                **result,
                "success": True,
                "source": "ollama",
                "model": (
                    llm_result["model"]
                ),
                "confidence": "high",
                "reason": (
                    "User request interpreted by "
                    "the local Ollama model and "
                    "validated against the dataset."
                ),
            }

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

        # ---------------------------------------------
        # Deterministic fallback
        # ---------------------------------------------

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

        fallback_result[
            "source"
        ] = "deterministic_fallback"

        fallback_result[
            "llm_error"
        ] = llm_error

        self.state.task_understanding = (
            fallback_result
        )

        self.state.task_type = (
            fallback_result["task_type"]
        )

        if (
            self.state.task_type
            == "classification"
        ):

            self.state.problem_type = (
                "classification"
            )

        elif (
            self.state.task_type
            == "regression"
        ):

            self.state.problem_type = (
                "regression"
            )

        elif (
            self.state.task_type
            == "clustering"
        ):

            self.state.problem_type = (
                "clustering"
            )

        elif (
            self.state.task_type
            == "anomaly_detection"
        ):

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

        IMPORTANT ORDER:
        1. Feature engineering on the original dataset.
        2. Target-leakage detection on the original dataset.
        3. Remove confirmed leakage features.
        4. Remove suspicious derived features.
        5. Train and evaluate models.

        This prevents removed columns such as Sales from
        being accidentally recreated as calculated_sales.
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

        # ---------------------------------------------
        # FEATURE ENGINEERING FIRST
        # ---------------------------------------------

        (
            engineered_data,
            self.state.feature_report,
        ) = prepare_ml_features(
            self.state.dataset
        )

        if target not in engineered_data.columns:

            raise ValueError(
                f"Target column '{target}' was not "
                "available after feature preparation."
            )

        # ---------------------------------------------
        # TARGET LEAKAGE DETECTION
        # ---------------------------------------------

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

        # ---------------------------------------------
        # REMOVE CONFIRMED LEAKAGE FEATURES
        # ---------------------------------------------

        (
            ml_data,
            removed_columns,
        ) = remove_leakage_features(
            engineered_data,
            leakage_report,
        )

        self.state.leakage_report[
            "removed_columns"
        ] = removed_columns

        # ---------------------------------------------
        # REMOVE SUSPICIOUS DERIVED FEATURES
        # ---------------------------------------------

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

        # ---------------------------------------------
        # MODEL TRAINING WITH RECOVERY
        # ---------------------------------------------

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

        # ---------------------------------------------
        # PREPARE EVALUATION DATA
        # ---------------------------------------------

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

        # ---------------------------------------------
        # MODEL EVALUATION
        # ---------------------------------------------

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

        # ---------------------------------------------
        # FEATURE IMPORTANCE
        # ---------------------------------------------

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

        # ---------------------------------------------
        # PREDICTIONS
        # ---------------------------------------------

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

        # ---------------------------------------------
        # SHAP
        # ---------------------------------------------

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

        # ---------------------------------------------
        # SUMMARY
        # ---------------------------------------------

        self.state.explainability_result[
            "summary"
        ] = summarize_evaluation(
            self.state.problem_type,
            self.state.evaluation_result,
            importance,
        )

    def run_unsupervised_workflow(
        self,
    ) -> None:
        """
        Execute clustering or anomaly detection.

        These are target-free workflows and therefore do not
        use the supervised training/evaluation pipeline.
        """

        if self.state.problem_type == "clustering":

            result = run_clustering(
                self.state.dataset
            )

            self.state.unsupervised_result = {
                "task": "clustering",
                **result,
            }

            self._record_step(
                "clustering"
            )

            return

        if self.state.problem_type == "anomaly_detection":

            result = run_anomaly_detection(
                self.state.dataset
            )

            self.state.unsupervised_result = {
                "task": "anomaly_detection",
                **result,
            }

            self._record_step(
                "anomaly_detection"
            )

            return


    def generate_deterministic_answer(
        self,
    ) -> str:
        """
        Generate a deterministic fallback answer.
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

            return (
                "Dataset analysis completed successfully. "
                f"The dataset contains {rows} rows and "
                f"{columns} columns."
            )

        if self.state.problem_type in {
            "classification",
            "regression",
        }:

            model_name = (
                self.state.training_result.get(
                    "best_model",
                    "Unknown",
                )
            )

            return (
                f"{self.state.problem_type.title()} "
                "workflow completed successfully. "
                f"Best model: {model_name}."
            )

        if (
            self.state.problem_type
            == "clustering"
        ):

            result = self.state.unsupervised_result

            cluster_count = result.get(
                "cluster_count",
                "Unknown",
            )

            silhouette = result.get(
                "silhouette_score",
                "Unknown",
            )

            quality = result.get(
                "cluster_quality",
                "Unknown",
            )

            warning = result.get(
                "warning"
            )

            answer = (
                f"Clustering completed using "
                f"{result.get('method', 'KMeans')}. "
                f"DataMind identified {cluster_count} "
                f"clusters with a silhouette score of "
                f"{silhouette}. Cluster quality: {quality}."
            )

            if warning:
                answer += f" {warning}"

            return answer

        if (
            self.state.problem_type
            == "anomaly_detection"
        ):

            result = self.state.unsupervised_result

            anomaly_count = result.get(
                "anomaly_count",
                0,
            )

            anomaly_rate = result.get(
                "anomaly_rate",
                0,
            )

            return (
                f"Anomaly detection completed using "
                f"{result.get('method', 'Isolation Forest')}. "
                f"DataMind identified {anomaly_count} "
                f"potential anomalies, representing "
                f"{float(anomaly_rate) * 100:.2f}% of analyzed rows."
            )

        return (
            "The requested analysis completed."
        )

    def generate_final_answer(
        self,
    ) -> str:
        """
        Generate the final user-facing answer.

        EDA uses deterministic verified reporting.

        Supervised ML uses Ollama for wording with a
        deterministic fallback.
        """

        # ---------------------------------------------
        # EDA reporting
        # ---------------------------------------------

        if (
            self.state.task_type
            == "exploratory_analysis"
        ):

            self.state.final_answer = (
                generate_llm_report(
                    self.state
                )
            )

            self.state.final_answer_source = (
                "deterministic_eda"
            )

            self._record_step(
                "verified_report_generation"
            )

            return self.state.final_answer

        # ---------------------------------------------
        # ML reporting
        # ---------------------------------------------

        if self.state.problem_type in {
            "classification",
            "regression",
        }:

            try:

                answer = generate_llm_report(
                    self.state
                )

                if not answer:

                    raise ValueError(
                        "Ollama returned an empty report."
                    )

                self.state.final_answer = (
                    answer
                )

                self.state.final_answer_source = (
                    "ollama"
                )

                self._record_step(
                    "llm_report_generation"
                )

                return answer

            except Exception as error:

                self.state.task_understanding[
                    "report_llm_error"
                ] = str(error)

                self.state.final_answer = (
                    self.generate_deterministic_answer()
                )

                self.state.final_answer_source = (
                    "deterministic_fallback"
                )

                self._record_step(
                    "deterministic_report_fallback"
                )

                return self.state.final_answer

        # ---------------------------------------------
        # Non-ML tasks
        # ---------------------------------------------

        self.state.final_answer = (
            self.generate_deterministic_answer()
        )

        self.state.final_answer_source = (
            "deterministic"
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

            elif self.state.problem_type in {
                "clustering",
                "anomaly_detection",
            }:

                self.run_unsupervised_workflow()

            self.generate_final_answer()

            self.state.status = "completed"

            return self.state

        except Exception as error:

            self._record_error(
                error
            )

            return self.state