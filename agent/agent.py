from pathlib import Path
from typing import Any, Callable

from agent.state import AgentState
from agent.task_understanding import understand_task
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

from tools.feature_tools import prepare_ml_features

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
    Deterministic orchestration engine for DataMind AI.

    Coordinates dataset understanding, data quality,
    EDA, leakage detection, feature engineering, model
    training, evaluation, and explainability.

    Failed model-training operations can be retried
    through the recovery engine.
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

        A failed operation is retried according to the
        recovery policy. Failed attempts and recovery
        actions are recorded in the shared agent state.
        """

        result = execute_with_recovery(
            operation,
            max_retries=max_retries,
        )

        # Keep a history when a retry was needed
        # or the operation ultimately failed.
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
        Understand the user's analytical request.
        """

        result = understand_task(
            self.state.question,
            self.state.dataset,
        )

        self.state.task_understanding = result

        if not result["success"]:
            raise ValueError(
                result["reason"]
            )

        self.state.task_type = (
            result["task_type"]
        )

        self.state.problem_type = (
            result["problem_type"]
        )

        self.state.target_column = (
            result["target_column"]
        )

        self._record_step(
            "understand_task"
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
        Analyze dataset quality.

        Clean the working dataset when quality issues
        are detected.
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
        1. Detect target leakage.
        2. Remove confirmed leakage features.
        3. Exclude suspicious target-derived features.
        4. Engineer features.
        5. Train models with retry support.
        6. Evaluate the selected model.
        7. Generate feature importance and predictions.
        8. Explain model behavior with SHAP.
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
        # 1. TARGET LEAKAGE DETECTION
        # ---------------------------------------------

        leakage_report = detect_target_leakage(
            self.state.dataset,
            target,
        )

        self.state.leakage_report = (
            leakage_report
        )

        self._record_step(
            "target_leakage_detection"
        )

        # ---------------------------------------------
        # 2. REMOVE CONFIRMED LEAKAGE FEATURES
        # ---------------------------------------------

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

        # ---------------------------------------------
        # 3. FEATURE ENGINEERING
        # ---------------------------------------------

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

        # Exclude columns whose names suggest that
        # they are derived from the target.
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
        # 4. MODEL TRAINING WITH RECOVERY
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
        # 5. PREPARE EVALUATION DATA
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
        # 6. MODEL EVALUATION
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
        # 7. FEATURE IMPORTANCE
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
        # 8. PREDICTIONS
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
        # 9. SHAP EXPLAINABILITY
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
        # 10. EVALUATION SUMMARY
        # ---------------------------------------------

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
                    entry.get("attempts", 1) - 1,
                    0,
                )
                for entry in self.state.recovery_history
            )

            answer = (
                f"{self.state.problem_type.title()} "
                "workflow completed successfully. "
                f"Best model: {model_name}."
            )

            if retry_count:
                answer += (
                    f" Recovery succeeded after "
                    f"{retry_count} retry attempt(s)."
                )

            self.state.final_answer = answer

        elif self.state.task_type == "clustering":

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