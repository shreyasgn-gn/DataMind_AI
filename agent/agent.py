from pathlib import Path

import pandas as pd

from agent.state import AgentState
from agent.task_understanding import understand_task

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

    Coordinates:
    - dataset loading
    - task understanding
    - inspection
    - profiling
    - data-quality analysis
    - cleaning
    - EDA
    - target-leakage detection
    - feature engineering
    - model training
    - model evaluation
    - feature importance
    - SHAP explainability
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
        Store an error in the agent state.
        """

        self.state.errors.append(
            str(error)
        )

        self.state.status = "failed"

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

        Clean the dataset only when quality issues
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

        if self.state.data_quality[
            "has_quality_issues"
        ]:

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
        Run the complete supervised ML workflow.

        Pipeline:
        leakage detection
        ↓
        leakage removal
        ↓
        feature engineering
        ↓
        model training
        ↓
        evaluation
        ↓
        feature importance
        ↓
        SHAP
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

        # -------------------------------------------------
        # TARGET LEAKAGE DETECTION
        # -------------------------------------------------

        leakage_report = (
            detect_target_leakage(
                self.state.dataset,
                self.state.target_column,
            )
        )

        self.state.leakage_report = (
            leakage_report
        )

        self._record_step(
            "target_leakage_detection"
        )

        # -------------------------------------------------
        # REMOVE CONFIRMED LEAKAGE
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
        # FEATURE ENGINEERING
        # -------------------------------------------------

        (
            ml_data,
            self.state.feature_report,
        ) = prepare_ml_features(
            ml_source_data
        )

        if self.state.target_column not in (
            ml_data.columns
        ):
            raise ValueError(
                f"Target column "
                f"'{self.state.target_column}' "
                "was not available after "
                "feature preparation."
            )

        # -------------------------------------------------
        # REMOVE SUSPICIOUS DERIVED FEATURES
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
                and column
                != self.state.target_column
            )
        ]

        if suspicious_columns:

            ml_data = ml_data.drop(
                columns=suspicious_columns
            )

        self.state.leakage_report[
            "excluded_suspicious_columns"
        ] = suspicious_columns

        self._record_step(
            "feature_engineering"
        )

        # -------------------------------------------------
        # MODEL TRAINING
        # -------------------------------------------------

        self.state.training_result = (
            train_models(
                ml_data,
                self.state.target_column,
                self.state.problem_type,
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
                "No valid model was "
                "successfully trained."
            )

        # -------------------------------------------------
        # PREPARE EVALUATION DATA
        # -------------------------------------------------

        (
            X,
            y,
        ) = prepare_training_data(
            ml_data,
            self.state.target_column,
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
        # MODEL EVALUATION
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
        # STANDARD FEATURE IMPORTANCE
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
        # PREDICTIONS
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
        # SHAP EXPLAINABILITY
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
        # FINAL EVALUATION SUMMARY
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
        Generate a concise deterministic result.

        Ollama will later replace this with richer
        natural-language reasoning.
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
                "Dataset analysis completed "
                "successfully. "
                f"The dataset contains "
                f"{rows} rows and "
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

            self.state.final_answer = (
                f"{self.state.problem_type.title()} "
                "workflow completed successfully. "
                f"Best model: {model_name}."
            )

        elif (
            self.state.task_type
            == "clustering"
        ):

            self.state.final_answer = (
                "Clustering was identified "
                "as the requested task, but "
                "the clustering engine has "
                "not been implemented yet."
            )

        elif (
            self.state.task_type
            == "anomaly_detection"
        ):

            self.state.final_answer = (
                "Anomaly detection was identified "
                "as the requested task, but the "
                "anomaly detection engine has not "
                "been implemented yet."
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

            if not Path(
                file_path
            ).exists():

                raise FileNotFoundError(
                    f"Dataset not found: "
                    f"{file_path}"
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