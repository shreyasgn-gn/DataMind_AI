from pathlib import Path
from uuid import uuid4

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.encoders import jsonable_encoder

from agent.agent import DataMindAgent

from tools.data_tools import (
    load_dataset,
    inspect_dataset,
)

from tools.profile_tools import (
    profile_dataset,
)

from tools.cleaning_tools import (
    analyze_data_quality,
)

from tools.eda_tools import (
    perform_eda,
)


app = FastAPI(
    title="DataMind AI",
    description="Autonomous AI Data Scientist",
    version="1.0.0",
)


UPLOAD_DIR = Path(
    "data/uploads"
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ALLOWED_EXTENSIONS = {
    ".csv",
    ".xlsx",
}


def save_uploaded_file(
    file: UploadFile,
) -> tuple[str, Path]:
    """
    Save an uploaded CSV/XLSX file safely.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only CSV and XLSX files are supported."
            ),
        )

    original_name = Path(
        file.filename
    ).name

    unique_name = (
        f"{Path(original_name).stem}_"
        f"{uuid4().hex[:8]}"
        f"{extension}"
    )

    file_path = (
        UPLOAD_DIR / unique_name
    )

    try:

        contents = file.file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        file_path.write_bytes(
            contents
        )

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to save uploaded file: "
                f"{str(error)}"
            ),
        )

    return (
        original_name,
        file_path,
    )


def build_upload_response(
    original_file_name: str,
    file_path: Path,
    inspection: dict,
    profile: dict,
    quality: dict,
    eda: dict,
) -> dict:
    """
    Build a compact response for the basic upload endpoint.
    """

    return {
        "message": (
            "Dataset uploaded and analyzed successfully."
        ),
        "file_name": original_file_name,
        "file_path": str(file_path),
        "dataset": {
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
            "categorical_columns": profile.get(
                "categorical_columns",
                [],
            ),
            "datetime_columns": profile.get(
                "datetime_columns",
                [],
            ),
        },
        "data_quality": {
            "has_quality_issues": quality.get(
                "has_quality_issues",
                False,
            ),
            "missing_values": quality.get(
                "total_missing_values",
                0,
            ),
            "duplicate_rows": quality.get(
                "duplicate_rows",
                0,
            ),
        },
        "eda": {
            "analysis_available": bool(eda),
        },
    }


def build_analysis_response(
    original_file_name: str,
    file_path: Path,
    state,
) -> dict:
    """
    Build a compact API response from DataMindAgent state.

    Internal model objects, raw DataFrames, and large diagnostic
    structures are intentionally excluded.
    """

    response = {
        "message": (
            "DataMind analysis completed successfully."
            if state.status == "completed"
            else "DataMind analysis failed."
        ),
        "status": state.status,
        "file_name": original_file_name,
        "file_path": str(file_path),
        "question": state.question,
        "task": {
            "task_type": state.task_type,
            "problem_type": state.problem_type,
            "target_column": state.target_column,
            "source": state.task_understanding.get(
                "source"
            ),
            "model": state.task_understanding.get(
                "model"
            ),
            "llm_task_correction": state.task_understanding.get(
                "llm_task_correction"
            ),
        },
        "dataset": {
            "rows": state.profile.get(
                "rows",
                0,
            ),
            "columns": state.profile.get(
                "columns",
                0,
            ),
        },
        "data_quality": {
            "has_quality_issues": state.data_quality.get(
                "has_quality_issues",
                False,
            ),
            "missing_values": state.data_quality.get(
                "total_missing_values",
                0,
            ),
            "duplicate_rows": state.data_quality.get(
                "duplicate_rows",
                0,
            ),
        },
        "evaluation": state.evaluation_result,
        "model": {
            "best_model": state.training_result.get(
                "best_model"
            ),
        },
        "leakage": {
            "detected": state.leakage_report.get(
                "leakage_detected",
                False,
            ),
            "removed_columns": state.leakage_report.get(
                "removed_columns",
                [],
            ),
            "excluded_columns": state.leakage_report.get(
                "excluded_suspicious_columns",
                [],
            ),
        },
        "explainability": {},
        "recovery": {
            "retries": sum(
                max(
                    item.get(
                        "attempts",
                        1,
                    ) - 1,
                    0,
                )
                for item in state.recovery_history
            ),
        },
        "completed_steps": state.completed_steps,
        "errors": state.errors,
        "final_answer": state.final_answer,
        "final_answer_source": (
            state.final_answer_source
        ),
    }

    # ---------------------------------------------
    # Feature importance
    # ---------------------------------------------

    importance = (
        state.explainability_result.get(
            "feature_importance",
            {},
        )
    )

    if importance.get(
        "available",
        False,
    ):

        response["explainability"][
            "top_features"
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

    else:

        response["explainability"][
            "top_features"
        ] = []

    # ---------------------------------------------
    # SHAP summary
    # ---------------------------------------------

    shap_result = (
        state.explainability_result.get(
            "shap",
            {},
        )
    )

    response["explainability"][
        "shap_available"
    ] = shap_result.get(
        "success",
        False,
    )

    if shap_result.get(
        "success",
        False,
    ):

        response["explainability"][
            "shap_top_features"
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

        response["explainability"][
            "shap_top_features"
        ] = []

    return jsonable_encoder(
        response
    )


@app.get("/")
def root():
    """
    Health/status endpoint.
    """

    return {
        "message": "DataMind AI is running",
        "status": "online",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    """
    Backend health check.
    """

    return {
        "status": "healthy",
        "service": "DataMind AI",
    }


@app.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
):
    """
    Upload and perform basic dataset analysis.
    """

    (
        original_file_name,
        file_path,
    ) = save_uploaded_file(
        file
    )

    try:

        df = load_dataset(
            str(file_path)
        )

        inspection = inspect_dataset(
            str(file_path)
        )

        profile = profile_dataset(
            df
        )

        quality = analyze_data_quality(
            df
        )

        eda = perform_eda(
            df
        )

    except Exception as error:

        if file_path.exists():
            file_path.unlink(
                missing_ok=True
            )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Dataset analysis failed: "
                f"{str(error)}"
            ),
        )

    return build_upload_response(
        original_file_name,
        file_path,
        inspection,
        profile,
        quality,
        eda,
    )


@app.post("/analyze")
async def analyze_dataset(
    question: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Run the complete DataMind autonomous workflow.

    Returns a compact, frontend-friendly JSON response.
    """

    if not question.strip():

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    (
        original_file_name,
        file_path,
    ) = save_uploaded_file(
        file
    )

    try:

        agent = DataMindAgent()

        state = agent.run(
            question=question,
            file_path=str(file_path),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"DataMind agent failed: "
                f"{str(error)}"
            ),
        )

    return build_analysis_response(
        original_file_name,
        file_path,
        state,
    )