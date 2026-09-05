from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException

from tools.data_tools import load_dataset, inspect_dataset
from tools.profile_tools import profile_dataset
from tools.cleaning_tools import analyze_data_quality
from tools.eda_tools import perform_eda


app = FastAPI(
    title="DataMind AI",
    description="Autonomous AI Data Scientist",
    version="1.0.0"
)


UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".csv", ".xlsx"}


@app.get("/")
def root():
    return {
        "message": "DataMind AI is running",
        "status": "online"
    }


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload and analyze a CSV or XLSX dataset.
    """

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only CSV and XLSX files are supported."
        )

    file_path = UPLOAD_DIR / Path(file.filename).name

    contents = await file.read()
    file_path.write_bytes(contents)

    try:
        df = load_dataset(str(file_path))

        inspection = inspect_dataset(str(file_path))
        profile = profile_dataset(df)
        quality = analyze_data_quality(df)
        eda = perform_eda(df)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Dataset analysis failed: {str(error)}"
        )

    return {
        "message": "Dataset uploaded and analyzed successfully",
        "file_name": file.filename,
        "file_path": str(file_path),
        "inspection": inspection,
        "profile": profile,
        "data_quality": quality,
        "eda": eda
    }