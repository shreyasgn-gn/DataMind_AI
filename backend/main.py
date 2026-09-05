from fastapi import FastAPI

app = FastAPI(
    title="DataMind AI",
    description="Autonomous AI Data Scientist",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "DataMind AI is running",
        "status": "online"
    }