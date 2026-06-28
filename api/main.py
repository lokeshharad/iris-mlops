import os
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
from src.reference import MLFLOW_MODEL_STAGE

app = FastAPI(title="Iris MLOps API", version="1.0.0")


class PredictionRequest(BaseModel):
    features: list[float]


class PredictionResponse(BaseModel):
    prediction: str
    probabilities: list[float] | None = None


MODEL = None


def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL

    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions("name LIKE 'iris-%'")
    if not versions:
        raise RuntimeError("No registered MLflow models found")

    latest_version = max(versions, key=lambda m: int(m.version))
    registry_uri = f"models:/{latest_version.name}/{MLFLOW_MODEL_STAGE}"
    MODEL = mlflow.sklearn.load_model(registry_uri)
    return MODEL


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if len(request.features) != 4:
        raise ValueError("Exactly 4 features are required")

    model = load_model()
    features = [[float(x) for x in request.features]]
    prediction = model.predict(features)[0]
    prediction = str(prediction)  # Convert to string for JSON serialization

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0].tolist()

    return PredictionResponse(prediction=prediction, probabilities=probabilities)
