"""FastAPI service that serves the best wine model from MLflow.

Run from the project root:
    uvicorn serving.app:app --port 8000
Then open http://localhost:8000/docs
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.datasets import load_wine

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("MLOPS_MLFLOW_DB", ROOT / "mlflow.db"))  # override: Docker and tests
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
EXPERIMENT = "wine-classifier"
MODEL_NAME = "wine-classifier"  # name in the MLflow model registry

_wine = load_wine()
FEATURE_NAMES = list(_wine.feature_names)  # the 13 inputs, in the order the model expects
CLASS_NAMES = list(_wine.target_names)  # "class_0", "class_1", "class_2"

# Holds the loaded model so we load it once at startup, not on every request.
state: dict = {}


def find_best_run_id() -> str:
    """Ask MLflow for finished runs sorted by F1 (best first); newest wins ties."""
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT],
        filter_string="status = 'FINISHED'",
        order_by=["metrics.f1 DESC", "start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError(f"No finished runs in experiment '{EXPERIMENT}'. Run training first.")
    return runs.iloc[0]["run_id"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the server starts. Prefer the model promoted to "production" in the registry;
    # if nothing has been promoted yet, fall back to the run with the best F1.
    # (The trusted types were saved with the model in train.py, so load_model picks them up.)
    try:
        prod = mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, "production")
        state["run_id"] = prod.run_id
        state["source"] = f"registry production (version {prod.version})"
        state["model"] = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@production")
    except Exception:
        run_id = find_best_run_id()
        state["run_id"] = run_id
        state["source"] = "best F1 (no production model promoted yet)"
        state["model"] = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    yield


app = FastAPI(title="Wine model service", lifespan=lifespan)


class PredictRequest(BaseModel):
    features: list[float] = Field(
        min_length=13,
        max_length=13,
        description=f"13 numbers in this order: {FEATURE_NAMES}",
        examples=[[13.2, 2.77, 2.51, 18.5, 98.0, 1.96, 0.8, 0.35, 1.1, 4.5, 0.95, 2.1, 600.0]],
    )


class PredictResponse(BaseModel):
    prediction: int
    class_name: str
    run_id: str  # which MLflow run's model produced this answer


@app.get("/health")
def health():
    return {"status": "ok", "run_id": state["run_id"], "source": state["source"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Wrap in a DataFrame with column names, the same shape the model was trained on.
    row = pd.DataFrame([req.features], columns=FEATURE_NAMES)
    try:
        pred = int(state["model"].predict(row)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    return PredictResponse(prediction=pred, class_name=CLASS_NAMES[pred], run_id=state["run_id"])
