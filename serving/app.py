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
from mlflow.exceptions import MlflowException
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

# The model currently being served. It is one dict that is replaced as a whole (never edited key by
# key), so a request that runs during a reload sees either the old model or the new one, never a mix.
state: dict = {}  # state["current"] = {"run_id": ..., "source": ..., "model": ...}


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


def select_and_load() -> dict:
    """Pick the model to serve and load it. Used at startup and by /reload.

    Prefer the model promoted to "production" in the registry; if nothing has been promoted yet,
    fall back to the run with the best F1. Raises if the chosen model cannot be loaded, so a failed
    reload never replaces a working model. (The trusted types were saved with the model in
    train.py, so load_model picks them up.)
    """
    try:
        prod = mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, "production")
    except MlflowException:
        prod = None  # no production alias (or nothing registered yet)

    if prod is not None:
        return {
            "run_id": prod.run_id,
            "source": f"registry production (version {prod.version})",
            "model": mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@production"),
        }
    run_id = find_best_run_id()
    return {
        "run_id": run_id,
        "source": "best F1 (no production model promoted yet)",
        "model": mlflow.sklearn.load_model(f"runs:/{run_id}/model"),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["current"] = select_and_load()  # runs once when the server starts
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
    cur = state["current"]
    return {"status": "ok", "run_id": cur["run_id"], "source": cur["source"]}


@app.post("/reload")
def reload_model():
    """Re-run the model selection and switch to the result without restarting the server.

    The new model is fully loaded before it replaces the old one. If loading fails, the old model
    keeps serving and this returns an error. Note: no authentication, so do not expose publicly.
    """
    previous = state["current"]
    try:
        new = select_and_load()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Reload failed, still serving run {previous['run_id']}: {e}",
        )
    state["current"] = new  # single assignment = atomic swap
    return {
        "reloaded": True,
        "changed": new["run_id"] != previous["run_id"],
        "previous": {"run_id": previous["run_id"], "source": previous["source"]},
        "now_serving": {"run_id": new["run_id"], "source": new["source"]},
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    cur = state["current"]  # read once, so model and run_id always belong together
    # Wrap in a DataFrame with column names, the same shape the model was trained on.
    row = pd.DataFrame([req.features], columns=FEATURE_NAMES)
    try:
        pred = int(cur["model"].predict(row)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    return PredictResponse(prediction=pred, class_name=CLASS_NAMES[pred], run_id=cur["run_id"])
