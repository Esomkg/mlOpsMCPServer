"""Shared fixtures. Every test gets its own throwaway MLflow database, so tests never touch
the real mlflow.db / registry / audit log."""
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import mlflow.sklearn
import pytest
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier

from mcp_server import actions


@pytest.fixture(scope="session")
def wine_model():
    """A tiny but real model, trained once per test session (5 trees train in milliseconds)."""
    X, y = load_wine(return_X_y=True, as_frame=True)
    return RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0).fit(X, y)


@pytest.fixture
def registry(tmp_path, monkeypatch, wine_model):
    """Fresh MLflow DB + experiment + audit log in tmp_path. Returns a `make_run(f1)` factory."""
    db = tmp_path / "test.db"
    monkeypatch.setattr(actions, "DB_PATH", db)
    monkeypatch.setattr(actions, "AUDIT_LOG", tmp_path / "audit.jsonl")
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    exp_id = mlflow.create_experiment("wine-classifier", artifact_location=(tmp_path / "art").as_uri())
    mlflow.set_experiment(experiment_id=exp_id)

    def make_run(f1: float) -> str:
        with mlflow.start_run() as run:
            mlflow.log_metrics({"f1": f1, "accuracy": f1})
            mlflow.sklearn.log_model(
                wine_model,
                name="model",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
                pip_requirements=["scikit-learn"],  # skips MLflow's slow environment inference
            )
        return run.info.run_id

    return make_run
