"""Tests for the FastAPI model service (serving/app.py)."""
import mlflow
import pytest
from fastapi.testclient import TestClient

import serving.app as api
from mcp_server import actions

GOOD_ROW = [14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065]


@pytest.fixture
def client(registry):
    """API wired to the throwaway DB from the `registry` fixture (which already set the tracking URI)."""
    def start():
        return TestClient(api.app)

    return start


def test_without_promotion_serves_best_f1(registry, client):
    registry(0.80)
    best = registry(0.95)
    with client() as c:
        health = c.get("/health").json()
    assert health["run_id"] == best
    assert health["source"].startswith("best F1")


def test_promoted_production_model_wins_over_best_f1(registry, client):
    chosen, better = registry(0.80), registry(0.99)
    for stage in ("staging", "production"):
        actions.promote_model(chosen, stage, confirm=True)
    with client() as c:
        health = c.get("/health").json()
    assert health["run_id"] == chosen  # not the 0.99 run: production is decided by promotion
    assert "registry production" in health["source"]


def test_predict_returns_a_class(registry, client):
    registry(0.9)
    with client() as c:
        body = c.post("/predict", json={"features": GOOD_ROW}).json()
    assert body["class_name"] == f"class_{body['prediction']}"
    assert body["prediction"] in (0, 1, 2)


@pytest.mark.parametrize("features", [[1.0, 2.0, 3.0], GOOD_ROW + [1.0], []])
def test_predict_rejects_wrong_number_of_features(registry, client, features):
    registry(0.9)
    with client() as c:
        assert c.post("/predict", json={"features": features}).status_code == 422
