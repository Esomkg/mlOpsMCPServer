"""Tests for the read-only MCP tools in server.py (called as plain Python functions)."""
import httpx

from mcp_server import server

WINE = {
    "alcohol": 14.23, "malic_acid": 1.71, "ash": 2.43, "alcalinity_of_ash": 15.6, "magnesium": 127,
    "total_phenols": 2.8, "flavanoids": 3.06, "nonflavanoid_phenols": 0.28, "proanthocyanins": 2.29,
    "color_intensity": 5.64, "hue": 1.04, "od280/od315_of_diluted_wines": 3.92, "proline": 1065,
}


class FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"prediction": 0, "class_name": "class_0", "run_id": "r1"}


def test_predict_reports_wrong_feature_names_without_calling_api(monkeypatch):
    monkeypatch.setattr(server.httpx, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no call")))
    result = server.predict({"alcohol": 13.0})
    assert result["error"] == "Wrong feature names."
    assert "proline" in result["missing"]


def test_predict_sends_features_in_model_order(monkeypatch):
    sent = {}

    def fake_post(url, json, timeout):
        sent["features"] = json["features"]
        return FakeResponse()

    monkeypatch.setattr(server.httpx, "post", fake_post)
    shuffled = dict(reversed(list(WINE.items())))  # dict order must not matter
    result = server.predict(shuffled)
    assert sent["features"] == list(WINE.values())
    assert result["class_name"] == "class_0" and "warnings" not in result


def test_predict_warns_about_out_of_range_values(monkeypatch):
    monkeypatch.setattr(server.httpx, "post", lambda *a, **k: FakeResponse())
    result = server.predict({**WINE, "proline": 6_000_000})
    assert len(result["warnings"]) == 1 and "proline" in result["warnings"][0]


def test_predict_handles_api_down(monkeypatch):
    def refuse(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(server.httpx, "post", refuse)
    assert "not reachable" in server.predict(WINE)["error"]


def test_list_runs_and_compare_runs(registry):
    low, high = registry(0.80), registry(0.95)
    assert {r["run_id"] for r in server.list_runs()} == {low, high}
    ranking = server.compare_runs(metric="f1", top_n=5)["ranking"]
    assert [r["run_id"] for r in ranking] == [high, low]
    assert "error" in server.compare_runs(metric="bogus")


def test_compare_runs_tie_prefers_newest_run(registry):
    older, newer = registry(1.0), registry(1.0)
    assert server.compare_runs()["ranking"][0]["run_id"] == newer


def test_get_registry_starts_empty(registry):
    registry(0.9)
    assert server.get_registry() == {
        "model": "wine-classifier", "aliases": {"staging": None, "production": None}, "versions": [],
    }
