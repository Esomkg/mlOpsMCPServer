"""Guardrail tests for trigger_training and promote_model (the tools that change things)."""
import json
from types import SimpleNamespace

import pytest

from mcp_server import actions


def audit_entries():
    if not actions.AUDIT_LOG.exists():
        return []
    return [json.loads(line) for line in actions.AUDIT_LOG.read_text().splitlines()]


# ---------------------------------------------------------------- trigger_training
@pytest.mark.parametrize(
    "kwargs",
    [{"n_estimators": 0}, {"n_estimators": 501}, {"n_estimators": 10, "max_depth": 0}, {"n_estimators": 10, "max_depth": 51}],
)
def test_training_rejects_out_of_range_settings(kwargs):
    assert "error" in actions.trigger_training(**kwargs, confirm=True)


def test_training_dry_run_does_not_start_a_process(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("dry run must not launch training")

    monkeypatch.setattr(actions.subprocess, "run", boom)
    result = actions.trigger_training(n_estimators=10, max_depth=3)  # confirm defaults to False
    assert result["dry_run"] is True
    assert result["would_train"] == {"n_estimators": 10, "max_depth": 3}


def test_training_confirm_runs_and_parses_result(registry, monkeypatch):
    out = "run_id=abc123 params={'n_estimators': 10} metrics={'accuracy': 0.9, 'f1': 0.88}\n"
    monkeypatch.setattr(
        actions.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=out, stderr="")
    )
    result = actions.trigger_training(n_estimators=10, confirm=True)
    assert result["trained"] is True
    assert (result["run_id"], result["f1"], result["accuracy"]) == ("abc123", 0.88, 0.9)
    assert audit_entries()[-1]["action"] == "trigger_training"


def test_training_failure_is_reported_and_audited(registry, monkeypatch):
    monkeypatch.setattr(
        actions.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="kaboom")
    )
    result = actions.trigger_training(n_estimators=10, confirm=True)
    assert "error" in result and "kaboom" in result["details"]
    assert audit_entries()[-1]["outcome"] == "failed"


# ------------------------------------------------------------------ promote_model
def test_promote_rejects_bad_stage_and_unknown_run(registry):
    run_id = registry(0.9)
    assert "error" in actions.promote_model(run_id, "archived")
    assert "error" in actions.promote_model("does-not-exist", "staging")


def test_dry_run_changes_nothing(registry):
    run_id = registry(0.9)
    result = actions.promote_model(run_id, "staging")  # confirm defaults to False
    assert result["dry_run"] is True
    assert actions.get_registry()["versions"] == []
    assert audit_entries() == []  # dry runs are not audited, only real actions and blocks


def test_production_requires_staging_first(registry):
    run_id = registry(0.9)
    result = actions.promote_model(run_id, "production", confirm=True)
    assert result["blocked"] is True
    assert "staging" in result["problems"][0]
    assert actions.get_registry()["versions"] == []  # even with confirm=True nothing was registered


def test_staging_then_production_happy_path(registry):
    run_id = registry(0.9)
    staged = actions.promote_model(run_id, "staging", confirm=True)
    assert staged["promoted"] is True
    promoted = actions.promote_model(run_id, "production", confirm=True)
    assert promoted["promoted"] is True
    reg = actions.get_registry()
    assert reg["aliases"] == {"staging": 1, "production": 1}
    assert [e["outcome"] for e in audit_entries()] == ["ok", "ok"]


def test_worse_model_cannot_replace_production(registry):
    good, weak = registry(0.95), registry(0.50)
    for stage in ("staging", "production"):
        assert actions.promote_model(good, stage, confirm=True)["promoted"] is True
    assert actions.promote_model(weak, "staging", confirm=True)["promoted"] is True  # staging is allowed
    result = actions.promote_model(weak, "production", confirm=True)
    assert result["blocked"] is True
    assert "worse" in result["problems"][0]
    assert actions.get_registry()["aliases"]["production"] == 1  # still the good model
    assert audit_entries()[-1]["outcome"] == "blocked"


def test_equal_or_better_model_can_replace_production(registry):
    old, new = registry(0.90), registry(0.95)
    for stage in ("staging", "production"):
        actions.promote_model(old, stage, confirm=True)
    actions.promote_model(new, "staging", confirm=True)
    assert actions.promote_model(new, "production", confirm=True)["promoted"] is True
    assert actions.get_registry()["aliases"]["production"] == 2
