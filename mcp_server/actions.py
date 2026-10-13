"""Tools that CHANGE things (train, promote), each wrapped in guardrails.

Guardrails used here:
  1. dry run by default: nothing happens unless confirm=True
  2. input validation: settings must be inside safe ranges
  3. business rules: production needs a prior staging step and no F1 regression
  4. audit log: every real action is appended to logs/audit.jsonl
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT =Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("MLOPS_MLFLOW_DB", ROOT / "mlflow.db"))
AUDIT_LOG = Path(os.environ.get("MLOPS_AUDIT_LOG", ROOT / "logs" / "audit.jsonl"))
MODEL_NAME = "wine-classifier"
STAGES = ("staging", "production")  # implemented as MLflow registry aliases

API_URL = os.environ.get("MODEL_API_URL", "http://127.0.0.1:8000")
RELOAD_TIMEOUT_S = 120  # loading a model can take a while on a cold machine
MAX_ESTIMATORS = 500
MAX_DEPTH = 50
TRAIN_TIMEOUT_S = 600


def audit(action: str, details: dict) -> None:
    AUDIT_LOG.parent.mkdir(exist_ok=True)
    entry = {"time": datetime.now(timezone.utc).isoformat(), "action": action, **details}
    with AUDIT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _client():
    # Lazy import: mlflow takes a long time to import (see server.py).
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
    return mlflow.MlflowClient()


# ---------------------------------------------------------------- trigger_training
def trigger_training(n_estimators: int = 100, max_depth: int | None = None, confirm: bool = False) -> dict:
    if not 1 <= n_estimators <= MAX_ESTIMATORS:
        return {"error": f"n_estimators must be between 1 and {MAX_ESTIMATORS}."}
    if max_depth is not None and not 1 <= max_depth <= MAX_DEPTH:
        return {"error": f"max_depth must be between 1 and {MAX_DEPTH}, or omitted for no limit."}

    settings = {"n_estimators": n_estimators, "max_depth": max_depth}
    if not confirm:
        return {
            "dry_run": True,
            "would_train": settings,
            "note": "Nothing was run. Call again with confirm=true to start this training run "
            "(it logs a new run to MLflow and can take a minute or two).",
        }

    cmd = [sys.executable, str(ROOT / "training" / "train.py"), "--n-estimators", str(n_estimators)]
    if max_depth is not None:
        cmd += ["--max-depth", str(max_depth)]
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1", "MLOPS_MLFLOW_DB": str(DB_PATH)}
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TRAIN_TIMEOUT_S, cwd=ROOT, env=env
        )
    except subprocess.TimeoutExpired:
        audit("trigger_training", {**settings, "outcome": "timeout"})
        return {"error": f"Training did not finish within {TRAIN_TIMEOUT_S}s."}

    match = re.search(r"run_id=(\w+).*?'accuracy': ([\d.]+).*?'f1': ([\d.]+)", proc.stdout, re.S)
    if proc.returncode != 0 or not match:
        audit("trigger_training", {**settings, "outcome": "failed"})
        return {"error": "Training failed.", "details": proc.stderr.strip()[-500:]}

    result = {
        "run_id": match.group(1),
        "settings": settings,
        "accuracy": round(float(match.group(2)), 4),
        "f1": round(float(match.group(3)), 4),
    }
    audit("trigger_training", {**result, "outcome": "ok"})
    return {"trained": True, **result,
            "note": "Compare it with compare_runs, then promote_model to staging if it is good."}


# ------------------------------------------------------------------ promote_model
def _versions(client) -> list:
    try:
        return list(client.search_model_versions(f"name='{MODEL_NAME}'"))
    except Exception:  # registered model does not exist yet
        return []


def _alias_version(client, alias: str):
    try:
        return client.get_model_version_by_alias(MODEL_NAME, alias)
    except Exception:
        return None


def _f1_of(client, run_id: str) -> float:
    return float(client.get_run(run_id).data.metrics["f1"])


def get_registry() -> dict:
    """Read-only view of registered versions and which alias (staging/production) points where."""
    client = _client()
    aliases = {a: _alias_version(client, a) for a in STAGES}
    return {
        "model": MODEL_NAME,
        "aliases": {a: (v.version if v else None) for a, v in aliases.items()},
        "versions": [
            {"version": v.version, "run_id": v.run_id, "f1": round(_f1_of(client, v.run_id), 4)}
            for v in sorted(_versions(client), key=lambda v: int(v.version))
        ],
    }


def promote_model(run_id: str, stage: str, confirm: bool = False) -> dict:
    if stage not in STAGES:
        return {"error": f"stage must be one of {STAGES}."}
    client = _client()
    try:
        run = client.get_run(run_id)
    except Exception:
        return {"error": f"Run '{run_id}' not found. Use list_runs to see valid run ids."}
    if run.info.status != "FINISHED" or "f1" not in run.data.metrics:
        return {"error": f"Run '{run_id}' is not a finished run with metrics."}

    candidate_f1 = _f1_of(client, run_id)
    existing = next((v for v in _versions(client) if v.run_id == run_id), None)
    problems, checks = [], [f"run {run_id} finished with f1={candidate_f1:.4f}"]

    if stage == "production":
        staged = _alias_version(client, "staging")
        if not (existing and staged and staged.version == existing.version):
            problems.append("This model is not in staging yet. Promote it to staging first.")
        else:
            checks.append(f"version {existing.version} is currently in staging")
        current = _alias_version(client, "production")
        if current:
            current_f1 = _f1_of(client, current.run_id)
            if candidate_f1 < current_f1:
                problems.append(
                    f"Candidate f1 {candidate_f1:.4f} is worse than production "
                    f"(version {current.version}, f1 {current_f1:.4f})."
                )
            else:
                checks.append(f"f1 is not worse than production version {current.version} ({current_f1:.4f})")
        else:
            checks.append("no current production model, so no regression check needed")

    plan = {
        "run_id": run_id,
        "stage": stage,
        "change": f"set alias '{stage}' on " + (f"version {existing.version}" if existing else "a new registered version"),
        "checks_passed": checks,
    }
    if problems:
        audit("promote_model", {**plan, "outcome": "blocked", "problems": problems})
        return {"blocked": True, "problems": problems, **plan}
    if not confirm:
        return {"dry_run": True, **plan, "note": "Nothing changed. Call again with confirm=true to apply."}

    import mlflow

    version = existing or mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, stage, str(version.version))
    result = {"promoted": True, "version": str(version.version), "stage": stage, "run_id": run_id}
    if stage == "production":
        result["api_reload"] = reload_api()  # make the live API switch to the new model now
    audit("promote_model", {**plan, "version": str(version.version), "outcome": "ok",
                            "api_reload": result.get("api_reload")})
    return result


def reload_api() -> dict:
    """Ask the running model API to re-select and load its model (POST /reload)."""
    try:
        resp = httpx.post(f"{API_URL}/reload", timeout=RELOAD_TIMEOUT_S)
        resp.raise_for_status()
    except httpx.ConnectError:
        return {"reloaded": False, "reason": f"Model API not reachable at {API_URL}. The promotion is saved; "
                                              "the API will use it the next time it starts."}
    except httpx.HTTPStatusError as e:
        return {"reloaded": False, "reason": f"API reload failed, so it is still serving its previous model: "
                                              f"{e.response.text[:300]}"}
    except httpx.HTTPError as e:
        return {"reloaded": False, "reason": f"API reload call failed: {e}"}
    return resp.json()
