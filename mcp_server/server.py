"""MLOps Copilot MCP server: read-only tools over MLflow and the model API.

Claude Code starts this file as a subprocess and talks to it over stdin/stdout (stdio transport).
So never print() to stdout in here: it would corrupt the protocol. Log to stderr instead.
"""
import os
import sys
from pathlib import Path

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from sklearn.datasets import load_wine

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `import monitoring...` works when run as a script
EXPERIMENT = "wine-classifier"
API_URL = os.environ.get("MODEL_API_URL", "http://127.0.0.1:8000")

# The 13 inputs the model expects, with the min/max seen in training (used to warn about odd values).
_X = load_wine(as_frame=True).data
FEATURE_NAMES = list(_X.columns)
FEATURE_RANGES = {c: (float(_X[c].min()), float(_X[c].max())) for c in FEATURE_NAMES}

mcp = MCPServer(
    "mlops-copilot",
    instructions="Tools to inspect wine-classifier training runs in MLflow and get predictions "
    "from the model API.",
)
READ_ONLY = ToolAnnotations(readOnlyHint=True)  # tells the client these tools change nothing


def _runs_df():
    # mlflow takes a long time to import, so we import it lazily. This keeps the server's
    # startup fast, so Claude Code doesn't time out waiting for it.
    import mlflow

    from mcp_server import actions

    mlflow.set_tracking_uri(f"sqlite:///{actions.DB_PATH}")  # same DB (and override) as the actions
    return mlflow.search_runs(
        experiment_names=[EXPERIMENT],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"],
    )


def _summarize(row) -> dict:
    return {
        "run_id": row["run_id"],
        "n_estimators": row.get("params.n_estimators"),
        "max_depth": row.get("params.max_depth"),
        "accuracy": round(float(row["metrics.accuracy"]), 4),
        "f1": round(float(row["metrics.f1"]), 4),
        "started": str(row["start_time"]),
    }


@mcp.tool(annotations=READ_ONLY)
def list_runs(limit: int = 10) -> list[dict]:
    """List the most recent finished training runs with their parameters and scores."""
    df = _runs_df().head(max(1, min(limit, 50)))
    return [_summarize(r) for _, r in df.iterrows()]


@mcp.tool(annotations=READ_ONLY)
def compare_runs(metric: str = "f1", top_n: int = 5) -> dict:
    """Rank finished training runs by a metric ('f1' or 'accuracy'), best first.

    Ties are broken in favour of the newest run.
    """
    if metric not in ("f1", "accuracy"):
        return {"error": f"Unknown metric '{metric}'. Use 'f1' or 'accuracy'."}
    df = _runs_df()  # already newest first, and sort_values(stable) keeps that order on ties
    df = df.sort_values(f"metrics.{metric}", ascending=False, kind="stable").head(max(1, top_n))
    return {"metric": metric, "ranking": [_summarize(r) for _, r in df.iterrows()]}


@mcp.tool(annotations=READ_ONLY)
def predict(features: dict[str, float]) -> dict:
    """Predict the wine class (class_0, class_1 or class_2) from 13 chemical measurements.

    `features` must contain exactly these keys: alcohol, malic_acid, ash, alcalinity_of_ash,
    magnesium, total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins,
    color_intensity, hue, od280/od315_of_diluted_wines, proline. Needs the model API running.
    """
    missing = [n for n in FEATURE_NAMES if n not in features]
    extra = [k for k in features if k not in FEATURE_NAMES]
    if missing or extra:
        return {"error": "Wrong feature names.", "missing": missing, "unexpected": extra}

    # Named dict -> list in the fixed order the API/model expects.
    ordered = [features[n] for n in FEATURE_NAMES]
    warnings = [
        f"{n}={features[n]} is outside the training range {lo:g}..{hi:g}; the prediction may be unreliable."
        for n, (lo, hi) in FEATURE_RANGES.items()
        if not lo <= features[n] <= hi
    ]

    try:
        resp = httpx.post(f"{API_URL}/predict", json={"features": ordered}, timeout=15)
        resp.raise_for_status()
    except httpx.ConnectError:
        return {"error": f"Model API not reachable at {API_URL}. Start it with uvicorn serving.app:app --port 8000"}
    except httpx.HTTPError as e:
        return {"error": f"Model API call failed: {e}"}

    result = resp.json()
    if warnings:
        result["warnings"] = warnings
    return result


@mcp.tool(annotations=READ_ONLY)
def check_drift(dataset: str = "recent_normal.csv") -> dict:
    """Check whether recent data has drifted away from the data the model was trained on.

    Runs Evidently statistical tests per feature. `dataset` is a file in data/ such as
    'recent_normal.csv' (looks like training data) or 'recent_drifted.csv' (shifted values).
    Pass an unknown name to get the list of available datasets.
    """
    from monitoring.drift import check_drift as run_check  # lazy: Evidently is slow to import

    return run_check(dataset)


@mcp.tool(annotations=READ_ONLY)
def get_registry() -> dict:
    """Show the model registry: which versions exist and which are in staging / production."""
    from mcp_server import actions

    return actions.get_registry()


# Tools below change state. They default to a dry run and need confirm=true to act.
# Claude Code also asks the human for permission before calling a non-read-only tool.
ACTION = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)
PROMOTE = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True)


@mcp.tool(annotations=ACTION)
def trigger_training(n_estimators: int = 100, max_depth: int | None = None, confirm: bool = False) -> dict:
    """Train a new wine model and log it as a new MLflow run.

    DRY RUN by default: with confirm=false it only reports what it would do. Always show the user
    the dry-run result and get their agreement before calling again with confirm=true.
    n_estimators: 1-500. max_depth: 1-50, or omit for unlimited.
    """
    from mcp_server import actions

    return actions.trigger_training(n_estimators, max_depth, confirm)


@mcp.tool(annotations=PROMOTE)
def promote_model(run_id: str, stage: str, confirm: bool = False) -> dict:
    """Promote a run's model to 'staging' or 'production' in the model registry.

    DRY RUN by default: with confirm=false it checks the rules and reports the plan without changing
    anything. Rules: production requires the model to be in staging first and must not have a worse
    F1 than the current production model. Never use confirm=true unless the user has explicitly
    approved this exact promotion after seeing the dry-run result.

    Promoting to 'production' also tells the running model API to switch to the new model right away
    (no restart needed). The result's `api_reload` field says whether that worked.
    """
    from mcp_server import actions

    return actions.promote_model(run_id, stage, confirm)


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
