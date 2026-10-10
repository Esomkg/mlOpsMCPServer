# MLOps Copilot

Ask Claude questions about an ML system in plain English. An MCP server exposes tools that talk to
MLflow (runs and model registry), a FastAPI model service, and Evidently (drift detection).

```
Claude Code ──MCP──▶ mcp_server/server.py ──▶ MLflow (runs, registry)
                                          ├──▶ FastAPI /predict  (serving/app.py)
                                          └──▶ Evidently drift   (monitoring/)
training/train.py ──▶ logs runs + models to MLflow
```

## Tools
| Tool | Kind | What it does |
|---|---|---|
| `list_runs`, `compare_runs` | read | Browse and rank training runs |
| `predict` | read | Classify a wine through the model API (warns on out-of-range values) |
| `check_drift` | read | Evidently drift report: recent data vs training data |
| `get_registry` | read | Which versions are in staging / production |
| `trigger_training` | **action** | Train a new model. Dry run unless `confirm=true`. |
| `promote_model` | **action** | Move a model to staging / production. Dry run unless `confirm=true`; production needs staging first and no F1 regression. Every real action is written to `logs/audit.jsonl`. |

## Run it with Docker
```
docker compose up --build        # trains a model, then API on :8000/docs and MLflow UI on :5000
docker compose run --rm -i mcp   # the MCP server over stdio (for an MCP client)
```

## Run it locally
```
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python training\train.py --n-estimators 100 --max-depth 5
.venv\Scripts\python -m uvicorn serving.app:app --port 8000
.venv\Scripts\python monitoring\make_drifted_data.py
```
`.mcp.json` registers the MCP server with Claude Code for this folder.

## Tests and CI
`pytest` runs the guardrail, tool, API and drift tests against a throwaway MLflow database.
`.github/workflows/ci.yml` runs them on every push and checks the Docker image builds.
