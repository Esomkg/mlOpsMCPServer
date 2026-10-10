FROM python:3.12-slim

# MLOPS_MLFLOW_DB / MLOPS_AUDIT_LOG point at /state, a volume shared by the training, API and MLflow UI
# containers, so they all see the same runs and models.
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    MLOPS_MLFLOW_DB=/state/mlflow.db \
    MLOPS_AUDIT_LOG=/state/logs/audit.jsonl

WORKDIR /app

# Dependencies first: this layer is cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY training/ training/
COPY serving/ serving/
COPY mcp_server/ mcp_server/
COPY monitoring/ monitoring/

# Generate the reference / recent datasets used by check_drift.
RUN python monitoring/make_drifted_data.py

# Run from /state so MLflow's default artifact folder (./mlruns) lands in the shared volume.
RUN mkdir -p /state
WORKDIR /state

EXPOSE 8000
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
