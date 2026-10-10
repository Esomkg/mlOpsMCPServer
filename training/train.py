"""Train a wine classifier and log every run to MLflow.

Usage:
    python training/train.py --n-estimators 100 --max-depth 5
"""
import argparse
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Store MLflow data in the project root (mlruns/ + mlflow.db), not wherever the script is run from.
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("MLOPS_MLFLOW_DB", ROOT / "mlflow.db"))  # override is used by tests
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
mlflow.set_experiment("wine-classifier")


def main(n_estimators: int, max_depth: int | None) -> None:
    # 1. Data: 178 wines, 13 chemical features, 3 classes.
    X, y = load_wine(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 2. Everything inside this block is recorded as ONE run in MLflow.
    with mlflow.start_run() as run:
        params = {"n_estimators": n_estimators, "max_depth": max_depth}
        mlflow.log_params(params)

        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds, average="macro"),
        }
        mlflow.log_metrics(metrics)

        # Save the trained model itself so we can load/serve it later.
        # MLflow saves sklearn models in the safe "skops" format and asks us to declare which
        # types we trust. Tree is the node storage inside random forests; we trained this model
        # ourselves, so trusting it is fine. Only load models from sources you trust.
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_train.head(2),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(f"run_id={run.info.run_id} params={params} metrics={metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    args = parser.parse_args()
    main(args.n_estimators, args.max_depth)
