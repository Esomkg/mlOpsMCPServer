"""Compare recent data against the training data with Evidently and summarize the result."""
import warnings
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def available_datasets() -> list[str]:
    """Recent-data files that can be checked (everything in data/ except the reference)."""
    return sorted(p.name for p in DATA_DIR.glob("*.csv") if p.name != "reference.csv")


def check_drift(dataset: str) -> dict:
    # Evidently is imported here, not at the top, to keep importing this module cheap.
    from evidently import Report
    from evidently.presets import DataDriftPreset

    if dataset not in available_datasets():
        return {"error": f"Unknown dataset '{dataset}'.", "available": available_datasets()}

    reference = pd.read_csv(DATA_DIR / "reference.csv")
    current = pd.read_csv(DATA_DIR / dataset)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = Report([DataDriftPreset()]).run(current, reference).dict()["metrics"]

    summary: dict = {"dataset": dataset, "rows_checked": len(current), "reference_rows": len(reference)}
    drifted_columns = []
    for m in metrics:
        cfg = m["config"]
        if cfg["type"].endswith("DriftedColumnsCount"):
            summary["drifted_features"] = int(m["value"]["count"])
            summary["share_drifted"] = round(m["value"]["share"], 3)
            summary["dataset_drift_threshold"] = cfg["drift_share"]
        elif cfg["type"].endswith("ValueDrift"):
            # Statistical test p-value: small (below threshold) means the distribution changed.
            if m["value"] < cfg["threshold"]:
                drifted_columns.append({"feature": cfg["column"], "p_value": float(f"{m['value']:.3g}")})

    summary["dataset_drift"] = summary["share_drifted"] >= summary["dataset_drift_threshold"]
    summary["drifted_columns"] = sorted(drifted_columns, key=lambda d: d["p_value"])
    # "dataset_drift" alone is easy to misread: 3 shifted features out of 13 is still worth acting on.
    if summary["dataset_drift"]:
        summary["status"] = "DATASET_DRIFT: most features changed, retrain recommended"
    elif drifted_columns:
        summary["status"] = "FEATURE_DRIFT: some features changed, investigate"
    else:
        summary["status"] = "OK: no drift detected"
    return summary


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(check_drift(sys.argv[1] if len(sys.argv) > 1 else "recent_normal.csv"), indent=2))
