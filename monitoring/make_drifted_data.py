"""Create the monitoring datasets in data/ (run once from the project root).

    reference.csv       what the model was trained on
    recent_normal.csv   "production" data that looks like the training data
    recent_drifted.csv  "production" data where the world has changed (shifted values)

Uses the same split as training/train.py (random_state=42), so reference == training set.
"""
from pathlib import Path

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    X, y = load_wine(return_X_y=True, as_frame=True)
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    drifted = X_test.copy()
    drifted["alcohol"] += 1.5          # e.g. riper grapes, stronger wines
    drifted["proline"] *= 1.6
    drifted["color_intensity"] += 2.0

    X_train.to_csv(DATA_DIR / "reference.csv", index=False)
    X_test.to_csv(DATA_DIR / "recent_normal.csv", index=False)
    drifted.to_csv(DATA_DIR / "recent_drifted.csv", index=False)
    print(f"Wrote reference ({len(X_train)} rows), recent_normal and recent_drifted "
          f"({len(X_test)} rows each) to {DATA_DIR}")


if __name__ == "__main__":
    main()
