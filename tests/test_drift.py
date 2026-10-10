"""Tests for drift detection (monitoring/). Uses freshly generated data in a temp folder."""
import pytest

from monitoring import drift, make_drifted_data


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("data")
    mp = pytest.MonkeyPatch()
    mp.setattr(make_drifted_data, "DATA_DIR", path)
    mp.setattr(drift, "DATA_DIR", path)
    make_drifted_data.main()
    yield path
    mp.undo()


def test_datasets_are_generated(data_dir):
    assert {"reference.csv", "recent_normal.csv", "recent_drifted.csv"} == {p.name for p in data_dir.glob("*.csv")}
    assert drift.available_datasets() == ["recent_drifted.csv", "recent_normal.csv"]  # reference excluded


def test_unchanged_data_reports_no_drift(data_dir):
    result = drift.check_drift("recent_normal.csv")
    assert result["status"].startswith("OK")
    assert result["drifted_columns"] == []


def test_shifted_data_flags_exactly_the_shifted_features(data_dir):
    result = drift.check_drift("recent_drifted.csv")
    assert result["status"].startswith("FEATURE_DRIFT")
    assert {c["feature"] for c in result["drifted_columns"]} == {"alcohol", "proline", "color_intensity"}
    assert result["dataset_drift"] is False  # 3 of 13 features is below Evidently's 50% dataset threshold


def test_unknown_or_path_like_dataset_is_refused(data_dir):
    for name in ("nope.csv", "../reference.csv", "reference.csv"):
        assert "error" in drift.check_drift(name)
