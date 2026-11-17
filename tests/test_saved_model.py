"""The saved dashboard model must only use registration-time fields."""
from pathlib import Path

import joblib
import pytest

from train_tabular import FEATURES

MODEL = Path(__file__).resolve().parents[1] / "models" / "final_model.joblib"
FORBIDDEN_EXACT = {"start_year", "days_registered_after_start", "registered_prospectively", "registered_by_start",
                   "overall_status", "why_stopped", "end_date", "stopped_early", "is_resolved", "y",
                   "terminated_5y", "status_verified_date", "last_update_date", "sponsor_prior_started",
                   "sponsor_prior_stop_rate", "sponsor_prior_ended_v1"}


def forbidden(features: list[str]) -> list[str]:
    return [f for f in features if f.startswith("leaky_") or f in FORBIDDEN_EXACT]


def test_feature_list_is_clean() -> None:
    assert forbidden(FEATURES) == []


@pytest.mark.skipif(not MODEL.exists(), reason="models/final_model.joblib not built")
def test_saved_model_features() -> None:
    bundle = joblib.load(MODEL)
    assert forbidden(bundle["features"]) == []
    assert set(bundle["features"]) <= set(FEATURES) | {"text_score"}
    assert list(bundle["model"].feature_name_) == bundle["features"]
