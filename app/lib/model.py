"""Scoring a hypothetical trial with the saved model (the what-if form)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from lib import data


@dataclass
class Score:
    raw: float
    calibrated: float
    base_value: float              # log-odds of an average trial
    contributions: pd.Series       # per-feature SHAP values (log-odds), indexed by feature


def frame(values: dict[str, Any]) -> pd.DataFrame:
    """One-row frame in the column order and category levels the model was trained with."""
    bundle = data.model_bundle()
    row = {}
    for f in bundle["features"]:
        v = values.get(f, np.nan)
        if f in bundle["categories"]:
            row[f] = pd.Categorical([v], categories=bundle["categories"][f])
        else:
            row[f] = [np.nan if v is None else float(v)]
    return pd.DataFrame(row)[bundle["features"]]


def score(values: dict[str, Any]) -> Score:
    bundle = data.model_bundle()
    X = frame(values)
    raw = float(bundle["model"].predict_proba(X)[:, 1][0])
    contrib = bundle["model"].predict(X, pred_contrib=True)[0]  # TreeSHAP, last column = expected value
    return Score(raw=raw, calibrated=float(bundle["calibrator"].predict([raw])[0]),
                 base_value=float(contrib[-1]),
                 contributions=pd.Series(contrib[:-1], index=bundle["features"]))
