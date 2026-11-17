"""Cached loaders. The app reads only committed files in models/, reports/ and app/data/."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
APP_DATA = ROOT / "app" / "data"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"


@st.cache_data
def read_json(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text())


@st.cache_data
def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(ROOT / path)


@st.cache_data
def read_csv(path: str) -> pd.DataFrame:
    # keep "NA" (the registry's "no phase" code) as text rather than a missing value
    return pd.read_csv(ROOT / path, keep_default_na=False, na_values=[""])


def headline() -> dict[str, Any]:
    return read_json("reports/headline_metrics.json")


@st.cache_resource
def model_bundle() -> dict[str, Any]:
    """The saved model: LightGBM classifier, isotonic calibrator, features and category levels."""
    return joblib.load(MODELS / "final_model.joblib")


@st.cache_data
def scored_sample() -> pd.DataFrame:
    """models/dashboard_sample.parquet (risk + SHAP) joined to the trial details in app/data."""
    sample = pd.read_parquet(MODELS / "dashboard_sample.parquet")
    details = pd.read_parquet(APP_DATA / "trials.parquet")
    df = sample.merge(details.drop(columns=["phase", "lead_sponsor_class"]), on="nct_id", how="left")
    df["brief_title"] = df.brief_title.fillna("(no title)")
    return df


def percentile_of(p: float) -> float:
    """Share of test trials (2018-2020) with a calibrated risk strictly below p."""
    levels = read_json("app/data/risk_distribution.json")["values"]
    n = sum(v["n"] for v in levels)
    return sum(v["n"] for v in levels if v["p"] < p - 1e-9) / n


def why_taxonomy() -> tuple[dict[str, list[str]], list[str]]:
    """The categoriser's patterns, imported from src/analysis so the page never drifts from the code."""
    sys.path.insert(0, str(ROOT / "src" / "analysis"))
    import why_stopped  # noqa: PLC0415

    return why_stopped.TAXONOMY, why_stopped.PRECEDENCE


@st.cache_data
def readme_section(heading: str) -> list[str]:
    """Bullet points of one README section, so the limitations shown stay in sync with the README."""
    text = (ROOT / "README.md").read_text()
    match = re.search(rf"^## {re.escape(heading)}\s*\n(.*?)(?=^## |\Z)", text, flags=re.M | re.S)
    if not match:
        return []
    bullets = re.split(r"^- ", match.group(1), flags=re.M)
    return [" ".join(b.split()) for b in bullets[1:]]
