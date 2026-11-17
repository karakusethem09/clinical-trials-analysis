"""Entry point: `.venv/bin/streamlit run app/Home.py` from the repository root."""
from __future__ import annotations

import streamlit as st

from lib import data

st.set_page_config(page_title="Will this trial finish?", layout="wide",
                   initial_sidebar_state="expanded")

pages = [
    st.Page("views/overview.py", title="Overview", default=True),
    st.Page("views/risk_explorer.py", title="Risk explorer", url_path="risk-explorer"),
    st.Page("views/why_stopped.py", title="Why trials stop", url_path="why-trials-stop"),
    st.Page("views/leakage.py", title="The leakage ladder", url_path="leakage-ladder"),
    st.Page("views/trust.py", title="How far to trust it", url_path="trust"),
]
nav = st.navigation(pages, position="sidebar")

with st.sidebar:
    pull = data.read_json("reports/survival_metrics.json")["pull_date"]
    st.caption("Early-termination risk of clinical trials, from ClinicalTrials.gov registry data "
               f"(interventional trials, current record versions, pulled {pull}).")

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1180px;}
  [data-testid="stMetricValue"] {font-size: 1.9rem;}
  [data-testid="stMetricLabel"] p {font-size: 0.92rem;}
  h1 {font-weight: 650; letter-spacing: -0.01em;}
  h2, h3 {font-weight: 600;}
</style>
""", unsafe_allow_html=True)

nav.run()
