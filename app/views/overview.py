from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from lib import data, style

h = data.headline()
best = h["primary_test_set"][h["headline_model"]]
surv = data.read_json("reports/survival_metrics.json")
dist = data.read_json("app/data/risk_distribution.json")
pal = style.palette()

style.page_intro(
    "Will this trial finish?",
    "Using only what is known on the day a clinical trial is registered, how well can we tell "
    "whether it will be terminated within five years of starting?")

st.markdown(
    f"The data are {surv['n']:,} interventional trials from ClinicalTrials.gov that started in 2008 or later. "
    f"The model learns from trials that started in 2008-2016 and is tested on {h['n_test']:,} trials that "
    "started in 2018-2020, so every score below is a forecast for years the model never saw.")

c1, c2, c3, c4 = st.columns(4)
auc = best["roc_auc"]
c1.metric("Test ROC AUC", style.num(auc["estimate"]),
          help="Probability that a randomly chosen terminated trial is ranked above a randomly "
               "chosen non-terminated one. 0.5 is chance, 1.0 is perfect.")
c1.caption(f"95% CI {style.num(auc['ci95'][0], 3)} to {style.num(auc['ci95'][1], 3)}")
prec = best["top_decile_precision"]
c2.metric("Terminated, top 10% of risk", style.pct(prec["estimate"]),
          help="Share terminated within five years among the 10% of test trials the model rated riskiest.")
c2.caption(f"{style.num(best['top_decile_lift'], 2)} times the base rate")
c3.metric("Base rate", style.pct(h["base_rate_test"]),
          help="Share of test trials terminated within five years of starting.")
c3.caption("terminated within 5 years")
c4.metric("Test cohort", f"{h['n_test']:,}")
c4.caption("trials started 2018-2020")

st.divider()
cal = data.read_parquet("app/data/calibration.parquet")
cal = cal[cal.model.str.startswith("Isotonic")].copy()
ratio = cal.observed.iloc[-1] / cal.observed.iloc[0]
base = alt.Chart(cal).encode(
    x=alt.X("decile:O", title="Decile of predicted risk (1 = lowest, 10 = highest)",
            axis=alt.Axis(labelAngle=0)),
    y=alt.Y("observed:Q", title="Terminated within 5 years", axis=alt.Axis(format=".0%")),
    tooltip=[alt.Tooltip("decile:O", title="Decile"),
             alt.Tooltip("observed:Q", title="Observed", format=".1%"),
             alt.Tooltip("predicted:Q", title="Mean predicted", format=".1%"),
             alt.Tooltip("n:Q", title="Trials", format=",")])
bars = base.mark_bar(color=pal.blue, cornerRadiusTopLeft=3, cornerRadiusTopRight=3, size=44)
vals = base.mark_text(dy=-8, fontSize=11, opacity=0.8, color=pal.ink).encode(text=alt.Text("observed:Q", format=".0%"))
rule = alt.Chart(pd.DataFrame({"y": [h["base_rate_test"]]})).mark_rule(
    color=pal.neutral, strokeDash=[4, 3]).encode(y="y:Q")
rule_text = alt.Chart(pd.DataFrame({"y": [h["base_rate_test"]], "decile": [1]})).mark_text(
    align="left", dx=-26, dy=-8, color=pal.neutral, fontSize=11).encode(
    y="y:Q", x="decile:O", text=alt.value(f"base rate {style.pct(h['base_rate_test'])}"))
style.chart((bars + vals + rule + rule_text).properties(title=style.title(
    f"The riskiest tenth of test trials were terminated {ratio:.1f} times as often as the safest tenth",
    "Test trials started 2018-2020, ranked by the saved dashboard model's calibrated risk; dashed line = base rate")),
    height=400)

left, right = st.columns(2, gap="large")
with left:
    st.subheader("What the project is")
    st.markdown(
        "- A **forecast at registration**: every feature had to be known on the day the trial was "
        "registered, which rules out the enrollment count, the number of sites and anything else the "
        "registry overwrites later.\n"
        "- A **ranking tool for portfolios**: it separates higher-risk from lower-risk groups of trials. "
        "It is not precise enough to make decisions about a single trial.\n"
        "- A **description of why trials stop**, from the sponsors' own free-text reasons.")
with right:
    st.subheader("What it is not")
    st.markdown(
        "- **Not a causal claim.** Features such as an industry sponsor or a Phase 2 design are "
        "associated with termination; changing them would not necessarily change the outcome.\n"
        "- **Not a reconstruction of the registry at registration.** ClinicalTrials.gov publishes only "
        "the current version of each record, so some fields may have been edited after the trial started, "
        "possibly because of what happened in it.")

st.divider()
st.subheader("Pages")
st.markdown(
    "- **Risk explorer**: look up a test trial, see its risk, what drove it and what actually happened; "
    "try a hypothetical design.\n"
    "- **Why trials stop**: the sponsors' stated reasons, grouped into categories by a transparent rule set.\n"
    "- **The leakage ladder**: why an apparently strong first model was mostly reading the answer, and "
    "what the defensible number for a different, harder question is.\n"
    "- **How far to trust it**: calibration, where the model is weak, survival curves, sensitivity checks "
    "and limitations.")
model_text = ("gradient boosting on registration-time fields plus a text score"
              if h["headline_model"] == "tabular_plus_text_gbm" else "gradient boosting on registration-time fields")
st.caption(f"Data: ClinicalTrials.gov API v2, pulled {surv['pull_date']}. Headline model, chosen on 2017 "
           f"starts before the test years were scored: {model_text}. Bootstrap intervals use "
           f"{h['bootstrap_resamples']:,} resamples of the test trials.")
