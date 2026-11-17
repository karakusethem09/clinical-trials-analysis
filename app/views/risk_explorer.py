from __future__ import annotations

import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from lib import data, labels, style
from lib.model import score

pal = style.palette()
h = data.headline()
df = data.scored_sample()
dist = data.read_json("app/data/risk_distribution.json")
defaults = data.read_json("app/data/whatif_defaults.json")
bundle = data.model_bundle()
FEATURES: list[str] = bundle["features"]
INTERVENTIONS = [f.removeprefix("intervention_") for f in FEATURES
                 if f.startswith("intervention_") and f != "intervention_model"]
final_auc = h["final_model_test"]["raw"]["roc_auc"]

style.page_intro(
    "Risk explorer",
    "Look up a trial from the test years (started 2018-2020), see the model's calibrated risk that it is "
    "terminated within five years, what pushed that risk up or down, and what actually happened.")
st.caption(f"A random sample of {len(df):,} of the {dist['n']:,} test trials, scored by the saved dashboard "
           f"model (trained on 2008-2016 starts, calibrated on 2017; ROC AUC {style.num(final_auc, 3)} on the test years). "
           "The model never saw these trials or their outcomes.")


def histogram(marker: float, marker_label: str) -> alt.LayerChart:
    """Share of test trials at each calibrated risk level.

    The isotonic calibration is a step function, so calibrated risk takes only a few
    dozen distinct values; each is drawn as its own spike rather than binned, which
    would leave empty bins between the levels and look like missing data."""
    lv = pd.DataFrame(dist["values"])
    lv["share"] = lv.n / dist["n"]
    lv["observed"] = lv.terminated / lv.n
    x = alt.X("p:Q", title="Calibrated 5-year termination risk (one spike per risk level)",
              axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 0.4], nice=False))
    tip = [alt.Tooltip("p:Q", title="Calibrated risk", format=".1%"),
           alt.Tooltip("n:Q", title="Test trials", format=","),
           alt.Tooltip("observed:Q", title="Actually terminated", format=".1%")]
    stems = alt.Chart(lv).mark_rule(color=pal.neutral, strokeWidth=3, opacity=0.75).encode(
        x=x, y=alt.Y("share:Q", title="Share of test trials", axis=alt.Axis(format=".0%")), y2=alt.datum(0),
        tooltip=tip)
    heads = alt.Chart(lv).mark_circle(color=pal.neutral, size=28, opacity=0.9).encode(x=x, y="share:Q", tooltip=tip)
    m = pd.DataFrame({"x": [marker], "label": [marker_label]})
    rule = alt.Chart(m).mark_rule(color=pal.orange, strokeWidth=2.5).encode(x="x:Q")
    text = alt.Chart(m).mark_text(align="right" if marker > 0.22 else "left", dx=-6 if marker > 0.22 else 6,
                                  baseline="top", color=pal.orange, fontWeight=600, fontSize=12).encode(
        x="x:Q", y=alt.value(2), text="label:N")
    base = alt.Chart(pd.DataFrame({"x": [dist["base_rate"]]})).mark_rule(
        color=pal.neutral, strokeDash=[4, 3]).encode(x="x:Q")
    return stems + heads + base + rule + text


def share_text(x: float) -> str:
    """A percentile as text; never rounds up to 100% or down to 0%."""
    return "over 99%" if x > 0.995 else "under 1%" if x < 0.005 else f"{x:.0%}"


def risk_text(p: float) -> str:
    return "under 0.5%" if p < 0.005 else style.pct(p)


def contributions_chart(contrib: pd.Series, values: dict, k: int = 6, title: str = "") -> alt.Chart:
    """Top-k features pushing risk up and top-k pushing it down, labelled in plain English."""
    s = contrib[contrib.abs() > 1e-4]
    picked = pd.concat([s[s > 0].nlargest(k), s[s < 0].nsmallest(k)])
    rows = pd.DataFrame({
        "feature": picked.index,
        "label": [labels.describe(f, values.get(f)) for f in picked.index],
        "value": picked.to_numpy(),
    })
    rows["direction"] = np.where(rows.value > 0, "Raises risk", "Lowers risk")
    rows["odds"] = np.exp(rows.value)
    order = rows.sort_values("value", ascending=False).label.tolist()
    return alt.Chart(rows).mark_bar(cornerRadiusEnd=3, height=16).encode(
        x=alt.X("value:Q", title="Contribution (log-odds)"),
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=340)),
        color=alt.Color("direction:N", title=None,
                        scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=[pal.orange, pal.blue])),
        tooltip=[alt.Tooltip("label:N", title="Feature"),
                 alt.Tooltip("value:Q", title="Log-odds", format="+.3f"),
                 alt.Tooltip("odds:Q", title="Multiplies the odds by", format=".2f")],
    ).properties(title=style.title(title, "Orange raises the risk, blue lowers it (SHAP values)"))


tab_trial, tab_whatif = st.tabs(["A test trial", "What-if illustration"])

# --------------------------------------------------------------------------- test trial
with tab_trial:
    f1, f2, f3, f4 = st.columns([1.4, 1, 1, 1])
    query = f1.text_input("Search by NCT id or words in the title", placeholder="e.g. NCT03 or asthma")
    phase_opts = sorted(df.phase.unique(), key=lambda p: list(labels.PHASE).index(p) if p in labels.PHASE else 99)
    phases = f2.multiselect("Phase", phase_opts, format_func=labels.PHASE.get)
    sponsor_opts = df.lead_sponsor_class.value_counts().index.tolist()
    sponsors = f3.multiselect("Lead sponsor type", sponsor_opts, format_func=lambda s: labels.SPONSOR.get(s, s))
    kinds = f4.multiselect("Intervention type", INTERVENTIONS, format_func=labels.INTERVENTION.get)

    view = df
    if query.strip():
        q = query.strip().lower()
        view = view[view.nct_id.str.lower().str.contains(q, regex=False)
                    | view.brief_title.str.lower().str.contains(q, regex=False)]
    if phases:
        view = view[view.phase.isin(phases)]
    if sponsors:
        view = view[view.lead_sponsor_class.isin(sponsors)]
    if kinds:
        view = view[view[[f"intervention_{k}" for k in kinds]].fillna(0).max(axis=1) >= 1]
    view = view.sort_values(["p_calibrated", "nct_id"], ascending=[False, True])

    if view.empty:
        st.info("No sampled trial matches these filters. Clear a filter or change the search.")
        st.stop()

    options = view.nct_id.tolist()
    titles = dict(zip(view.nct_id, view.brief_title))
    risks = dict(zip(view.nct_id, view.p_calibrated))
    pick = st.selectbox(
        f"Trial ({len(view):,} match, riskiest first)", options,
        format_func=lambda n: f"{n}  ·  risk {style.pct(risks[n], 0)}  ·  {titles[n][:95]}")
    t = view.set_index("nct_id").loc[pick]

    st.markdown(f"### {t.brief_title}")
    st.caption(f"[{pick}](https://clinicaltrials.gov/study/{pick}) · {labels.PHASE.get(t.phase, t.phase)} · "
               f"{labels.SPONSOR.get(t.lead_sponsor_class, t.lead_sponsor_class)} sponsor "
               f"({t.lead_sponsor_name}) · started {int(t.start_year)}")

    risk = float(t.p_calibrated)
    below = data.percentile_of(risk)
    m1, m2, m3 = st.columns(3)
    m1.metric("Calibrated 5-year termination risk", risk_text(risk))
    m1.caption(f"{style.num(risk / dist['base_rate'], 1)} times the base rate of {style.pct(dist['base_rate'])}")
    m2.metric("Riskier than", share_text(below))
    m2.caption("of test trials" + (" (in the top 10% of risk)" if risk >= dist["top_decile_threshold"] else ""))
    status = labels.STATUS.get(t.overall_status, t.overall_status)
    if t.y == 1:
        outcome, note = "Terminated", "within 5 years of starting"
    elif t.overall_status == "TERMINATED":
        outcome, note = "Not terminated", "within 5 years; terminated later"
    else:
        outcome, note = "Not terminated", f"within 5 years; status now: {status.lower()}"
    m3.metric("What happened", outcome)
    m3.caption(note)

    if isinstance(t.why_stopped, str) and t.why_stopped.strip():
        st.markdown(f"**Reason given by the sponsor:** \"{t.why_stopped.strip()}\"  \n"
                    f"Category assigned on the *Why trials stop* page: {t.why_category}")

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        style.chart(histogram(risk, f"this trial: {risk_text(risk)}").properties(title=style.title(
            f"This trial is riskier than {share_text(below)} of test trials",
            "All 2018-2020 test trials; dashed line = base rate", wrap=58)), height=430)
        st.caption(f"The calibration maps scores onto {len(dist['values'])} risk levels (a step function), "
                   "so the distribution is a set of spikes, one per level.")
    with right:
        feats = [f for f in FEATURES]
        contrib = pd.Series({f: float(t[f"shap_{f}"]) for f in feats})
        values = {f: t.get(f) for f in feats}
        style.chart(contributions_chart(contrib, values, title="What pushed this trial's risk up and down"),
                    height=470)

    with st.expander("All design fields used by the model for this trial"):
        rows = [{"Field": labels.FEATURE.get(f, f), "Value": labels.value_text(f, values.get(f)),
                 "Contribution (log-odds)": round(contrib[f], 3)} for f in feats]
        st.dataframe(pd.DataFrame(rows).sort_values("Contribution (log-odds)", key=abs, ascending=False),
                     hide_index=True, width="stretch")
        if "text_score" in FEATURES:
            st.caption("The text score is the prediction of a separate text model that reads the title, "
                       "conditions, summary and eligibility criteria, after statements that the study "
                       f"stopped are removed. Conditions listed: {t.conditions_text or 'none'}.")
        else:
            st.caption(f"Conditions listed: {t.conditions_text or 'none'}.")

# --------------------------------------------------------------------------- what-if
with tab_whatif:
    set_by_you = "the text score and the sponsor's history are" if "text_score" in FEATURES else \
        "the sponsor's history is"
    st.info("**Illustration only.** This scores a hypothetical trial with the same saved model. "
            f"Fields not shown are set to typical test-year values, and {set_by_you} "
            "set by you rather than read from a real record. The result shows how the model "
            "responds to design choices; it is an association learned from past trials, not the effect "
            "that changing a design would have.")
    nm = defaults["numeric_median"]
    tq = defaults.get("text_score_quantiles")
    st_ = defaults["sponsor_typical"]

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown("**Design**")
        phase = st.selectbox("Phase", ["PHASE2", "PHASE1", "EARLY_PHASE1", "PHASE1/PHASE2", "PHASE2/PHASE3",
                                       "PHASE3", "PHASE4", "NA"], format_func=labels.PHASE.get)
        allocation = st.selectbox("Allocation", ["RANDOMIZED", "NON_RANDOMIZED", "NA"],
                                  format_func=labels.ALLOCATION.get)
        masking = st.selectbox("Masking", ["NONE", "SINGLE", "DOUBLE", "TRIPLE", "QUADRUPLE"],
                               format_func=labels.MASKING.get)
        model_ = st.selectbox("Study design", ["PARALLEL", "SINGLE_GROUP", "CROSSOVER", "FACTORIAL",
                                               "SEQUENTIAL"], format_func=labels.MODEL.get)
        purpose = st.selectbox("Primary purpose", ["TREATMENT", "PREVENTION", "SUPPORTIVE_CARE", "DIAGNOSTIC",
                                                   "BASIC_SCIENCE", "SCREENING", "HEALTH_SERVICES_RESEARCH",
                                                   "OTHER"], format_func=labels.PURPOSE.get)
    with c2:
        st.markdown("**What is studied and who can join**")
        kinds_w = st.multiselect("Intervention types", INTERVENTIONS, default=["drug"],
                                 format_func=labels.INTERVENTION.get)
        n_cond = st.number_input("Conditions listed", 1, 20, int(nm["n_conditions"]))
        healthy = st.checkbox("Accepts healthy volunteers", value=False)
        sex = st.selectbox("Sex eligible", ["ALL", "FEMALE", "MALE"], format_func=labels.SEX.get)
        ages = st.slider("Age range (years)", 0, 100, (18, 75))
        no_max = st.checkbox("No upper age limit", value=True)
        chars = st.slider("Length of eligibility criteria (characters)", 200, 8000,
                          int(round(nm["criteria_chars"], -2)), step=100)
    with c3:
        st.markdown("**Sponsor**")
        sponsor = st.selectbox("Lead sponsor type", ["INDUSTRY", "OTHER", "NIH", "FED", "OTHER_GOV", "NETWORK"],
                               format_func=labels.SPONSOR.get)
        ind_collab = st.checkbox("Industry collaborator")
        nih_collab = st.checkbox("NIH collaborator")
        other_collab = st.checkbox("Other collaborator")
        history = st.radio("Sponsor's track record", ["Typical", "No earlier trials", "Set it"], horizontal=True,
                           help="Typical = median among test-year sponsors with earlier trials whose outcome "
                                "was public.")
        if history == "Set it":
            prior_ended = st.slider("Earlier trials with a known outcome", 1, 500, int(st_["sponsor_prior_ended"]))
            term_rate = st.slider("Share of those terminated", 0.0, 0.6,
                                  round(st_["sponsor_prior_terminated_rate"], 2), 0.01, format="%.2f")
            wd_rate = st.slider("Share of those withdrawn", 0.0, 0.3,
                                round(st_["sponsor_prior_withdrawn_rate"], 2), 0.01, format="%.2f")
            prior_reg = prior_ended * st_["sponsor_prior_registered"] / max(st_["sponsor_prior_ended"], 1)
        elif history == "Typical":
            prior_reg, prior_ended = st_["sponsor_prior_registered"], st_["sponsor_prior_ended"]
            term_rate, wd_rate = st_["sponsor_prior_terminated_rate"], st_["sponsor_prior_withdrawn_rate"]
        else:
            prior_reg, prior_ended, term_rate, wd_rate = 0, 0, math.nan, math.nan
        text_level = None
        if "text_score" in FEATURES and tq:
            text_level = st.select_slider(
                "Text score (what the text model reads into the description)",
                options=["10", "25", "50", "75", "90"], value="50",
                format_func=lambda q: {"10": "Low (10th pct)", "25": "25th pct", "50": "Typical (median)",
                                       "75": "75th pct", "90": "High (90th pct)"}[q])

    values = {
        "phase": phase, "allocation": allocation, "masking": masking, "primary_purpose": purpose,
        "intervention_model": model_, "lead_sponsor_class": sponsor, "sex": sex,
        "n_conditions": n_cond, "n_intervention_types": len(kinds_w),
        "min_age_years": ages[0], "max_age_years": math.nan if no_max else ages[1],
        "criteria_chars": chars,
        # bullets scale with length at the typical test-year ratio
        "criteria_bullets": round(chars * nm["criteria_bullets"] / nm["criteria_chars"]),
        "sponsor_prior_registered": prior_reg, "sponsor_prior_ended": prior_ended,
        "sponsor_prior_terminated_rate": term_rate, "sponsor_prior_withdrawn_rate": wd_rate,
        "has_collaborator": float(ind_collab or nih_collab or other_collab),
        "has_industry_collaborator": float(ind_collab), "has_nih_collaborator": float(nih_collab),
        "healthy_volunteers": float(healthy),
        **{f"intervention_{k}": float(k in kinds_w) for k in INTERVENTIONS},
    }
    if text_level is not None:
        values["text_score"] = tq[text_level]
    s = score(values)
    below_w = data.percentile_of(s.calibrated)
    st.divider()
    r1, r2 = st.columns([1, 1.15], gap="large")
    with r1:
        a, b = st.columns(2)
        a.metric("Illustrative 5-year risk", risk_text(s.calibrated))
        a.caption(f"{style.num(s.calibrated / dist['base_rate'], 1)} times the base rate")
        b.metric("Riskier than", share_text(below_w))
        b.caption("of real test trials")
        style.chart(histogram(s.calibrated, f"hypothetical: {risk_text(s.calibrated)}").properties(
            title=style.title("Where the hypothetical trial would sit",
                              "All 2018-2020 test trials; dashed line = base rate", wrap=58)), height=400)
        st.caption("Calibrated risks take a limited set of values because the isotonic calibration is a step "
                   "function; small changes to the inputs can leave the risk unchanged.")
    with r2:
        style.chart(contributions_chart(s.contributions, values, title="What drives the hypothetical score"),
                    height=470)
