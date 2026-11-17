from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from lib import data, labels, style

pal = style.palette()
h = data.headline()
calj = h["calibration"]
best_lift = h["primary_test_set"][h["headline_model"]]["top_decile_lift"]
surv = data.read_json("reports/survival_metrics.json")
defaults = data.read_json("app/data/whatif_defaults.json")

style.page_intro(
    "How far to trust it",
    "A ROC AUC around 0.66 means the model ranks groups of trials better than chance, not that it can call "
    "individual outcomes. This page shows where its probabilities are reliable, where it is weak, and how much "
    "the result depends on judgment calls.")

# ------------------------------------------------------------------ calibration
st.header("Calibration")
cal = data.read_parquet("app/data/calibration.parquet")
saved = defaults["calibration_summary"]
l, r = st.columns([1.2, 1], gap="large")
with l:
    top = float(max(cal.predicted.max(), cal.observed.max())) + 0.02
    diag = alt.Chart(pd.DataFrame({"x": [0, top], "y": [0, top]})).mark_line(
        color=pal.neutral, strokeDash=[4, 3], strokeWidth=1.5).encode(x="x:Q", y="y:Q")
    order = ["Raw score", "Isotonic (fitted on 2017)"]
    lines = alt.Chart(cal).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=50, filled=True)).encode(
        x=alt.X("predicted:Q", title="Mean predicted risk in decile", axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domain=[0, top])),
        y=alt.Y("observed:Q", title="Observed share terminated within 5 years", axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domain=[0, top])),
        color=alt.Color("model:N", title=None, sort=order,
                        scale=alt.Scale(domain=order, range=[pal.orange, pal.blue])),
        order="decile:O",
        tooltip=[alt.Tooltip("model:N", title="Score"), alt.Tooltip("decile:O", title="Decile"),
                 alt.Tooltip("predicted:Q", title="Predicted", format=".1%"),
                 alt.Tooltip("observed:Q", title="Observed", format=".1%"),
                 alt.Tooltip("n:Q", title="Trials", format=",")])
    style.chart((diag + lines).properties(title=style.title(
        "Raw scores under-predict; recalibration moves the deciles closer to the diagonal",
        "Saved dashboard model on test years 2018-2020, deciles of predicted risk; dashed = perfect calibration", wrap=58)),
        height=480)
with r:
    tbl = pd.DataFrame([
        {"Model": "Headline model, raw", "ECE": calj["raw"]["ece"], "Brier": calj["raw"]["brier"],
         "Mean predicted": calj["raw"]["mean_predicted"]},
        {"Model": "Headline model, isotonic", "ECE": calj["isotonic"]["ece"], "Brier": calj["isotonic"]["brier"],
         "Mean predicted": calj["isotonic"]["mean_predicted"]},
        {"Model": "Dashboard model, raw", "ECE": saved["Raw score"]["ece"],
         "Brier": h["final_model_test"]["raw"]["brier"],
         "Mean predicted": saved["Raw score"]["mean_predicted"]},
        {"Model": "Dashboard model, isotonic", "ECE": saved["Isotonic (fitted on 2017)"]["ece"],
         "Brier": h["final_model_test"]["calibrated_brier"],
         "Mean predicted": saved["Isotonic (fitted on 2017)"]["mean_predicted"]},
    ])
    st.markdown(f"Observed rate on the test years: **{style.pct(calj['observed_rate'])}**.")
    st.dataframe(tbl, hide_index=True, width="stretch", column_config={
        "ECE": st.column_config.NumberColumn(format="%.3f", help="Expected calibration error, 10 equal-sized bins"),
        "Brier": st.column_config.NumberColumn(format="%.4f"),
        "Mean predicted": st.column_config.NumberColumn(format="percent")})
    st.markdown(
        "The isotonic calibration is fitted on 2017 starts, which neither training nor early stopping saw. "
        "It reduces the calibration error but cannot close the gap fully: 2018-2020 had a slightly "
        "higher termination rate than 2017, so even the calibrated risks are a little low on average. "
        f"Because it maps scores onto a few steps, it also ties some trials and lowers the ROC AUC slightly "
        f"({style.num(calj['raw']['roc_auc'], 4)} to {style.num(calj['isotonic']['roc_auc'], 4)}).")

# ------------------------------------------------------------------ decision curve
st.header("Is it useful for a decision?")
dc = data.read_csv("reports/decision_curve.csv")
long = dc.melt(id_vars=["threshold", "share_flagged"],
               value_vars=["net_benefit_model", "net_benefit_treat_all", "net_benefit_treat_none"],
               var_name="strategy", value_name="net_benefit")
names_dc = {"net_benefit_model": "Flag trials above the threshold (model)",
            "net_benefit_treat_all": "Flag every trial", "net_benefit_treat_none": "Flag no trial"}
long["strategy"] = long.strategy.map(names_dc)
order_dc = list(names_dc.values())
better = dc[(dc.net_benefit_model > dc.net_benefit_treat_all) & (dc.net_benefit_model > 0)]
lo_dc = float(long.net_benefit.min())
l2, r2 = st.columns([1.3, 1], gap="large")
with l2:
    c = alt.Chart(long).mark_line(strokeWidth=2, clip=True).encode(
        x=alt.X("threshold:Q", title="Risk threshold for flagging a trial", axis=alt.Axis(format=".0%")),
        y=alt.Y("net_benefit:Q", title="Net benefit per trial",
                scale=alt.Scale(domain=[max(lo_dc, -0.03), float(dc.net_benefit_treat_all.max()) + 0.01])),
        color=alt.Color("strategy:N", title=None, sort=order_dc,
                        scale=alt.Scale(domain=order_dc, range=[pal.blue, pal.orange, pal.neutral]),
                        legend=alt.Legend(orient="bottom", direction="vertical")),
        strokeDash=alt.StrokeDash("strategy:N", sort=order_dc, legend=None,
                                  scale=alt.Scale(domain=order_dc, range=[[1, 0], [5, 3], [2, 2]])),
        tooltip=[alt.Tooltip("strategy:N", title="Strategy"),
                 alt.Tooltip("threshold:Q", title="Threshold", format=".0%"),
                 alt.Tooltip("net_benefit:Q", title="Net benefit", format=".4f"),
                 alt.Tooltip("share_flagged:Q", title="Share flagged by the model", format=".0%")])
    head = ("The model adds net benefit over flagging all or none for thresholds of "
            f"{style.pct(better.threshold.min(), 0)}-{style.pct(better.threshold.max(), 0)}") if len(better) else \
        "The model does not beat flagging all or none at any threshold"
    style.chart(c.properties(title=style.title(
        head, "Recalibrated headline model, test years 2018-2020; net benefit = true positives minus "
              "false positives weighted by the threshold odds, per trial", wrap=58)), height=430)
with r2:
    st.markdown(
        "A decision curve asks: if trials above a risk threshold were flagged for, say, an extra feasibility "
        "review, would that do more good than reviewing every trial or none? The threshold expresses how many "
        "unnecessary reviews one caught termination is worth (at 20%, one catch is worth four false alarms).")
    st.markdown(
        f"The top tenth of trials by risk is terminated {style.num(best_lift, 2)} times as often as average. "
        "That supports **prioritising** a review of the riskiest decile of a portfolio. It does not support a "
        "verdict about any single trial: most flagged trials still finish.")

# ------------------------------------------------------------------ subgroups
st.header("Where the model is weak")
sub = data.read_csv("reports/subgroup_metrics.csv")
overall = sub[sub.dimension == "all"].iloc[0]
sub = sub[sub.dimension != "all"].copy()
ph = sub[sub.dimension == "phase"].set_index("group")
st.warning(
    f"**Weakest where termination is most common.** For Phase 2 trials (the largest drug-trial group, "
    f"{style.pct(ph.loc['PHASE2', 'base_rate'], 0)} terminated) the ROC AUC is only "
    f"{style.num(ph.loc['PHASE2', 'roc_auc'], 2)}, and {style.num(ph.loc['PHASE1/PHASE2', 'roc_auc'], 2)} for "
    f"Phase 1/2 ({style.pct(ph.loc['PHASE1/PHASE2', 'base_rate'], 0)} terminated), against "
    f"{style.num(overall.roc_auc, 2)} overall. Even after recalibration the model under-predicts Phase 1/2 by "
    f"{style.num(100 * ph.loc['PHASE1/PHASE2', 'calibration_in_the_large'], 1)} and Phase 1 by "
    f"{style.num(100 * ph.loc['PHASE1', 'calibration_in_the_large'], 1)} percentage points.")
dim_names = {"phase": "Phase", "lead_sponsor_class": "Lead sponsor", "intervention_type": "Intervention"}
sub["Dimension"] = sub.dimension.map(dim_names)
sub["Group"] = [labels.PHASE.get(g, g) if d == "phase" else labels.SPONSOR.get(g, g) if d == "lead_sponsor_class"
                else labels.INTERVENTION.get(g, g) for d, g in zip(sub.dimension, sub.group)]
sub["Group"] = sub.Group.str.replace(" (not a drug trial)", "", regex=False)
dim = st.segmented_control("Break down by", list(dim_names.values()), default="Phase")
dim = dim or "Phase"
part = sub[sub.Dimension == dim].sort_values("roc_auc")
part["weak"] = part.roc_auc < overall.roc_auc - 0.03
dots = alt.Chart(part).mark_point(filled=True, opacity=1).encode(
    x=alt.X("roc_auc:Q", title="ROC AUC on test trials (0.5 = chance)", scale=alt.Scale(domain=[0.5, 0.75])),
    y=alt.Y("Group:N", sort=part.Group.tolist(), title=None, axis=alt.Axis(labelLimit=240)),
    size=alt.Size("n:Q", title="Test trials", scale=alt.Scale(range=[60, 500]),
                  legend=alt.Legend(format=",.0f", orient="bottom", values=[500, 2000, 5000, 15000])),
    color=alt.condition(alt.datum.weak, alt.value(pal.orange), alt.value(pal.blue)),
    tooltip=[alt.Tooltip("Group:N"), alt.Tooltip("n:Q", title="Trials", format=","),
             alt.Tooltip("events:Q", title="Terminated", format=","),
             alt.Tooltip("base_rate:Q", title="Base rate", format=".1%"),
             alt.Tooltip("roc_auc:Q", title="ROC AUC", format=".3f"),
             alt.Tooltip("calibration_in_the_large:Q", title="Observed minus predicted", format="+.1%")])
AUC_SCALE = alt.Scale(domain=[0.5, 0.75])
ref = alt.Chart(pd.DataFrame({"x": [overall.roc_auc]})).mark_rule(color=pal.neutral, strokeDash=[4, 3]).encode(
    x=alt.X("x:Q", scale=AUC_SCALE))
reftext = alt.Chart(pd.DataFrame({"x": [overall.roc_auc]})).mark_text(
    align="left", dx=4, dy=-4, baseline="bottom", fontSize=11, color=pal.neutral).encode(
    x=alt.X("x:Q", scale=AUC_SCALE), y=alt.value(0), text=alt.value(f"all test trials {style.num(overall.roc_auc, 3)}"))
w = part.nsmallest(2, "roc_auc")
heading = (f"Weakest for {w.Group.iloc[0]} ({style.num(w.roc_auc.iloc[0], 2)}) and {w.Group.iloc[1]} "
           f"({style.num(w.roc_auc.iloc[1], 2)}); best for {part.Group.iloc[-1]} ({style.num(part.roc_auc.iloc[-1], 2)})")
style.chart((ref + reftext + dots).properties(title=style.title(
    heading, "Headline model, test years 2018-2020; orange = more than 0.03 below the overall ROC AUC; "
             "groups with at least 200 trials and 20 terminations")), height=max(300, 42 * len(part) + 170))
with st.expander("Subgroup table"):
    st.dataframe(part[["Group", "n", "events", "base_rate", "roc_auc", "pr_auc", "mean_predicted_calibrated",
                       "calibration_in_the_large"]].rename(columns={
        "n": "Trials", "events": "Terminated", "base_rate": "Base rate", "roc_auc": "ROC AUC", "pr_auc": "PR AUC",
        "mean_predicted_calibrated": "Mean predicted", "calibration_in_the_large": "Observed minus predicted"}),
        hide_index=True, width="stretch", column_config={
            "Trials": st.column_config.NumberColumn(format="localized"),
            "Base rate": st.column_config.NumberColumn(format="percent"),
            "ROC AUC": st.column_config.NumberColumn(format="%.3f"),
            "PR AUC": st.column_config.NumberColumn(format="%.3f"),
            "Mean predicted": st.column_config.NumberColumn(format="percent"),
            "Observed minus predicted": st.column_config.NumberColumn(format="%.3f")})
st.markdown(
    "The phases with the most terminations are the hardest to rank: Phase 2 and Phase 1/2 trials stop about "
    f"{style.pct(sub[(sub.dimension == 'phase') & (sub.group == 'PHASE2')].base_rate.iloc[0], 0)} of the time, "
    "and the registration record says little about which of them will.")

# ------------------------------------------------------------------ survival
st.header("Time to termination")
curves = data.read_parquet("app/data/survival_curves.parquet")
aj5 = surv["terminated_by_5_years"]["aalen_johansen"]
km5 = surv["terminated_by_5_years"]["one_minus_kaplan_meier"]
st.markdown(
    f"Survival analysis can use all {surv['n']:,} started trials, including the "
    f"{surv['n_started_2021_or_later']:,} that started in 2021 or later and are mostly still running. "
    "A completed trial can no longer be terminated, so completion is a competing event, not censoring.")
l, r = st.columns(2, gap="large")
with l:
    ov = curves[curves.dimension == "overall"]
    order = ["Aalen-Johansen (completion as competing risk)", "1 - Kaplan-Meier (completion censored)"]
    c = alt.Chart(ov).mark_line(strokeWidth=2, interpolate="step-after").encode(
        x=alt.X("years:Q", title="Years since start", scale=alt.Scale(domain=[0, 10])),
        y=alt.Y("incidence:Q", title="Cumulative share terminated", axis=alt.Axis(format=".0%")),
        color=alt.Color("group:N", title=None, sort=order, scale=alt.Scale(domain=order, range=[pal.blue, pal.orange]),
                        legend=alt.Legend(orient="bottom", direction="vertical")),
        strokeDash=alt.StrokeDash("group:N", sort=order, scale=alt.Scale(domain=order, range=[[1, 0], [5, 3]]),
                                  legend=None),
        tooltip=[alt.Tooltip("group:N", title="Method"), alt.Tooltip("years:Q", title="Years", format=".1f"),
                 alt.Tooltip("incidence:Q", title="Terminated", format=".1%")])
    style.chart(c.properties(title=style.title(
        f"Ignoring the competing risk nearly doubles the 5-year estimate: {style.pct(km5)} vs {style.pct(aj5)}",
        f"All {surv['n']:,} started trials, 2008 onwards", wrap=58)), height=450)
with r:
    which = st.segmented_control("Curves by", ["Phase", "Lead sponsor"], default="Phase") or "Phase"
    dim = "phase" if which == "Phase" else "sponsor"
    cs = curves[curves.dimension == dim].copy()
    at5 = cs[cs.years == 5.0].sort_values("incidence", ascending=False)
    keep = at5.group.tolist()
    if len(keep) > 3:  # three colours at most; the rest are grey context lines
        highlight = keep[:2] + keep[-1:]
    else:
        highlight = keep
    cs["shown"] = cs.group.where(cs.group.isin(highlight), "Other groups")
    cs["label"] = cs.group + " (n=" + cs.n.map("{:,}".format) + ")"
    dom = highlight + ["Other groups"]
    c = alt.Chart(cs).mark_line(interpolate="step-after").encode(
        x=alt.X("years:Q", title="Years since start", scale=alt.Scale(domain=[0, 10])),
        y=alt.Y("incidence:Q", title="Cumulative share terminated", axis=alt.Axis(format=".0%")),
        detail="group:N",
        color=alt.Color("shown:N", title=None, sort=dom,
                        scale=alt.Scale(domain=dom, range=[pal.orange, pal.aqua, pal.blue, pal.faint]),
                        legend=alt.Legend(orient="bottom", columns=2)),
        strokeWidth=alt.condition(alt.datum.shown == "Other groups", alt.value(1.2), alt.value(2.2)),
        tooltip=[alt.Tooltip("label:N", title="Group"), alt.Tooltip("years:Q", title="Years", format=".1f"),
                 alt.Tooltip("incidence:Q", title="Terminated", format=".1%")])
    hi, lo = at5.iloc[0], at5.iloc[-1]
    style.chart(c.properties(title=style.title(
        f"By year 5, {style.pct(hi.incidence, 0)} of {hi.group} trials are terminated vs "
        f"{style.pct(lo.incidence, 0)} of {lo.group}",
        "Aalen-Johansen cumulative incidence; highest two and lowest group coloured, others grey", wrap=58)), height=450)
    with st.expander("Five-year values for every group"):
        st.dataframe(at5[["group", "n", "incidence"]].rename(columns={
            "group": "Group", "n": "Trials", "incidence": "Terminated by year 5"}), hide_index=True,
            width="stretch", column_config={
                "Terminated by year 5": st.column_config.NumberColumn(format="percent"),
                "Trials": st.column_config.NumberColumn(format="localized")})
ci_ = surv["c_index"]
st.caption(f"A penalised Cox model fitted on starts up to 2017 reaches a concordance index of "
           f"{style.num(ci_['test_2018_2020'], 3)} on 2018-2020 starts (hazard ratios in `reports/survival_hazard_ratios.csv`).")

# ------------------------------------------------------------------ sensitivity
st.header("Sensitivity to judgment calls")
sens = data.read_csv("reports/sensitivity_metrics.csv")
names = {"primary": "Primary set-up",
         "no_registration_filter": "No registration-timing filter",
         "legacy_registration_filter": "Old 21-day filter on day-1 dates",
         "unknown_as_not_terminated": "UNKNOWN status counted as not terminated",
         "unknown_verified_after_5y": "UNKNOWN counted only if verified running after year 5",
         "drop_stale_running": "Still running, not verified in the last year: left out",
         "plus_registration_delay": "Plus registration delay as a feature",
         "plus_start_year": "Plus start year as a feature"}
sens["Variant"] = sens.variant.map(names).fillna(sens.variant)
prim = sens[sens.variant == "primary"].roc_auc_on_primary_test.iloc[0]
sens["Change on primary test set"] = sens.roc_auc_on_primary_test - prim
sens["own"] = [f"{style.num(a)} ({' to '.join(style.num(float(x)) for x in str(c).strip('[]').split(','))})"
               for a, c in zip(sens.roc_auc, sens.roc_auc_ci95)]
st.dataframe(sens[["Variant", "n_test", "base_rate_test", "own", "roc_auc_on_primary_test",
                   "Change on primary test set"]].rename(columns={
    "n_test": "Test trials", "base_rate_test": "Terminated", "own": "ROC AUC, own test set (95% CI)",
    "roc_auc_on_primary_test": "ROC AUC, primary test set"}),
    hide_index=True, width="stretch", column_config={
        "Test trials": st.column_config.NumberColumn(format="localized"),
        "Terminated": st.column_config.NumberColumn(format="percent"),
        "ROC AUC, primary test set": st.column_config.NumberColumn(format="%.3f"),
        "Change on primary test set": st.column_config.NumberColumn("Change vs primary", format="%+.3f")})
spread = sens.roc_auc_on_primary_test.max() - sens.roc_auc_on_primary_test.min()
st.markdown(f"Tabular model, one decision changed at a time. On the same primary test set every variant lands "
            f"within {style.num(spread, 3)} of each other, so the headline does not hinge on any single choice. The "
            "variant without the registration filter scores higher on its own test set because that set "
            "contains easier, retrospectively registered trials.")

# ------------------------------------------------------------------ limitations
st.header("Limitations")
items = data.readme_section("Limitations")
for item in items:
    st.markdown(f"- {item}")
if not items:
    st.markdown("See the README.")
