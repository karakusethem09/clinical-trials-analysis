from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from lib import data, style

pal = style.palette()
h = data.headline()
ladder = h["ladder_own_test_sets"]
primary = h["primary_test_set"]
ev = data.read_json("app/data/leakage_evidence.json")

style.page_intro(
    "The leakage ladder",
    "The registry shows each trial as it is today, not as it was when it was registered. A model built "
    "carelessly on that snapshot learns the outcome from its own consequences. The same question, asked "
    "three ways, gives three very different answers.")

RUNGS = [
    ("naive", "1. Naive",
     "Old label (terminated or withdrawn vs completed, finished trials only) and fields the registry "
     "overwrites after a trial ends: enrollment count, number of sites and countries."),
    ("partly_fixed", "2. Partly fixed",
     "Overwritten fields removed, but the old label is kept, together with the registration delay, the start "
     "year and a sponsor track record that counts outcomes before they were public."),
    ("defensible", "3. Defensible",
     "Primary task: trials that actually started, terminated within five years, registration-time fields only, "
     "sponsor history lagged by 12 months."),
]
rows = pd.DataFrame([{"key": k, "rung": name, "auc": ladder[k]["roc_auc"]["estimate"],
                      "lo": ladder[k]["roc_auc"]["ci95"][0], "hi": ladder[k]["roc_auc"]["ci95"][1],
                      "definition": d} for k, name, d in RUNGS])
ablation_name = h["ablation_model"]
ablation_auc = primary[ablation_name]["roc_auc"]

cols = st.columns(3)
for col, r in zip(cols, rows.itertuples()):
    col.metric(r.rung, f"{style.num(r.auc, 3)}")
    col.caption(f"95% CI {style.num(r.lo, 3)} to {style.num(r.hi, 3)}. {r.definition}")
st.caption(("Rung 3 is the pre-specified headline model (tabular only: the text score did not earn its place "
            "on the 2017 selection year). With the text score added it scores "
            if ablation_name == "tabular_plus_text_gbm" else
            "Rung 3 is the pre-specified headline model (with the text score). Without it, it scores ")
           + f"{style.num(ablation_auc['estimate'], 3)} ({style.num(ablation_auc['ci95'][0], 3)} to "
           f"{style.num(ablation_auc['ci95'][1], 3)}). ROC AUC: 0.5 is chance, 1.0 is perfect.")

rows["label"] = rows.auc.map(style.num)
base = alt.Chart(rows).encode(
    y=alt.Y("rung:N", sort=[r[1] for r in RUNGS], title=None, axis=alt.Axis(labelFontSize=13)))
err = base.mark_rule(strokeWidth=2, color=pal.neutral).encode(
    x=alt.X("lo:Q", title="Test ROC AUC with 95% bootstrap interval (0.5 = chance)",
            scale=alt.Scale(domain=[0.5, 0.9])),
    x2="hi:Q")
dots = base.mark_point(filled=True, size=160, opacity=1).encode(
    x="auc:Q",
    color=alt.condition(alt.datum.key == "defensible", alt.value(pal.blue), alt.value(pal.orange)),
    tooltip=[alt.Tooltip("rung:N", title="Set-up"), alt.Tooltip("auc:Q", title="ROC AUC", format=".3f"),
             alt.Tooltip("lo:Q", title="95% CI from", format=".3f"), alt.Tooltip("hi:Q", title="to", format=".3f")])
lbl = base.mark_text(dy=-16, fontSize=12, fontWeight=600, color=pal.ink).encode(x="auc:Q", text="label:N")
chance = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(color=pal.neutral, strokeDash=[3, 3]).encode(
    x=alt.X("x:Q", scale=alt.Scale(domain=[0.5, 0.9])))
style.chart((chance + err + dots + lbl).properties(title=style.title(
    f"Removing leaks cuts the apparent ROC AUC from {style.num(rows.auc.iloc[0], 2)} to {style.num(rows.auc.iloc[2], 2)}",
    "Each set-up on its own test set: rungs 1-2 on 2018-2019 starts with the old label, rung 3 on 2018-2020 "
    "starts with the primary label")), height=300)

p_fixed = primary["partly_fixed_scored_on_primary"]["roc_auc"]["estimate"]
p_naive = primary["naive_scored_on_primary"]["roc_auc"]["estimate"]
st.markdown(
    f"Scored as-is on the primary test set, the partly fixed model reaches {style.num(p_fixed, 3)}, about the same as the "
    f"defensible one ({style.num(rows.auc.iloc[2], 3)}). So most of the drop from rung 2 to rung 3 comes from measuring "
    f"a different, harder estimand, not a worse model. The naive model still scores {style.num(p_naive, 3)} there, because "
    "it keeps reading the overwritten fields.")

st.divider()
st.header("The traps, with the evidence")

# ------------------------------------------------------------------ trap 1
st.subheader("1. Fields that are overwritten when a trial ends")
enr = pd.DataFrame(ev["enrollment_by_status"])
enr["Status"] = enr.overall_status.str.capitalize()
med = enr.set_index("overall_status").median_enrollment
l, r = st.columns([1, 1.2], gap="large")
with l:
    st.markdown(
        "For a finished trial the registry replaces the *planned* number of participants with the *actual* "
        f"number. A terminated trial therefore shows a small enrollment: the median is {style.num(med['TERMINATED'], 0)} "
        f"participants against {style.num(med['COMPLETED'], 0)} for completed trials, and {style.num(med['WITHDRAWN'], 0)} for "
        "withdrawn ones, which never enrolled anyone. A model that sees this field is told the outcome. "
        "The number of sites has the same problem: "
        f"{style.pct(enr.set_index('overall_status').share_no_sites['WITHDRAWN'], 0)} of withdrawn trials list "
        "no site at all.")
with r:
    bars = alt.Chart(enr).mark_bar(cornerRadiusEnd=3, height=22).encode(
        x=alt.X("median_enrollment:Q", title="Median enrollment shown in the registry (participants)"),
        y=alt.Y("Status:N", sort=["Completed", "Terminated", "Withdrawn"], title=None),
        color=alt.condition(alt.datum.overall_status == "COMPLETED", alt.value(pal.blue), alt.value(pal.orange)),
        tooltip=[alt.Tooltip("Status:N"), alt.Tooltip("n:Q", title="Trials", format=","),
                 alt.Tooltip("median_enrollment:Q", title="Median enrollment", format=",.0f")])
    txt = alt.Chart(enr).mark_text(align="left", dx=5, fontSize=12, color=pal.ink).encode(
        x="median_enrollment:Q", y=alt.Y("Status:N", sort=["Completed", "Terminated", "Withdrawn"]),
        text=alt.Text("median_enrollment:Q", format=",.0f"))
    style.chart((bars + txt).properties(title=style.title(
        f"Terminated trials show a median of {style.num(med['TERMINATED'], 0)} participants, completed ones "
        f"{style.num(med['COMPLETED'], 0)}", "Interventional trials started 2008-2019, finished, registered within 21 days "
        "of start", wrap=58)), height=250)

# ------------------------------------------------------------------ trap 2
st.subheader("2. Planned start dates of withdrawn trials")
sd = pd.DataFrame(ev["start_date_by_status"]).set_index("overall_status")
l, r = st.columns([1, 1.2], gap="large")
with l:
    st.markdown(
        "A withdrawn trial never started, so its start date is only a plan, and "
        f"{style.pct(sd.share_estimated['WITHDRAWN'], 0)} of them say so explicitly (date type *estimated*; "
        f"none of the completed or terminated trials do). Planned dates sit in the future at registration, so "
        f"withdrawn trials look unusually prospective: the median one was registered "
        f"{style.num(-sd.median_days_registered_after_start['WITHDRAWN'], 0)} days *before* its start date, against "
        f"{style.num(sd.median_days_registered_after_start['COMPLETED'], 0)} days *after* for completed trials. The "
        "registration delay, a harmless-looking feature, was really a withdrawal detector. The defensible "
        "set-up keeps only trials that started and treats withdrawal as a separate question.")
with r:
    sd2 = sd.reset_index()
    sd2["Status"] = sd2.overall_status.str.capitalize()
    sd2["days"] = sd2.median_days_registered_after_start
    bars = alt.Chart(sd2).mark_bar(cornerRadiusEnd=3, height=22).encode(
        x=alt.X("days:Q", title="Median days from start date to registration (negative = registered before)"),
        y=alt.Y("Status:N", sort=["Completed", "Terminated", "Withdrawn"], title=None),
        color=alt.condition(alt.datum.overall_status == "COMPLETED", alt.value(pal.blue), alt.value(pal.orange)),
        tooltip=[alt.Tooltip("Status:N"), alt.Tooltip("n:Q", title="Trials", format=","),
                 alt.Tooltip("days:Q", title="Median days", format=",.0f"),
                 alt.Tooltip("share_estimated:Q", title="Start date estimated", format=".0%")])
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=pal.neutral).encode(x="x:Q")
    style.chart((bars + zero).properties(title=style.title(
        f"Withdrawn trials were registered a median {style.num(-sd.median_days_registered_after_start['WITHDRAWN'], 0)} "
        "days before a start that never happened", "Interventional trials with a start date in 2008-2020", wrap=58)),
        height=250)

# ------------------------------------------------------------------ trap 3
st.subheader("3. Dropping trials that are still running")
yr = pd.DataFrame(ev["label_rate_by_start_year"])
long = yr.melt(id_vars="start_year", value_vars=["old_label_rate", "primary_label_rate"],
               var_name="label", value_name="rate")
names = {"old_label_rate": "Old label: stopped among finished trials",
         "primary_label_rate": "Primary label: terminated within 5 years"}
long["label"] = long.label.map(names)
first, last = yr.iloc[0], yr.iloc[-1]
l, r = st.columns([1, 1.2], gap="large")
with l:
    st.markdown(
        "The old label kept only trials that had finished. In recent start years many long trials are still "
        f"running ({style.pct(last.share_still_running, 0)} of {int(last.start_year)} starts), and those are mostly "
        "trials that have not stopped, so the trials left over are disproportionately the ones that did. The old "
        f"stop rate climbs to {style.pct(last.old_label_rate, 0)} for {int(last.start_year)} starts. A fixed "
        "five-year horizon gives every trial in an eligible year a known label, and the rate stays roughly "
        f"flat ({style.pct(yr.primary_label_rate.min(), 0)} to {style.pct(yr.primary_label_rate.max(), 0)}).")
with r:
    lines = alt.Chart(long).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=36, filled=True)).encode(
        x=alt.X("start_year:O", title="Start year", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("rate:Q", title="Share of trials", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 0.25])),
        color=alt.Color("label:N", title=None, sort=list(names.values()),
                        scale=alt.Scale(domain=list(names.values()), range=[pal.orange, pal.blue]),
                        legend=alt.Legend(orient="bottom", direction="vertical")),
        tooltip=[alt.Tooltip("start_year:O", title="Start year"), alt.Tooltip("label:N", title="Label"),
                 alt.Tooltip("rate:Q", title="Rate", format=".1%")])
    style.chart(lines.properties(title=style.title(
        "The old label's stop rate rises in recent years; the five-year label does not",
        "Prospectively registered interventional trials by start year", wrap=58)), height=390)

# ------------------------------------------------------------------ trap 4
st.subheader("4. A sponsor's track record, counted too early")
st.markdown(
    "How often a sponsor's earlier trials were terminated is a useful feature, but only outcomes that were "
    "public at the time count. The registry keeps no history of status changes, so the first version counted "
    "earlier trials whose termination was only reported later. The defensible version anchors the history at "
    "the earlier of start and registration, counts an earlier trial only from 12 months after its end date, "
    "and only if it was registered before the anchor (DECISIONS.md, rows 10 and 24).")
