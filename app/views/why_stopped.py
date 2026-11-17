from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from lib import data, style

pal = style.palette()
summary = data.read_json("app/data/why_summary.json")
taxonomy, precedence = data.why_taxonomy()
OTHER = "Other / unclear"
CATS = precedence + [OTHER]
counts = summary["n_by_status"]

style.page_intro(
    "Why trials stop",
    "When a trial is terminated, withdrawn or suspended, the sponsor can give a short free-text reason. "
    "These reasons are grouped into categories by a transparent keyword rule set, so every assignment "
    "can be traced to a pattern.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reasons categorised", f"{summary['n_reasons']:,}")
c1.caption(", ".join(f"{v:,} {k.lower()}" for k, v in counts.items()))
top = max(precedence, key=lambda c: summary["primary_category_share"][c])
c2.metric(top, style.pct(summary['primary_category_share'][top]))
c2.caption("of reasons (primary category)")
c3.metric("Other / unclear", style.pct(summary["other_unclear_share"]))
c3.caption("matched no pattern")
c4.metric("More than one category", style.pct(summary["multi_label_share"]))
c4.caption("primary category chosen by precedence")
st.caption("Accuracy has not been measured yet. A stratified sample of 300 reasons is in "
           "`reports/why_stopped_label_sample.csv` for hand-checking.")


def complete(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Add zero rows for categories that do not occur in a group."""
    groups = df[[by, "group_n"]].drop_duplicates()
    full = groups.merge(pd.DataFrame({"primary_category": CATS}), how="cross")
    out = full.merge(df[[by, "primary_category", "n", "share"]], on=[by, "primary_category"], how="left")
    return out.fillna({"n": 0, "share": 0.0})


# ---------------------------------------------------------------- overall mix
overall = data.read_parquet("app/data/why_overall.parquet")
status = st.segmented_control("Status", ["All", "TERMINATED", "WITHDRAWN", "SUSPENDED"], default="All",
                              format_func=lambda s: s.capitalize() if s != "All" else "All stopped trials")
status = status or "All"
mix = overall[overall.status == status].copy()
mix["kind"] = mix.primary_category.map(lambda c: "Other / unclear" if c == OTHER else "Category")
lead = mix[mix.primary_category != OTHER].sort_values("share", ascending=False).iloc[0]
mix_order = mix.sort_values("share", ascending=False).primary_category.tolist()
bars = alt.Chart(mix).mark_bar(cornerRadiusEnd=3, height=18).encode(
    x=alt.X("share:Q", title="Share of reasons (primary category)", axis=alt.Axis(format=".0%")),
    y=alt.Y("primary_category:N", title=None, sort=mix_order, axis=alt.Axis(labelLimit=320)),
    color=alt.Color("kind:N", legend=None, scale=alt.Scale(domain=["Category", "Other / unclear"],
                                                           range=[pal.blue, pal.neutral])),
    tooltip=[alt.Tooltip("primary_category:N", title="Category"),
             alt.Tooltip("n:Q", title="Reasons", format=","),
             alt.Tooltip("share:Q", title="Share", format=".1%")])
labels_ = alt.Chart(mix).mark_text(align="left", dx=4, fontSize=11, opacity=0.8, color=pal.ink).encode(
    x="share:Q", y=alt.Y("primary_category:N", sort=mix_order), text=alt.Text("share:Q", format=".0%"))
who = "stopped trials" if status == "All" else f"{status.lower()} trials"
style.chart((bars + labels_).properties(title=style.title(
    f"{lead.primary_category} is the most common reason, given for {style.pct(lead.share, 0)} of {who}",
    f"{int(mix.n.sum()):,} reasons; each reason counted once, under its primary category")), height=470)


# ---------------------------------------------------------------- heatmaps
def heatmap(df: pd.DataFrame, by: str, order: list[str], heading: str, sub: str) -> alt.Chart:
    df = complete(df, by)
    df = df[df[by].isin(order)]
    df["col"] = df[by] + "\n(n=" + df.group_n.map("{:,.0f}".format) + ")"
    col_order = [f"{g}\n(n={int(df[df[by] == g].group_n.iloc[0]):,})" for g in order if g in set(df[by])]
    base = alt.Chart(df).encode(
        x=alt.X("col:N", sort=col_order, title=None,
                axis=alt.Axis(labelAngle=0, orient="top", labelExpr="split(datum.label, '\\n')", labelLimit=110,
                              labelOverlap=False, labelPadding=6)),
        y=alt.Y("primary_category:N", sort=CATS, title=None, axis=alt.Axis(labelLimit=320)))
    rect = base.mark_rect(stroke=None, cornerRadius=2).encode(
        color=alt.Color("share:Q", title="Share within column", scale=alt.Scale(range=list(pal.sequential)),
                        legend=alt.Legend(format=".0%", orient="bottom", gradientLength=220)),
        tooltip=[alt.Tooltip(f"{by}:N", title="Group"), alt.Tooltip("primary_category:N", title="Category"),
                 alt.Tooltip("n:Q", title="Reasons", format=","),
                 alt.Tooltip("share:Q", title="Share of group", format=".1%")])
    text = base.transform_filter("datum.share >= 0.03").mark_text(fontSize=11).encode(
        text=alt.Text("share:Q", format=".0%"),
        color=alt.condition("datum.share > 0.2", alt.value(pal.ink_inverse), alt.value(pal.ink)))
    return (rect + text).properties(title=style.title(heading, sub))


phase = data.read_parquet("app/data/why_by_phase.parquet")
phase_order = ["Early phase 1", "Phase 1", "Phase 1/2", "Phase 2", "Phase 2/3", "Phase 3", "Phase 4", "No phase"]
p = complete(phase, "phase_label").set_index(["phase_label", "primary_category"]).share
eff = {g: p.get((g, "Lack of efficacy or futility"), 0) for g in phase_order}
style.chart(heatmap(
    phase, "phase_label", phase_order,
    f"Futility stops trials late: {style.pct(eff['Phase 3'], 0)} of Phase 3 reasons, "
    f"{style.pct(eff['Phase 1'], 0)} of Phase 1",
    "Share of each phase's reasons in each primary category; columns sum to 100%"), height=560)

sponsor = data.read_parquet("app/data/why_by_sponsor.parquet")
sp_order = ["Industry", "Academic / other", "NIH", "US federal", "Other government", "Network", "Individual"]
s = complete(sponsor, "sponsor_label").set_index(["sponsor_label", "primary_category"]).share
biz_ind = s.get(("Industry", "Sponsor, investigator or business decision"), 0)
fund_ac = s.get(("Academic / other", "Funding"), 0)
fund_ind = s.get(("Industry", "Funding"), 0)
style.chart(heatmap(
    sponsor[sponsor.group_n >= 100], "sponsor_label", sp_order,
    f"Industry trials stop for business reasons ({style.pct(biz_ind, 0)}), academic ones more often for "
    f"money ({style.pct(fund_ac, 0)} vs {style.pct(fund_ind, 0)})",
    "Share of each lead sponsor type's reasons in each primary category; groups with at least 100 reasons"),
    height=560)

# ---------------------------------------------------------------- trend
years = data.read_parquet("app/data/why_by_year.parquet")
basis = st.radio("Year", ["Year stopped (terminated trials)", "Start year (all stopped trials)"], horizontal=True,
                 help="The year a terminated trial ended is close to when it stopped. Start years show the "
                      "cohort a trial belonged to; for withdrawn trials it is only the planned start.")
picked = st.multiselect("Categories to show", CATS,
                        default=["COVID-19", "Slow or low enrollment", "Funding"], max_selections=3)
yr = years[years.basis == basis].rename(columns={"year": "yr"})
yr = complete(yr, "yr")
yr = yr[yr.primary_category.isin(picked) & yr.yr.between(2010, 2025)]
cov = yr[(yr.primary_category == "COVID-19")]
peak = cov.loc[cov.share.idxmax()] if not cov.empty else None
colors = [pal.blue, pal.orange, pal.aqua]
heading = (f"COVID-19 appears abruptly: {style.pct(peak.share, 0)} of reasons in {int(peak.yr)}"
           if peak is not None and peak.share > 0.05 else "Share of reasons by year")
line = alt.Chart(yr).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=36, filled=True)).encode(
    x=alt.X("yr:O", title=basis.split(" (")[0], axis=alt.Axis(labelAngle=0)),
    y=alt.Y("share:Q", title="Share of reasons that year", axis=alt.Axis(format=".0%")),
    color=alt.Color("primary_category:N", title=None, sort=picked,
                    scale=alt.Scale(domain=picked, range=colors[:len(picked)])),
    tooltip=[alt.Tooltip("yr:O", title="Year"), alt.Tooltip("primary_category:N", title="Category"),
             alt.Tooltip("n:Q", title="Reasons", format=","), alt.Tooltip("group_n:Q", title="All reasons", format=","),
             alt.Tooltip("share:Q", title="Share", format=".1%")])
if picked:
    style.chart(line.properties(title=style.title(
        heading, f"Primary category share by {basis.split(' (')[0].lower()}, 2010-2025; {basis.split('(')[1][:-1]}")),
        height=430)

# ---------------------------------------------------------------- conditions and examples
st.subheader("Look inside a category")
cat = st.selectbox("Category", CATS, index=CATS.index("Lack of efficacy or futility"))
cond = data.read_parquet("app/data/why_top_conditions.parquet")
ex = data.read_parquet("app/data/why_examples.parquet")
l, r = st.columns([1, 1.4], gap="large")
with l:
    cc = cond[cond.primary_category == cat][["condition", "n", "share_of_category"]]
    st.markdown("**Most frequent conditions**")
    st.dataframe(cc.rename(columns={"condition": "Condition", "n": "Reasons",
                                    "share_of_category": "Share of category"}),
                 hide_index=True, width="stretch",
                 column_config={"Share of category": st.column_config.NumberColumn(format="percent")})
with r:
    st.markdown("**Example reasons (random)**")
    st.dataframe(ex[ex.primary_category == cat][["nct_id", "why_stopped"]].rename(
        columns={"nct_id": "Trial", "why_stopped": "Reason as written"}), hide_index=True, width="stretch")

with st.expander("How the categoriser works"):
    st.markdown(
        "Each category is a list of case-insensitive regular expressions. A reason can match several "
        "categories; its **primary** category is the first match in the order below, which puts specific "
        "root causes (safety, efficacy, COVID-19, regulators, money, supply) before the generic outcomes "
        "they often explain (slow enrollment, a sponsor decision). For safety, efficacy and regulatory "
        "patterns a match is ignored when a negation (not, no, nor, non, without, unrelated, neither) appears earlier "
        "in the same clause, because many sponsors write \"not due to safety concerns\". Reasons that match "
        "nothing, including bare statements such as \"study never started\", are *Other / unclear*.\n\n"
        "Scope: interventional trials that started in 2008 or later and are terminated, withdrawn or "
        "suspended with a reason given. Source: `src/analysis/why_stopped.py`.")
    rows = [{"Precedence": i + 1, "Category": c, "Patterns": "  ·  ".join(taxonomy[c])}
            for i, c in enumerate(precedence)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"Patterns": st.column_config.TextColumn(width="large")})
