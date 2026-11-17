"""Rule-based categories for the free-text `why_stopped` field.

Scope      : interventional trials in the analysis cohort (start 2008 onwards,
             dbt model trial_cohort) whose status is TERMINATED, WITHDRAWN or
             SUSPENDED and that give a reason. No registration-timing filter is
             applied: this describes stopped trials, it does not feed a model.
Method     : every category is a list of case-insensitive regular expressions
             (TAXONOMY below). A reason can match several categories (multi-label).
             Its primary category is the first match in PRECEDENCE, which puts
             specific root causes (safety, efficacy, COVID-19, regulators, drug
             supply, money) before the generic ones they often explain (slow
             enrollment, "sponsor decision"). Reasons that match nothing are
             "Other / unclear".
Negation   : sponsors often write "not due to safety concerns". For the categories in
             NEGATABLE, a match is ignored when a negation word (not, no, nor,
             without, unrelated, ...) appears earlier in the same clause within
             80 characters.
Validation : a stratified random sample of 300 reasons (by primary category) is
             written to reports/why_stopped_label_sample.csv with an empty
             `human_label` column for hand-checking. No accuracy figure is claimed
             until that sample has been labelled.

Run        : .venv/bin/python src/analysis/why_stopped.py
             (also run by src/analysis/build_app_data.py)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "trials.duckdb"
APP_DATA = ROOT / "app" / "data"
REPORTS = ROOT / "reports"
OTHER = "Other / unclear"
SAMPLE_SIZE = 300

# category -> patterns. Patterns are matched with re.IGNORECASE on the raw reason.
TAXONOMY: dict[str, list[str]] = {
    "Safety or adverse events": [
        r"\bsafety\b", r"\badverse (event|effect|reaction|drug)", r"\btoxicit", r"\btoxic\b",
        r"\bside[- ]effect", r"\b(serious|unexpected|unacceptable) (adverse|events?|risk)",
        r"\bSAEs?\b", r"\bdeaths?\b", r"\bmortality\b", r"\bharm(ful)?\b",
        r"\bdose[- ]limiting\b", r"\bDLTs?\b", r"\bclinical hold\b", r"\btolerab", r"\bintolerab",
        r"\bcomplications?\b",
    ],
    "Lack of efficacy or futility": [
        r"\bfutil", r"\bineffective",
        r"\b(lack|absence|loss) of (sufficient |adequate |clinical |demonstrated )?efficacy",
        r"\b(insufficient|limited|low|poor|inadequate|no|minimal|modest) (clinical )?efficacy",
        r"\befficacy (was |were |is )?(not|insufficient|lower|below|inadequate|lacking|concerns?|reasons?|data)", r"\bnot effective",
        r"\black of (effect|benefit|response|activity)", r"\b(no|insufficient|limited) (clinical )?(benefit|response|activity)",
        r"\b(did not|failed to|unlikely to) (show|demonstrate|meet|reach|achieve)\b",
        r"\b(primary )?end ?point (was )?not (met|reached)", r"\binterim (analysis|results?|review|data)",
        r"\bnegative (results?|data|outcome)", r"\bstopping (rule|boundary|criteria)",
        r"\bpoor (clinical )?(outcome|results?|response)", r"\bnot promising", r"\b(treatment|clinical) response",
        r"\bnot (statistically )?significant", r"\bno significant", r"\binconclusive",
    ],
    "COVID-19": [
        r"\bcovid", r"\bsars[- ]?cov", r"\bcorona ?virus",
        # "pandemic" alone, unless the reason is about the 2009 influenza pandemic
        r"(?s)^(?!.*\b(influenza|h1n1|flu|swine)\b).*\bpandemi",
    ],
    "Regulatory or ethics": [
        r"\bregulat", r"\bIRBs?\b", r"\bethic",
        r"\b(FDA|EMA|MHRA)\b.{0,15}\b(request|requir|hold|letter|feedback|decision|review|guidance|concern|did not|denied)",
        r"\b(request|requir|instruct)\w* (by|of|from) (the )?(FDA|EMA|MHRA)\b",
        r"\bhealth canada\b", r"\bcompetent authorit", r"\bauthorit(y|ies)\b",
        r"\b(not|never|wasn'?t) (been )?approved\b", r"\b(lack|absence) of (an? )?(ethical |regulatory |irb )?approval",
        r"\bapproval (was )?(not|never) (obtained|granted|received)", r"\bdid not (get|obtain|receive) (the )?approval",
        r"\bpermit\b", r"\blicen[cs]e", r"\bGCP\b", r"\bcompliance with (the )?regulation", r"\binspection",
    ],
    "Drug or device supply": [
        r"\b(drug|product|device|medication|vaccine|study agent|investigational product|imp|supply|supplies) "
        r"(was |is |were )?(not |no longer |un)?(available|availability|supplied|provided|shortage)",
        r"\bshortage", r"\bsupply\b", r"\bsupplies\b", r"\bmanufactur", r"\bexpir(ed|y|ation)",
        r"\b(drug|product|device|medication|vaccine|agent|imp)s? (was |were )?unavailab", r"\bstock\b", r"\bstability\b",
    ],
    "Funding": [
        r"\bfund", r"\bgrant\b", r"\bfinanc", r"\bbudget", r"\bmoney\b", r"\bcosts?\b", r"\bcostly\b",
        r"\bexpens", r"\bresources?\b",
    ],
    "Investigator, staff or site logistics": [
        r"\b(principal )?investigators?\b.{0,40}\b(left|leav|retir|mov|relocat|died|passed|deceased|"
        r"departure|depart|change|illness|sick|maternity|no longer)",
        r"\bPI\b", r"\bP\.I\.", r"\bstaff", r"\bpersonnel\b", r"\bcoordinator", r"\blogistic",
        r"\badministrative", r"\binfrastructure", r"\bsite (closure|closed|issues?|problems?|unable)",
        r"\btechnical (issue|problem|difficult)", r"\bequipment\b", r"\bfacilit(y|ies)\b",
        r"\bfeasib", r"\bworkload\b", r"\bcapacity\b", r"\bpassed away\b",
        r"\b(researchers?|study team|investigators?|team)\b.{0,20}\b(left|leaving|moved)\b",
        r"\binvestigator\b.{0,20}\bnot available", r"\bcontract", r"\bhuman ress?ources",
    ],
    "Design changed or superseded": [
        r"\bprotocol (change|amendment|revision|redesign)", r"\b(re)?design", r"\bamend",
        r"\breplaced\b", r"\bsuperseded", r"\bnew (study|protocol|trial|version)", r"\banother (study|protocol|trial)",
        r"\bcompeting (study|studies|trial|trials)", r"\b(landscape|standard of care)", r"\bno longer (needed|necessary|relevant|of interest|required)",
        r"\bother (studies|trials)\b", r"\bapproved (by|in|for)\b", r"\b(market|marketing) (approval|authori[sz]ation)",
        r"\bmethodolog", r"\bscientific (reasons?|rationale|question)", r"\bchanges? in (the )?(study|trial|treatment|clinical) ",
        r"\brevis", r"\bmerged\b", r"\bcombined (with|into)\b", r"\bout of date\b", r"\bchange in (clinical )?practice",
        r"\bnewly published", r"\b(results|data) (were |was )?(obtained|available) (in|from) (another|other)",
    ],
    "Slow or low enrollment": [
        r"\benrol", r"\baccru", r"\brecruit", r"\binclusions?\b", r"\binclud(e|ing) (patients|subjects|participants)",
        r"\b(no|few|insufficient|not enough|lack of|low number of|limited) (eligible |suitable )?"
        r"(patients|subjects|participants|volunteers|candidates|cases)",
        r"\bparticipants? (were|was) (not )?(enrolled|randomi[sz]ed|identified)", r"\beligible (patients|subjects|participants)",
        r"\bsample size\b", r"\bscreen ?fail", r"\bpatient (availability|population)", r"\bretention\b",
        r"\bloss to follow", r"\blost to follow", r"\bdrop ?outs?\b", r"\badherence\b",
        r"\b(finding|find|identify|identifying) (enough |eligible |suitable )?(patients|subjects|participants)",
        r"\beligibility rate", r"\bincidence\b", r"\bparticipation\b", r"\bnot enough patients?\b", r"\brecrut",
    ],
    "Sponsor, investigator or business decision": [
        r"\bbusiness", r"\bsponsor('s|s)?\b.{0,25}\b(decision|decided|request|terminat|withdr|discontinu|choice|stop)",
        r"\bcompany\b", r"\bcorporate", r"\bcommercial", r"\bstrateg", r"\bportfolio", r"\bprioriti",
        r"\bdevelopment (program|programme|plan|decision|project)", r"\bacqui(red|sition)", r"\bmerger\b",
        r"\bbankrupt", r"\binsolven", r"\binternal (decision|reasons?|review)", r"\bsponsor decision",
        r"\bdecision (of|by) the sponsor", r"\bat the (request|discretion) of the sponsor", r"\bnot pursued\b",
        r"\bby (the )?sponsor\b", r"\bsponsor'?s? discretion", r"^\W*sponsor\W*$", r"\bmanagement decision",
        r"\b(investigator|institution|institutional|organi[sz]ational|team)('s)? (decision|choice|reasons?)",
        r"\bdecision of the (investigator|institution)", r"\bwind(ing)? down", r"\bchapter 11\b",
        r"\b(stop|discontinue|terminate|end)\w* (the )?(clinical )?development\b",
    ],
}

PRECEDENCE: list[str] = [
    "Safety or adverse events",
    "Lack of efficacy or futility",
    "COVID-19",
    "Regulatory or ethics",
    "Funding",
    "Drug or device supply",
    "Investigator, staff or site logistics",
    "Design changed or superseded",
    "Slow or low enrollment",
    "Sponsor, investigator or business decision",
]
CATEGORIES: list[str] = PRECEDENCE + [OTHER]
NEGATABLE = {"Safety or adverse events", "Lack of efficacy or futility", "Regulatory or ethics"}
NEGATION = re.compile(r"\b(not|no|nor|non|without|unrelated|neither)\b", re.IGNORECASE)
CLAUSE_BREAK = re.compile(r"[.;:!?]")
COMPILED = {c: [re.compile(p, re.IGNORECASE) for p in pats] for c, pats in TAXONOMY.items()}


def _negated(text: str, start: int, window: int = 80) -> bool:
    """True when a negation word precedes position `start` in the same clause."""
    before = text[max(0, start - window):start]
    before = CLAUSE_BREAK.split(before)[-1]
    return bool(NEGATION.search(before))


def categorise(reason: str | None) -> list[str]:
    """All categories a reason matches, in precedence order; [OTHER] when none."""
    if not reason or not reason.strip():
        return [OTHER]
    hits = []
    for cat in PRECEDENCE:
        for pat in COMPILED[cat]:
            if any(not (cat in NEGATABLE and _negated(reason, m.start())) for m in pat.finditer(reason)):
                hits.append(cat)
                break
    return hits or [OTHER]


PHASE_LABELS = {"EARLY_PHASE1": "Early phase 1", "PHASE1": "Phase 1", "PHASE1/PHASE2": "Phase 1/2",
                "PHASE2": "Phase 2", "PHASE2/PHASE3": "Phase 2/3", "PHASE3": "Phase 3",
                "PHASE4": "Phase 4", "NA": "No phase", "NONE": "No phase"}
SPONSOR_LABELS = {"INDUSTRY": "Industry", "OTHER": "Academic / other", "NIH": "NIH",
                  "FED": "US federal", "OTHER_GOV": "Other government", "NETWORK": "Network",
                  "INDIV": "Individual", "UNKNOWN": "Unknown", "MISSING": "Unknown"}


def load() -> pd.DataFrame:
    import duckdb  # only needed to rebuild; the dashboard imports TAXONOMY without it

    con = duckdb.connect(str(DB), read_only=True)
    df = con.sql("""
        select nct_id, overall_status, why_stopped, start_year, year(end_date) as stop_year,
               phase, lead_sponsor_class, conditions
        from trial_cohort
        where overall_status in ('TERMINATED', 'WITHDRAWN', 'SUSPENDED')
          and why_stopped is not null and trim(why_stopped) <> ''
        order by nct_id
    """).df()
    df["phase_label"] = df.phase.map(PHASE_LABELS).fillna("No phase")
    df["sponsor_label"] = df.lead_sponsor_class.map(SPONSOR_LABELS).fillna("Unknown")
    labels = [categorise(r) for r in df.why_stopped]
    df["categories"] = labels
    df["primary_category"] = [c[0] for c in labels]
    df["n_categories"] = [0 if c == [OTHER] else len(c) for c in labels]
    return df


def shares(df: pd.DataFrame, by: str | None) -> pd.DataFrame:
    """Primary-category share within each level of `by` (or overall)."""
    keys = [by] if by else []
    counts = df.groupby(keys + ["primary_category"]).size().rename("n").reset_index()
    totals = counts.groupby(keys).n.transform("sum") if keys else counts.n.sum()
    counts["share"] = counts.n / totals
    counts["group_n"] = totals
    return counts


def any_mention(df: pd.DataFrame) -> pd.DataFrame:
    """Share of reasons that mention each category at all (multi-label)."""
    rows = [{"category": c, "n": int(df.categories.map(lambda x, c=c: c in x).sum())} for c in CATEGORIES]
    out = pd.DataFrame(rows)
    out["share"] = out.n / len(df)
    return out


def top_conditions(df: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    ex = df[["primary_category", "conditions"]].explode("conditions").dropna()
    ex["condition"] = (ex.conditions.str.strip().str.lower()
                       .str.replace(r"\bcovid[\s-]?19\b|\bcovid\b", "covid-19", regex=True))
    ex = ex[ex.condition != ""]
    counts = ex.groupby(["primary_category", "condition"]).size().rename("n").reset_index()
    cat_n = df.primary_category.value_counts()
    counts["share_of_category"] = counts.n / counts.primary_category.map(cat_n)
    counts = counts.sort_values(["primary_category", "n", "condition"], ascending=[True, False, True])
    return counts.groupby("primary_category").head(k).reset_index(drop=True)


def label_sample(df: pd.DataFrame, n: int = SAMPLE_SIZE, seed: int = 0) -> pd.DataFrame:
    """Stratified by primary category: proportional allocation with at least 10 per category."""
    rng = np.random.default_rng(seed)
    counts = df.primary_category.value_counts()
    alloc = np.maximum(10, np.floor(counts / counts.sum() * n)).astype(int).clip(upper=counts)
    while alloc.sum() > n:  # trim the largest strata until the total fits
        alloc[alloc.idxmax()] -= 1
    while alloc.sum() < n:
        room = (counts - alloc)[lambda s: s > 0]
        alloc[room.idxmax()] += 1
    parts = []
    for cat, k in alloc.items():
        idx = df.index[df.primary_category == cat]
        parts.append(df.loc[rng.choice(idx, size=int(k), replace=False)])
    out = pd.concat(parts).sample(frac=1, random_state=seed)
    return pd.DataFrame({
        "nct_id": out.nct_id, "overall_status": out.overall_status, "why_stopped": out.why_stopped,
        "primary_category": out.primary_category,
        "all_categories": out.categories.map(" | ".join), "human_label": "",
    })


def main() -> dict:
    """Write the aggregates to app/data/, the summary and the label sample to reports/."""
    APP_DATA.mkdir(parents=True, exist_ok=True)
    df = load()

    overall = shares(df, None)
    overall["status"] = "All"
    by_status = shares(df, "overall_status").rename(columns={"overall_status": "status"})
    pd.concat([overall, by_status]).to_parquet(APP_DATA / "why_overall.parquet", index=False)

    term = df[df.overall_status == "TERMINATED"]
    shares(df, "phase_label").to_parquet(APP_DATA / "why_by_phase.parquet", index=False)
    shares(df, "sponsor_label").to_parquet(APP_DATA / "why_by_sponsor.parquet", index=False)
    start = shares(df[df.start_year.between(2008, 2025)], "start_year").assign(basis="Start year (all stopped trials)")
    stop = (shares(term[term.stop_year.between(2008, 2025)], "stop_year").rename(columns={"stop_year": "year"})
            .assign(basis="Year stopped (terminated trials)"))
    pd.concat([start.rename(columns={"start_year": "year"}), stop]).to_parquet(
        APP_DATA / "why_by_year.parquet", index=False)
    any_mention(df).to_parquet(APP_DATA / "why_any_mention.parquet", index=False)
    top_conditions(df).to_parquet(APP_DATA / "why_top_conditions.parquet", index=False)

    examples = (df.groupby("primary_category", group_keys=False)
                [["primary_category", "nct_id", "why_stopped"]]
                .apply(lambda g: g.sample(min(6, len(g)), random_state=2)))
    examples["why_stopped"] = examples.why_stopped.str.slice(0, 220)
    examples.to_parquet(APP_DATA / "why_examples.parquet", index=False)

    label_sample(df).to_csv(REPORTS / "why_stopped_label_sample.csv", index=False)

    primary = df.primary_category.value_counts(normalize=True)
    summary = {
        "scope": "trial_cohort, status TERMINATED/WITHDRAWN/SUSPENDED with a why_stopped reason",
        "n_reasons": int(len(df)),
        "n_by_status": {k: int(v) for k, v in df.overall_status.value_counts().items()},
        "primary_category_share": {c: round(float(primary.get(c, 0.0)), 4) for c in CATEGORIES},
        "other_unclear_share": round(float(primary.get(OTHER, 0.0)), 4),
        "multi_label_share": round(float((df.n_categories > 1).mean()), 4),
        "precedence": PRECEDENCE,
        "accuracy": "not measured; see reports/why_stopped_label_sample.csv (human_label is empty)",
    }
    (REPORTS / "why_stopped_summary.json").write_text(json.dumps(summary, indent=2))
    (APP_DATA / "why_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
