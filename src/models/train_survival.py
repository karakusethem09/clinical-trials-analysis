"""Time to termination, using the recent trials the classifier has to drop.

The classifier needs five years of follow-up, so it stops at 2020 starts. A survival
model can also learn from trials that are still running: they count as "not
terminated so far" up to the day of the data pull.

Population : interventional trials started 2008 onward that actually began enrolling
             (not withdrawn, not waiting to recruit, start date not ESTIMATED), first
             registered no later than 21 days after their start: the same cohort rule
             as the classifier, without the five-year follow-up requirement.
Clock      : years from start_date.
Outcomes   : TERMINATED   -> event of interest at end_date (the date the trial stopped).
             COMPLETED    -> competing event at end_date. A completed trial can no
                             longer be terminated, so it is not "censored" in the
                             usual sense: treating it as censored and plotting
                             1 - Kaplan-Meier would overstate the chance of termination.
                             Curves therefore show the Aalen-Johansen cumulative
                             incidence of termination with completion as a competing
                             risk. The Cox model is a cause-specific hazard model,
                             where completion does end follow-up (censoring is the
                             right treatment for a cause-specific hazard).
             RECRUITING, ACTIVE_NOT_RECRUITING, ENROLLING_BY_INVITATION, SUSPENDED
                          -> censored at the pull date (max last_update_date).
                             Suspension is a pause, not a stop.
             UNKNOWN      -> censored at status_verified_date, the last day the sponsor
                             confirmed the trial was still going. Some of these have
                             quietly stopped, so this censoring is probably
                             informative and termination risk is slightly understated.
             WITHDRAWN and NOT_YET_RECRUITING -> excluded. They never enrolled
                             anyone, their start_date is only the planned date, and
                             a "time to stopping" of ~0 says nothing about how long
                             running trials last. Withdrawal has its own analysis
                             (train_withdrawn.py).
Model      : penalised Cox proportional hazards on registration-time features.
             Hazard ratios come from a fit on every start year; the concordance
             index comes from a temporal hold-out (fit on starts <= 2017,
             evaluate on 2018-2020 starts), with the penalty chosen by fitting on
             starts <= 2014 and scoring 2015-2016. The hold-out model's risk score is
             also scored as a classifier of the primary label (terminated within 5
             years) on the classifier's test trials, for a like-for-like comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import AalenJohansenFitter, CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "reports" / "figures"
ONGOING = ("RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION", "SUSPENDED")
CENSORED, TERMINATED, COMPLETED = 0, 1, 2
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

PHASE_GROUPS = {"EARLY_PHASE1": "Phase 1", "PHASE1": "Phase 1", "PHASE1/PHASE2": "Phase 1/2",
                "PHASE2": "Phase 2", "PHASE2/PHASE3": "Phase 2/3", "PHASE3": "Phase 3",
                "PHASE4": "Phase 4", "NA": "No phase (NA)", "NONE": "No phase (NA)"}
SPONSOR_GROUPS = {"INDUSTRY": "Industry", "OTHER": "Academic / other", "NIH": "NIH",
                  "FED": "US federal", "OTHER_GOV": "Other government", "NETWORK": "Network"}
CATEGORICAL = {"phase_group": "No phase (NA)", "sponsor_group": "Academic / other",
               "allocation": "RANDOMIZED", "masking": "NONE", "purpose": "TREATMENT"}
FLAGS = ["healthy_volunteers", "has_industry_collaborator", "has_nih_collaborator",
         "intervention_drug", "intervention_device", "intervention_biological",
         "intervention_procedure", "intervention_behavioral", "intervention_radiation"]


def load() -> tuple[pd.DataFrame, pd.Timestamp]:
    con = duckdb.connect(str(ROOT / "data" / "trials.duckdb"), read_only=True)
    pull = pd.Timestamp(con.sql("select max(pull_date) from trial_cohort").fetchall()[0][0])
    df = con.sql("""
        select c.*, s.sponsor_prior_registered, s.sponsor_prior_terminated_rate
        from trial_cohort c join sponsor_history s using (nct_id)
        where started and registered_by_start
        order by nct_id
    """).df()
    for c in ("start_date", "end_date", "last_update_date", "status_verified_date"):
        df[c] = pd.to_datetime(df[c])
    df = df[df.start_date <= pull]

    status = df.overall_status
    stop = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    stop[status.isin(["TERMINATED", "COMPLETED"])] = df.end_date
    stop[status.isin(ONGOING)] = pull
    stop[status == "UNKNOWN"] = df.status_verified_date.fillna(df.last_update_date)
    df["event"] = np.select([status == "TERMINATED", status == "COMPLETED"],
                            [TERMINATED, COMPLETED], CENSORED)
    df["years"] = (stop.clip(upper=pull) - df.start_date).dt.days / 365.25
    # missing or impossible end dates are dropped; same-day stops get half a day
    df = df[df.years.notna() & (df.years >= 0)].copy()
    df["years"] = df.years.clip(lower=0.5 / 365.25)
    return df, pull


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Registration-time covariates, one-hot encoded against a fixed reference level."""
    purpose = df.primary_purpose.where(
        df.primary_purpose.isin(["TREATMENT", "PREVENTION", "SUPPORTIVE_CARE", "BASIC_SCIENCE",
                                 "DIAGNOSTIC"]), "OTHER")
    raw = pd.DataFrame({
        "phase_group": df.phase.map(PHASE_GROUPS),
        "sponsor_group": df.lead_sponsor_class.map(SPONSOR_GROUPS).fillna("Individual / unknown"),
        "allocation": df.allocation.where(df.allocation != "NA", "MISSING"),
        "masking": df.masking, "purpose": purpose,
    }, index=df.index)
    X = pd.concat([pd.get_dummies(raw[c], prefix=c, dtype=float).drop(columns=f"{c}_{ref}")
                   for c, ref in CATEGORICAL.items()], axis=1)
    for f in FLAGS:
        X[f] = df[f].fillna(False).astype(float)
    X["log_sponsor_prior_trials"] = np.log1p(df.sponsor_prior_registered.astype(float))
    X["sponsor_prior_terminated_rate_missing"] = df.sponsor_prior_terminated_rate.isna().astype(float)
    # an unknown track record is set to the typical rate; the indicator carries the difference
    X["sponsor_prior_terminated_rate"] = df.sponsor_prior_terminated_rate.fillna(0.09)
    X["log_criteria_chars"] = np.log1p(df.criteria_chars.fillna(0))
    X["n_conditions"] = df.n_conditions.clip(upper=10)
    return X.loc[:, X.std() > 0]


def fit_cox(X: pd.DataFrame, df: pd.DataFrame, penalizer: float) -> CoxPHFitter:
    data = X.assign(years=df.years, terminated=(df.event == TERMINATED).astype(int))
    return CoxPHFitter(penalizer=penalizer).fit(data, "years", "terminated")


def c_index(model: CoxPHFitter, X: pd.DataFrame, df: pd.DataFrame) -> float:
    risk = model.predict_partial_hazard(X)
    return concordance_index(df.years, -risk, df.event == TERMINATED)


def plot_incidence(df: pd.DataFrame, group: str, order: list[str], title: str, path: Path) -> dict:
    """Cumulative incidence of termination (completion as competing risk), one line per group."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    at5 = {}
    for colour, name in zip(SERIES, order):
        g = df[df[group] == name]
        ajf = AalenJohansenFitter(calculate_variance=False, jitter_level=1e-5, seed=0)
        ajf.fit(g.years, g.event, event_of_interest=TERMINATED)
        cif = ajf.cumulative_density_.iloc[:, 0]
        cif = cif[cif.index <= 10]
        at5[name] = round(float(cif[cif.index <= 5].iloc[-1]), 4)
        ax.step(cif.index, cif.values, where="post", color=colour, lw=2,
                label=f"{name} (n={len(g):,})")
    ax.set(xlabel="Years since start", ylabel="Cumulative incidence of termination",
           title=title, xlim=(0, 10), ylim=(0, None))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.text(0.01, 0.01, "Aalen-Johansen estimate; completion treated as a competing event, "
             "running trials censored.", fontsize=7.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return at5


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df, pull = load()
    df["phase_group"] = df.phase.map(PHASE_GROUPS)
    df["sponsor_group"] = df.lead_sponsor_class.map(SPONSOR_GROUPS)

    results: dict = {
        "pull_date": str(pull.date()),
        "n": int(len(df)),
        "n_by_outcome": {k: int(v) for k, v in df.overall_status.value_counts().items()},
        "n_started_2021_or_later": int((df.start_year >= 2021).sum()),
    }

    # How much does ignoring the competing risk overstate termination?
    kmf = KaplanMeierFitter().fit(df.years, df.event == TERMINATED)
    ajf = AalenJohansenFitter(calculate_variance=False, jitter_level=1e-5, seed=0).fit(
        df.years, df.event, event_of_interest=TERMINATED)
    cif = ajf.cumulative_density_.iloc[:, 0]
    results["terminated_by_5_years"] = {
        "one_minus_kaplan_meier": round(float(1 - kmf.predict(5.0)), 4),
        "aalen_johansen": round(float(cif[cif.index <= 5].iloc[-1]), 4),
    }

    phases = ["Phase 1", "Phase 1/2", "Phase 2", "Phase 2/3", "Phase 3", "Phase 4", "No phase (NA)"]
    sponsors = ["Industry", "Academic / other", "NIH", "US federal", "Other government", "Network"]
    results["terminated_by_5_years_by_phase"] = plot_incidence(
        df, "phase_group", phases, "Termination by phase", FIG / "survival_by_phase.png")
    results["terminated_by_5_years_by_sponsor_class"] = plot_incidence(
        df, "sponsor_group", sponsors, "Termination by lead sponsor class",
        FIG / "survival_by_sponsor_class.png")

    X = features(df)

    # Penalty chosen on a temporal split inside the training years.
    fit_part, tune_part = df.start_year <= 2014, df.start_year.between(2015, 2016)
    tuning = {}
    for pen in (0.001, 0.01, 0.1):
        m = fit_cox(X[fit_part], df[fit_part], pen)
        tuning[pen] = round(c_index(m, X[tune_part], df[tune_part]), 4)
    penalizer = max(tuning, key=tuning.get)
    results["penalizer_tuning_c_index_2015_2016"] = {str(k): v for k, v in tuning.items()}
    results["penalizer"] = penalizer

    train, test = df.start_year <= 2017, df.start_year.between(2018, 2020)
    holdout = fit_cox(X[train], df[train], penalizer)
    results["c_index"] = {
        "train_2008_2017": round(c_index(holdout, X[train], df[train]), 4),
        "test_2018_2020": round(c_index(holdout, X[test], df[test]), 4),
    }
    results["n_holdout"] = {"train": int(train.sum()), "test": int(test.sum()),
                            "test_terminated": int((df[test].event == TERMINATED).sum())}
    labelled = test & df.terminated_5y.notna()
    risk = holdout.predict_partial_hazard(X[labelled])
    results["holdout_risk_as_classifier_of_terminated_5y"] = {
        "roc_auc": round(roc_auc_score(df.terminated_5y[labelled], risk), 4),
        "n": int(labelled.sum())}

    full = fit_cox(X, df, penalizer)
    hr = full.summary[["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
    hr.columns = ["coef", "hazard_ratio", "hr_lower_95", "hr_upper_95", "p"]
    hr = hr.sort_values("hazard_ratio", ascending=False)
    hr.round(4).to_csv(ROOT / "reports" / "survival_hazard_ratios.csv", index_label="feature")
    results["hazard_ratios_full_cohort"] = hr.hazard_ratio.round(3).to_dict()

    # Proportional-hazards check on a sample (Schoenfeld residual test, rank time).
    sample = df.sample(20_000, random_state=0).index
    ph = proportional_hazard_test(fit_cox(X.loc[sample], df.loc[sample], penalizer),
                                  X.loc[sample].assign(years=df.years[sample],
                                                       terminated=(df.event[sample] == TERMINATED).astype(int)),
                                  time_transform="rank")
    results["ph_violations_p_below_0_01"] = sorted(ph.summary.index[ph.summary.p < 0.01].tolist())

    (ROOT / "reports" / "survival_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
