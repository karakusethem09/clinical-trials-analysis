"""Tabular models for termination within five years, plus a leakage ladder.

Task   : among interventional trials that actually started (2008-2020), registered no
         later than 21 days after their start, predict whether the trial is TERMINATED
         within 5 years of its start date. Every start year used has a full 5 years of
         follow-up before the data pull, so "not terminated" is known, not assumed.
         Withdrawn trials are a separate question (train_withdrawn.py); UNKNOWN status
         trials are excluded (sensitivity.py tries the alternatives).
Split  : by start year. Train 2008-2014, early stopping 2015-2016, calibration 2017,
         test 2018-2020.
Models : logistic regression baseline and gradient boosting on registration-time
         features, without the start year (it extrapolates under a temporal split) or
         the registration delay (it uses the actual start date, which is set after
         registration).

The ladder reproduces two earlier, flawed set-ups so the effect of each fix is visible:
  naive        : old label (terminated or withdrawn vs completed, resolved trials only,
                 2008-2019), old features plus fields the registry overwrites after a
                 trial ends (enrollment, sites, countries).
  partly_fixed : same old label and cohort, old "honest" features, which still include
                 the registration delay and start year and the unlagged sponsor history.
  defensible   : the primary task above.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "trials.duckdb"
SPLITS = {"train": (2008, 2014), "valid": (2015, 2016), "calib": (2017, 2017), "test": (2018, 2020)}

CATEGORICAL = ["phase", "allocation", "masking", "primary_purpose", "intervention_model",
               "lead_sponsor_class", "sex"]
NUMERIC = ["n_conditions", "n_intervention_types", "min_age_years", "max_age_years",
           "criteria_chars", "criteria_bullets", "sponsor_prior_registered", "sponsor_prior_ended",
           "sponsor_prior_terminated_rate", "sponsor_prior_withdrawn_rate"]
BOOLEAN = ["has_collaborator", "has_industry_collaborator", "has_nih_collaborator",
           "healthy_volunteers", "intervention_drug", "intervention_device",
           "intervention_biological", "intervention_procedure", "intervention_behavioral",
           "intervention_radiation", "intervention_dietary_supplement", "intervention_genetic",
           "intervention_diagnostic_test", "intervention_combination_product", "intervention_other"]
FEATURES = CATEGORICAL + NUMERIC + BOOLEAN
LEAKY = ["leaky_enrollment_count", "leaky_n_locations", "leaky_n_countries"]

# the earlier set-up, kept only for the ladder
LEGACY_SPLITS = {"train": (2008, 2015), "valid": (2016, 2017), "test": (2018, 2019)}
LEGACY_NUMERIC = ["start_year", "days_registered_after_start", "n_conditions", "n_intervention_types",
                  "min_age_years", "max_age_years", "criteria_chars", "criteria_bullets",
                  "sponsor_prior_started", "sponsor_prior_ended_v1", "sponsor_prior_stop_rate"]
LEGACY_FEATURES = CATEGORICAL + LEGACY_NUMERIC + BOOLEAN

LABELS = ("terminated_5y", "terminated_5y_unknown_as_0", "terminated_5y_unknown_verified")


def typed(df: pd.DataFrame) -> pd.DataFrame:
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    for c in BOOLEAN:
        df[c] = df[c].astype("float")
    return df


def query_cohort(where: str, label: str) -> pd.DataFrame:
    con = duckdb.connect(str(DB), read_only=True)
    return con.sql(f"""
        select c.*, {label} as y,
               s.sponsor_prior_registered, s.sponsor_prior_ended,
               s.sponsor_prior_terminated_rate, s.sponsor_prior_withdrawn_rate,
               v.sponsor_prior_started, v.sponsor_prior_ended as sponsor_prior_ended_v1,
               v.sponsor_prior_stop_rate
        from trial_cohort c
        join sponsor_history s using (nct_id)
        join sponsor_history_v1 v using (nct_id)
        where {where}
        order by nct_id  -- fixed row order: DuckDB's is not, and bagging depends on it
    """).df()


RUNNING = "('RECRUITING', 'ACTIVE_NOT_RECRUITING', 'ENROLLING_BY_INVITATION', 'SUSPENDED')"
# still running at the pull but not verified by the sponsor in the 12 months before it
STALE_RUNNING = (f"overall_status in {RUNNING} and "
                 "coalesce(status_verified_date, date '1900-01-01') < pull_date - interval 12 month")


def load(label: str = "terminated_5y", registration_filter: str | None = "registered_by_start",
         drop_stale_running: bool = False) -> pd.DataFrame:
    """Primary cohort with label column `y`.

    `registration_filter` is the boolean column that defines prospective registration:
    registered_by_start (current), registered_prospectively (legacy) or None (no filter).
    `drop_stale_running` removes trials labelled 0 only because they are still running
    when their status has not been verified in the year before the pull."""
    assert label in LABELS and registration_filter in ("registered_by_start", "registered_prospectively", None)
    where = f"{label} is not null and start_year between 2008 and {SPLITS['test'][1]}"
    if registration_filter:
        where += f" and {registration_filter}"
    if drop_stale_running:
        where += f" and not ({STALE_RUNNING})"
    return typed(query_cohort(where, label))


def load_legacy() -> pd.DataFrame:
    """The earlier cohort: resolved trials 2008-2019, old 21-day filter, label = terminated or withdrawn."""
    where = (f"is_resolved and registered_prospectively and start_year between "
             f"{LEGACY_SPLITS['train'][0]} and {LEGACY_SPLITS['test'][1]}")
    return typed(query_cohort(where, "stopped_early"))


def part(df: pd.DataFrame, name: str, splits: dict = SPLITS) -> pd.DataFrame:
    return df[df.start_year.between(*splits[name])]


def top_decile_precision(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p)
    k = max(1, int(round(0.1 * len(y))))
    return float(y[np.argsort(-p, kind="stable")[:k]].mean())


def scores(y, p) -> dict:
    base = float(np.mean(y))
    prec = top_decile_precision(y, p)
    return {"roc_auc": round(roc_auc_score(y, p), 4),
            "pr_auc": round(average_precision_score(y, p), 4),
            "brier": round(brier_score_loss(y, p), 4),
            "top_decile_precision": round(prec, 4),
            "top_decile_lift": round(prec / base, 3),
            "base_rate": round(base, 4),
            "n": int(len(y))}


BOOT_METRICS = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score,
                "top_decile_precision": top_decile_precision}


def bootstrap(y, preds: dict[str, np.ndarray], n_boot: int = 1000, seed: int = 0,
              pairs: tuple[tuple[str, str], ...] = (), groups: np.ndarray | None = None) -> dict:
    """Percentile 95% intervals from resampling trials; every model sees the same resamples,
    so `pairs` (a, b) also get an interval for the paired difference a - b.

    With `groups` (e.g. the normalised sponsor of each trial) whole clusters are resampled
    instead of single trials, which allows for trials of one sponsor being alike."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    clusters = None
    if groups is not None:
        codes = pd.factorize(np.asarray(groups))[0]
        order = np.argsort(codes, kind="stable")
        clusters = np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)
    draws = {name: {m: np.empty(n_boot) for m in BOOT_METRICS} for name in preds}
    for b in range(n_boot):
        if clusters is None:
            i = rng.integers(0, len(y), len(y))
        else:
            i = np.concatenate([clusters[c] for c in rng.integers(0, len(clusters), len(clusters))])
        for name, p in preds.items():
            for m, f in BOOT_METRICS.items():
                draws[name][m][b] = f(y[i], np.asarray(p)[i])

    def ci(point: float, samples: np.ndarray) -> dict:
        lo, hi = np.percentile(samples, [2.5, 97.5])
        return {"estimate": round(float(point), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}

    out: dict = {}
    for name, p in preds.items():
        out[name] = {m: ci(f(y, np.asarray(p)), draws[name][m]) for m, f in BOOT_METRICS.items()}
        out[name]["top_decile_lift"] = round(out[name]["top_decile_precision"]["estimate"] / y.mean(), 3)
    for a, b in pairs:
        out[f"difference_{a}_minus_{b}"] = {
            m: ci(out[a][m]["estimate"] - out[b][m]["estimate"], draws[a][m] - draws[b][m])
            for m in BOOT_METRICS}
    return out


def fit_logit(train: pd.DataFrame, features: list[str] = FEATURES) -> Pipeline:
    cat = [c for c in features if c in CATEGORICAL]
    num = [c for c in features if c not in CATEGORICAL]
    model = make_pipeline(
        ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat),
            ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                  StandardScaler()), num),
        ]),
        LogisticRegression(max_iter=2000, C=0.5),
    )
    return model.fit(train[features], train.y)


def gbm_params() -> dict:
    return dict(learning_rate=0.03, num_leaves=63, min_child_samples=50, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)


def fit_gbm(train: pd.DataFrame, valid: pd.DataFrame, features: list[str]) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(n_estimators=3000, **gbm_params())
    model.fit(train[features], train.y,
              eval_set=[(valid[features], valid.y)], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


def ladder() -> dict[str, dict]:
    """Test-set labels and predictions for each rung, on its own test set and on the primary one."""
    legacy, primary = load_legacy(), load()
    tr, va, te = (part(legacy, s, LEGACY_SPLITS) for s in ("train", "valid", "test"))
    p_test = part(primary, "test")
    rungs = {}
    for name, feats in (("naive", LEGACY_FEATURES + LEAKY), ("partly_fixed", LEGACY_FEATURES)):
        m = fit_gbm(tr, va, feats)
        rungs[name] = {"y": te.y.to_numpy(), "p": m.predict_proba(te[feats])[:, 1],
                       "y_primary": p_test.y.to_numpy(),
                       "p_primary": m.predict_proba(p_test[feats])[:, 1]}
    m = fit_gbm(part(primary, "train"), part(primary, "valid"), FEATURES)
    p = m.predict_proba(p_test[FEATURES])[:, 1]
    rungs["defensible"] = {"y": p_test.y.to_numpy(), "p": p, "y_primary": p_test.y.to_numpy(), "p_primary": p}
    return rungs


def main() -> None:
    df = load()
    train, valid, calib, test = (part(df, s) for s in SPLITS)
    results: dict = {"label": "terminated within 5 years of start",
                     "n": {s: int(len(part(df, s))) for s in SPLITS},
                     "base_rate": {s: round(float(part(df, s).y.mean()), 4) for s in SPLITS}}

    results["logistic_regression"] = scores(test.y, fit_logit(train).predict_proba(test[FEATURES])[:, 1])
    gbm = fit_gbm(train, valid, FEATURES)
    p_test = gbm.predict_proba(test[FEATURES])[:, 1]
    results["gradient_boosting"] = scores(test.y, p_test)
    results["gradient_boosting_best_iteration"] = int(gbm.best_iteration_)

    results["ladder"] = {
        name: {"own_test_set": scores(r["y"], r["p"]),
               "primary_test_set": scores(r["y_primary"], r["p_primary"])}
        for name, r in ladder().items()}

    bins = pd.qcut(p_test, 10, duplicates="drop")
    cal = (pd.DataFrame({"p": p_test, "y": test.y.values, "bin": bins})
           .groupby("bin", observed=True).agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")))
    results["calibration_deciles_raw"] = cal.round(4).reset_index(drop=True).to_dict("records")

    imp = pd.Series(gbm.booster_.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    results["top_features_by_gain"] = (imp / imp.sum()).round(4).head(12).to_dict()

    (ROOT / "reports" / "tabular_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
