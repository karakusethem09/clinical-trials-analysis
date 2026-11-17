"""Tabular models for early stopping, with a deliberate leakage comparison.

Cohort : interventional trials started 2008-2019, registered no later than 21 days
         after their start date, whose outcome is known.
Split  : by start year, so the model is always tested on later trials.
Models : logistic regression baseline, gradient boosting on registration-time
         features, and a "naive" gradient boosting that also sees fields the
         registry overwrites after a trial ends.
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
TRAIN_YEARS, VALID_YEARS, TEST_YEARS = (2008, 2015), (2016, 2017), (2018, 2019)

CATEGORICAL = ["phase", "allocation", "masking", "primary_purpose", "intervention_model",
               "lead_sponsor_class", "sex"]
NUMERIC = ["start_year", "days_registered_after_start", "n_conditions", "n_intervention_types",
           "min_age_years", "max_age_years", "criteria_chars", "criteria_bullets",
           "sponsor_prior_started", "sponsor_prior_ended", "sponsor_prior_stop_rate"]
BOOLEAN = ["has_collaborator", "has_industry_collaborator", "has_nih_collaborator",
           "healthy_volunteers", "intervention_drug", "intervention_device",
           "intervention_biological", "intervention_procedure", "intervention_behavioral",
           "intervention_radiation", "intervention_dietary_supplement", "intervention_genetic",
           "intervention_diagnostic_test", "intervention_combination_product", "intervention_other"]
LEAKY = ["leaky_enrollment_count", "leaky_n_locations", "leaky_n_countries"]
HONEST = CATEGORICAL + NUMERIC + BOOLEAN


def load() -> pd.DataFrame:
    con = duckdb.connect(str(ROOT / "data" / "trials.duckdb"), read_only=True)
    cols = ", ".join(["nct_id", "stopped_early"] + CATEGORICAL + NUMERIC + BOOLEAN + LEAKY)
    df = con.sql(f"""
        select {cols}
        from trial_cohort join sponsor_history using (nct_id)
        where is_resolved and registered_prospectively and start_year between {TRAIN_YEARS[0]} and {TEST_YEARS[1]}
    """).df()
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    for c in BOOLEAN:
        df[c] = df[c].astype("float")
    return df


def part(df: pd.DataFrame, years: tuple[int, int]) -> pd.DataFrame:
    return df[df.start_year.between(*years)]


def scores(y, p) -> dict:
    return {"roc_auc": round(roc_auc_score(y, p), 4),
            "pr_auc": round(average_precision_score(y, p), 4),
            "brier": round(brier_score_loss(y, p), 4),
            "base_rate": round(float(np.mean(y)), 4)}


def fit_logit(train: pd.DataFrame):
    model = make_pipeline(
        ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL),
            ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                  StandardScaler()), NUMERIC + BOOLEAN),
        ]),
        LogisticRegression(max_iter=2000, C=0.5),
    )
    return model.fit(train[HONEST], train.stopped_early)


def fit_gbm(train, valid, features):
    model = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=63,
                               min_child_samples=50, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
    model.fit(train[features], train.stopped_early,
              eval_set=[(valid[features], valid.stopped_early)], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


def main() -> None:
    df = load()
    train, valid, test = (part(df, y) for y in (TRAIN_YEARS, VALID_YEARS, TEST_YEARS))
    results = {"n": {"train": len(train), "valid": len(valid), "test": len(test)}}

    logit = fit_logit(train)
    results["logistic_regression"] = scores(test.stopped_early, logit.predict_proba(test[HONEST])[:, 1])

    gbm = fit_gbm(train, valid, HONEST)
    p_test = gbm.predict_proba(test[HONEST])[:, 1]
    results["gradient_boosting"] = scores(test.stopped_early, p_test)

    naive = fit_gbm(train, valid, HONEST + LEAKY)
    results["gradient_boosting_with_leaky_fields"] = scores(
        test.stopped_early, naive.predict_proba(test[HONEST + LEAKY])[:, 1])

    # calibration of the honest model on the test years, in ten equal-sized bins
    bins = pd.qcut(p_test, 10, duplicates="drop")
    cal = (pd.DataFrame({"p": p_test, "y": test.stopped_early.values, "bin": bins})
           .groupby("bin", observed=True).agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")))
    results["calibration_deciles"] = cal.round(4).reset_index(drop=True).to_dict("records")

    imp = pd.Series(gbm.booster_.feature_importance("gain"), index=HONEST).sort_values(ascending=False)
    results["top_features_by_gain"] = (imp / imp.sum()).round(4).head(12).to_dict()

    out = ROOT / "reports" / "tabular_metrics.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
