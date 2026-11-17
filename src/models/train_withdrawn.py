"""Secondary analysis: will a newly registered trial be withdrawn before it starts?

Withdrawn trials never enrol anyone, and their start date is usually a plan that
never happened, so they are kept out of the primary "terminated within 5 years"
task. This asks the other question, anchored on the registration date instead.

Cohort : interventional trials first submitted 2008-2020, registered no later than
         21 days after their (planned or actual) start, that either were WITHDRAWN
         (label 1) or actually started (label 0). Trials still waiting to recruit or
         with only a planned start date and an UNKNOWN status are left out: whether
         they will start is not known.
Split  : by year of first submission: train 2008-2014, early stopping 2015-2016,
         test 2018-2020.
Model  : the same gradient boosting set-up and registration-time features as the
         primary model.
"""
from __future__ import annotations

import json

import pandas as pd

from train_tabular import (FEATURES, ROOT, query_cohort, typed, bootstrap, fit_gbm, scores)

SPLITS = {"train": (2008, 2014), "valid": (2015, 2016), "test": (2018, 2020)}


def load() -> pd.DataFrame:
    df = typed(query_cohort("""registered_by_start
        and year(first_submit_date) between 2008 and 2020
        and (overall_status = 'WITHDRAWN' or started)""",
                       "(overall_status = 'WITHDRAWN')::int"))
    df["registration_year"] = pd.to_datetime(df.first_submit_date).dt.year
    return df


def part(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return df[df.registration_year.between(*SPLITS[name])]


def main() -> None:
    df = load()
    train, valid, test = (part(df, s) for s in SPLITS)
    model = fit_gbm(train, valid, FEATURES)
    p = model.predict_proba(test[FEATURES])[:, 1]

    imp = pd.Series(model.booster_.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    by_class = (test.assign(p=p).groupby("lead_sponsor_class", observed=True)
                .agg(n=("y", "size"), withdrawn_rate=("y", "mean"), mean_predicted=("p", "mean")))
    results = {
        "label": "withdrawn before starting, among prospectively registered trials",
        "n": {s: int(len(part(df, s))) for s in SPLITS},
        "base_rate_by_registration_year": df.groupby("registration_year").y.mean().round(4)
                                            .rename(index=str).to_dict(),
        "test": scores(test.y, p),
        "test_bootstrap": bootstrap(test.y, {"withdrawn_gbm": p})["withdrawn_gbm"],
        "top_features_by_gain": (imp / imp.sum()).round(4).head(10).to_dict(),
        "test_by_sponsor_class": by_class[by_class.n >= 200].round(4).to_dict("index"),
    }
    (ROOT / "reports" / "withdrawn_metrics.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
