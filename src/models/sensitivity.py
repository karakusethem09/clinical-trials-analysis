"""Sensitivity of the primary result to the judgment calls behind it.

Each variant changes one thing, refits the tabular gradient boosting model on the
same year split, and is scored two ways: on its own test set (its own cohort and
label) and on the primary test set, so differences in who is counted can be told
apart from differences in what the model learned.

  primary                       : as in train_tabular.py
  no_registration_filter        : keep trials registered any time after their start
  legacy_registration_filter    : the earlier 21-day rule on the day-1 start date
  unknown_as_not_terminated     : UNKNOWN status counted as "not terminated by year 5"
  unknown_verified_after_5y     : UNKNOWN counted as not terminated only if the sponsor
                                  last verified the trial as running after year 5
  drop_stale_running            : leave out trials labelled "not terminated" only because
                                  they are still running, when their status was not
                                  verified in the 12 months before the pull
  plus_registration_delay       : add days from registration to actual start
  plus_start_year               : add the start year
"""
from __future__ import annotations

import json

import pandas as pd

from train_tabular import FEATURES, ROOT, bootstrap, fit_gbm, load, part

VARIANTS: dict[str, dict] = {
    "primary": {},
    "no_registration_filter": {"registration_filter": None},
    "legacy_registration_filter": {"registration_filter": "registered_prospectively"},
    "unknown_as_not_terminated": {"label": "terminated_5y_unknown_as_0"},
    "unknown_verified_after_5y": {"label": "terminated_5y_unknown_verified"},
    "drop_stale_running": {"drop_stale_running": True},
    "plus_registration_delay": {"extra": ["days_registered_after_start"]},
    "plus_start_year": {"extra": ["start_year"]},
}


def main() -> None:
    primary_test = part(load(), "test")
    rows = []
    for name, spec in VARIANTS.items():
        df = load(**{k: v for k, v in spec.items() if k != "extra"})
        features = FEATURES + spec.get("extra", [])
        model = fit_gbm(part(df, "train"), part(df, "valid"), features)
        test = part(df, "test")
        own = bootstrap(test.y, {"m": model.predict_proba(test[features])[:, 1]})["m"]
        on_primary = bootstrap(primary_test.y,
                               {"m": model.predict_proba(primary_test[features])[:, 1]})["m"]
        rows.append({
            "variant": name, "n_train": int(len(part(df, "train"))), "n_test": int(len(test)),
            "base_rate_test": round(float(test.y.mean()), 4),
            "roc_auc": own["roc_auc"]["estimate"],
            "roc_auc_ci95": own["roc_auc"]["ci95"],
            "top_decile_precision": own["top_decile_precision"]["estimate"],
            "roc_auc_on_primary_test": on_primary["roc_auc"]["estimate"],
            "roc_auc_on_primary_test_ci95": on_primary["roc_auc"]["ci95"],
        })
        print(rows[-1], flush=True)
    (ROOT / "reports" / "sensitivity_metrics.json").write_text(json.dumps(rows, indent=2))
    pd.DataFrame(rows).to_csv(ROOT / "reports" / "sensitivity_metrics.csv", index=False)


if __name__ == "__main__":
    main()
