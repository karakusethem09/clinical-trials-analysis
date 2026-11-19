"""One-shot confirmation on 2021 starts. NOT RUN: importing this module does nothing.

Cohort    : trial_cohort rows with start year 2021, terminated_5y known and
            registered_by_start. With the pull date 2026-11-13 (max last_update_date,
            as in the trial_cohort model) only starts 2021-01-01 .. 2021-11-13 have a
            complete 5-year horizon; trial_cohort already sets the label to null for the
            rest. Count when this was written: 12,944 trials (outcomes not looked at).
            Caveat: a termination counts only once the sponsor reports it, and these
            trials reached their horizon 0-10 months before the pull, so late reporting
            will hide some terminations (the 2020 starts already had 15% still running).
Training  : starts 2008-2015 (label-knowable for 2021: 2015 + 6 = 2021), tree count by the
            same sponsor-grouped CV as run.py, 5 seeds averaged, same arm definitions.
Scoring   : the chosen arm against the refitted baseline, paired sponsor-clustered bootstrap.

    from confirm import confirm
    confirm("C")      # only after deciding, from the fold results, which arm to confirm
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

import metrics
from cohort import CACHE, FEATURES, _select, connect, sponsor_groups, typed
from run import ARMS, SEEDS, choose_trees, features_for, lgb_params
from text_score import fold_text_score, load_documents

TRAIN_YEARS = (2008, 2015)
CONFIRM_YEAR = 2021


def load_confirmation_frame() -> pd.DataFrame:
    con = connect()
    where = (f"(c.start_year between {TRAIN_YEARS[0]} and {TRAIN_YEARS[1]}) "
             f"or c.start_year = {CONFIRM_YEAR}")
    df = con.sql(_select("trial_cohort", "sponsor_history", f"({where})")).df()
    assert not df.start_year.between(2016, 2020).any()
    return typed(df)


def confirm(arm: str, n_boot: int = 1000) -> dict:
    assert arm in ARMS and arm != "A"
    df = features_for(load_confirmation_frame(), "confirm2021")
    pool = df[df.start_year.between(*TRAIN_YEARS)].reset_index(drop=True)
    test = df[df.start_year == CONFIRM_YEAR].reset_index(drop=True)
    n_trees, _ = choose_trees(pool)
    if "text_score" in ARMS[arm][1]:
        docs = load_documents(df.nct_id.tolist())
        pool["text_score"], test["text_score"] = fold_text_score(pool, test, docs)
    preds = {}
    for name in ("A", arm):
        extra = ARMS[name][1]
        cols = FEATURES + extra
        p = np.zeros(len(test))
        for seed in SEEDS:
            real = FEATURES + [c for c in extra if c != "noise"]
            X_tr, X_te = pool[real].copy(), test[real].copy()
            if "noise" in extra:
                z = np.random.default_rng(1000 + seed).standard_normal(len(pool) + len(test))
                X_tr["noise"], X_te["noise"] = z[:len(pool)], z[len(pool):]
            model = lgb.LGBMClassifier(n_estimators=n_trees, **lgb_params(seed)).fit(X_tr[cols], pool.y)
            p += model.predict_proba(X_te[cols])[:, 1] / len(SEEDS)
        preds[name] = p
    res = metrics.fold_bootstrap(test.y.to_numpy(), preds, (test.phase.astype(str) == "PHASE2").to_numpy(),
                                 sponsor_groups(test), base="A", n_boot=n_boot, seed=CONFIRM_YEAR)
    out = {"arm": arm, "n_train": len(pool), "n_trees": n_trees, "n_test": len(test),
           "positives": int(test.y.sum()), "result": res}
    (CACHE / f"confirm_{arm}.json").write_text(json.dumps(out, indent=2, default=float))
    return out
