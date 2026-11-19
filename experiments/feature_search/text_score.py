"""Arm F (partly EXPLORATORY: title, summary and eligibility text are the current record).

The repo's text score (src/models/train_text.py) rebuilt per fold: the same masked
documents (`documents`), the same TF-IDF + logistic model (`text_model`) and the same
sponsor-grouped StratifiedGroupKFold(5, shuffle=True, random_state=0). The text model only
ever sees the fold's training pool: training rows get out-of-fold scores, validation rows
get the score of one text model fitted on the whole pool.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold

from cohort import connect, sponsor_groups
from train_text import _fit_predict, documents, text_model


def raw_text(nct_ids: list[str]) -> pd.DataFrame:
    """Same columns as train_text.raw_text, read from stg_studies so 2005-2007 starts work too
    (trial_cohort copies these fields unchanged from stg_studies)."""
    con = connect()
    con.register("ids", pd.DataFrame({"nct_id": nct_ids}))
    return con.sql("""select nct_id, brief_title, array_to_string(conditions, '; ') as conditions_text,
                             brief_summary, eligibility_criteria
                      from stg_studies join ids using (nct_id) order by nct_id""").df()


def load_documents(nct_ids: list[str]) -> pd.Series:
    docs = documents(raw_text(nct_ids)).set_index("nct_id").text
    return docs.reindex(nct_ids).fillna("")


def fold_text_score(pool: pd.DataFrame, valid: pd.DataFrame, docs: pd.Series,
                    n_jobs: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold scores for the pool rows and pool-fitted scores for the validation rows."""
    text = docs.reindex(pool.nct_id).to_numpy()
    y = pool.y.to_numpy()
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    folds = list(splitter.split(np.arange(len(pool)), y, groups=sponsor_groups(pool)))
    oof = np.full(len(pool), np.nan)
    preds = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict)(pd.Series(text[f]), pd.Series(y[f]), pd.Series(text[p]))
        for f, p in folds)
    for (_, p), pred in zip(folds, preds):
        oof[p] = pred
    assert not np.isnan(oof).any()
    final = text_model().fit(pd.Series(text), pd.Series(y))
    return oof, final.predict_proba(docs.reindex(valid.nct_id))[:, 1]
