"""Does free text add signal beyond the structured fields?

A TF-IDF + logistic regression model reads the title, summary, conditions and
eligibility criteria. Its prediction is then given to the gradient boosting
model as one extra feature. On the training years that prediction is produced
out-of-fold, so the boosting model never sees a text score that was fitted on
the same trial.

The text model uses the liblinear solver: on ~55k documents x 300k n-grams it
converges in about 15 seconds, where saga needed over a minute per fit and had
not converged after 300 passes. The five folds are fitted in parallel.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline

from train_tabular import (HONEST, ROOT, TEST_YEARS, TRAIN_YEARS, VALID_YEARS, fit_gbm,
                           load, part, scores)

TEXT_FEATURES = HONEST + ["text_score"]


def text_model() -> Pipeline:
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=300_000,
                        sublinear_tf=True, strip_accents="unicode", dtype=np.float32),
        LogisticRegression(C=2.0, max_iter=1000, solver="liblinear"),
    )


def load_text(df: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect(str(ROOT / "data" / "trials.duckdb"), read_only=True)
    text = con.sql("""
        select nct_id,
               concat_ws(' \n ', brief_title, array_to_string(conditions, '; '),
                         brief_summary, eligibility_criteria) as text
        from trial_cohort
    """).df()
    df = df.merge(text, on="nct_id", how="left")
    df["text"] = df.text.fillna("")
    return df


def _fit_predict(fit_text: pd.Series, fit_y: pd.Series, pred_text: pd.Series) -> np.ndarray:
    return text_model().fit(fit_text, fit_y).predict_proba(pred_text)[:, 1]


@dataclass
class TextScored:
    df: pd.DataFrame           # cohort with a `text_score` column (out-of-fold on train years)
    text_model: Pipeline       # text model fitted on all training years


def add_text_score(df: pd.DataFrame, n_jobs: int = 3) -> TextScored:
    """Attach the text model's score: out-of-fold for train years, final model for later years."""
    df = load_text(df).reset_index(drop=True)
    df["text_score"] = np.nan

    train_idx = df.index[df.start_year.between(*TRAIN_YEARS)]
    later_idx = df.index[~df.start_year.between(*TRAIN_YEARS)]
    y_train = df.loc[train_idx, "stopped_early"]
    text = df.text

    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(train_idx, y_train))
    oof = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict)(text[train_idx[f]], y_train.iloc[f], text[train_idx[p]])
        for f, p in folds)
    for (_, p), pred in zip(folds, oof):
        df.loc[train_idx[p], "text_score"] = pred
    final = text_model().fit(text[train_idx], y_train)
    df.loc[later_idx, "text_score"] = final.predict_proba(text[later_idx])[:, 1]
    return TextScored(df=df.drop(columns="text"), text_model=final)


def fit_combined(df: pd.DataFrame) -> lgb.LGBMClassifier:
    """Gradient boosting on the tabular features plus the text score."""
    train, valid = part(df, TRAIN_YEARS), part(df, VALID_YEARS)
    return fit_gbm(train, valid, TEXT_FEATURES)


def main() -> None:
    scored = add_text_score(load())
    df = scored.df
    train, valid, test = (part(df, y) for y in (TRAIN_YEARS, VALID_YEARS, TEST_YEARS))
    results = {
        "text_only": scores(test.stopped_early, test.text_score),
        "tabular_only": scores(test.stopped_early,
                               fit_gbm(train, valid, HONEST).predict_proba(test[HONEST])[:, 1]),
    }
    combined = fit_combined(df)
    p = combined.predict_proba(test[TEXT_FEATURES])[:, 1]
    results["tabular_plus_text"] = scores(test.stopped_early, p)

    final = scored.text_model
    vec, clf = final.named_steps["tfidfvectorizer"], final.named_steps["logisticregression"]
    terms = np.array(vec.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    results["terms_raising_risk"] = terms[order[-25:]][::-1].tolist()
    results["terms_lowering_risk"] = terms[order[:25]].tolist()

    out = test[["nct_id", "stopped_early", "text_score"]].assign(p_combined=p)
    out.to_parquet(ROOT / "reports" / "test_predictions.parquet")
    (ROOT / "reports" / "text_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
