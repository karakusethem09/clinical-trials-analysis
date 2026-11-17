"""Does free text add signal beyond the structured fields?

A TF-IDF + logistic regression model reads the title, conditions, summary and
eligibility criteria. Its prediction is given to the gradient boosting model as one
extra feature. On the training years that prediction is produced out-of-fold, with
folds grouped by normalised lead sponsor: a sponsor's trials are never split between
the fitting and the predicting folds, so the text model cannot score a trial by
recognising the drug programme or house style of its own sponsor. Later years are
scored by one text model fitted on all training years.

Before the text is read, statements that the trial itself stopped are removed from
the title and summary (text_masking.py).

This script reports the text ablation on the selection year (2017) and, as a
description, on the test years, for grouped and for ungrouped folds, plus how many
trials lose a sentence to masking under the current and the earlier, broader rule.
Whether the text score goes into the headline model is decided in evaluate.py on 2017
only.

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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline

from text_masking import BROAD_RULE, document, split_sentences, states_stopping
from train_tabular import DB, FEATURES, ROOT, SPLITS, fit_gbm, load, part, scores

TEXT_FEATURES = FEATURES + ["text_score"]


def text_model() -> Pipeline:
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=300_000,
                        sublinear_tf=True, strip_accents="unicode", dtype=np.float32),
        LogisticRegression(C=2.0, max_iter=1000, solver="liblinear"),
    )


def raw_text(ids: list[str] | None = None) -> pd.DataFrame:
    """Title, conditions, summary and criteria per trial, as stored in trial_cohort."""
    con = duckdb.connect(str(DB), read_only=True)
    sql = """select nct_id, brief_title, array_to_string(conditions, '; ') as conditions_text,
                    brief_summary, eligibility_criteria
             from trial_cohort"""
    if ids is not None:
        con.register("ids", pd.DataFrame({"nct_id": ids}))
        sql += " join ids using (nct_id)"
    return con.sql(sql + " order by nct_id").df()


def documents(raw: pd.DataFrame) -> pd.DataFrame:
    """The masked document each trial contributes, with the sentences that were removed."""
    built = [document(r.brief_title, r.conditions_text, r.brief_summary, r.eligibility_criteria)
             for r in raw.itertuples()]
    return pd.DataFrame({"nct_id": raw.nct_id, "text": [b[0] for b in built],
                         "masked_sentences": [b[1] for b in built]})


def load_text(df: pd.DataFrame) -> pd.DataFrame:
    return df.merge(documents(raw_text()), on="nct_id", how="left").assign(
        text=lambda d: d.text.fillna(""),
        masked_sentences=lambda d: d.masked_sentences.map(lambda x: x if isinstance(x, list) else []))


def _fit_predict(fit_text: pd.Series, fit_y: pd.Series, pred_text: pd.Series) -> np.ndarray:
    return text_model().fit(fit_text, fit_y).predict_proba(pred_text)[:, 1]


@dataclass
class TextScored:
    df: pd.DataFrame           # cohort with `text_score` (out-of-fold on train years)
    text_model: Pipeline       # text model fitted on all training years


def sponsor_groups(df: pd.DataFrame) -> np.ndarray:
    """Normalised lead sponsor; a trial without one is its own group."""
    return df.lead_sponsor_key.fillna("trial:" + df.nct_id).to_numpy()


def add_text_score(df: pd.DataFrame, grouped: bool = True, n_jobs: int = 3) -> TextScored:
    """Attach the text model's score: out-of-fold for train years, final model for later years."""
    df = load_text(df).reset_index(drop=True)
    df["text_score"] = np.nan

    train_idx = df.index[df.start_year.between(*SPLITS["train"])]
    later_idx = df.index[~df.start_year.between(*SPLITS["train"])]
    y_train = df.loc[train_idx, "y"]
    text = df.text

    if grouped:
        splitter = StratifiedGroupKFold(5, shuffle=True, random_state=0)
        folds = list(splitter.split(train_idx, y_train, groups=sponsor_groups(df.loc[train_idx])))
    else:
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
    return fit_gbm(part(df, "train"), part(df, "valid"), TEXT_FEATURES)


def masking_report(df: pd.DataFrame, seed: int = 0) -> tuple[dict, pd.DataFrame]:
    """Share of cohort trials losing a sentence, by outcome and rule, plus 20 random masked sentences."""
    raw = raw_text(df.nct_id.tolist()).merge(df[["nct_id", "y", "start_year"]], on="nct_id")
    fields = {"title": "brief_title", "summary": "brief_summary", "criteria": "eligibility_criteria"}
    hit = {name: raw[col].map(lambda t: any(states_stopping(s) for s in split_sentences(t or "")))
           for name, col in fields.items()}
    whole = raw[["brief_title", "conditions_text", "brief_summary", "eligibility_criteria"]].fillna("").agg(
        " \n ".join, axis=1)
    broad = whole.map(lambda t: any(BROAD_RULE.search(s) for s in split_sentences(t)))

    def share(mask: pd.Series) -> dict:
        g = mask.groupby(raw.y).mean()
        return {"terminated_5y": round(float(g.get(1, 0)), 4), "not_terminated": round(float(g.get(0, 0)), 4)}

    summary = {
        "cohort": "primary cohort, start years 2008-2020",
        "n_trials": int(len(raw)),
        "current_rule_title_or_summary": share(hit["title"] | hit["summary"]),
        "current_rule_if_applied_to_criteria": share(hit["criteria"]),
        "earlier_broad_rule_all_text": share(broad),
    }
    rows = [(r.nct_id, field, int(r.y), s) for field, col in (("title", "brief_title"), ("summary", "brief_summary"))
            for r in raw[hit[field]].itertuples() for s in split_sentences(getattr(r, col) or "")
            if states_stopping(s)]
    sample = pd.DataFrame(rows, columns=["nct_id", "field", "terminated_5y", "masked_sentence"])
    sample = sample.sample(min(20, len(sample)), random_state=seed).sort_values("nct_id")
    return summary, sample


def ablation(df: pd.DataFrame, year_set: str) -> dict:
    train, valid, ev = part(df, "train"), part(df, "valid"), part(df, year_set)
    tab = fit_gbm(train, valid, FEATURES).predict_proba(ev[FEATURES])[:, 1]
    both = fit_combined(df).predict_proba(ev[TEXT_FEATURES])[:, 1]
    t, b = scores(ev.y, tab), scores(ev.y, both)
    return {"text_only": scores(ev.y, ev.text_score), "tabular_only": t, "tabular_plus_text": b,
            "roc_auc_gain": round(b["roc_auc"] - t["roc_auc"], 4)}


def main() -> None:
    base = load()
    grouped = add_text_score(base)
    ungrouped = add_text_score(base, grouped=False)
    summary, sample = masking_report(base)
    sample.to_csv(ROOT / "reports" / "masked_sentences_sample.csv", index=False)
    (ROOT / "reports" / "masking_summary.json").write_text(json.dumps(summary, indent=2))

    results: dict = {"masking": summary}
    for name, scored in (("grouped_by_sponsor", grouped), ("ungrouped", ungrouped)):
        results[name] = {"selection_year_2017": ablation(scored.df, "calib"),
                         "test_2018_2020": ablation(scored.df, "test")}

    final = grouped.text_model
    vec, clf = final.named_steps["tfidfvectorizer"], final.named_steps["logisticregression"]
    terms = np.array(vec.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    results["terms_raising_risk"] = terms[order[-25:]][::-1].tolist()
    results["terms_lowering_risk"] = terms[order[:25]].tolist()

    df = grouped.df
    test = part(df, "test")
    p = fit_combined(df).predict_proba(test[TEXT_FEATURES])[:, 1]
    test[["nct_id", "y", "text_score"]].assign(p_combined=p).to_parquet(
        ROOT / "reports" / "test_predictions.parquet")
    (ROOT / "reports" / "text_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
