"""Evaluation of the primary classifier (terminated within 5 years of start).

Selection   : decided before the test years are touched. Both candidates (tabular
              gradient boosting, and the same plus the text score) are fitted on
              2008-2014 with early stopping on 2015-2016 and compared on 2017. The text
              score is in the headline model only if the paired bootstrap interval for
              its ROC AUC gain on 2017 excludes zero. The other candidate is reported
              as an ablation.
Test        : every model is scored once on the test years (2018-2020).
Ladder      : the naive and partly fixed earlier set-ups next to the defensible one,
              each on its own test set and, scored as-is, on the primary test set.
Calibration : the raw headline scores against an isotonic recalibration fitted on
              2017, applied unchanged to the test years. Expected calibration error
              (ECE) uses ten equal-sized bins of predicted risk.
Net benefit : decision curve of the recalibrated headline model against treating all
              or no trials as high risk, over risk thresholds of 1-40%.
Subgroups   : discrimination and calibration-in-the-large (observed minus mean
              recalibrated prediction) on the test years by phase, lead sponsor class
              and intervention type. Groups under 200 trials or 20 events are skipped.
SHAP        : TreeExplainer on 5,000 random test trials.
Uncertainty : 1000 bootstrap resamples of the test trials, shared by all models so
              differences get their own (paired) interval; once resampling trials and
              once resampling whole sponsors (normalised lead sponsor), since trials of
              one sponsor are not independent.
Final model : refitted on train + early-stopping years (2008-2016) with the number of
              trees found by early stopping, recalibrated on 2017, saved with its
              feature list and category levels for the dashboard, together with
              predictions and SHAP values for a sample of test trials.
"""
from __future__ import annotations

import json

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from train_tabular import (BOOLEAN, CATEGORICAL, FEATURES, ROOT, SPLITS, bootstrap, fit_gbm,
                           fit_logit, gbm_params, ladder, load, part, scores)
from train_text import TEXT_FEATURES, add_text_score, sponsor_groups

FIG = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
MIN_TRIALS, MIN_EVENTS = 200, 20
SHAP_SAMPLE, DASHBOARD_SAMPLE = 5000, 3000


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error over equal-sized bins of predicted risk."""
    order = np.argsort(p, kind="stable")
    gaps = [abs(p[b].mean() - y[b].mean()) * len(b) for b in np.array_split(order, n_bins)]
    return float(sum(gaps) / len(p))


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    order = np.argsort(p, kind="stable")
    return pd.DataFrame([{"predicted": p[b].mean(), "observed": y[b].mean(), "n": len(b)}
                         for b in np.array_split(order, n_bins)])


def fit_calibrator(y: np.ndarray, p: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)


def calibration(y_test: np.ndarray, p_test: np.ndarray, p_cal: np.ndarray) -> dict:
    summary = {}
    for name, p in (("raw", p_test), ("isotonic", p_cal)):
        summary[name] = {"brier": round(brier_score_loss(y_test, p), 4),
                         "ece": round(ece(y_test, p), 4),
                         "mean_predicted": round(float(p.mean()), 4),
                         "roc_auc": round(roc_auc_score(y_test, p), 4)}
    summary["observed_rate"] = round(float(y_test.mean()), 4)

    fig, (ax, hx) = plt.subplots(2, 1, figsize=(6, 6.6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
    top = float(min(1.0, max(p_test.max(), p_cal.max()) + 0.05))
    ax.plot([0, top], [0, top], color="#9a9890", lw=1, ls="--", label="Perfect calibration")
    for key, label, p, colour in (("raw", "Raw boosting", p_test, "#2a78d6"),
                                  ("isotonic", "Isotonic (fitted on 2017)", p_cal, "#eb6834")):
        r = reliability(y_test, p)
        ax.plot(r.predicted, r.observed, marker="o", ms=5, lw=2, color=colour,
                label=f"{label}: ECE {summary[key]['ece']:.3f}, Brier {summary[key]['brier']:.3f}")
        hx.hist(p, bins=50, range=(0, top), histtype="step", lw=1.5, color=colour)
    ax.set(ylabel="Observed share terminated by year 5", xlim=(0, top), ylim=(0, top),
           title="Calibration on test years 2018-2020 (deciles of predicted risk)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    hx.set(xlabel="Predicted probability of termination within 5 years", ylabel="Trials")
    for a in (ax, hx):
        a.grid(alpha=0.25, lw=0.6)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png", dpi=150)
    plt.close(fig)
    return summary


def subgroups(test: pd.DataFrame, p_raw: np.ndarray, p_cal: np.ndarray) -> pd.DataFrame:
    y = test.y.to_numpy()
    groups: list[tuple[str, str, np.ndarray]] = [("all", "all", np.ones(len(y), bool))]
    for col in ("phase", "lead_sponsor_class"):
        for level in sorted(test[col].astype(str).unique()):
            groups.append((col, level, (test[col].astype(str) == level).to_numpy()))
    for flag in (b for b in BOOLEAN if b.startswith("intervention_")):
        groups.append(("intervention_type", flag.removeprefix("intervention_"),
                       (test[flag] == 1).to_numpy()))
    rows = []
    for dim, level, mask in groups:
        n, events = int(mask.sum()), int(y[mask].sum())
        if n < MIN_TRIALS or events < MIN_EVENTS:
            continue
        rows.append({"dimension": dim, "group": level, "n": n, "events": events,
                     "base_rate": y[mask].mean(),
                     "roc_auc": roc_auc_score(y[mask], p_raw[mask]),
                     "pr_auc": average_precision_score(y[mask], p_raw[mask]),
                     "mean_predicted_raw": p_raw[mask].mean(),
                     "mean_predicted_calibrated": p_cal[mask].mean(),
                     "calibration_in_the_large": y[mask].mean() - p_cal[mask].mean()})
    return pd.DataFrame(rows).round(4)


def shap_values(model: lgb.LGBMClassifier, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):  # older shap returns one array per class
        values = values[1]
    base = explainer.expected_value
    return values, float(base[1] if np.ndim(base) else base)


def shap_report(model: lgb.LGBMClassifier, test: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    sample = test[features].sample(min(SHAP_SAMPLE, len(test)), random_state=0)
    values, _ = shap_values(model, sample)
    shap.summary_plot(values, sample, max_display=20, show=False, plot_size=(8, 7))
    plt.title("SHAP values, headline model (5,000 test trials, log-odds)", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG / "shap_summary.png", dpi=150)
    plt.close("all")
    return (pd.DataFrame({"feature": features, "mean_abs_shap": np.abs(values).mean(0)})
            .sort_values("mean_abs_shap", ascending=False).round(5))


def final_model(df: pd.DataFrame, features: list[str], n_trees: int, text_model) -> dict:
    """Refit on 2008-2016, recalibrate on 2017, save model and a dashboard sample."""
    fit_rows = df[df.start_year.between(SPLITS["train"][0], SPLITS["valid"][1])]
    calib, test = part(df, "calib"), part(df, "test")
    model = lgb.LGBMClassifier(n_estimators=n_trees, **gbm_params()).fit(fit_rows[features], fit_rows.y)
    calibrator = fit_calibrator(calib.y.to_numpy(), model.predict_proba(calib[features])[:, 1])

    p_raw = model.predict_proba(test[features])[:, 1]
    p_cal = calibrator.predict(p_raw)
    bundle = {
        "model": model, "calibrator": calibrator, "features": features,
        "categories": {c: list(df[c].cat.categories) for c in CATEGORICAL},
        "text_model": text_model,  # None when the headline model is tabular only
        # text_score = text_model.predict_proba(masked text)[:, 1], where the text is
        # title, conditions, summary and criteria joined as in train_text.load_text and
        # passed through train_text.mask_stopping first
        "text_preprocessing": "train_text.load_text + train_text.mask_stopping" if text_model else None,
        "label": "terminated within 5 years of start (among trials that started)",
        "fit_years": [SPLITS["train"][0], SPLITS["valid"][1]], "calibration_years": list(SPLITS["calib"]),
        "test_metrics": {"raw": scores(test.y, p_raw), "calibrated_brier": round(brier_score_loss(test.y, p_cal), 4)},
    }
    MODELS.mkdir(exist_ok=True)
    joblib.dump(bundle, MODELS / "final_model.joblib", compress=3)

    idx = test.sample(min(DASHBOARD_SAMPLE, len(test)), random_state=1).index
    values, base = shap_values(model, test.loc[idx, features])
    sample = test.loc[idx, ["nct_id", "brief_title", "start_year", "phase", "lead_sponsor_class", "y"]].copy()
    sample["phase"] = sample.phase.astype(str)
    sample["lead_sponsor_class"] = sample.lead_sponsor_class.astype(str)
    sample["p_raw"] = model.predict_proba(test.loc[idx, features])[:, 1]
    sample["p_calibrated"] = calibrator.predict(sample.p_raw.to_numpy())
    sample["shap_base_value"] = base
    shap_cols = pd.DataFrame(values, columns=[f"shap_{f}" for f in features], index=idx)
    pd.concat([sample, shap_cols], axis=1).reset_index(drop=True).to_parquet(
        MODELS / "dashboard_sample.parquet")
    return bundle["test_metrics"]


def decision_curve(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """Net benefit per trial of flagging trials with risk >= t, against flagging all or none.

    Net benefit = TP/n - FP/n * t / (1 - t): true positives found, minus false
    positives weighted by the odds of the threshold (how many false alarms one
    termination found is worth at that threshold)."""
    n, prev = len(y), float(np.mean(y))
    rows = []
    for t in np.round(np.arange(0.01, 0.401, 0.01), 2):
        flag = p >= t
        tp, fp = float((flag & (y == 1)).sum()) / n, float((flag & (y == 0)).sum()) / n
        w = t / (1 - t)
        rows.append({"threshold": t, "net_benefit_model": tp - fp * w,
                     "net_benefit_treat_all": prev - (1 - prev) * w, "net_benefit_treat_none": 0.0,
                     "share_flagged": float(flag.mean())})
    return pd.DataFrame(rows).round(5)


def plot_decision_curve(dc: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(dc.threshold, dc.net_benefit_model, color="#2a78d6", lw=2, label="Headline model (recalibrated)")
    ax.plot(dc.threshold, dc.net_benefit_treat_all, color="#eb6834", lw=2, ls="--", label="Flag every trial")
    ax.axhline(0, color="#9a9890", lw=1.2, label="Flag no trial")
    ax.set(xlabel="Risk threshold for flagging a trial", ylabel="Net benefit per trial",
           title="Decision curve, test years 2018-2020",
           ylim=(min(-0.02, float(dc.net_benefit_model.min()) - 0.005), float(dc.net_benefit_treat_all.max()) + 0.01))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "decision_curve.png", dpi=150)
    plt.close(fig)


def select_headline(calib: pd.DataFrame, tab: lgb.LGBMClassifier, txt: lgb.LGBMClassifier) -> dict:
    """Decide on 2017 only whether the text score earns its place."""
    y = calib.y.to_numpy()
    preds = {"tabular_gbm": tab.predict_proba(calib[FEATURES])[:, 1],
             "tabular_plus_text_gbm": txt.predict_proba(calib[TEXT_FEATURES])[:, 1]}
    boot = bootstrap(y, preds, pairs=(("tabular_plus_text_gbm", "tabular_gbm"),))
    gain = boot["difference_tabular_plus_text_gbm_minus_tabular_gbm"]["roc_auc"]
    return {"rule": "include the text score if the paired 95% bootstrap interval of its ROC AUC gain "
                    "on 2017 starts excludes zero",
            "selection_year": 2017, "n": int(len(y)),
            "roc_auc_2017": {k: v["roc_auc"]["estimate"] for k, v in boot.items() if k in preds},
            "roc_auc_gain_2017": gain,
            "headline_model": "tabular_plus_text_gbm" if gain["ci95"][0] > 0 else "tabular_gbm"}


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    scored = add_text_score(load())
    df = scored.df
    train, valid, calib, test = (part(df, s) for s in SPLITS)

    tab = fit_gbm(train, valid, FEATURES)
    txt = fit_gbm(train, valid, TEXT_FEATURES)
    selection = select_headline(calib, tab, txt)
    headline = selection["headline_model"]
    ablation = "tabular_gbm" if headline == "tabular_plus_text_gbm" else "tabular_plus_text_gbm"
    model, features = (txt, TEXT_FEATURES) if headline == "tabular_plus_text_gbm" else (tab, FEATURES)
    calibrator = fit_calibrator(calib.y.to_numpy(), model.predict_proba(calib[features])[:, 1])

    # ---- from here on the test years are scored, once per model
    y_test = test.y.to_numpy()
    rungs = ladder()
    preds = {
        "logistic_regression": fit_logit(train).predict_proba(test[FEATURES])[:, 1],
        "tabular_gbm": tab.predict_proba(test[FEATURES])[:, 1],
        "tabular_plus_text_gbm": txt.predict_proba(test[TEXT_FEATURES])[:, 1],
        "partly_fixed_scored_on_primary": rungs["partly_fixed"]["p_primary"],
        # leakage comparison only: fields overwritten after a trial ends
        "naive_scored_on_primary": rungs["naive"]["p_primary"],
    }
    assert (rungs["defensible"]["y"] == y_test).all()
    pair = (("tabular_plus_text_gbm", "tabular_gbm"),)
    primary = bootstrap(y_test, preds, pairs=pair)
    groups = sponsor_groups(test)
    clustered = bootstrap(y_test, {k: preds[k] for k in ("logistic_regression", "tabular_gbm",
                                                         "tabular_plus_text_gbm")},
                          pairs=pair, groups=groups)

    own_ladder = {name: bootstrap(rungs[name]["y"], {name: rungs[name]["p"]})[name]
                  for name in ("naive", "partly_fixed")}
    own_ladder["defensible"] = primary[headline]

    p_test = preds[headline]
    p_cal = calibrator.predict(p_test)
    cal_summary = calibration(y_test, p_test, p_cal)
    dc = decision_curve(y_test, p_cal)
    dc.to_csv(ROOT / "reports" / "decision_curve.csv", index=False)
    plot_decision_curve(dc)
    subgroups(test, p_test, p_cal).to_csv(ROOT / "reports" / "subgroup_metrics.csv", index=False)
    shap_report(model, test, features).to_csv(ROOT / "reports" / "shap_importance.csv", index=False)
    final = final_model(df, features, int(model.best_iteration_),
                        scored.text_model if headline == "tabular_plus_text_gbm" else None)

    beats = dc[(dc.net_benefit_model > dc.net_benefit_treat_all) & (dc.net_benefit_model > 0)].threshold
    results = {
        "task": "terminated within 5 years of start; test = starts 2018-2020",
        "n_test": int(len(test)), "base_rate_test": round(float(y_test.mean()), 4),
        "bootstrap_resamples": 1000,
        "n_sponsor_clusters_test": int(len(np.unique(groups))),
        "selection": selection,
        "headline_model": headline,
        "ablation_model": ablation,
        "ladder_own_test_sets": {
            "naive": {"definition": "old label/cohort 2018-2019 test, with leaky fields", **own_ladder["naive"]},
            "partly_fixed": {"definition": "old label/cohort 2018-2019 test, registration delay + start year",
                             **own_ladder["partly_fixed"]},
            "defensible": {"definition": "terminated within 5 years, started trials, 2018-2020 test, headline model",
                           **own_ladder["defensible"]},
        },
        "primary_test_set": primary,
        "primary_test_set_sponsor_clustered": clustered,
        "calibration": cal_summary,
        "decision_curve": {"model_beats_both_defaults_at_thresholds":
                           [float(beats.min()), float(beats.max())] if len(beats) else None,
                           "file": "reports/decision_curve.csv"},
        "final_model_test": final,
    }
    (ROOT / "reports" / "headline_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
