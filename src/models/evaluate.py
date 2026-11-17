"""Evaluation of the best classifier: tabular + text-score gradient boosting.

Calibration : the raw boosting scores are compared with an isotonic recalibration
              fitted on the validation years (2016-2017) and applied unchanged to
              the test years (2018-2019). Expected calibration error (ECE) uses ten
              equal-sized bins of predicted risk: the size-weighted average gap
              between mean predicted and observed stop rate.
Subgroups   : discrimination and calibration-in-the-large (observed minus mean
              predicted stop rate, after recalibration) on the test years by phase,
              lead sponsor class and intervention type. Small groups are skipped.
SHAP        : TreeExplainer on a random sample of test trials.
Uncertainty : 1000 bootstrap resamples of the test trials. Every model is scored
              on the same resamples, so the difference between two models gets its
              own (paired) interval. The leaky model is included only to show how
              much fields overwritten after the trial ends would inflate the metrics.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from train_tabular import (BOOLEAN, HONEST, LEAKY, ROOT, TEST_YEARS, TRAIN_YEARS, VALID_YEARS,
                           fit_gbm, fit_logit, load, part)
from train_text import TEXT_FEATURES, add_text_score, fit_combined

FIG = ROOT / "reports" / "figures"
MIN_TRIALS, MIN_EVENTS = 200, 20
N_BOOT, SHAP_SAMPLE = 1000, 5000


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error over equal-sized bins of predicted risk."""
    order = np.argsort(p, kind="stable")
    gaps = [abs(p[b].mean() - y[b].mean()) * len(b) for b in np.array_split(order, n_bins)]
    return float(sum(gaps) / len(p))


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    order = np.argsort(p, kind="stable")
    return pd.DataFrame([{"predicted": p[b].mean(), "observed": y[b].mean(), "n": len(b)}
                         for b in np.array_split(order, n_bins)])


def calibration(y_valid: np.ndarray, p_valid: np.ndarray, y_test: np.ndarray,
                p_test: np.ndarray) -> tuple[np.ndarray, dict]:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_valid, y_valid)
    p_cal = iso.predict(p_test)
    summary = {}
    for name, p in (("raw", p_test), ("isotonic", p_cal)):
        summary[name] = {"brier": round(brier_score_loss(y_test, p), 4),
                         "ece": round(ece(y_test, p), 4),
                         "mean_predicted": round(float(p.mean()), 4),
                         "roc_auc": round(roc_auc_score(y_test, p), 4)}
    summary["observed_rate"] = round(float(y_test.mean()), 4)

    fig, (ax, hx) = plt.subplots(2, 1, figsize=(6, 6.6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
    ax.plot([0, 1], [0, 1], color="#9a9890", lw=1, ls="--", label="Perfect calibration")
    for name, p, colour in (("Raw boosting", p_test, "#2a78d6"),
                            ("Isotonic (fitted on 2016-17)", p_cal, "#eb6834")):
        r = reliability(y_test, p)
        key = "raw" if name.startswith("Raw") else "isotonic"
        ax.plot(r.predicted, r.observed, marker="o", ms=5, lw=2, color=colour,
                label=f"{name}: ECE {summary[key]['ece']:.3f}, Brier {summary[key]['brier']:.3f}")
        hx.hist(p, bins=50, range=(0, 1), histtype="step", lw=1.5, color=colour)
    top = float(max(p_test.max(), p_cal.max()))
    ax.set(ylabel="Observed stop rate", xlim=(0, min(1, top + 0.05)), ylim=(0, min(1, top + 0.05)),
           title="Calibration on test years 2018-2019 (deciles of predicted risk)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    hx.set(xlabel="Predicted probability of stopping early", ylabel="Trials")
    for a in (ax, hx):
        a.grid(alpha=0.25, lw=0.6)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png", dpi=150)
    plt.close(fig)
    return p_cal, summary


def subgroups(test: pd.DataFrame, p_raw: np.ndarray, p_cal: np.ndarray) -> pd.DataFrame:
    y = test.stopped_early.to_numpy()
    groups: list[tuple[str, str, np.ndarray]] = []
    for col in ("phase", "lead_sponsor_class"):
        for level in test[col].astype(str).unique():
            groups.append((col, level, (test[col].astype(str) == level).to_numpy()))
    for flag in (b for b in BOOLEAN if b.startswith("intervention_")):
        groups.append(("intervention_type", flag.removeprefix("intervention_"),
                       (test[flag] == 1).to_numpy()))
    rows = []
    for dim, level, mask in [("all", "all", np.ones(len(y), bool))] + groups:
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


def shap_importance(model, test: pd.DataFrame) -> pd.DataFrame:
    sample = test[TEXT_FEATURES].sample(min(SHAP_SAMPLE, len(test)), random_state=0)
    values = shap.TreeExplainer(model).shap_values(sample)
    if isinstance(values, list):  # older shap returns one array per class
        values = values[1]
    shap.summary_plot(values, sample, max_display=20, show=False, plot_size=(8, 7))
    plt.title("SHAP values, tabular + text model (5,000 test trials, log-odds)", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG / "shap_summary.png", dpi=150)
    plt.close("all")
    return (pd.DataFrame({"feature": TEXT_FEATURES, "mean_abs_shap": np.abs(values).mean(0)})
            .sort_values("mean_abs_shap", ascending=False).round(5))


def bootstrap(y: np.ndarray, preds: dict[str, np.ndarray], seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    draws = {name: {"roc_auc": [], "pr_auc": []} for name in preds}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        for name, p in preds.items():
            draws[name]["roc_auc"].append(roc_auc_score(y[i], p[i]))
            draws[name]["pr_auc"].append(average_precision_score(y[i], p[i]))

    def summarise(point: float, samples: np.ndarray) -> dict:
        lo, hi = np.percentile(samples, [2.5, 97.5])
        return {"estimate": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)]}

    out: dict = {}
    for name, p in preds.items():
        out[name] = {"roc_auc": summarise(roc_auc_score(y, p), np.array(draws[name]["roc_auc"])),
                     "pr_auc": summarise(average_precision_score(y, p), np.array(draws[name]["pr_auc"]))}
    a, b = "tabular_plus_text_gbm", "tabular_gbm"
    out[f"difference_{a}_minus_{b}"] = {
        m: summarise(out[a][m]["estimate"] - out[b][m]["estimate"],
                     np.array(draws[a][m]) - np.array(draws[b][m]))
        for m in ("roc_auc", "pr_auc")}
    return out


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = add_text_score(load()).df
    train, valid, test = (part(df, y) for y in (TRAIN_YEARS, VALID_YEARS, TEST_YEARS))
    y_valid, y_test = valid.stopped_early.to_numpy(), test.stopped_early.to_numpy()

    combined = fit_combined(df)
    p_valid = combined.predict_proba(valid[TEXT_FEATURES])[:, 1]
    p_test = combined.predict_proba(test[TEXT_FEATURES])[:, 1]

    p_cal, cal_summary = calibration(y_valid, p_valid, y_test, p_test)
    subgroups(test, p_test, p_cal).to_csv(ROOT / "reports" / "subgroup_metrics.csv", index=False)
    shap_importance(combined, test).to_csv(ROOT / "reports" / "shap_importance.csv", index=False)

    preds = {
        "logistic_regression": fit_logit(train).predict_proba(test[HONEST])[:, 1],
        "tabular_gbm": fit_gbm(train, valid, HONEST).predict_proba(test[HONEST])[:, 1],
        "tabular_plus_text_gbm": p_test,
        # leakage comparison only: these fields are overwritten after a trial ends
        "leaky_gbm": fit_gbm(train, valid, HONEST + LEAKY).predict_proba(test[HONEST + LEAKY])[:, 1],
    }
    results = {"n_test": int(len(test)), "base_rate_test": round(float(y_test.mean()), 4),
               "bootstrap_resamples": N_BOOT, **bootstrap(y_test, preds),
               "calibration": cal_summary}
    (ROOT / "reports" / "headline_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
