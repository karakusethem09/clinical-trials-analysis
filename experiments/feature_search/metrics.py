"""Metrics and the paired, sponsor-clustered bootstrap.

Resampling follows bootstrap() in src/models/train_tabular.py (used by evaluate.py):
clusters are the normalised lead sponsors (a trial without one is its own cluster), each
resample draws len(clusters) clusters with replacement from numpy's default_rng(seed),
and every arm is scored on the same resample. Instead of materialising the resampled
rows, each row gets a weight equal to how often it was drawn; weighted ROC AUC and
average precision are exactly the values of the replicated rows (checked against
sklearn in check_against_repo()).

Pooled result: the three validation years are stacked, sponsors are resampled across the
stacked data (a sponsor's trials in all three years move together), and in each resample
the arm-minus-baseline delta is the mean of the three per-year deltas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class Scorer:
    """Weighted ROC AUC / average precision for one fixed score vector, fast under reweighting."""

    def __init__(self, y: np.ndarray, p: np.ndarray):
        order = np.argsort(-p, kind="stable")          # descending score
        self.order = order
        ps = p[order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(ps)) + 1]   # tie groups
        self.y = y[order].astype(float)

    def _groups(self, w: np.ndarray):
        ws = w[self.order]
        pos = np.add.reduceat(ws * self.y, self.starts)
        neg = np.add.reduceat(ws * (1 - self.y), self.starts)
        return pos, neg

    def roc_auc(self, w: np.ndarray) -> float:
        pos, neg = self._groups(w)
        # descending order: a positive beats the negatives in all later (lower) groups
        neg_below = neg.sum() - np.cumsum(neg)
        return float((pos * (neg_below + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))

    def avg_precision(self, w: np.ndarray) -> float:
        pos, neg = self._groups(w)
        tp, fp = np.cumsum(pos), np.cumsum(neg)
        precision = tp / np.maximum(tp + fp, 1e-300)
        return float((pos / pos.sum() * precision).sum())


def cluster_index(groups: np.ndarray) -> tuple[np.ndarray, int]:
    codes = pd.factorize(np.asarray(groups))[0]
    return codes, int(codes.max()) + 1


def resample_weights(codes: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """Row weights for one cluster resample (same draws as train_tabular.bootstrap)."""
    draw = rng.integers(0, n_clusters, n_clusters)
    cluster_mult = np.bincount(draw, minlength=n_clusters).astype(float)
    return cluster_mult[codes]


METRICS = ("roc_auc", "pr_auc", "phase2_roc_auc")


def _scorers(y, preds: dict, phase2: np.ndarray) -> dict:
    out = {}
    for arm, p in preds.items():
        out[arm] = {"all": Scorer(y, p), "ph2": Scorer(y[phase2], p[phase2])}
    return out


def _metric_values(sc: dict, w: np.ndarray, phase2: np.ndarray) -> dict:
    return {"roc_auc": sc["all"].roc_auc(w), "pr_auc": sc["all"].avg_precision(w),
            "phase2_roc_auc": sc["ph2"].roc_auc(w[phase2])}


def fold_bootstrap(y, preds: dict, phase2, groups, base: str = "A", n_boot: int = 1000,
                   seed: int = 0) -> dict:
    """Point values and paired clustered 95% intervals of arm - base, one validation year."""
    y, phase2 = np.asarray(y), np.asarray(phase2, bool)
    sc = _scorers(y, preds, phase2)
    codes, k = cluster_index(groups)
    rng = np.random.default_rng(seed)
    ones = np.ones(len(y))
    point = {a: _metric_values(s, ones, phase2) for a, s in sc.items()}
    draws = {a: {m: np.empty(n_boot) for m in METRICS} for a in preds}
    for b in range(n_boot):
        w = resample_weights(codes, k, rng)
        for a, s in sc.items():
            for m, v in _metric_values(s, w, phase2).items():
                draws[a][m][b] = v
    out = {}
    for a in preds:
        out[a] = {}
        for m in METRICS:
            out[a][m] = point[a][m]
            if a != base:
                d = draws[a][m] - draws[base][m]
                out[a][f"delta_{m}"] = point[a][m] - point[base][m]
                out[a][f"delta_{m}_lo"], out[a][f"delta_{m}_hi"] = np.percentile(d, [2.5, 97.5])
    return out


def pooled_bootstrap(frames: dict[int, pd.DataFrame], arms: list[str], base: str = "A",
                     n_boot: int = 1000, seed: int = 1) -> dict:
    """Mean over years of per-year (arm - base), with sponsors resampled across the stacked years.

    frames[year] has columns y, phase2, group and one prediction column per arm."""
    years = sorted(frames)
    stacked = pd.concat([frames[t].assign(year=t) for t in years], ignore_index=True)
    codes, k = cluster_index(stacked.group.to_numpy())
    rows = {t: np.flatnonzero(stacked.year.to_numpy() == t) for t in years}
    sc, ph2 = {}, {}
    for t in years:
        f = stacked.iloc[rows[t]]
        ph2[t] = f.phase2.to_numpy(bool)
        sc[t] = _scorers(f.y.to_numpy(), {a: f[a].to_numpy() for a in arms}, ph2[t])

    def values(w):  # arm -> metric -> per-year array
        return {a: {m: np.array([_metric_values(sc[t][a], w[rows[t]], ph2[t])[m] for t in years])
                    for m in METRICS} for a in arms}

    point = values(np.ones(len(stacked)))
    rng = np.random.default_rng(seed)
    draws = {a: {m: np.empty(n_boot) for m in METRICS} for a in arms if a != base}
    level = {a: {m: np.empty(n_boot) for m in METRICS} for a in arms}
    for b in range(n_boot):
        v = values(resample_weights(codes, k, rng))
        for a in arms:
            for m in METRICS:
                level[a][m][b] = v[a][m].mean()
                if a != base:
                    draws[a][m][b] = (v[a][m] - v[base][m]).mean()
    out = {}
    for a in arms:
        out[a] = {}
        for m in METRICS:
            out[a][m] = float(point[a][m].mean())
            out[a][f"{m}_lo"], out[a][f"{m}_hi"] = np.percentile(level[a][m], [2.5, 97.5])
            if a != base:
                per_year = point[a][m] - point[base][m]
                out[a][f"delta_{m}"] = float(per_year.mean())
                out[a][f"delta_{m}_lo"], out[a][f"delta_{m}_hi"] = np.percentile(draws[a][m], [2.5, 97.5])
                out[a][f"delta_{m}_by_year"] = [round(float(x), 4) for x in per_year]
                out[a][f"delta_{m}_positive_years"] = int((per_year > 0).sum())
    return out


def check_against_repo(y, p_base, p_arm, groups, seed: int = 0, n_boot: int = 200) -> dict:
    """Compare the weighted shortcut with the repo helper on the same resamples."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    from train_tabular import bootstrap
    y = np.asarray(y)
    repo = bootstrap(y, {"base": p_base, "arm": p_arm}, n_boot=n_boot, seed=seed,
                     pairs=(("arm", "base"),), groups=groups)
    mine = fold_bootstrap(y, {"base": p_base, "arm": p_arm}, np.ones(len(y), bool), groups,
                          base="base", n_boot=n_boot, seed=seed)
    w = resample_weights(*cluster_index(groups), np.random.default_rng(5))
    s = Scorer(y, p_arm)
    return {"repo_delta_roc_ci": repo["difference_arm_minus_base"]["roc_auc"]["ci95"],
            "this_delta_roc_ci": [round(float(mine["arm"]["delta_roc_auc_lo"]), 4),
                                  round(float(mine["arm"]["delta_roc_auc_hi"]), 4)],
            "repo_delta_pr_ci": repo["difference_arm_minus_base"]["pr_auc"]["ci95"],
            "this_delta_pr_ci": [round(float(mine["arm"]["delta_pr_auc_lo"]), 4),
                                 round(float(mine["arm"]["delta_pr_auc_hi"]), 4)],
            "weighted_vs_sklearn_roc": [s.roc_auc(w), roc_auc_score(y, p_arm, sample_weight=w)],
            "weighted_vs_sklearn_ap": [s.avg_precision(w), average_precision_score(y, p_arm, sample_weight=w)]}
