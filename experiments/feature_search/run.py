"""Run the feature search: 2 configurations x 3 validation years x 8 arms x 5 seeds.

    .venv/bin/python experiments/feature_search/run.py [--config main|early2005|all]

See README.md in this directory for the design and the pre-specified selection rule.
Writes results.csv, pooled.csv, run_info.json here and validation predictions to
data/feature_search/ (not committed).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import competition
import disease_areas
import metrics
from cohort import CACHE, CONFIGS, FEATURES, HERE, VALID_YEARS, gbm_params, load_focal, sponsor_groups
from text_score import fold_text_score, load_documents

SEEDS = (0, 1, 2, 3, 4)
N_THREADS = 4
MAX_TREES = 1500
NOISE = ["noise"]
C = disease_areas.COLUMNS
D = competition.D_COLUMNS
E = competition.E_COLUMNS
F = ["text_score"]
ARMS = {
    "A": ("baseline", []),
    "B": ("noise control", NOISE),
    "C": ("disease area", C),
    "D": ("competition P1", D),
    "E": ("competition P1 same country (exploratory)", E),
    "F": ("text score (exploratory)", F),
    "G": ("C + D", C + D),
    "H": ("C + D + E + F (exploratory)", C + D + E + F),
}
EXPLORATORY = {"E", "F", "H"}


def lgb_params(seed: int) -> dict:
    return dict(gbm_params(), random_state=seed, n_jobs=N_THREADS, deterministic=True,
                force_col_wise=True)


def choose_trees(pool: pd.DataFrame) -> tuple[int, float]:
    """Tree count for the baseline features: argmax of the mean AUC over 5 sponsor-grouped folds."""
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(
        np.arange(len(pool)), pool.y, groups=sponsor_groups(pool)))
    p = gbm_params()
    params = {"objective": "binary", "metric": "auc", "learning_rate": p["learning_rate"],
              "num_leaves": p["num_leaves"], "min_child_samples": p["min_child_samples"],
              "subsample": p["subsample"], "subsample_freq": p["subsample_freq"],
              "colsample_bytree": p["colsample_bytree"], "reg_lambda": p["reg_lambda"],
              "verbose": -1, "seed": 0, "num_threads": N_THREADS, "deterministic": True,
              "force_col_wise": True}
    data = lgb.Dataset(pool[FEATURES], pool.y, free_raw_data=False)
    res = lgb.cv(params, data, num_boost_round=MAX_TREES, folds=folds, stratified=False)
    curve = np.asarray(res["valid auc-mean"])
    best = int(np.argmax(curve)) + 1
    assert best < MAX_TREES, "tree count hit the ceiling"
    return best, float(curve[best - 1])


def features_for(df: pd.DataFrame, config: str) -> pd.DataFrame:
    """Arms C, D, E for every focal trial of the configuration (cached in data/)."""
    path = CACHE / f"features_{config}.parquet"
    if path.exists():
        feats = pd.read_parquet(path)
    else:
        comp = competition.build(df.nct_id.tolist())
        area = disease_areas.area_features(df).assign(nct_id=df.nct_id.to_numpy())
        feats = area.merge(comp, on="nct_id", validate="1:1")
        feats.to_parquet(path)
    out = df.merge(feats, on="nct_id", how="left", validate="1:1")
    assert len(out) == len(df) and (out.nct_id.to_numpy() == df.nct_id.to_numpy()).all()
    return out


def run_fold(df: pd.DataFrame, docs: pd.Series, year: int, lo: int, log) -> tuple[pd.DataFrame, dict]:
    pool = df[df.start_year.between(lo, year - 6)].reset_index(drop=True)
    valid = df[df.start_year == year].reset_index(drop=True)
    assert pool.start_year.max() <= year - 6 and (valid.start_year == year).all()
    info = {"valid_year": year, "train_years": f"{lo}-{year - 6}", "n_train": len(pool),
            "train_positives": int(pool.y.sum()), "n_valid": len(valid),
            "valid_positives": int(valid.y.sum())}
    t0 = time.time()
    info["n_trees"], info["cv_auc_at_n_trees"] = choose_trees(pool)
    log(f"  {year}: pool {len(pool)}, valid {len(valid)}, trees {info['n_trees']} "
        f"(cv auc {info['cv_auc_at_n_trees']:.4f}, {time.time() - t0:.0f}s)")
    t0 = time.time()
    pool["text_score"], valid["text_score"] = fold_text_score(pool, valid, docs)
    log(f"  {year}: text score {time.time() - t0:.0f}s")

    preds = pd.DataFrame({"nct_id": valid.nct_id, "y": valid.y.astype(int),
                          "phase2": (valid.phase.astype(str) == "PHASE2").to_numpy(),
                          "group": sponsor_groups(valid)})
    for arm, (_, extra) in ARMS.items():
        cols = FEATURES + extra
        p = np.zeros(len(valid))
        for seed in SEEDS:
            real = FEATURES + [c for c in extra if c != "noise"]
            X_tr, X_va = pool[real].copy(), valid[real].copy()
            if "noise" in extra:
                # one standard-normal column per seed, drawn for pool and validation rows
                rng = np.random.default_rng(1000 + seed)
                z = rng.standard_normal(len(pool) + len(valid))
                X_tr["noise"], X_va["noise"] = z[:len(pool)], z[len(pool):]
            X_tr, X_va = X_tr[cols], X_va[cols]
            model = lgb.LGBMClassifier(n_estimators=info["n_trees"], **lgb_params(seed))
            model.fit(X_tr, pool.y)
            p += model.predict_proba(X_va)[:, 1] / len(SEEDS)
        preds[arm] = p
    log(f"  {year}: arms fitted")
    return preds, info


def run_config(config: str, log) -> tuple[list[dict], list[dict], list[dict]]:
    lo = CONFIGS[config]
    df = features_for(load_focal(config), config)
    docs = load_documents(df.nct_id.tolist())
    frames, infos, rows = {}, [], []
    for year in VALID_YEARS:
        preds, info = run_fold(df, docs, year, lo, log)
        preds.to_parquet(CACHE / f"predictions_{config}_{year}.parquet")
        frames[year] = preds
        infos.append({"config": config, **info})
        res = metrics.fold_bootstrap(preds.y.to_numpy(), {a: preds[a].to_numpy() for a in ARMS},
                                     preds.phase2.to_numpy(), preds.group.to_numpy(), seed=year)
        for arm, r in res.items():
            rows.append({"config": config, "valid_year": year, "arm": arm, "arm_name": ARMS[arm][0],
                         "n": len(preds), "positives": int(preds.y.sum()),
                         "n_phase2": int(preds.phase2.sum()),
                         "positives_phase2": int(preds.y[preds.phase2].sum()),
                         "n_train": info["n_train"], "n_trees": info["n_trees"], **r})
        log(f"  {year}: bootstrap done")
    pooled = metrics.pooled_bootstrap(frames, list(ARMS), seed=1)
    prow = [{"config": config, "arm": a, "arm_name": ARMS[a][0], **r} for a, r in pooled.items()]
    return rows, prow, infos


def apply_rule(pooled: pd.DataFrame) -> pd.DataFrame:
    """The pre-specified selection rule (README), per configuration."""
    out = []
    for config, g in pooled.groupby("config", sort=False):
        noise = g.set_index("arm").loc["B"]
        noise_half = (noise.delta_roc_auc_hi - noise.delta_roc_auc_lo) / 2
        for _, r in g.iterrows():
            if r.arm in ("A", "B"):
                out.append("")
                continue
            ok = (r.delta_roc_auc_lo > 0 and r.delta_roc_auc_positive_years == 3
                  and r.delta_roc_auc > noise.delta_roc_auc + noise_half)
            out.append(("selected (exploratory)" if r.arm in EXPLORATORY else "selected")
                       if ok else "not selected")
    return pooled.assign(rule=out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="all", choices=["all", *CONFIGS])
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()
    configs = list(CONFIGS) if args.config == "all" else [args.config]
    start = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - start:6.0f}s] {msg}", flush=True)

    rows, prow, infos = [], [], []
    for config in configs:
        log(f"config {config}")
        r, p, i = run_config(config, log)
        rows += r
        prow += p
        infos += i
    results = pd.DataFrame(rows)
    pooled = apply_rule(pd.DataFrame(prow))
    for frame in (results, pooled):
        num = frame.select_dtypes("number").columns
        frame[num] = frame[num].round(4)
    out_dir = Path(args.out)
    results.to_csv(out_dir / "results.csv", index=False)
    pooled.to_csv(out_dir / "pooled.csv", index=False)
    (out_dir / "run_info.json").write_text(json.dumps(
        {"folds": infos, "seeds": SEEDS, "threads": N_THREADS, "max_trees": MAX_TREES,
         "lightgbm": lgb.__version__, "minutes": round((time.time() - start) / 60, 1)}, indent=2))
    log("done")


if __name__ == "__main__":
    main()
