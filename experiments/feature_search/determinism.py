"""Determinism check: refit one fold from scratch and compare with the saved predictions.

All randomness is seeded: StratifiedGroupKFold(random_state=0) for the tree-count CV and
the text folds, LightGBM seeds 0-4 with deterministic=True and force_col_wise=True on a
fixed thread count, noise columns from default_rng(1000 + seed), bootstrap rngs seeded by
the validation year (per fold) and 1 (pooled); rows are ordered by nct_id.

    .venv/bin/python experiments/feature_search/determinism.py [--config main] [--year 2015]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from cohort import CACHE, CONFIGS, HERE, load_focal
from run import ARMS, features_for, run_fold
from text_score import load_documents

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="main", choices=list(CONFIGS))
ap.add_argument("--year", type=int, default=2015)
args = ap.parse_args()

saved = pd.read_parquet(CACHE / f"predictions_{args.config}_{args.year}.parquet")
df = features_for(load_focal(args.config), args.config)
again, info = run_fold(df, load_documents(df.nct_id.tolist()), args.year, CONFIGS[args.config], print)
diff = {a: float(np.max(np.abs(saved[a].to_numpy() - again[a].to_numpy()))) for a in ARMS}
result = {"config": args.config, "valid_year": args.year, "n_trees_rerun": info["n_trees"],
          "max_abs_prediction_difference": diff,
          "identical": bool(all(v == 0 for v in diff.values())
                            and (saved.nct_id.to_numpy() == again.nct_id.to_numpy()).all())}
(HERE / "determinism.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
