"""Regenerate every file in app/data/ from data/trials.duckdb and the saved model.

The dashboard runs on Streamlit Community Cloud, where the DuckDB file and the raw
pull are not available, so everything it needs beyond models/ and reports/ is
precomputed here into small parquet and JSON files.

Writes
  app/data/trials.parquet          details of the 3,000 scored test trials in
                                   models/dashboard_sample.parquet (features, status,
                                   reason for stopping); the SHAP values stay in models/
  app/data/risk_distribution.json  calibrated risk of the saved model on all test trials
                                   (2018-2020): percentiles, histogram, observed rates
  app/data/calibration.parquet     reliability deciles of the saved model, raw and isotonic
  app/data/whatif_defaults.json    typical feature values for the what-if form
  app/data/survival_curves.parquet Aalen-Johansen cumulative incidence of termination
                                   (completion as competing risk) overall, by phase and
                                   by sponsor class, plus 1 - Kaplan-Meier overall
  app/data/leakage_evidence.json   the numbers behind each trap on the leakage page
  app/data/why_*.parquet|json      why_stopped categories (src/analysis/why_stopped.py)

Run from the repository root:
  .venv/bin/python src/analysis/build_app_data.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "analysis"))

from lifelines import AalenJohansenFitter, KaplanMeierFitter  # noqa: E402

import train_survival  # noqa: E402
import why_stopped  # noqa: E402
from evaluate import ece, reliability  # noqa: E402
from train_tabular import BOOLEAN, CATEGORICAL, NUMERIC, SPLITS, load, part  # noqa: E402
from train_text import documents, raw_text  # noqa: E402

DB = ROOT / "data" / "trials.duckdb"
OUT = ROOT / "app" / "data"
INTERVENTIONS = [b.removeprefix("intervention_") for b in BOOLEAN if b.startswith("intervention_")]


def text_for(ids: list[str]) -> pd.Series:
    """Masked text exactly as train_text builds it, for the given trials only."""
    docs = documents(raw_text(ids)).set_index("nct_id").text
    return docs.reindex(ids).fillna("")


def score_test(bundle: dict) -> pd.DataFrame:
    """All primary test trials with the saved model's raw and calibrated risk (and text score, if used)."""
    test = part(load(), "test").reset_index(drop=True)
    if bundle.get("text_model") is not None:
        test["text_score"] = bundle["text_model"].predict_proba(text_for(test.nct_id.tolist()))[:, 1]
    X = test[bundle["features"]]
    test["p_raw"] = bundle["model"].predict_proba(X)[:, 1]
    test["p_calibrated"] = bundle["calibrator"].predict(test.p_raw.to_numpy())
    return test


def trial_details(test: pd.DataFrame) -> pd.DataFrame:
    sample = pd.read_parquet(ROOT / "models" / "dashboard_sample.parquet")
    merged = sample[["nct_id", "p_raw"]].merge(test, on="nct_id", suffixes=("_saved", ""))
    gap = float(np.abs(merged.p_raw - merged.p_raw_saved).max())
    assert len(merged) == len(sample) and gap < 1e-6, f"rescored sample differs from saved one ({gap})"

    con = duckdb.connect(str(DB), read_only=True)
    con.register("ids", sample[["nct_id"]])
    extra = con.sql("""
        select nct_id, lead_sponsor_name, array_to_string(conditions, '; ') as conditions_text,
               start_date, end_date, why_stopped
        from trial_cohort join ids using (nct_id)
    """).df()
    cols = ["nct_id", "overall_status"] + CATEGORICAL + NUMERIC + BOOLEAN + (
        ["text_score"] if "text_score" in test else [])
    out = merged[cols].merge(extra, on="nct_id")
    for c in CATEGORICAL:
        out[c] = out[c].astype(str)
    out["why_category"] = [why_stopped.categorise(r)[0] if isinstance(r, str) and r.strip() else None
                           for r in out.why_stopped]
    return out


def risk_distribution(test: pd.DataFrame) -> dict:
    p, y = test.p_calibrated.to_numpy(), test.y.to_numpy()
    edges = np.round(np.arange(0, 0.42, 0.02), 2)
    idx = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
    hist = [{"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": int((idx == i).sum()),
             "terminated": int(y[idx == i].sum())} for i in range(len(edges) - 1)]
    order = np.argsort(-p, kind="stable")
    k = int(round(0.1 * len(p)))
    # the isotonic calibration is a step function, so calibrated risk takes few distinct values
    levels = (pd.DataFrame({"p": p.round(6), "y": y}).groupby("p").y.agg(["size", "sum"]).reset_index())
    values = [{"p": float(r.p), "n": int(r["size"]), "terminated": int(r["sum"])} for _, r in levels.iterrows()]
    return {"n": int(len(p)), "base_rate": float(y.mean()), "values": values,
            "percentiles": np.percentile(p, np.arange(101)).round(5).tolist(),
            "histogram": hist,
            "top_decile_threshold": float(p[order[k - 1]]),
            "top_decile_observed": float(y[order[:k]].mean()),
            "bottom_half_observed": float(y[p <= np.median(p)].mean())}


def calibration_table(test: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    y = test.y.to_numpy()
    rows, summary = [], {}
    for name, col in (("Raw score", "p_raw"), ("Isotonic (fitted on 2017)", "p_calibrated")):
        r = reliability(y, test[col].to_numpy()).assign(model=name, decile=range(1, 11))
        rows.append(r)
        summary[name] = {"ece": round(ece(y, test[col].to_numpy()), 4),
                         "mean_predicted": round(float(test[col].mean()), 4)}
    summary["observed_rate"] = round(float(y.mean()), 4)
    return pd.concat(rows, ignore_index=True), summary


def whatif_defaults(test: pd.DataFrame) -> dict:
    with_history = test[test.sponsor_prior_ended > 0]
    out = {
        "numeric_median": {c: float(test[c].median()) for c in NUMERIC},
        "sponsor_typical": {c: float(with_history[c].median()) for c in
                            ("sponsor_prior_registered", "sponsor_prior_ended",
                             "sponsor_prior_terminated_rate", "sponsor_prior_withdrawn_rate")},
        "share_without_history": float((test.sponsor_prior_ended == 0).mean()),
    }
    if "text_score" in test:
        out["text_score_quantiles"] = {str(q): float(test.text_score.quantile(q / 100))
                                       for q in (10, 25, 50, 75, 90)}
    return out


def survival_curves() -> pd.DataFrame:
    df, _ = train_survival.load()
    df["phase_group"] = df.phase.map(train_survival.PHASE_GROUPS)
    df["sponsor_group"] = df.lead_sponsor_class.map(train_survival.SPONSOR_GROUPS)
    grid = np.round(np.arange(0, 10.001, 0.05), 2)

    warnings.filterwarnings("ignore", message="Tied event times")

    def aj(g: pd.DataFrame) -> np.ndarray:
        f = AalenJohansenFitter(calculate_variance=False, jitter_level=1e-5, seed=0)
        f.fit(g.years, g.event, event_of_interest=train_survival.TERMINATED)
        cif = f.cumulative_density_.iloc[:, 0]
        return np.interp(grid, cif.index.to_numpy(), cif.to_numpy(), left=0.0)

    parts = [pd.DataFrame({"dimension": "overall", "group": "Aalen-Johansen (completion as competing risk)",
                           "years": grid, "incidence": aj(df), "n": len(df)})]
    km = KaplanMeierFitter().fit(df.years, df.event == train_survival.TERMINATED)
    parts.append(pd.DataFrame({"dimension": "overall", "group": "1 - Kaplan-Meier (completion censored)",
                               "years": grid, "incidence": 1 - km.survival_function_at_times(grid).to_numpy(),
                               "n": len(df)}))
    for dim, col in (("phase", "phase_group"), ("sponsor", "sponsor_group")):
        for name, g in df.groupby(col):
            if len(g) >= 1000:
                parts.append(pd.DataFrame({"dimension": dim, "group": name, "years": grid,
                                           "incidence": aj(g), "n": len(g)}))
    return pd.concat(parts, ignore_index=True)


def leakage_evidence() -> dict:
    con = duckdb.connect(str(DB), read_only=True)
    legacy = "is_resolved and registered_prospectively and start_year between 2008 and 2019"
    enrol = con.sql(f"""
        select overall_status, count(*) as n,
               median(leaky_enrollment_count) as median_enrollment,
               median(leaky_n_locations) as median_sites,
               avg((coalesce(leaky_n_locations, 0) = 0)::int) as share_no_sites
        from trial_cohort where {legacy} group by 1 order by 1
    """).df()
    start_type = con.sql("""
        select overall_status, count(*) as n,
               avg((start_date_type = 'ESTIMATED')::int) as share_estimated,
               avg((start_date_type is null)::int) as share_untyped,
               avg(registered_prospectively::int) as share_registered_by_start_21d,
               median(days_registered_after_start) as median_days_registered_after_start
        from trial_cohort
        where start_year between 2008 and 2020 and overall_status in ('COMPLETED', 'TERMINATED', 'WITHDRAWN')
        group by 1 order by 1
    """).df()
    by_year = con.sql("""
        select start_year,
               avg(stopped_early) filter (where is_resolved and registered_prospectively) as old_label_rate,
               avg(terminated_5y) filter (where registered_by_start) as primary_label_rate,
               avg((overall_status in ('RECRUITING', 'ACTIVE_NOT_RECRUITING', 'ENROLLING_BY_INVITATION',
                                       'SUSPENDED'))::int) filter (where registered_by_start and started)
                   as share_still_running
        from trial_cohort where start_year between 2008 and 2020
        group by 1 order by 1
    """).df()
    return {"enrollment_by_status": enrol.round(4).to_dict("records"),
            "start_date_by_status": start_type.round(4).to_dict("records"),
            "label_rate_by_start_year": by_year.round(4).to_dict("records")}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("why_stopped categories ...")
    why_stopped.main()

    print("scoring the test years with the saved model ...")
    bundle = joblib.load(ROOT / "models" / "final_model.joblib")
    test = score_test(bundle)
    trial_details(test).to_parquet(OUT / "trials.parquet", index=False)
    (OUT / "risk_distribution.json").write_text(json.dumps(risk_distribution(test), indent=2))
    cal, cal_summary = calibration_table(test)
    cal.to_parquet(OUT / "calibration.parquet", index=False)
    defaults = whatif_defaults(test)
    defaults["calibration_summary"] = cal_summary
    (OUT / "whatif_defaults.json").write_text(json.dumps(defaults, indent=2))

    print("survival curves ...")
    survival_curves().to_parquet(OUT / "survival_curves.parquet", index=False)
    print("leakage evidence ...")
    (OUT / "leakage_evidence.json").write_text(json.dumps(leakage_evidence(), indent=2, default=str))
    total = sum(f.stat().st_size for f in OUT.iterdir())
    print(f"done: {len(list(OUT.iterdir()))} files, {total / 1e6:.2f} MB in {OUT}")


if __name__ == "__main__":
    main()
