"""Focal trials for the feature search, read-only from data/trials.duckdb.

Two training-pool configurations share the validation years 2015-2017:
  main      : starts 2008-2017 straight from the trial_cohort and sponsor_history marts
  early2005 : the same plus starts 2005-2007, rebuilt by running copies of the compiled
              dbt models (sql/) against stg_studies with the start cut-off moved to 2005.
              The copies are checked to reproduce the marts row for row on 2008-2017.

Start years 2018-2020 are never selected here (they are the project's test years). Start
year 2021 is only read by confirm.py, which is not run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src" / "models"))

from train_tabular import BOOLEAN, CATEGORICAL, DB, FEATURES, gbm_params, typed  # noqa: E402,F401

CACHE = ROOT / "data" / "feature_search"
CACHE.mkdir(parents=True, exist_ok=True)

VALID_YEARS = (2015, 2016, 2017)
CONFIGS = {"main": 2008, "early2005": 2005}
SPONSOR_COLS = ["sponsor_prior_registered", "sponsor_prior_ended",
                "sponsor_prior_terminated_rate", "sponsor_prior_withdrawn_rate"]
KEEP = (["nct_id", "y", "start_year", "start_date", "first_submit_date", "lead_sponsor_key",
         "lead_sponsor_name", "conditions"] + [c for c in FEATURES if c not in SPONSOR_COLS])
FORBIDDEN_YEARS = (2018, 2019, 2020)


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(DB), read_only=True)
    con.sql("set threads = 4")
    con.sql("set enable_progress_bar = false")
    return con


def _sql(name: str) -> str:
    return (HERE / "sql" / name).read_text()


def _create_extended(con: duckdb.DuckDBPyConnection, min_start: str) -> None:
    con.sql("create or replace temp table ext_cohort as "
            + _sql("extended_cohort.sql").replace("{min_start}", min_start))
    con.sql("create or replace temp table ext_sponsor as " + _sql("extended_sponsor_history.sql"))


def _select(source_cohort: str, source_sponsor: str, where: str) -> str:
    cols = ", ".join(f"c.{c}" for c in KEEP if c != "y")
    sp = ", ".join(f"s.{c}" for c in SPONSOR_COLS)
    return f"""select {cols}, c.terminated_5y as y, {sp}
               from {source_cohort} c join {source_sponsor} s using (nct_id)
               where c.terminated_5y is not null and c.registered_by_start and {where}
               order by c.nct_id"""


def load_focal(config: str = "main", max_year: int = 2017) -> pd.DataFrame:
    """Cohort-eligible trials (primary label known, registered by start) with start year in
    [CONFIGS[config], max_year]; never 2018-2020."""
    assert max_year <= 2017
    lo = CONFIGS[config]
    con = connect()
    where = f"c.start_year between {lo} and {max_year}"
    if lo >= 2008:
        df = con.sql(_select("trial_cohort", "sponsor_history", where)).df()
    else:
        _create_extended(con, f"{lo}-01-01")
        df = con.sql(_select("ext_cohort", "ext_sponsor", where)).df()
    assert not df.start_year.isin(FORBIDDEN_YEARS).any()
    return typed(df)


def check_extended_matches_marts() -> dict:
    """The rebuilt cohort and sponsor history must equal the marts on 2008-2017."""
    con = connect()
    _create_extended(con, "2005-01-01")
    where = "c.start_year between 2008 and 2017"
    a = con.sql(_select("trial_cohort", "sponsor_history", where)).df()
    b = con.sql(_select("ext_cohort", "ext_sponsor", where)).df()
    cols = [c for c in a.columns if c != "conditions"]
    pd.testing.assert_frame_equal(a[cols], b[cols])
    assert (a.conditions.map(list) == b.conditions.map(list)).all()
    early = con.sql("""select start_year, count(*) n_interventional,
                              count_if(terminated_5y is not null) label_known,
                              count_if(terminated_5y is not null and registered_by_start) eligible,
                              avg((start_date_type is null)::int) share_start_type_missing,
                              avg((start_date_precision = 'MONTH')::int) share_month_only
                       from ext_cohort where start_year between 2005 and 2007
                       group by 1 order by 1""").df()
    return {"rows_compared_2008_2017": int(len(a)), "identical": True,
            "early_years": early.round(4).to_dict("records")}


def sponsor_groups(df: pd.DataFrame) -> np.ndarray:
    """Normalised lead sponsor; a trial without one is its own group (as in train_text.py)."""
    return df.lead_sponsor_key.fillna("trial:" + df.nct_id).to_numpy()


if __name__ == "__main__":
    import json
    print(json.dumps(check_extended_matches_marts(), indent=2, default=str))
