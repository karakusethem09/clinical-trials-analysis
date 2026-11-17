"""The fixed-horizon label (dbt/macros/terminated_within.sql) on a tiny synthetic cohort."""
import pytest

from conftest import render_macro

ROWS = """
    (1, true,  'TERMINATED', date '2015-01-01', date '2017-06-01', null),               -- stopped in year 3
    (2, true,  'TERMINATED', date '2015-01-01', date '2021-06-01', null),               -- stopped in year 7
    (3, true,  'COMPLETED',  date '2015-01-01', date '2016-01-01', null),
    (4, true,  'RECRUITING', date '2015-01-01', null,              date '2026-10-01'),  -- still running
    (5, true,  'UNKNOWN',    date '2015-01-01', null,              date '2021-03-01'),  -- verified after year 5
    (6, true,  'UNKNOWN',    date '2015-01-01', null,              date '2017-03-01'),  -- verified before year 5
    (7, false, 'WITHDRAWN',  date '2015-01-01', null,              null),
    (8, true,  'TERMINATED', date '2023-01-01', date '2024-01-01', null),               -- under 5 years of follow-up
    (9, true,  'TERMINATED', date '2015-01-01', null,              null)                -- no end date
"""
EXPECTED = {
    "exclude":  [1, 0, 0, 0, None, None, None, None, None],
    "as_zero":  [1, 0, 0, 0, 0, 0, None, None, None],
    "verified": [1, 0, 0, 0, 0, None, None, None, None],
}


@pytest.mark.parametrize("unknown", list(EXPECTED))
def test_terminated_within_5_years(duck, unknown: str) -> None:
    label = render_macro("terminated_within.sql", f"terminated_within(5, unknown='{unknown}')")
    got = duck.execute(f"""
        select {label}
        from (values {ROWS}) t(id, started, overall_status, start_date, end_date, status_verified_date),
             (select date '2026-11-13' as pull_date)
        order by id
    """).fetchall()
    assert [r[0] for r in got] == EXPECTED[unknown]
