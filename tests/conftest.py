"""Make the project modules importable and give tests a way to run dbt macros in DuckDB."""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import jinja2
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src" / "models"), str(ROOT / "src" / "analysis")]


def render_macro(macro_file: str, call: str) -> str:
    """Render one dbt macro call to SQL with plain Jinja (the macros use no dbt-specific functions)."""
    source = (ROOT / "dbt" / "macros" / macro_file).read_text()
    return jinja2.Environment().from_string(source + "{{ " + call + " }}").render()


@pytest.fixture
def duck() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()  # in memory: tests never need data/trials.duckdb
