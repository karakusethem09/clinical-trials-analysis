"""The SQL macro is the only implementation of the sponsor-name rule; test it directly."""
import pytest

from conftest import render_macro

CASES = [
    ("Novartis Pharmaceuticals", "novartis"),
    ("Novartis", "novartis"),
    ("Eisai Co., Ltd.", "eisai"),
    ("Eisai Inc.", "eisai"),
    ("Merck Sharp & Dohme LLC", "merck sharp and dohme"),
    ("Eli Lilly and Company", "eli lilly"),
    ("Janssen Research & Development, LLC", "janssen"),
    ("Hoffmann-La Roche", "roche"),
    ("Seagen, a wholly owned subsidiary of Pfizer", "seagen"),
    ("The University of Texas Health Science Center", "university of texas health science center"),
    ("Genentech, Inc.", "genentech"),
    ("Université de Montréal", "universite de montreal"),
]


@pytest.mark.parametrize("raw, expected", CASES)
def test_normalise_sponsor(duck, raw: str, expected: str) -> None:
    sql = render_macro("normalise_sponsor.sql", "normalise_sponsor('name')")
    assert duck.execute(f"select {sql} from (select ? as name)", [raw]).fetchone()[0] == expected


def test_null_and_punctuation_only(duck) -> None:
    sql = render_macro("normalise_sponsor.sql", "normalise_sponsor('name')")
    assert duck.execute(f"select {sql} from (select null::varchar as name)").fetchone()[0] is None
    assert duck.execute(f"select {sql} from (select ' ., ' as name)").fetchone()[0] is None
