import pytest
from why_stopped import OTHER, categorise

CASES = [
    ("Slow accrual", "Slow or low enrollment"),
    ("Low recruitment rate", "Slow or low enrollment"),
    ("Sponsor decision", "Sponsor, investigator or business decision"),
    ("Business reasons", "Sponsor, investigator or business decision"),
    ("Lack of funding", "Funding"),
    ("Futility at interim analysis", "Lack of efficacy or futility"),
    ("Safety concerns", "Safety or adverse events"),
    ("Terminated due to the COVID-19 pandemic", "COVID-19"),
    ("PI left the institution", "Investigator, staff or site logistics"),
    ("Drug supply issues", "Drug or device supply"),
    ("IRB approval was not obtained", "Regulatory or ethics"),
    ("Study replaced by a new protocol", "Design changed or superseded"),
]


@pytest.mark.parametrize("reason, category", CASES)
def test_primary_category(reason: str, category: str) -> None:
    assert categorise(reason)[0] == category


def test_negated_safety_is_not_safety() -> None:
    assert "Safety or adverse events" not in categorise("Not due to safety reasons; sponsor decision")


def test_empty_reason_is_other() -> None:
    assert categorise("") == [OTHER] and categorise(None) == [OTHER]
