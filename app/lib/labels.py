"""Plain-English names for model features and registry codes."""
from __future__ import annotations

import math

PHASE = {"EARLY_PHASE1": "Early phase 1", "PHASE1": "Phase 1", "PHASE1/PHASE2": "Phase 1/2",
         "PHASE2": "Phase 2", "PHASE2/PHASE3": "Phase 2/3", "PHASE3": "Phase 3",
         "PHASE4": "Phase 4", "NA": "No phase (not a drug trial)", "NONE": "No phase given"}
SPONSOR = {"INDUSTRY": "Industry", "OTHER": "Academic / other", "NIH": "NIH",
           "FED": "US federal (non-NIH)", "OTHER_GOV": "Other government", "NETWORK": "Network",
           "INDIV": "Individual", "UNKNOWN": "Unknown"}
ALLOCATION = {"RANDOMIZED": "Randomised", "NON_RANDOMIZED": "Non-randomised",
              "NA": "Not applicable (single arm)", "MISSING": "Not stated"}
MASKING = {"NONE": "Open label", "SINGLE": "Single blind", "DOUBLE": "Double blind",
           "TRIPLE": "Triple blind", "QUADRUPLE": "Quadruple blind", "MISSING": "Not stated"}
PURPOSE = {"TREATMENT": "Treatment", "PREVENTION": "Prevention", "SUPPORTIVE_CARE": "Supportive care",
           "DIAGNOSTIC": "Diagnostic", "BASIC_SCIENCE": "Basic science", "SCREENING": "Screening",
           "HEALTH_SERVICES_RESEARCH": "Health services research",
           "DEVICE_FEASIBILITY": "Device feasibility", "OTHER": "Other", "MISSING": "Not stated"}
MODEL = {"PARALLEL": "Parallel groups", "SINGLE_GROUP": "Single group", "CROSSOVER": "Crossover",
         "FACTORIAL": "Factorial", "SEQUENTIAL": "Sequential", "MISSING": "Not stated"}
SEX = {"ALL": "All sexes", "FEMALE": "Female only", "MALE": "Male only", "MISSING": "Not stated"}
INTERVENTION = {"drug": "Drug", "device": "Device", "biological": "Biological", "procedure": "Procedure",
                "behavioral": "Behavioural", "radiation": "Radiation",
                "dietary_supplement": "Dietary supplement", "genetic": "Genetic",
                "diagnostic_test": "Diagnostic test", "combination_product": "Combination product",
                "other": "Other"}
STATUS = {"COMPLETED": "Completed", "TERMINATED": "Terminated", "WITHDRAWN": "Withdrawn",
          "SUSPENDED": "Suspended", "RECRUITING": "Recruiting",
          "ACTIVE_NOT_RECRUITING": "Active, not recruiting",
          "ENROLLING_BY_INVITATION": "Enrolling by invitation", "UNKNOWN": "Unknown"}

CODES = {"phase": PHASE, "lead_sponsor_class": SPONSOR, "allocation": ALLOCATION, "masking": MASKING,
         "primary_purpose": PURPOSE, "intervention_model": MODEL, "sex": SEX}

FEATURE = {
    "phase": "Phase", "allocation": "Allocation", "masking": "Masking",
    "primary_purpose": "Primary purpose", "intervention_model": "Study design",
    "lead_sponsor_class": "Lead sponsor type", "sex": "Sex eligible",
    "n_conditions": "Conditions listed", "n_intervention_types": "Intervention types",
    "min_age_years": "Minimum age", "max_age_years": "Maximum age",
    "criteria_chars": "Length of eligibility criteria", "criteria_bullets": "Eligibility criteria items",
    "sponsor_prior_registered": "Sponsor's earlier trials registered",
    "sponsor_prior_ended": "Sponsor's earlier trials with a known outcome",
    "sponsor_prior_terminated_rate": "Sponsor's past termination rate",
    "sponsor_prior_withdrawn_rate": "Sponsor's past withdrawal rate",
    "has_collaborator": "Has a collaborator", "has_industry_collaborator": "Industry collaborator",
    "has_nih_collaborator": "NIH collaborator", "healthy_volunteers": "Accepts healthy volunteers",
    "text_score": "Text score (title, summary, criteria)",
    **{f"intervention_{k}": f"{v} intervention" for k, v in INTERVENTION.items()},
}


def code(column: str, value: object) -> str:
    return CODES.get(column, {}).get(str(value), str(value))


def _missing(v: object) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def value_text(feature: str, value: object) -> str:
    """The value of one feature as a short phrase."""
    if feature in CODES:
        return code(feature, value)
    if _missing(value):
        if feature == "max_age_years":
            return "no upper limit"
        return "none known" if feature.startswith("sponsor_prior") else "not stated"
    if feature.startswith("intervention_") and feature != "intervention_model" or feature in (
            "has_collaborator", "has_industry_collaborator", "has_nih_collaborator", "healthy_volunteers"):
        return "yes" if float(value) >= 0.5 else "no"
    if feature.endswith("_rate"):
        return f"{100 * float(value):.0f}%"
    if feature == "text_score":
        return f"{float(value):.2f}"
    if feature in ("min_age_years", "max_age_years"):
        return f"{float(value):.0f} years"
    if feature == "criteria_chars":
        return f"{float(value):,.0f} characters"
    return f"{float(value):,.0f}"


def describe(feature: str, value: object) -> str:
    return f"{FEATURE.get(feature, feature)}: {value_text(feature, value)}"
