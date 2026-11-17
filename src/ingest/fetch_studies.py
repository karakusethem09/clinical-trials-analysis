"""Download study records from the ClinicalTrials.gov API v2 into gzipped JSONL pages.

Usage:
    python src/ingest/fetch_studies.py --out data/raw --max-pages 2      # smoke test
    python src/ingest/fetch_studies.py --out data/raw                    # full pull (~612k studies)

The pull is resumable: the next page token is stored in <out>/_state.json.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://clinicaltrials.gov/api/v2/studies"

# Only what the project needs; full records average 17 KB each.
FIELDS = [
    "NCTId", "BriefTitle", "OfficialTitle", "BriefSummary",
    "OverallStatus", "WhyStopped",
    "StudyFirstSubmitDate", "StartDate", "PrimaryCompletionDate", "CompletionDate",
    "LastUpdateSubmitDate",
    # Actual vs. estimated: planned start dates of withdrawn trials never happened.
    "StartDateType", "PrimaryCompletionDateType", "CompletionDateType",
    "StudyFirstPostDate", "StudyFirstSubmitQCDate", "LastUpdatePostDate",
    "StatusVerifiedDate", "ResultsFirstSubmitDate",
    "StudyType", "Phase", "EnrollmentCount", "EnrollmentType",
    "DesignAllocation", "DesignMasking", "DesignPrimaryPurpose", "DesignInterventionModel",
    "LeadSponsorName", "LeadSponsorClass", "CollaboratorClass",
    "Condition", "InterventionType",
    "EligibilityCriteria", "Sex", "MinimumAge", "MaximumAge", "HealthyVolunteers",
    "LocationCountry",
]


def fetch_page(token: str | None, page_size: int) -> dict:
    params = {"pageSize": page_size, "fields": ",".join(FIELDS), "format": "json"}
    if token:
        params["pageToken"] = token
    url = f"{API}?{urllib.parse.urlencode(params)}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.load(resp)
        except Exception:  # network hiccup or rate limit: back off and retry
            if attempt == 4:
                raise
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--page-size", type=int, default=1000)
    ap.add_argument("--max-pages", type=int, default=None)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    state_path = args.out / "_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"page": 0, "token": None, "done": False}
    if state["done"]:
        print("Already complete; delete _state.json to start over.")
        return

    pages_this_run = 0
    while args.max_pages is None or pages_this_run < args.max_pages:
        payload = fetch_page(state["token"], args.page_size)
        studies = payload.get("studies", [])
        path = args.out / f"studies_{state['page']:05d}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for s in studies:
                f.write(json.dumps(s) + "\n")
        state["page"] += 1
        state["token"] = payload.get("nextPageToken")
        state["done"] = state["token"] is None
        state_path.write_text(json.dumps(state))
        pages_this_run += 1
        print(f"page {state['page']:>4}: {len(studies)} studies -> {path.name}")
        if state["done"]:
            break
        time.sleep(0.3)


if __name__ == "__main__":
    main()
