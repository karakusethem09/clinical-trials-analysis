# Will This Trial Finish?

Predicting early termination of clinical trials from what is known on the day a trial is registered, using public ClinicalTrials.gov data (~612k studies).

## Questions

1. **Why do trials stop?** Classify the free-text `whyStopped` field into categories (low enrollment, funding, safety, sponsor decision, ...) and describe the patterns by phase, sponsor type and disease area.
2. **Can we tell in advance?** Estimate the risk that a newly registered trial is terminated or withdrawn, using registration-time information only.

## Why it is hard: leakage

The registry shows the *current* state of each record. A terminated trial's `EnrollmentCount` is the actual number enrolled (often tiny), not the planned number, so a naive model "predicts" termination from its own consequence. Every feature here must be justified as known at registration.

## Layout

```
src/ingest/   download from the ClinicalTrials.gov API v2 (resumable)
dbt/          staging -> cleaned -> feature tables (DuckDB)
notebooks/    exploration and write-ups
src/models/   baselines, gradient boosting, survival models, evaluation
app/          risk explorer
data/         local only, not committed
```

## Plan

| Week | Work | Output |
|---|---|---|
| 1 | Full data pull, DuckDB + dbt staging models, data tests | Clean tables, data-quality notes |
| 2 | Cohort definition, label definition, leakage audit of every feature | Decision log |
| 3 | Baselines (logistic regression) and gradient boosting, temporal split | First honest metrics |
| 4 | Text features: embeddings of summary and eligibility criteria | Ablation: tabular vs. tabular + text |
| 5 | Survival analysis: time to termination with censoring | Risk curves |
| 6 | Calibration, subgroup performance, SHAP | Evaluation report |
| 7 | LLM classification of `whyStopped`, scored on a hand-labelled sample | Category analysis |
| 8 | Risk explorer app, scheduled refresh, final write-up | Public demo |

## Quick start

```bash
python src/ingest/fetch_studies.py --out data/raw --max-pages 2
```
