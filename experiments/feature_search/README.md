# Feature search: disease area, competition and text score

Does adding a feature group to the headline tabular model (`FEATURES` in
`src/models/train_tabular.py`) improve discrimination of termination within five years?
This directory is self-contained: it reads `data/trials.duckdb` read-only and changes
nothing in `models/`, `reports/`, `app/`, `dbt/` or `src/`.

## Design

**Rows.** The primary cohort (`terminated_5y` known, `registered_by_start`). Start years
2018-2020 are never selected, for any purpose. Start year 2021 is only touched by
`confirm.py`, which has not been run.

**Folds (label-knowable).** Validation years 2015, 2016 and 2017. For validation year Y the
model is fitted only on starts up to Y-6 (a start on 31 Dec Y-6 reaches its 5-year horizon
on 31 Dec Y-1), so every target-derived quantity (tree count, text model) uses only those
rows.

| Configuration | Training pool for Y = 2015 / 2016 / 2017 |
|---|---|
| `main` | starts 2008-2009 / 2008-2010 / 2008-2011, from the `trial_cohort` and `sponsor_history` marts |
| `early2005` | starts 2005-2009 / 2005-2010 / 2005-2011 |

`early2005` adds 2005-2007 starts by running copies of the compiled dbt models
(`sql/`, only the 2008 cut-off changed) against `stg_studies` read-only; `cohort.py` checks
that the copies reproduce the marts exactly for 2008-2017 (84,813 rows identical). In
2005-2007 about 91% of start-date types are missing and 93% of start dates give only a
month, so the "started" and "registered by start" rules are applied to much rougher dates
there, and only 1,878 / 3,713 / 4,391 trials per year pass the registration filter.

**Noise control.**
- Tree count: chosen once per fold for the baseline features, as the argmax (up to 1,500
  trees) of the mean ROC AUC over 5 sponsor-grouped folds of the training pool
  (`StratifiedGroupKFold`, seed 0); every arm in that fold uses it. Other hyperparameters
  are the repo's `gbm_params()`.
- Each arm is fitted with LightGBM seeds 0-4 and the predicted probabilities are averaged.
- Arm B adds one standard-normal column, a different one for each seed.
- Same rows in every arm; the baseline is refitted on exactly those rows.

**Arms** (each added to the baseline `FEATURES`):

| Arm | Features | Status |
|---|---|---|
| A | baseline | |
| B | + one random normal column | negative control |
| C | 19 therapeutic-area flags, `no_mapped_area`, `n_areas` (`disease_areas.py`) | |
| D | competition from first-posted dates: 12-month count, other-sponsor count, busiest-condition count, relative volume (`competition.py`) | |
| E | D's count restricted to overlapping countries (NaN without a country) + `geo_missing` | exploratory: country lists are the current site list |
| F | repo TF-IDF + logistic text score, text model fitted on the training pool only, sponsor-grouped out-of-fold on training rows (`text_score.py`) | exploratory: text is the current record |
| G | C + D | |
| H | C + D + E + F | exploratory |

**Metrics.** Per fold and arm: ROC AUC, PR AUC (average precision), Phase 2 ROC AUC
(`phase == 'PHASE2'` only), n, positives. Deltas against A with a paired sponsor-clustered
bootstrap (normalised lead sponsor resampled with replacement, 1,000 resamples, same
resample for all arms; same scheme as the repo's `bootstrap()` helper, checked against it).
Pooled: the three years are stacked, sponsors are resampled across them, and in each
resample the delta is the mean of the three per-year deltas.

## Pre-specified selection rule

Written before any arm was fitted. Applied separately to each configuration; `main` is the
primary configuration and `early2005` is a secondary one, reported alongside.

An arm is **selected** only if all three hold:
1. the 95% interval of its pooled ROC AUC delta against A lies entirely above zero;
2. its ROC AUC delta is positive in each of 2015, 2016 and 2017;
3. its pooled ROC AUC delta exceeds arm B's pooled delta by more than the half-width of
   B's pooled 95% interval.

Arms E, F and H can be **selected (exploratory)** at most. PR AUC and Phase 2 ROC AUC are
reported but do not enter the rule.

## How to run

```bash
.venv/bin/python experiments/feature_search/cohort.py          # check the 2005 rebuild
.venv/bin/python experiments/feature_search/check_mapper.py    # 40 random condition strings
.venv/bin/python experiments/feature_search/run.py             # both configurations
.venv/bin/python experiments/feature_search/determinism.py     # refit one fold, compare
```
