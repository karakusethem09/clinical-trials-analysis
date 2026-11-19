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
# confirmation, NOT run yet: cd experiments/feature_search && python -c "from confirm import confirm; confirm('F')"
```

Runtime about 10 minutes on 4 cores (nothing else was running). Outputs: `results.csv`
(per configuration x year x arm), `pooled.csv` (per configuration x arm, with the rule
column), `run_info.json` (training sizes, tree counts), `determinism.json`,
`mapper_check_sample.csv`; predictions in `data/feature_search/` (not committed).

## Results

Training sizes and tree counts (sponsor-grouped CV on the training pool):

| Config | Year | Train years | n train | Trees | n valid | Positives | Phase 2 n |
|---|---|---|---|---|---|---|---|
| main | 2015 | 2008-2009 | 12,432 | 81 | 10,097 | 1,105 | 1,644 |
| main | 2016 | 2008-2010 | 19,502 | 64 | 10,555 | 1,132 | 1,606 |
| main | 2017 | 2008-2011 | 26,932 | 81 | 11,391 | 1,283 | 1,755 |
| early2005 | 2015 | 2005-2009 | 22,414 | 72 | 10,097 | 1,105 | 1,644 |
| early2005 | 2016 | 2005-2010 | 29,484 | 107 | 10,555 | 1,132 | 1,606 |
| early2005 | 2017 | 2005-2011 | 36,914 | 123 | 11,391 | 1,283 | 1,755 |

Pooled over 2015-2017, configuration `main` (delta = mean of per-year deltas vs A,
sponsor-clustered paired 95% interval):

| Arm | ROC AUC | Delta | 95% CI | Per-year delta 2015/16/17 | PR AUC (delta) | Phase 2 ROC AUC | Phase 2 delta (95% CI) | Rule |
|---|---|---|---|---|---|---|---|---|
| A baseline | 0.6484 | | | | 0.1682 | 0.5772 | | |
| B noise | 0.6496 | +0.0013 | -0.0003, +0.0028 | +0.0029 / +0.0010 / -0.0001 | 0.1688 (+0.0006) | 0.5791 | +0.0019 (-0.0027, +0.0068) | control |
| C area | 0.6506 | +0.0023 | -0.0003, +0.0051 | +0.0032 / +0.0024 / +0.0013 | 0.1707 (+0.0025) | 0.5832 | +0.0060 (-0.0020, +0.0149) | not selected |
| D competition | 0.6477 | -0.0007 | -0.0031, +0.0019 | -0.0009 / +0.0027 / -0.0038 | 0.1682 (-0.0000) | 0.5781 | +0.0010 (-0.0073, +0.0099) | not selected |
| E same country | 0.6485 | +0.0002 | -0.0021, +0.0027 | -0.0017 / +0.0008 / +0.0014 | 0.1701 (+0.0019) | 0.5783 | +0.0011 (-0.0069, +0.0098) | not selected |
| F text | 0.6561 | +0.0078 | +0.0017, +0.0141 | +0.0051 / +0.0170 / +0.0012 | 0.1719 (+0.0037) | 0.5977 | +0.0205 (+0.0076, +0.0348) | selected (exploratory) |
| G C+D | 0.6497 | +0.0014 | -0.0019, +0.0050 | +0.0024 / +0.0036 / -0.0019 | 0.1700 (+0.0017) | 0.5842 | +0.0071 (-0.0023, +0.0185) | not selected |
| H all | 0.6582 | +0.0099 | +0.0030, +0.0171 | +0.0083 / +0.0189 / +0.0024 | 0.1744 (+0.0061) | 0.5976 | +0.0204 (+0.0048, +0.0387) | selected (exploratory) |

Rule threshold for condition 3 in `main`: B's delta + B's half-width = 0.0013 + 0.0016 = 0.0028.
C (+0.00229) has 3/3 positive years but fails conditions 1 and 3; D, E and G fail all three.

Configuration `early2005` (secondary; rule threshold 0.0009 + 0.0015 = 0.0024):

| Arm | ROC AUC | Delta | 95% CI | Per-year delta | Phase 2 delta (95% CI) | Rule |
|---|---|---|---|---|---|---|
| A | 0.6514 | | | | | |
| B | 0.6523 | +0.0009 | -0.0005, +0.0024 | +0.0003 / +0.0025 / -0.0000 | -0.0036 (-0.0085, +0.0008) | control |
| C | 0.6537 | +0.0023 | -0.0005, +0.0051 | +0.0047 / +0.0033 / -0.0011 | +0.0052 (-0.0032, +0.0137) | not selected |
| D | 0.6517 | +0.0003 | -0.0021, +0.0027 | -0.0003 / +0.0024 / -0.0012 | -0.0016 (-0.0103, +0.0076) | not selected |
| E | 0.6537 | +0.0024 | +0.0001, +0.0049 | +0.0018 / +0.0031 / +0.0022 | +0.0005 (-0.0076, +0.0080) | not selected (fails 3 by 0.000002: delta 0.002367 vs threshold 0.002369) |
| F | 0.6598 | +0.0084 | +0.0017, +0.0158 | +0.0072 / +0.0146 / +0.0034 | +0.0296 (+0.0142, +0.0456) | selected (exploratory) |
| G | 0.6543 | +0.0030 | -0.0002, +0.0061 | +0.0060 / +0.0039 / -0.0009 | +0.0026 (-0.0086, +0.0147) | not selected |
| H | 0.6620 | +0.0106 | +0.0037, +0.0185 | +0.0117 / +0.0170 / +0.0031 | +0.0243 (+0.0069, +0.0407) | selected (exploratory) |

**Summary.** No non-exploratory arm (C, D, G) passes the rule in either configuration.
Only the text score (F) and the combination containing it (H) pass, and only as
exploratory. The noise control itself moves ROC AUC by +0.001 on average (+0.003 in 2015),
which is the scale of the C and D effects.

## Checks

- **Mapper, 40 random condition strings** (`mapper_check_sample.csv`, seed 20261119). The
  pilot mapper had two clear errors in the sample: "Pregnant Women Undergoing Cesarean
  Delivery" went to gastro ("liver" inside "delivery") and "Neurologic Disorder" went to
  renal/urology ("urolog" inside "neurologic") and not to neurology; "Stasis Ulcer" also went
  to gastro. Omissions: abscess, alveolar ridge (dental), deafness, lymphangiomatosis, quality
  improvement and other non-disease strings. A second sample (seed 7) showed "transthyretin"
  going to ophthalmology ("retin"). All listed errors were fixed before any model was fitted
  (see `disease_areas.py`); 12.9% of focal trials still map to no area. There is no ENT or
  rare-disease area, and atopic dermatitis goes to immunology, not dermatology.
- **Competition features** were recomputed independently in pandas for 4 random focal
  trials: all counts and ratios match.
- **2005 rebuild**: identical to the marts on 84,813 rows of 2008-2017.
- **Bootstrap**: the weighted shortcut gives the same intervals as the repo's `bootstrap()`
  helper and the same weighted ROC/PR AUC as sklearn on the same resamples.
- **Determinism**: all seeds fixed (see `determinism.py`). Two complete runs gave
  byte-identical `results.csv` and `pooled.csv` and identical predictions; `determinism.json`
  is a further refit of `main` 2015 (max prediction difference 0 for all arms).

## Leakage flags

- E uses the current site list (countries): sites are added and dropped after registration,
  and trials that stop early may never add sites. Exploratory.
- F uses the current title, summary and eligibility text; explicit stopping statements are
  masked as in the repo, subtler later edits are not. Exploratory.
- C uses current condition strings (conditions can be edited, but rarely change with outcome).
- D uses only first-posted dates, which do not change after posting; the competitor pool is
  today's registry, so studies deleted from the registry are missing.
- Relative volume in early years: the registry is sparse before 2005, so for 2005-2009 the
  months 13-60 count is too low and the ratio is inflated (median 2.1-3.4 in 2005-2006
  against about 1.0-1.2 from 2010).

## Confirmation cohort (not run)

Pull date 2026-11-13 (max `last_update_date`, the definition used by `trial_cohort`; the raw
files were written 2026-11-17, `_state.json` holds no date). `trial_cohort` already contains
2021 starts and sets the label to null when the horizon is incomplete, so no extra query is
needed. Cohort-eligible (label known, registered by start) 2021 starts: **12,944**, start
dates 2021-01-01 to 2021-11-13. Only this count was looked at. Caveat: these trials reached
the 5-year horizon 0-10 months before the pull, so terminations not yet reported are counted
as not terminated (15% of 2020 starts were still running at the pull). `confirm.py`
defines `confirm(arm)` (train on 2008-2015, same tree-count CV, 5 seeds, score once); it has
not been called.

## Confirmation on 2021 starts (run once)

Arm F (text score) was the only single feature that passed the selection rule, so it was
the arm taken to the untouched 2021 cohort. `confirm("F")` was called once; no other arm
has been scored on 2021 starts.

Training: starts 2008-2015 (62,867 trials, 125 trees). Confirmation cohort: 12,944 trials
started 2021-01-01 to 2021-11-13, 1,408 terminated within five years.

| Arm | ROC AUC | Delta (95% CI, paired, sponsor-clustered) | PR AUC | Phase 2 ROC AUC | Phase 2 delta (95% CI) |
|---|---|---|---|---|---|
| A baseline | 0.6977 | | 0.1924 | 0.5746 | |
| F text | 0.6971 | -0.0006 (-0.0098, +0.0080) | 0.1937 | 0.5844 | +0.0098 (-0.0243, +0.0415) |

The gain seen on 2015-2017 (+0.0078) did not repeat. The interval still allows a gain of up
to about +0.008, so the result is "not confirmed", not "shown to be zero". No feature tested
here is supported strongly enough to change the headline model.

Caveat: these trials reached their five-year horizon 0-10 months before the data pull, so
terminations not yet reported count as not terminated.
