# Will This Trial Finish?

Predicting early termination of clinical trials from what is known on the day a trial is registered, using public ClinicalTrials.gov data (~612k studies).

## Questions

1. **Why do trials stop?** Classify the free-text `whyStopped` field into categories (low enrollment, funding, safety, sponsor decision, ...) and describe the patterns by phase, sponsor type and disease area.
2. **Can we tell in advance?** For a trial that starts, estimate the risk that it is terminated within five years, using registration-time information only. Withdrawal before the start is a separate, secondary question.

## Why it is hard: leakage

The registry shows the *current* state of each record. A terminated trial's `EnrollmentCount` is the actual number enrolled (often tiny), not the planned number, so a naive model "predicts" termination from its own consequence. Every feature here must be justified as known at registration. Less obvious traps found along the way: the start date of a withdrawn trial is usually a plan that never happened, dropping still-running trials inflates the stop rate of recent years, and a sponsor's "prior" trials may not have had their outcome posted yet. See [DECISIONS.md](DECISIONS.md).

## Layout

```
src/ingest/    download from the ClinicalTrials.gov API v2 (resumable)
dbt/           staging -> cohort, labels, point-in-time sponsor history (DuckDB), with data tests
src/models/    train_tabular.py    primary task, baselines and the leakage ladder
               text_masking.py     removes statements that a trial stopped from its text
               train_text.py       TF-IDF text score and the text ablation
               train_survival.py   time to termination with competing risks
               train_withdrawn.py  secondary task: withdrawn before starting
               sensitivity.py      one-change-at-a-time robustness checks
               evaluate.py         model selection, bootstrap intervals, calibration, decision curve,
                                   subgroups, SHAP, final model
src/analysis/  why_stopped.py      rule-based categories of the sponsors' stated reasons
               build_app_data.py   regenerates app/data/ from DuckDB and the saved model
models/        final model (joblib) and a scored sample for the dashboard
reports/       metrics (JSON/CSV) and figures
app/           Streamlit dashboard
tests/         pytest suite (runs without the database)
data/          local only, not committed
```

## Running it

```bash
python src/ingest/fetch_studies.py --out data/raw_v2            # full pull, resumable
cd dbt && ../.venv/bin/dbt build --profiles-dir . && cd ..       # tables + tests
for s in train_tabular train_text train_survival train_withdrawn sensitivity evaluate; do
    .venv/bin/python src/models/$s.py
done
.venv/bin/python src/analysis/build_app_data.py
.venv/bin/python -m pytest -q tests
```

All runs are deterministic (fixed row order, folds and seeds; two full reruns give identical report files). The chain takes about 30 minutes on 4 cores.

## Results

**Primary task.** Among interventional trials that started in 2008-2020 and were registered no later than 21 days after their start, predict termination within five years of the start. Train on 2008-2014 starts, stop early on 2015-2016, choose the model and calibrate on 2017, test on 2018-2020 (36,452 trials, 12.1% terminated by year 5).

**Pre-specified headline model.** Whether the text score enters the headline model was decided on 2017 alone: the rule was to include it if the paired bootstrap interval of its ROC AUC gain on 2017 excludes zero. The gain there was +0.0025 (95% CI -0.0067 to 0.0118), so **the headline model is tabular gradient boosting**, and the text model is reported as an ablation. The test years were scored once, after that choice.

| Model (test 2018-2020) | ROC AUC, trial bootstrap | ROC AUC, sponsor-clustered bootstrap | PR AUC | Precision in top 10% (lift) |
|---|---|---|---|---|
| Logistic regression | 0.639 (0.630-0.647) | (0.628-0.649) | 0.180 | 0.206 (1.69x) |
| **Gradient boosting, tabular (headline)** | **0.655 (0.647-0.664)** | **(0.644-0.666)** | 0.188 | 0.215 (1.77x) |
| Ablation: tabular + text score | 0.667 (0.658-0.674) | (0.656-0.677) | 0.198 | 0.233 (1.92x) |

The sponsor-clustered intervals resample the 7,746 normalised lead sponsors in the test years instead of single trials, because trials of one sponsor are alike. On the test years the text score does add +0.011 ROC AUC (0.006 to 0.016), more than on 2017; it stays an ablation because the rule was fixed before the test years were looked at. Out-of-fold text scores use folds grouped by sponsor; with ungrouped folds the gain was +0.0056 on 2017 and +0.0085 on the test years. The model refitted on 2008-2016 for the dashboard reaches 0.665 on the test years.

**Text masking.** Sentences in the title or summary that say the study was terminated, withdrawn, halted, closed or suspended, or that it enrolled fewer participants than planned, are removed before the text model reads them (`src/models/text_masking.py`). The rule now catches 0.56% of terminated and 0.11% of other trials; the earlier, broad rule removed a sentence from 22.0% and 17.5% of them, mostly innocent ones ("discontinue NSAIDs"). Eligibility criteria are left alone: the same rule fires on them equally often for both outcomes (0.33% vs 0.30%), and what it finds there is about participants. Twenty random masked sentences are in `reports/masked_sentences_sample.csv`.

**What it supports.** The riskiest tenth of test trials is terminated 1.77 times as often as average (21.5% vs 12.1%). That is enough to prioritise, for example, an extra feasibility review of the top-risk decile of a portfolio; it is not a verdict on any single trial, since about four in five flagged trials do not terminate. A decision curve (`reports/figures/decision_curve.png`) shows positive net benefit over flagging every trial or none for risk thresholds between 3% and 21%, and none above that.

**The leakage ladder.** The same project, three ways:

| Set-up | Test set | ROC AUC |
|---|---|---|
| Naive: old label (terminated or withdrawn vs completed, resolved trials only) and fields overwritten after a trial ends | 2018-2019, old label | 0.874 (0.868-0.880) |
| Partly fixed: old label, no overwritten fields, but registration delay, start year and an unlagged sponsor history | 2018-2019, old label | 0.721 (0.713-0.729) |
| Defensible: the primary task, headline model | 2018-2020, primary label | 0.655 (0.647-0.664) |

The rungs answer different questions, so the drop is not just leakage removed: rung 3 is a different, harder estimand (trials that started, a fixed horizon, no withdrawn trials whose planned start dates made them easy to spot). Scored as-is on the primary test set, the partly fixed model reaches 0.660, about the same as the defensible one.

**Calibration.** Raw scores under-predict (mean 0.108 against an observed 0.121). Isotonic recalibration on 2017 lowers the expected calibration error from 0.017 to 0.013 and the Brier score from 0.1037 to 0.1036 (`reports/figures/calibration.png`). Because it maps scores onto a few dozen steps it ties some trials and lowers the ROC AUC slightly, from 0.6553 to 0.6546. It cannot remove the gap fully because 2018-2020 had a slightly higher termination rate than 2017; by phase, recalibrated risks remain too low for Phase 1/2 (by 4.4 points) and Phase 1 (3.5 points).

**Where the model is weak: Phase 2.** ROC AUC is 0.58 for Phase 2 and 0.57 for Phase 1/2 trials, exactly where termination is most common (19% of both are terminated within five years), against 0.655 overall. It is also weak for radiation (0.58) and behavioural (0.59) trials, and best for Phase 1 (0.68) and Phase 2/3 (0.68) (`reports/subgroup_metrics.csv`).

**What drives predictions.** By mean absolute SHAP value: healthy-volunteer trials (lower risk), the sponsor's prior termination rate, behavioural interventions (lower risk), primary purpose, drug interventions, the intervention model, and the sponsor's prior withdrawal rate (`reports/shap_importance.csv`).

**Survival analysis.** Using all 231,240 started trials, including the 94,679 started from 2021 onwards and still largely running, with completion as a competing risk: 10.6% are terminated within five years (1 - Kaplan-Meier, which wrongly censors completions, says 19.7%). A penalised Cox model fitted on starts up to 2017 has a concordance index of 0.604 on 2018-2020 starts. Hazard ratios: prior sponsor termination rate (0 to 1) 2.35, industry vs academic sponsor 1.55, Phase 1 vs no phase 1.32, drug intervention 1.29, Phase 3 0.82, behavioural intervention 0.71, NIH collaborator 0.68.

**Withdrawal before the start (secondary).** Among prospectively registered trials, 4.5-6.4% per registration year are withdrawn. A model of the same design, split by registration year, scores ROC AUC 0.650 (0.639-0.660) on 2018-2020 registrations, with a top-decile precision of 0.138 (2.3x the 5.9% base rate). The sponsor's prior withdrawal rate is the strongest feature.

**Sensitivity.** Test ROC AUC of the tabular model when one decision changes (`reports/sensitivity_metrics.csv`):

| Variant | Test trials | Terminated | ROC AUC, own test set | ROC AUC, primary test set |
|---|---|---|---|---|
| Primary | 36,452 | 12.1% | 0.655 | 0.655 |
| No registration-timing filter | 52,043 | 9.8% | 0.696 | 0.663 |
| Old 21-day filter on day-1 dates | 36,452 | 12.1% | 0.656 | 0.656 |
| UNKNOWN status counted as not terminated | 42,775 | 10.4% | 0.667 | 0.663 |
| UNKNOWN counted only if verified running after year 5 | 36,650 | 12.1% | 0.658 | 0.659 |
| Still running and not verified in the year before the pull: left out | 34,941 | 12.7% | 0.664 | 0.657 |
| Plus registration delay | 36,452 | 12.1% | 0.662 | 0.662 |
| Plus start year | 36,452 | 12.1% | 0.658 | 0.658 |

## Dashboard

`.venv/bin/streamlit run app/Home.py` opens five pages: an overview with headline results; a risk explorer for 3,000 scored test trials (risk, SHAP drivers, actual outcome, plus a labelled what-if); why trials stop (rule-based categories of `why_stopped`, `src/analysis/why_stopped.py`); the leakage ladder with its evidence; and calibration, the decision curve, subgroups, competing-risk curves, sensitivity and limitations. It reads only `models/`, `reports/` and `app/data/`, so it deploys on Streamlit Community Cloud without the database. Rebuild `app/data/` with `.venv/bin/python src/analysis/build_app_data.py`.

The why-stopped categories are rule-based and their accuracy has not been measured yet; a 300-row sample for hand-labelling is in `reports/why_stopped_label_sample.csv`.

Deployment: Community Cloud installs `requirements.txt` (run-time packages only, pinned to the versions that saved the model). Community Cloud is assumed to take the Python version from the deploy dialog's advanced settings rather than from a file in the repository (not checked against its current documentation), so no runtime file is included; choose Python 3.12 or 3.13, for which all pinned packages have wheels. The theme is in `.streamlit/config.toml`.

## Limitations

- **Only current record versions are public.** ClinicalTrials.gov has no API for historical versions, so registration-time values cannot be reconstructed. Eligibility criteria, summaries, phase, design and collaborators may have been edited after registration, possibly in response to what happened in the trial. Explicit statements that the study stopped are masked in the text, but subtler edits cannot be detected.
- **Training and test years differ in date quality.** The start-date type (actual or estimated) is missing for 88% of 2008 starts in the primary cohort and for none of the 2018-2020 test trials; 90% of 2008 start dates give only a month, against none in the test years. Rules that use dates (who started, registration timing) are therefore applied to cleaner data in the test years than in training.
- **Status lags and stale records.** A termination counts only once the sponsor reports it. 15.1% of 2020 starts are still running at the pull and labelled "not terminated", against under 3% in the training years, and 651 of them (5.4%) have not had their status verified in the year before the pull. If 10% of those 651 had in fact been terminated within five years, the 2020 termination rate would rise from 12.4% to 13.0% (about 0.5 points; 0.4 points across all test years). Leaving such trials out changes the test ROC AUC by +0.002 on the primary test set.
- **UNKNOWN status.** 9-19% of started trials per year have not been verified for two years and are excluded. Some of them quietly stopped; the sensitivity analysis bounds the effect on AUC but not on the base rate.
- **Sponsor history is approximate.** The 12-month lag is a proxy for when an outcome became public, and name normalisation only merges obvious variants.
- **Dates.** 36% of all start dates carry only a month (mostly before 2017). They are set to day 1 for the label and to the month's end for the registration filter.
- **Modest signal.** With a ROC AUC around 0.66 the model ranks portfolios better than chance, but it is not precise enough to make decisions about individual trials, and it is weakest for Phase 2 trials.
