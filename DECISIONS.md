# Decision log

Judgment calls made while building the cohort and models, with the reason for each.

| # | Decision | Why |
|---|---|---|
| 1 | Interventional trials only | Observational studies do not "terminate" in the same sense and follow different registration rules. |
| 2 | Start date 2008 or later | Registration became a legal requirement (FDAAA) in late 2007; earlier records are a self-selected sample. |
| 3 | Outcome: terminated or withdrawn = stopped early, completed = not | Withdrawn trials stopped before enrolling anyone; both are failures to deliver a result. Status is kept so they can be analysed separately. |
| 4 | Trials with status UNKNOWN are excluded from classification | The registry sets this when a record has not been updated for two years. The true outcome is not known. This removes 8-25% of trials depending on start year and is a limitation. |
| 5 | Classification uses start years 2008-2019 only | For later years many trials are still running. Among recent trials the ones already resolved are disproportionately the ones that stopped, which would bias the label. |
| 6 | Split by start year: train 2008-2015, validate 2016-2017, test 2018-2019 | A random split lets the model learn from the future. |
| 7 | Enrollment count is not a feature | For finished trials the registry replaces the planned number with the actual one. Stopped trials have a median of 19 participants versus 60 for completed ones: the field encodes the outcome. |
| 8 | Number of sites and countries are not features | Sites are added as a trial progresses; withdrawn trials often list none. |
| 9 | Only prospectively registered trials (registered within 21 days of start) | A trial registered long after it began may already have finished at registration, so "predicting" its outcome is meaningless. Registration delay was the single strongest feature before this filter. |
| 10 | Sponsor track record is computed as of each trial's start date | Using a sponsor's overall stop rate would include the trial being predicted and trials that ended later. |
| 11 | Sponsor name is not a direct feature | Very high cardinality; the point-in-time track record carries the signal without memorising names. |
| 12 | Survival analysis treats completion as a competing event, not as censoring | A completed trial can no longer be terminated. Censoring it and plotting 1 - Kaplan-Meier gives 19.3% terminated by year 5 versus 10.6% from the Aalen-Johansen cumulative incidence. Curves use Aalen-Johansen; the Cox model is a cause-specific hazard model, for which censoring completers is correct. |
| 13 | Survival cohort excludes WITHDRAWN and NOT_YET_RECRUITING trials | Neither enrolled anyone and their start date is only a plan, so a "time to stopping" is meaningless. Withdrawal stays in the classifier's outcome. |
| 14 | In the survival analysis, UNKNOWN trials are censored at their last update and running or suspended trials at the data pull date | The last update is the last day the trial was confirmed to be running. Some UNKNOWN trials have probably stopped quietly, so this censoring is likely informative and slightly understates termination risk. |
| 15 | Cox hazard ratios are reported from a fit on all start years; the concordance index comes from a separate fit on starts up to 2017 scored on 2018-2019 starts, with the penalty chosen on 2016-2017 | Using every year, including still-running trials, is the reason to do survival analysis; the hold-out fit keeps the performance number honest. |
| 16 | Isotonic recalibration is fitted on the validation years, which were also used for early stopping | The test years stay untouched. Reusing the validation years makes the calibrator slightly optimistic about the scores it sees, which is small next to the gain from fitting it on years close to the test period. |
| 17 | Subgroups with fewer than 200 trials or fewer than 20 events on the test years are not reported; calibration-in-the-large is computed on recalibrated probabilities | Below that size the AUC confidence interval is too wide to say anything; the recalibrated model is the one that would be used. |
