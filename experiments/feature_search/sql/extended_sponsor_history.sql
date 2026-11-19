-- Copy of the compiled sponsor_history model (dbt/models/marts/sponsor_history.sql), read-only use.
-- Only change: the focal trials come from ext_cohort (extended_cohort.sql) instead of trial_cohort.
-- Sponsor track record as it could have been known when each trial was registered.
--
-- Anchor date : the earlier of the trial's start date and its first submission.
-- History     : every interventional study of the same (normalised) lead sponsor,
--               including studies started before 2008.
-- Registered  : studies first submitted before the anchor date (public by then).
-- Ended       : studies that are now COMPLETED, TERMINATED or WITHDRAWN and whose end
--               date is at least 12 months before the anchor date (and that were
--               registered before it). The 12-month lag is a proxy for "the outcome had
--               been posted": the registry only shows the current record, not when its
--               status changed.
with c as (
    select nct_id, lead_sponsor_key, least(start_date, first_submit_date) as anchor_date
    from ext_cohort
),

hist as (
    select
        lead_sponsor_key,
        first_submit_date,
        overall_status,
        coalesce(completion_date, primary_completion_date) as end_date
    from "stg_studies"
    where study_type = 'INTERVENTIONAL' and lead_sponsor_key is not null
),

registered as (
    select
        lead_sponsor_key,
        first_submit_date,
        sum(count(*)) over (partition by lead_sponsor_key order by first_submit_date) as cum_registered
    from hist
    where first_submit_date is not null
    group by 1, 2
),

ended as (
    select
        lead_sponsor_key,
        greatest(end_date + interval 12 month, first_submit_date)               as visible_from,
        count(*)                                                                 as n_ended,
        count_if(overall_status = 'COMPLETED')                                   as n_completed,
        count_if(overall_status = 'TERMINATED')                                  as n_terminated,
        count_if(overall_status = 'WITHDRAWN')                                   as n_withdrawn
    from hist
    where overall_status in ('COMPLETED', 'TERMINATED', 'WITHDRAWN')
      and end_date is not null and first_submit_date is not null
    group by 1, 2
),

ended_running as (
    select
        lead_sponsor_key,
        visible_from,
        sum(n_ended)      over w as cum_ended,
        sum(n_completed)  over w as cum_completed,
        sum(n_terminated) over w as cum_terminated,
        sum(n_withdrawn)  over w as cum_withdrawn
    from ended
    window w as (partition by lead_sponsor_key order by visible_from)
)

select
    c.nct_id,
    c.lead_sponsor_key,
    c.anchor_date                                                         as sponsor_anchor_date,
    coalesce(r.cum_registered, 0)::bigint                                         as sponsor_prior_registered,
    coalesce(e.cum_ended, 0)::bigint                                              as sponsor_prior_ended,
    coalesce(e.cum_terminated, 0)::bigint                                         as sponsor_prior_terminated,
    coalesce(e.cum_withdrawn, 0)::bigint                                          as sponsor_prior_withdrawn,
    e.cum_terminated / nullif(e.cum_completed + e.cum_terminated, 0)      as sponsor_prior_terminated_rate,
    e.cum_withdrawn / nullif(e.cum_ended, 0)                              as sponsor_prior_withdrawn_rate
from c
asof left join registered r
    on c.lead_sponsor_key = r.lead_sponsor_key
   and c.anchor_date > r.first_submit_date
asof left join ended_running e
    on c.lead_sponsor_key = e.lead_sponsor_key
   and c.anchor_date > e.visible_from