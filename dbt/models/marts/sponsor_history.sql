-- Point-in-time sponsor track record: for each trial, what was known about its
-- lead sponsor on the day the trial started. Only trials that had already
-- ended before that day count towards the stop rate.
with c as (
    select * from {{ ref('trial_cohort') }}
),

ended_by_day as (
    select
        lead_sponsor_name,
        end_date,
        count(*)            as n_ended,
        sum(stopped_early)  as n_stopped
    from c
    where is_resolved and end_date is not null and lead_sponsor_name is not null
    group by 1, 2
),

running as (
    select
        lead_sponsor_name,
        end_date,
        sum(n_ended)   over w as cum_ended,
        sum(n_stopped) over w as cum_stopped
    from ended_by_day
    window w as (partition by lead_sponsor_name order by end_date)
),

started_before as (
    select
        nct_id,
        count(*) over (
            partition by lead_sponsor_name order by start_date
            range between unbounded preceding and interval 1 day preceding
        ) as sponsor_prior_started
    from c
)

select
    c.nct_id,
    sb.sponsor_prior_started,
    coalesce(r.cum_ended, 0)                           as sponsor_prior_ended,
    coalesce(r.cum_stopped, 0)                         as sponsor_prior_stopped,
    r.cum_stopped / nullif(r.cum_ended, 0)             as sponsor_prior_stop_rate
from c
join started_before sb using (nct_id)
asof left join running r
    on c.lead_sponsor_name = r.lead_sponsor_name
   and c.start_date > r.end_date
