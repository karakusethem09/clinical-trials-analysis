-- A positive label is only ever a TERMINATED trial that ended within five years.
select nct_id
from {{ ref('trial_cohort') }}
where terminated_5y = 1
  and (overall_status <> 'TERMINATED' or end_date > start_date + interval 5 year)
