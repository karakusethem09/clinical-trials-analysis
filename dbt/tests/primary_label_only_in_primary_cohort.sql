-- The primary label must be missing for trials that did not start, lack five years of
-- follow-up, or have UNKNOWN or WITHDRAWN status.
select nct_id
from {{ ref('trial_cohort') }}
where terminated_5y is not null
  and (not started or not has_5y_follow_up or overall_status in ('UNKNOWN', 'WITHDRAWN'))
