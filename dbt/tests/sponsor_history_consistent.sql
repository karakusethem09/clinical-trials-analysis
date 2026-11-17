-- A sponsor's ended trials are a subset of its registered trials, and rates are proportions.
select nct_id
from {{ ref('sponsor_history') }}
where sponsor_prior_ended > sponsor_prior_registered
   or sponsor_prior_terminated + sponsor_prior_withdrawn > sponsor_prior_ended
   or sponsor_prior_terminated_rate not between 0 and 1
   or sponsor_prior_withdrawn_rate not between 0 and 1
