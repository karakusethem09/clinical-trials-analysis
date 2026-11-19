-- Copy of the compiled trial_cohort model (dbt/models/marts/trial_cohort.sql), read-only use.
-- Only change: the earliest start date is a parameter ({min_start}) instead of 2008-01-01.
-- Analysis cohort: interventional trials that started in 2008 or later
-- (registration became mandatory under FDAAA in late 2007).
--
-- Columns are split into three groups:
--   * outcome columns, including the primary label terminated_5y (DECISIONS.md #18-#21)
--   * features considered known when the trial was registered
--   * columns prefixed leaky_: values the registry overwrites after the trial ends.
--     They are kept only to measure how much they inflate a naive model.
with pull as (
    -- the data pull date: the latest update anyone submitted
    select max(last_update_date) as pull_date from "stg_studies"
),

s as (
    select
        *,
        coalesce(completion_date, primary_completion_date)                   as end_date,
        -- a month-only start date could be any day of that month: use its last day
        -- when checking registration timing, so the day-1 default never penalises a trial
        case when start_date_precision = 'MONTH' then last_day(start_date) else start_date end
                                                                             as start_date_latest,
        -- began enrolling: not withdrawn, not still waiting, and no planned-only start date
        overall_status not in ('WITHDRAWN', 'NOT_YET_RECRUITING')
            and coalesce(start_date_type, 'ACTUAL') = 'ACTUAL'               as started
    from "stg_studies", pull
    where study_type = 'INTERVENTIONAL'
      and start_date >= date '{min_start}'
      and start_date <= pull_date
)

select
    nct_id,

    -- primary outcome: terminated within 5 years of starting, among trials that started
    -- (definition in macros/terminated_within.sql)
    pull_date,
    started,
    start_date + interval 5 year <= pull_date                                as has_5y_follow_up,
    
    case
        when not started or start_date + interval 5 year > pull_date then null
        
        when overall_status = 'TERMINATED' and end_date is null then null
        when overall_status = 'TERMINATED' then (end_date <= start_date + interval 5 year)::int
        when overall_status in ('COMPLETED', 'RECRUITING', 'ACTIVE_NOT_RECRUITING',
                                'ENROLLING_BY_INVITATION', 'SUSPENDED') then 0
    end
                                                as terminated_5y,
    -- sensitivity versions for UNKNOWN status (last verified status was "ongoing")
    
    case
        when not started or start_date + interval 5 year > pull_date then null
        when overall_status = 'UNKNOWN' then 0
        when overall_status = 'TERMINATED' and end_date is null then null
        when overall_status = 'TERMINATED' then (end_date <= start_date + interval 5 year)::int
        when overall_status in ('COMPLETED', 'RECRUITING', 'ACTIVE_NOT_RECRUITING',
                                'ENROLLING_BY_INVITATION', 'SUSPENDED') then 0
    end
                             as terminated_5y_unknown_as_0,
    
    case
        when not started or start_date + interval 5 year > pull_date then null
        when overall_status = 'UNKNOWN' then
            case when status_verified_date >= start_date + interval 5 year then 0 end
        when overall_status = 'TERMINATED' and end_date is null then null
        when overall_status = 'TERMINATED' then (end_date <= start_date + interval 5 year)::int
        when overall_status in ('COMPLETED', 'RECRUITING', 'ACTIVE_NOT_RECRUITING',
                                'ENROLLING_BY_INVITATION', 'SUSPENDED') then 0
    end
                            as terminated_5y_unknown_verified,
    -- registered no later than 21 days after the (latest possible) start date
    date_diff('day', start_date_latest, first_submit_date) <= 21             as registered_by_start,

    -- status, and the legacy outcome (DECISIONS.md #3, superseded by #18)
    overall_status,
    overall_status in ('COMPLETED', 'TERMINATED', 'WITHDRAWN')  as is_resolved,
    case
        when overall_status in ('TERMINATED', 'WITHDRAWN') then 1
        when overall_status = 'COMPLETED' then 0
    end                                                         as stopped_early,
    why_stopped,
    start_date,
    start_date_type,
    start_date_precision,
    end_date,
    completion_date_type,
    first_submit_date,
    first_post_date,
    status_verified_date,
    last_update_date,

    -- registration-time features: design
    year(start_date)                                            as start_year,
    case when len(phases) = 0 then 'NONE' else array_to_string(list_sort(phases), '/') end as phase,
    coalesce(allocation, 'MISSING')                             as allocation,
    coalesce(masking, 'MISSING')                                as masking,
    coalesce(primary_purpose, 'MISSING')                        as primary_purpose,
    coalesce(intervention_model, 'MISSING')                     as intervention_model,
    -- legacy (DECISIONS.md #9, superseded by #22): the start date is the actual date for
    -- trials that started but a planned one for withdrawn trials, so these two columns
    -- are linked to the outcome. Not used by the current models.
    date_diff('day', start_date, first_submit_date)             as days_registered_after_start,
    date_diff('day', start_date, first_submit_date) <= 21       as registered_prospectively,

    -- registration-time features: sponsor
    lead_sponsor_name,
    lead_sponsor_key,
    coalesce(lead_sponsor_class, 'MISSING')                     as lead_sponsor_class,
    len(collaborator_classes) > 0                               as has_collaborator,
    list_contains(collaborator_classes, 'INDUSTRY')             as has_industry_collaborator,
    list_contains(collaborator_classes, 'NIH')                  as has_nih_collaborator,

    -- registration-time features: what is studied
    conditions,
    len(conditions)                                             as n_conditions,
    
    list_contains(intervention_types, 'DRUG')                as intervention_drug,
    
    list_contains(intervention_types, 'DEVICE')                as intervention_device,
    
    list_contains(intervention_types, 'BIOLOGICAL')                as intervention_biological,
    
    list_contains(intervention_types, 'PROCEDURE')                as intervention_procedure,
    
    list_contains(intervention_types, 'BEHAVIORAL')                as intervention_behavioral,
    
    list_contains(intervention_types, 'RADIATION')                as intervention_radiation,
    
    list_contains(intervention_types, 'DIETARY_SUPPLEMENT')                as intervention_dietary_supplement,
    
    list_contains(intervention_types, 'GENETIC')                as intervention_genetic,
    
    list_contains(intervention_types, 'DIAGNOSTIC_TEST')                as intervention_diagnostic_test,
    
    list_contains(intervention_types, 'COMBINATION_PRODUCT')                as intervention_combination_product,
    
    list_contains(intervention_types, 'OTHER')                as intervention_other,
    
    len(intervention_types)                                     as n_intervention_types,

    -- registration-time features: who can join
    coalesce(sex, 'MISSING')                                    as sex,
    
    case
        when minimum_age is null then null
        else try_cast(split_part(minimum_age, ' ', 1) as double) /
            case
                when minimum_age ilike '%year%'   then 1
                when minimum_age ilike '%month%'  then 12
                when minimum_age ilike '%week%'   then 52.18
                when minimum_age ilike '%day%'    then 365.25
                when minimum_age ilike '%hour%'   then 8766
                when minimum_age ilike '%minute%' then 525960
                else null
            end
    end
                           as min_age_years,
    
    case
        when maximum_age is null then null
        else try_cast(split_part(maximum_age, ' ', 1) as double) /
            case
                when maximum_age ilike '%year%'   then 1
                when maximum_age ilike '%month%'  then 12
                when maximum_age ilike '%week%'   then 52.18
                when maximum_age ilike '%day%'    then 365.25
                when maximum_age ilike '%hour%'   then 8766
                when maximum_age ilike '%minute%' then 525960
                else null
            end
    end
                           as max_age_years,
    healthy_volunteers,
    length(eligibility_criteria)                                as criteria_chars,
    len(regexp_extract_all(eligibility_criteria, '\n\s*(\*|-|\d+\.)\s'))  as criteria_bullets,

    -- text
    brief_title,
    brief_summary,
    eligibility_criteria,

    -- overwritten after the trial ends: do not use for prediction
    enrollment_count                                            as leaky_enrollment_count,
    enrollment_type                                             as leaky_enrollment_type,
    n_locations                                                 as leaky_n_locations,
    len(location_countries)                                     as leaky_n_countries,
    results_first_submit_date                                   as leaky_results_first_submit_date
from s