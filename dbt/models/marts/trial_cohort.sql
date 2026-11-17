-- Analysis cohort: interventional trials that started in 2008 or later
-- (registration became mandatory under FDAAA in late 2007).
--
-- Columns are split into three groups:
--   * outcome columns
--   * features considered known when the trial was registered
--   * columns prefixed leaky_: values the registry overwrites after the trial ends.
--     They are kept only to measure how much they inflate a naive model.
with s as (
    select * from {{ ref('stg_studies') }}
    where study_type = 'INTERVENTIONAL'
      and start_date >= date '2008-01-01'
      and start_date <= current_date
)

select
    nct_id,

    -- outcome
    overall_status,
    overall_status in ('COMPLETED', 'TERMINATED', 'WITHDRAWN')  as is_resolved,
    case
        when overall_status in ('TERMINATED', 'WITHDRAWN') then 1
        when overall_status = 'COMPLETED' then 0
    end                                                         as stopped_early,
    why_stopped,
    start_date,
    coalesce(completion_date, primary_completion_date)          as end_date,
    last_update_date,

    -- registration-time features: design
    year(start_date)                                            as start_year,
    case when len(phases) = 0 then 'NONE' else array_to_string(list_sort(phases), '/') end as phase,
    coalesce(allocation, 'MISSING')                             as allocation,
    coalesce(masking, 'MISSING')                                as masking,
    coalesce(primary_purpose, 'MISSING')                        as primary_purpose,
    coalesce(intervention_model, 'MISSING')                     as intervention_model,
    date_diff('day', start_date, first_submit_date)             as days_registered_after_start,
    -- FDAAA allows registration up to 21 days after the first participant is enrolled.
    -- Trials registered later may already have finished when they were registered.
    date_diff('day', start_date, first_submit_date) <= 21       as registered_prospectively,

    -- registration-time features: sponsor
    lead_sponsor_name,
    coalesce(lead_sponsor_class, 'MISSING')                     as lead_sponsor_class,
    len(collaborator_classes) > 0                               as has_collaborator,
    list_contains(collaborator_classes, 'INDUSTRY')             as has_industry_collaborator,
    list_contains(collaborator_classes, 'NIH')                  as has_nih_collaborator,

    -- registration-time features: what is studied
    conditions,
    len(conditions)                                             as n_conditions,
    {% for t in ['DRUG', 'DEVICE', 'BIOLOGICAL', 'PROCEDURE', 'BEHAVIORAL', 'RADIATION', 'DIETARY_SUPPLEMENT', 'GENETIC', 'DIAGNOSTIC_TEST', 'COMBINATION_PRODUCT', 'OTHER'] %}
    list_contains(intervention_types, '{{ t }}')                as intervention_{{ t | lower }},
    {% endfor %}
    len(intervention_types)                                     as n_intervention_types,

    -- registration-time features: who can join
    coalesce(sex, 'MISSING')                                    as sex,
    {{ age_in_years('minimum_age') }}                           as min_age_years,
    {{ age_in_years('maximum_age') }}                           as max_age_years,
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
    len(location_countries)                                     as leaky_n_countries
from s
