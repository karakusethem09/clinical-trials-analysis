-- One row per registered study, flattened from the raw API JSON.
-- Partial dates such as "2007-08" are set to the first of the month.
with raw as (
    select json as j
    from read_ndjson_objects('{{ var("raw_glob") }}')
),

parsed as (
    select
        j ->> '$.protocolSection.identificationModule.nctId'                as nct_id,
        j ->> '$.protocolSection.identificationModule.briefTitle'           as brief_title,
        j ->> '$.protocolSection.identificationModule.officialTitle'        as official_title,
        j ->> '$.protocolSection.descriptionModule.briefSummary'            as brief_summary,
        j ->> '$.protocolSection.statusModule.overallStatus'                as overall_status,
        j ->> '$.protocolSection.statusModule.whyStopped'                   as why_stopped,
        j ->> '$.protocolSection.statusModule.studyFirstSubmitDate'         as first_submit_raw,
        j ->> '$.protocolSection.statusModule.startDateStruct.date'         as start_raw,
        j ->> '$.protocolSection.statusModule.primaryCompletionDateStruct.date' as primary_completion_raw,
        j ->> '$.protocolSection.statusModule.completionDateStruct.date'    as completion_raw,
        j ->> '$.protocolSection.statusModule.lastUpdateSubmitDate'         as last_update_raw,
        j ->> '$.protocolSection.designModule.studyType'                    as study_type,
        from_json(j -> '$.protocolSection.designModule.phases', '["VARCHAR"]') as phases,
        try_cast(j ->> '$.protocolSection.designModule.enrollmentInfo.count' as bigint) as enrollment_count,
        j ->> '$.protocolSection.designModule.enrollmentInfo.type'          as enrollment_type,
        j ->> '$.protocolSection.designModule.designInfo.allocation'        as allocation,
        j ->> '$.protocolSection.designModule.designInfo.maskingInfo.masking' as masking,
        j ->> '$.protocolSection.designModule.designInfo.primaryPurpose'    as primary_purpose,
        j ->> '$.protocolSection.designModule.designInfo.interventionModel' as intervention_model,
        j ->> '$.protocolSection.sponsorCollaboratorsModule.leadSponsor.name'  as lead_sponsor_name,
        j ->> '$.protocolSection.sponsorCollaboratorsModule.leadSponsor.class' as lead_sponsor_class,
        from_json(j -> '$.protocolSection.sponsorCollaboratorsModule.collaborators[*].class', '["VARCHAR"]') as collaborator_classes,
        from_json(j -> '$.protocolSection.conditionsModule.conditions', '["VARCHAR"]') as conditions,
        from_json(j -> '$.protocolSection.armsInterventionsModule.interventions[*].type', '["VARCHAR"]') as intervention_types,
        j ->> '$.protocolSection.eligibilityModule.eligibilityCriteria'     as eligibility_criteria,
        j ->> '$.protocolSection.eligibilityModule.sex'                     as sex,
        j ->> '$.protocolSection.eligibilityModule.minimumAge'              as minimum_age,
        j ->> '$.protocolSection.eligibilityModule.maximumAge'              as maximum_age,
        try_cast(j ->> '$.protocolSection.eligibilityModule.healthyVolunteers' as boolean) as healthy_volunteers,
        from_json(j -> '$.protocolSection.contactsLocationsModule.locations[*].country', '["VARCHAR"]') as location_countries
    from raw
)

select
    nct_id,
    brief_title,
    official_title,
    brief_summary,
    overall_status,
    why_stopped,
    try_cast(first_submit_raw as date) as first_submit_date,
    {% for c in ['start', 'primary_completion', 'completion'] %}
    try_cast(case when length({{ c }}_raw) = 7 then {{ c }}_raw || '-01' else {{ c }}_raw end as date) as {{ c }}_date,
    {% endfor %}
    try_cast(last_update_raw as date) as last_update_date,
    study_type,
    coalesce(phases, []) as phases,
    enrollment_count,
    enrollment_type,
    allocation,
    masking,
    primary_purpose,
    intervention_model,
    lead_sponsor_name,
    lead_sponsor_class,
    coalesce(list_distinct(collaborator_classes), []) as collaborator_classes,
    coalesce(conditions, []) as conditions,
    coalesce(list_distinct(intervention_types), []) as intervention_types,
    eligibility_criteria,
    sex,
    minimum_age,
    maximum_age,
    healthy_volunteers,
    coalesce(len(location_countries), 0) as n_locations,
    coalesce(list_distinct(location_countries), []) as location_countries
from parsed
