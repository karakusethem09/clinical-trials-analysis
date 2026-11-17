{#
  Fixed-horizon label: was the trial terminated within `years` of its start?
    1    = TERMINATED with an end date within the horizon
    0    = known not terminated by then: completed, still running or suspended at the
           pull, or terminated after the horizon
    null = outside the cohort: did not start, horizon not yet reached at the pull,
           terminated without an end date, or UNKNOWN status (see `unknown`)
  `unknown` sets what UNKNOWN status means (its last verified status was "ongoing"):
    'exclude'  -> null (primary definition)
    'as_zero'  -> 0
    'verified' -> 0 if the status was verified after the horizon, else null
  Columns used: started, overall_status, start_date, end_date, status_verified_date, pull_date.
#}
{% macro terminated_within(years=5, unknown='exclude') %}
    case
        when not started or start_date + interval {{ years }} year > pull_date then null
        {% if unknown == 'as_zero' -%}
        when overall_status = 'UNKNOWN' then 0
        {%- elif unknown == 'verified' -%}
        when overall_status = 'UNKNOWN' then
            case when status_verified_date >= start_date + interval {{ years }} year then 0 end
        {%- endif %}
        when overall_status = 'TERMINATED' and end_date is null then null
        when overall_status = 'TERMINATED' then (end_date <= start_date + interval {{ years }} year)::int
        when overall_status in ('COMPLETED', 'RECRUITING', 'ACTIVE_NOT_RECRUITING',
                                'ENROLLING_BY_INVITATION', 'SUSPENDED') then 0
    end
{% endmacro %}
