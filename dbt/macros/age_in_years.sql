{% macro age_in_years(col) %}
    case
        when {{ col }} is null then null
        else try_cast(split_part({{ col }}, ' ', 1) as double) /
            case
                when {{ col }} ilike '%year%'   then 1
                when {{ col }} ilike '%month%'  then 12
                when {{ col }} ilike '%week%'   then 52.18
                when {{ col }} ilike '%day%'    then 365.25
                when {{ col }} ilike '%hour%'   then 8766
                when {{ col }} ilike '%minute%' then 525960
                else null
            end
    end
{% endmacro %}
