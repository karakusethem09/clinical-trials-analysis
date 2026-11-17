{#
  Collapse obvious spellings of the same sponsor ("Novartis Pharmaceuticals" and
  "Novartis", "Eisai Inc." and "Eisai Co., Ltd."):
    1. lowercase, strip accents, "&" -> "and", punctuation -> space
    2. drop "a subsidiary of ..." / "an affiliate of ..." tails and a leading "the"
    3. drop trailing corporate words (inc, ltd, gmbh, pharmaceuticals, usa, ...)
    4. a short list of groups registered under many names: janssen, roche
  It does not merge different companies that share a parent (Genentech vs Roche).
#}
{% macro normalise_sponsor(col) %}
    {%- set cleaned -%}
        trim(regexp_replace(regexp_replace(lower(strip_accents({{ col }})), '&', ' and ', 'g'), '[^a-z0-9]+', ' ', 'g'))
    {%- endset -%}
    {%- set no_parent -%}
        regexp_replace(regexp_replace({{ cleaned }}, ' (a|an) (wholly owned )?(subsidiary|affiliate) of .*$', ''), '^the ', '')
    {%- endset -%}
    {%- set no_suffix -%}
        nullif(regexp_replace({{ no_parent }}, '( (inc|incorporated|ltd|limited|llc|llp|corp|corporation|co|company|companies|gmbh|ag|kgaa|sa|s a|spa|s p a|srl|bv|b v|nv|plc|kk|k k|ab|as|a s|oy|pty|lp|holding|holdings|group|pharmaceutical|pharmaceuticals|pharma|usa|us|international|and))+$', ''), '')
    {%- endset -%}
    case
        when {{ no_suffix }} like 'janssen%' then 'janssen'
        when {{ no_suffix }} like 'hoffmann la roche%' then 'roche'
        else {{ no_suffix }}
    end
{% endmacro %}
