"""Arms D and E: how crowded the focal trial's conditions were when it was registered.

Anchor      : the focal trial's first-submitted date.
Competitors : every OTHER interventional study in stg_studies (any status, any start year)
              that shares at least one normalised condition string with the focal trial.
              Only their first-posted date, conditions, lead sponsor and (for E) country
              list are read, never their outcome or later dates.
Normalised condition: lowercase, non-alphanumeric runs -> one space, trimmed; empty
              strings dropped. (The snapshot has no MeSH terms.)

Arm D (no future information: first-posted dates are fixed once a record is public)
  comp_12m              competitors first posted in [anchor - 365 days, anchor)
  comp_12m_other_sponsor  the same, excluding competitors with the same normalised lead
                        sponsor (a missing sponsor never counts as the same)
  comp_busiest_cond_12m the largest 12-month count of any single focal condition
  comp_rel_volume       (comp_12m + 1) / (comp_13_60m / 4 + 1), where comp_13_60m counts
                        competitors first posted in [anchor - 60 months, anchor - 365 days):
                        this year's volume against the average of the four years before
  All four are NaN for a focal trial without conditions.

Arm E (EXPLORATORY: country lists come from the current record's site list, which sponsors
       edit after registration, e.g. adding or dropping sites)
  comp_12m_same_country comp_12m restricted to competitors whose country set overlaps the
                        focal trial's; competitors without countries never overlap. NaN when
                        the focal trial has no country (or no condition).
  geo_missing           1 when the focal trial lists no country
"""
from __future__ import annotations

import pandas as pd

from cohort import connect

D_COLUMNS = ["comp_12m", "comp_12m_other_sponsor", "comp_busiest_cond_12m", "comp_rel_volume"]
E_COLUMNS = ["comp_12m_same_country", "geo_missing"]

NORM = "trim(regexp_replace(lower({x}), '[^a-z0-9]+', ' ', 'g'))"


def build(nct_ids: list[str]) -> pd.DataFrame:
    """Competition features for the given focal trials, one row per nct_id (sorted)."""
    con = connect()
    con.register("focal_ids", pd.DataFrame({"nct_id": sorted(nct_ids)}))
    conds = f"list_distinct(list_filter(list_transform(conditions, lambda x: {NORM.format(x='x')}), lambda x: x <> ''))"
    con.sql(f"""create temp table pool as
        select nct_id, lead_sponsor_key sp, first_post_date post, location_countries ctry, {conds} conds
        from stg_studies where study_type = 'INTERVENTIONAL' and first_post_date is not null""")
    con.sql(f"""create temp table focal as
        select s.nct_id, s.first_submit_date anchor, s.lead_sponsor_key sp,
               s.location_countries ctry, {conds} conds
        from stg_studies s join focal_ids using (nct_id)""")
    assert con.sql("select count(*) from focal").fetchall()[0][0] == len(nct_ids)
    con.sql("create temp table pc as select nct_id, sp, post, ctry, unnest(conds) cond from pool")
    con.sql("create temp table fc as select nct_id, anchor, sp, ctry, unnest(conds) cond from focal")
    # every (focal, competitor, shared condition) within 60 months before the anchor
    con.sql("""create temp table pairs as
        select f.nct_id, f.cond, p.nct_id comp,
               p.post >= f.anchor - interval 365 day as in12,
               (p.sp is null or f.sp is null or p.sp <> f.sp) as other_sponsor,
               list_has_any(p.ctry, f.ctry) as same_country
        from fc f join pc p on p.cond = f.cond and p.nct_id <> f.nct_id
         and p.post < f.anchor and p.post >= f.anchor - interval 60 month""")
    con.sql("""create temp table per_trial as
        select nct_id,
               count(distinct case when in12 then comp end) c12,
               count(distinct case when in12 and other_sponsor then comp end) c12_other,
               count(distinct case when in12 and same_country then comp end) c12_ctry,
               count(distinct case when not in12 then comp end) c13_60
        from pairs group by 1""")
    con.sql("""create temp table per_cond as
        select nct_id, max(n) busiest from (
            select nct_id, cond, count(*) n from pairs where in12 group by 1, 2) group by 1""")
    out = con.sql("""
        select f.nct_id,
          case when len(f.conds) > 0 then coalesce(t.c12, 0) end::double comp_12m,
          case when len(f.conds) > 0 then coalesce(t.c12_other, 0) end::double comp_12m_other_sponsor,
          case when len(f.conds) > 0 then coalesce(c.busiest, 0) end::double comp_busiest_cond_12m,
          case when len(f.conds) > 0 then (coalesce(t.c12, 0) + 1.0) / (coalesce(t.c13_60, 0) / 4.0 + 1.0) end
              comp_rel_volume,
          case when len(f.conds) > 0 and len(f.ctry) > 0 then coalesce(t.c12_ctry, 0) end::double
              comp_12m_same_country,
          (len(f.ctry) = 0)::double geo_missing,
          (len(f.conds) = 0)::int cond_missing
        from focal f left join per_trial t using (nct_id) left join per_cond c using (nct_id)
        order by f.nct_id""").df()
    return out
