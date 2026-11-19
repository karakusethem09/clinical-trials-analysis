"""Print the therapeutic areas of 40 random distinct condition strings (seed 20261119), plus
coverage of the mapper among focal trials of 2008-2017. Reads conditions only, no outcomes."""
from __future__ import annotations

import disease_areas
from cohort import HERE, connect

con = connect()
conds = [r[0] for r in con.sql("""select distinct unnest(conditions) from trial_cohort
    where start_year between 2008 and 2017 and registered_by_start and terminated_5y is not null
    order by 1""").fetchall()]
sample = disease_areas.mapper_check(conds)
sample.to_csv(HERE / "mapper_check_sample.csv", index=False)
print(sample.to_string())
print(f"{len(conds)} distinct condition strings")
