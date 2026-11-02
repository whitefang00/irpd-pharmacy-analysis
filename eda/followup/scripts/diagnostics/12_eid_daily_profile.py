"""Daily store-wide units around each Eid relative to the same weekday's mean over days
-63..-15 (agent.duckdb, data to 30 Jun 2025; 2025 is DIAGNOSTIC). Shows where the
pre-Eid peak falls and how Monday-start weeks split it."""
import os
from datetime import date, timedelta
import duckdb
import pandas as pd

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402

DATA = fp.DATA
OUT = fp.out("diagnostics")
con = duckdb.connect(f"{DATA}/warehouse/agent.duckdb", read_only=True)
dd = con.execute("SELECT date, sum(units) u, sum(lines) l FROM daily_demand GROUP BY 1").df()
dd["date"] = pd.to_datetime(dd.date).dt.date
s = dd.set_index("date")
rows = []
for e in [date(2023, 4, 22), date(2024, 4, 10), date(2025, 3, 31)]:
    base = s.loc[[e + timedelta(days=k) for k in range(-63, -14) if e + timedelta(days=k) in s.index]]
    bw = base.groupby([d.weekday() for d in base.index]).mean()
    for k in range(-14, 15):
        d = e + timedelta(days=k)
        v = s.u.get(d, 0.0)
        rows.append({"eid": e.isoformat(), "k": k, "date": d.isoformat(), "dow": d.strftime("%a"),
                     "units_rel": v / bw.u[d.weekday()],
                     "week_monday": (d - timedelta(days=d.weekday())).isoformat()})
r = pd.DataFrame(rows)
r.to_csv(f"{OUT}/eid_daily_profile.csv", index=False)
print(r.pivot(index="k", columns="eid", values="units_rel").round(2).to_string())
wk = r.groupby(["eid", "week_monday"]).agg(days=("k", lambda x: f"{x.min()}..{x.max()}"),
                                          rel=("units_rel", "mean")).reset_index()
print(wk.round(2).to_string(index=False))
