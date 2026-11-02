"""Rolling H2 2025 forecasts (ml_rolling.csv, 4-week totals per weekly origin) vs realized
4-week totals from irpd.duckdb. DIAGNOSTIC: H2 2025 has been inspected before.
Checks how fast each forecaster's level follows the realized level (store and drug).
"""
import os
import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402

DATA = fp.DATA
OUT = fp.out("diagnostics")
r = pd.read_csv(f"{DATA}/warehouse/ml_rolling.csv", dtype={"drug_code": str})
r["week_start"] = pd.to_datetime(r.week_start)
con = duckdb.connect(f"{DATA}/warehouse/irpd.duckdb", read_only=True)
wk = con.execute("SELECT drug_code, week_start, units FROM weekly_demand "
                 "WHERE week_start >= DATE '2025-06-30'").df()
wk["week_start"] = pd.to_datetime(wk.week_start)
p = wk.pivot(index="drug_code", columns="week_start", values="units").fillna(0)
cols = sorted(p.columns)
tot4 = {}
for i, c in enumerate(cols[:-3]):
    tot4[c] = p[cols[i:i + 4]].sum(axis=1)
A = pd.DataFrame(tot4).stack().rename("actual").reset_index()
A.columns = ["drug_code", "week_start", "actual"]
m = r.merge(A, on=["drug_code", "week_start"], how="left").fillna({"actual": 0})
both = m[m.ml.notna()]
s = both.groupby("week_start")[["current", "ml", "actual"]].sum()
print(s.round(0).to_string())
print("store level: ml/actual", (s.ml / s.actual).round(3).describe().round(3).to_dict())
# drug level: correlation of forecast change with realized change of the level
both = both.sort_values(["drug_code", "week_start"])
for col in ["current", "ml"]:
    both[f"d_{col}"] = both.groupby("drug_code")[col].diff(4)
both["d_act_prev"] = both.groupby("drug_code").actual.shift(4)  # the 4 weeks just before
# lag test: does the forecast at origin t correlate more with actual(t-4..t-1) or actual(t..t+3)?
rows = []
for col in ["current", "ml"]:
    z = both.dropna(subset=["d_act_prev"])
    z = z[z.groupby("drug_code").actual.transform("mean") >= 8]  # >= 2 units/week
    rel = lambda v: v / z.groupby("drug_code")[col].transform("mean")
    f = z[col] - z.groupby("drug_code")[col].transform("mean")
    a_now = z.actual - z.groupby("drug_code").actual.transform("mean")
    a_prev = z.d_act_prev - z.groupby("drug_code").d_act_prev.transform("mean")
    rows.append({"forecaster": col, "drugs": z.drug_code.nunique(),
                 "corr_with_future_4wk": np.corrcoef(f, a_now)[0, 1],
                 "corr_with_past_4wk": np.corrcoef(f, a_prev)[0, 1],
                 "sd_fc_over_sd_actual_within_drug": f.std() / a_now.std(),
                 "corr_future_with_past_actual": np.corrcoef(a_now, a_prev)[0, 1]})
t = pd.DataFrame(rows)
print(t.round(3).to_string(index=False))
t.to_csv(f"{OUT}/rolling_tracking.csv", index=False)
s.to_csv(f"{OUT}/rolling_store_level.csv")
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(s.index, s.actual, "k-o", ms=3, label="actual 4-week total")
ax.plot(s.index, s.ml, c="#2a6fdb", label="ML")
ax.plot(s.index, s.current, "--", c="#d9822b", label="current")
ax.set_title("Rolling 4-week totals, summed over drugs the ML covers (H2 2025, diagnostic)", fontsize=9)
ax.legend(fontsize=8)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(f"{fp.fig('diagnostics')}/rolling_store_level_h2_2025.png", dpi=120)
