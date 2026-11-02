"""Oracle test: is week-to-week variation inside an 8-week window explained by a
compound-Poisson process with a CONSTANT rate (customers arrive at random, each buys a
random quantity)? If yes, an oracle that knew the true mean would forecast a flat path.
Selection data only (2024-01-01 .. 2024-12-08), read-only.

Per drug: s = base units per sale line (sales_lines.qty x pack_sizes.base_per_unit).
Compound-Poisson index of dispersion D_cp = E[s^2] / E[s].
Observed: weekly units in non-overlapping 8-week blocks (stock-out weeks dropped,
blocks with >= 6 weeks), pooled per drug as sum(within-block var) / sum(block mean)
with block var using ddof=1, so a constant-rate compound Poisson gives D_obs ~ D_cp.
Also the same for weekly sale LINES (Poisson arrivals give ~1).

Run twice from the repo root: once as is, and once with the two blocks around Eid
2024 left out (writes the *_noeid.csv files that 11_figures.py reads):
    uv run python eda/followup/scripts/diagnostics/10_compound_poisson.py
    SKIP_BLOCKS=1,2 uv run python eda/followup/scripts/diagnostics/10_compound_poisson.py
"""
import os
import duckdb
import numpy as np
import pandas as pd

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402

DATA = fp.DATA
OUT = fp.out("diagnostics")
con = duckdb.connect(f"{DATA}/warehouse/selection.duckdb", read_only=True)
s = con.execute("""
  SELECT l.drug_code, sum(l.qty * coalesce(k.base_per_unit, 1)) AS s1,
         sum(power(l.qty * coalesce(k.base_per_unit, 1), 2)) AS s2, count(*) AS n
  FROM sales_lines l LEFT JOIN pack_sizes k ON k.drug_code = l.drug_code AND k.sales_unit = l.unit
  WHERE l.date BETWEEN DATE '2024-01-01' AND DATE '2024-12-08' AND l.qty > 0 AND NOT l.is_service
  GROUP BY 1""").df()
s["D_cp"] = s.s2 / s.s1
s["mean_size"] = s.s1 / s.n
w = con.execute("""
  SELECT w.drug_code, w.week_start, w.units, w.lines,
         EXISTS (SELECT 1 FROM daily_stock d WHERE d.drug_code = w.drug_code AND d.location='total'
                 AND d.closing <= 0 AND d.date BETWEEN w.week_start AND w.week_end) AS out
  FROM weekly_demand w
  WHERE w.week_start >= DATE '2024-01-01' AND w.week_end <= DATE '2024-12-08'""").df()
w["week_start"] = pd.to_datetime(w.week_start)
weeks = pd.date_range("2024-01-01", "2024-12-02", freq="7D")
blk = {wk: i // 8 for i, wk in enumerate(weeks)}
U = w.pivot(index="drug_code", columns="week_start", values="units").reindex(columns=weeks)
L = w.pivot(index="drug_code", columns="week_start", values="lines").reindex(columns=weeks)
O = w.pivot(index="drug_code", columns="week_start", values="out").reindex(columns=weeks)
first = U.notna().idxmax(axis=1)
U, L = U.fillna(0), L.fillna(0)
mask = O.fillna(False).astype(bool).to_numpy() | (U.columns.values[None, :] < first.values[:, None])
Uv = np.where(mask, np.nan, U.to_numpy(float))
Lv = np.where(mask, np.nan, L.to_numpy(float))
res = []
nb = len(weeks) // 8
SKIP = {int(x) for x in os.environ.get("SKIP_BLOCKS", "").split(",") if x}
for arr, name in [(Uv, "units"), (Lv, "lines")]:
    vs, ms = np.zeros(len(arr)), np.zeros(len(arr))
    for b in [b for b in range(nb) if b not in SKIP]:
        x = arr[:, b * 8:(b + 1) * 8]
        ok = (~np.isnan(x)).sum(1) >= 6
        v = np.nanvar(x, axis=1, ddof=1)
        m = np.nanmean(x, axis=1)
        vs += np.where(ok, v, 0)
        ms += np.where(ok, m, 0)
    res.append(pd.Series(vs / np.where(ms > 0, ms, np.nan), index=U.index, name=f"D_obs_{name}"))
d = pd.concat(res, axis=1).join(s.set_index("drug_code")[["D_cp", "mean_size", "n"]])
tiers = dict(con.execute("SELECT drug_code, tier FROM demand_tiers").fetchall())
d["tier"] = d.index.map(tiers)
d["weekly_mean"] = np.nanmean(Uv, axis=1)
d = d[d.weekly_mean > 0.2].dropna(subset=["D_obs_units", "D_cp"])
d["ratio"] = d.D_obs_units / d.D_cp
out = d.groupby("tier").agg(drugs=("ratio", "size"), weekly_mean=("weekly_mean", "median"),
                            mean_size=("mean_size", "median"),
                            D_obs_units=("D_obs_units", "median"), D_cp=("D_cp", "median"),
                            obs_over_cp=("ratio", "median"),
                            obs_over_cp_p25=("ratio", lambda r: r.quantile(.25)),
                            obs_over_cp_p75=("ratio", lambda r: r.quantile(.75)),
                            D_obs_lines=("D_obs_lines", "median"))
pd.set_option("display.width", 200)
print(out.round(3).to_string())
out.to_csv(f"{OUT}/compound_poisson_check{os.environ.get('SKIP_BLOCKS', '') and '_noeid'}.csv")
d.to_csv(f"{OUT}/compound_poisson_by_drug{os.environ.get('SKIP_BLOCKS', '') and '_noeid'}.csv")
# implied share of within-window variance that is "signal" (rate movement)
d["signal_share"] = (1 - 1 / d.ratio).clip(lower=0)
print("median signal share of within-window variance by tier:",
      d.groupby("tier").signal_share.median().round(3).to_dict())
print("unit-weighted signal share by tier:",
      d.groupby("tier").apply(lambda z: np.average(z.signal_share, weights=z.weekly_mean),
                              include_groups=False).round(3).to_dict())
