"""Served ML forecast (origin 29 Jun 2025, weeks 30 Jun - 18 Aug 2025): per-drug path
flatness, store-level path, and the realized H2 2025 weeks for the same drugs.
The realized part reads irpd.duckdb (full data, H2 2025 already inspected): DIAGNOSTIC.
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
f = pd.read_csv(f"{DATA}/warehouse/ml_forecast.csv", dtype={"drug_code": str})
p = f.pivot(index="drug_code", columns="h", values="mean")
m = p.mean(axis=1)
cv = p.std(axis=1) / m
rng = (p.max(axis=1) - p.min(axis=1)) / m
print("served drugs", len(p))
print("per-drug path CV quantiles:", cv.quantile([.1, .5, .9, .99]).round(4).to_dict())
print("per-drug (max-min)/mean quantiles:", rng.quantile([.1, .5, .9, .99]).round(4).to_dict())
print("ratio h8/h1 quantiles:", (p[8] / p[1]).quantile([.1, .5, .9]).round(4).to_dict())
print("store-level path (sum over drugs):", p.sum().round(0).to_dict())

con = duckdb.connect(f"{DATA}/warehouse/irpd.duckdb", read_only=True)
wk = con.execute("""SELECT drug_code, week_start, units FROM weekly_demand
                    WHERE week_start BETWEEN DATE '2025-06-30' AND DATE '2025-08-18'""").df()
out = con.execute("""SELECT DISTINCT drug_code, date_trunc('week', date)::DATE AS week_start
                     FROM daily_stock WHERE location='total' AND closing <= 0
                     AND date BETWEEN DATE '2025-06-30' AND DATE '2025-08-24'""").df()
wk["week_start"] = pd.to_datetime(wk.week_start)
out["week_start"] = pd.to_datetime(out.week_start)
out["cens"] = True
wk = wk.merge(out, how="left", on=["drug_code", "week_start"])
weeks = pd.date_range("2025-06-30", periods=8, freq="7D")
a = wk.pivot(index="drug_code", columns="week_start", values="units").reindex(
    index=p.index, columns=weeks).fillna(0)
c = wk.pivot(index="drug_code", columns="week_start", values="cens").reindex(
    index=p.index, columns=weeks).fillna(False).astype(bool)
a.columns = range(1, 9)
c.columns = range(1, 9)
au = a.where(~c)
print("store-level actual (all drugs, censored included):", a.sum().round(0).to_dict())
print("store-level actual, uncensored only:", au.sum().round(0).to_dict())
print("store-level ML on uncensored cells:", p.where(~c).sum().round(0).to_dict())
acv = au.std(axis=1) / au.mean(axis=1)
print("per-drug actual CV median:", round(acv.median(), 3))
# July 2025 store change: compare level
print("ML total / actual total (uncensored):", round(p.where(~c).sum().sum() / au.sum().sum(), 3))
pd.DataFrame({"ml": p.sum(), "actual_all": a.sum(), "actual_uncens": au.sum(),
              "ml_uncens": p.where(~c).sum()}).to_csv(f"{OUT}/served_store_path.csv")
pd.DataFrame({"path_cv": cv, "path_range": rng, "actual_cv": acv, "mean": m}).to_csv(
    f"{OUT}/served_path_flatness.csv")
