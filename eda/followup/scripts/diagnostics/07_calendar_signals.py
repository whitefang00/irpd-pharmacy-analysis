"""EXPLORATORY: are there calendar signals the model lacks? Selection data only
(selection.duckdb, weeks ending <= 8 Dec 2024).

A. Store level: weekly total units, divided by a centred 9-week rolling median,
   regressed on payday share (days with day-of-month >= 25 or <= 2), national-holiday
   days in the week, and a year-end flag. Eid windows (-7..+28) are excluded.
B. Drug level yearly pattern: for fast+regular drugs, monthly units / the drug's mean
   in 2023 vs 2024 (Mar-Apr-May excluded because Eid moved), correlation of the two
   years' month profiles per drug vs a month-permutation null.
"""
import os
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

from irpd import forecast as fc

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402

DATA = fp.DATA
OUT = fp.out("diagnostics")
con = duckdb.connect(f"{DATA}/warehouse/selection.duckdb", read_only=True)

HOL = [  # main Indonesian national holidays 2023-2024 (approximate list, exploratory)
    "2023-01-01", "2023-01-22", "2023-02-18", "2023-03-22", "2023-04-07", "2023-05-01",
    "2023-05-18", "2023-06-01", "2023-06-04", "2023-06-29", "2023-07-19", "2023-08-17",
    "2023-09-28", "2023-12-25", "2023-12-26",
    "2024-01-01", "2024-02-08", "2024-02-10", "2024-02-14", "2024-03-11", "2024-03-29",
    "2024-05-01", "2024-05-09", "2024-05-23", "2024-06-01", "2024-06-17", "2024-07-07",
    "2024-08-17", "2024-09-16", "2024-11-27"]
HOL = {pd.Timestamp(h).date() for h in HOL}

wk = con.execute("""SELECT week_start, sum(units) u FROM weekly_demand
                    WHERE week_end <= DATE '2024-12-08' GROUP BY 1 ORDER BY 1""").df()
wk["week_start"] = pd.to_datetime(wk.week_start).dt.date
wk["rel"] = wk.u / wk.u.rolling(9, center=True).median()
wk["eid"] = [any(e + timedelta(days=-7) <= w + timedelta(days=6) and w <= e + timedelta(days=28)
                 for e in fc.EID) for w in wk.week_start]
days = lambda w: [w + timedelta(days=i) for i in range(7)]
wk["payday"] = [sum(d.day >= 25 or d.day <= 2 for d in days(w)) / 7 for w in wk.week_start]
wk["hol"] = [sum(d in HOL for d in days(w)) for w in wk.week_start]
wk["yearend"] = [any((d.month == 12 and d.day >= 24) or (d.month == 1 and d.day <= 3)
                     for d in days(w)) for w in wk.week_start]
z = wk[~wk.eid & wk.rel.notna()].iloc[1:]  # drop the partial first week
X = sm.add_constant(z[["payday", "hol"]].assign(yearend=z.yearend.astype(float)))
fit = sm.OLS(z.rel, X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
print(fit.summary().tables[1])
print("R2", round(fit.rsquared, 3), "n", len(z), "sd of rel (non-Eid):", round(z.rel.std(), 4))
r = z.rel - 1
print("lag-1 autocorr of detrended store weeks:", round(r.autocorr(1), 3))
wk.to_csv(f"{OUT}/store_weekly_calendar.csv", index=False)
with open(f"{OUT}/store_calendar_regression.txt", "w") as fh:
    fh.write(fit.summary().as_text())

# B. yearly pattern at drug level
m = con.execute("""SELECT drug_code, date_trunc('month', week_start)::DATE mo, sum(units) u
                   FROM weekly_demand WHERE week_end <= DATE '2024-11-30' GROUP BY ALL""").df()
m["mo"] = pd.to_datetime(m.mo)
m = m[m.mo >= "2023-01-01"]
p = m.pivot(index="drug_code", columns="mo", values="u").fillna(0)
months = [1, 2, 6, 7, 8, 9, 10, 11]
a = p[[pd.Timestamp(2023, k, 1) for k in months]].to_numpy(float)
b = p[[pd.Timestamp(2024, k, 1) for k in months]].to_numpy(float)
keep = (a.mean(1) >= 20) & (b.mean(1) >= 20)  # roughly >= 5 units a week in both years
a, b = a[keep] / a[keep].mean(1, keepdims=True), b[keep] / b[keep].mean(1, keepdims=True)
# remove the store-wide month pattern so the test is about drug-specific seasonality
a2, b2 = a - a.mean(0), b - b.mean(0)


def rowcorr(x, y):
    x = x - x.mean(1, keepdims=True)
    y = y - y.mean(1, keepdims=True)
    return (x * y).sum(1) / np.sqrt((x * x).sum(1) * (y * y).sum(1))


obs_raw = np.nanmedian(rowcorr(a, b))
obs = np.nanmedian(rowcorr(a2, b2))
rng = np.random.default_rng(0)
null = [np.nanmedian(rowcorr(a2, b2[:, rng.permutation(len(months))])) for _ in range(200)]
print(f"drugs {keep.sum()}: median year-to-year corr of month profile raw {obs_raw:.3f}, "
      f"store-pattern removed {obs:.3f}; null {np.mean(null):.3f} +- {np.std(null):.3f}")
store = pd.Series(a.mean(0), index=months)
print("store-wide month profile 2023 (rel):", store.round(3).to_dict())
print("store-wide month profile 2024 (rel):", pd.Series(b.mean(0), index=months).round(3).to_dict())
cc = rowcorr(a2, b2)
print("share of drugs with corr > 0.5:", round(np.nanmean(cc > 0.5), 3))
pd.DataFrame({"drug_code": p.index[keep], "corr_raw": rowcorr(a, b), "corr_resid": cc}).to_csv(
    f"{OUT}/drug_month_profile_corr.csv", index=False)
