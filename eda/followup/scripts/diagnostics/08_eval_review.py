"""Evaluation methodology checks. Uses the 2025 eval predictions (DIAGNOSTIC) and the
selection-period wave3 predictions (A2 vs baseline). Censored-week sales come from
agent.duckdb (data to 30 Jun 2025) and selection.duckdb, read-only.

A. Concentration: share of each tier's absolute error from its top-10% drugs by volume,
   and how much of ML's WAPE gain comes from them.
B. Censoring: share of target weeks dropped; forecast vs the recorded sales floor in
   dropped weeks; whether the drop changes the ML-vs-baseline comparison.
"""
import os
from datetime import date

import duckdb
import numpy as np
import pandas as pd

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402

DATA = fp.DATA
OUT = fp.out("diagnostics")
pd.set_option("display.width", 220)


def concentration(d, a_name, b_name, label):
    a = d[d.model == a_name].set_index(["origin", "drug_code", "h"])
    b = d[d.model == b_name].set_index(["origin", "drug_code", "h"])
    m = a[["tier", "actual", "mean"]].join(b[["mean"]], rsuffix="_b").dropna(subset=["actual"])
    m["ea"], m["eb"] = (m["mean"] - m.actual).abs(), (m["mean_b"] - m.actual).abs()
    m = m.reset_index()
    rows = []
    for t, z in m.groupby("tier", observed=True):
        vol = z.groupby("drug_code").actual.sum().sort_values(ascending=False)
        top = set(vol.index[: max(1, len(vol) // 10)])
        zt = z[z.drug_code.isin(top)]
        gain = z.eb.sum() - z.ea.sum()
        rows.append({"set": label, "tier": t, "drugs": len(vol),
                     "top10_share_of_units": zt.actual.sum() / z.actual.sum(),
                     "top10_share_of_abs_err_A": zt.ea.sum() / z.ea.sum(),
                     "wape_A": z.ea.sum() / z.actual.sum(), "wape_B": z.eb.sum() / z.actual.sum(),
                     "wape_A_rest": (z.ea.sum() - zt.ea.sum()) / (z.actual.sum() - zt.actual.sum()),
                     "wape_B_rest": (z.eb.sum() - zt.eb.sum()) / (z.actual.sum() - zt.actual.sum()),
                     "share_of_gain_from_top10": (zt.eb.sum() - zt.ea.sum()) / gain if gain else np.nan,
                     # mean per-drug WAPE-like ratio: unweighted view
                     "drugs_A_better": (z.groupby("drug_code").ea.sum() <
                                        z.groupby("drug_code").eb.sum()).mean()})
    allz = m
    rows.append({"set": label, "tier": "ALL", "drugs": allz.drug_code.nunique(),
                 "top10_share_of_units": np.nan,
                 "top10_share_of_abs_err_A": np.nan,
                 "wape_A": allz.ea.sum() / allz.actual.sum(),
                 "wape_B": allz.eb.sum() / allz.actual.sum(),
                 "fast_share_of_units": allz[allz.tier == "fast"].actual.sum() / allz.actual.sum(),
                 "fast_share_of_abs_err_A": allz[allz.tier == "fast"].ea.sum() / allz.ea.sum()})
    return pd.DataFrame(rows)


ev = pd.read_parquet(f"{OUT}/eval_predictions_combined.parquet")
se = pd.read_parquet(f"{OUT}/selection_predictions_combined.parquet")
c = pd.concat([concentration(ev, "ml", "current", "eval2025 ML vs current"),
               concentration(ev, "ml", "baseline_reselected", "eval2025 ML vs baseline"),
               concentration(se, "ml_A2", "baseline", "selection ML_A2 vs baseline")])
print(c.round(3).to_string(index=False))
c.to_csv(f"{OUT}/eval_concentration.csv", index=False)


def censor_check(d, a_name, b_name, db, end, label):
    con = duckdb.connect(f"{DATA}/warehouse/{db}", read_only=True)
    wk = con.execute(f"SELECT drug_code, week_start, units FROM weekly_demand "
                     f"WHERE week_end <= DATE '{end}'").df()
    wk["week_start"] = pd.to_datetime(wk.week_start)
    x = d[d.model.isin([a_name, b_name])].copy()
    x["target_week"] = x.origin + pd.to_timedelta(1 + 7 * (x.h - 1), unit="D")
    x = x[x.target_week + pd.Timedelta(days=6) <= pd.Timestamp(end)]
    x = x.merge(wk.rename(columns={"week_start": "target_week"}), how="left",
                on=["drug_code", "target_week"])
    x["units"] = x.units.fillna(0)
    x["dropped"] = x.actual.isna()
    rows = []
    for (m, t), z in x.groupby(["model", "tier"], observed=True):
        dz = z[z.dropped]
        rows.append({"set": label, "model": m, "tier": t,
                     "share_rows_dropped": z.dropped.mean(),
                     "share_units_in_dropped": dz.units.sum() / z.units.sum(),
                     "fc/sales_floor_dropped": dz["mean"].sum() / max(dz.units.sum(), 1),
                     "share_dropped_fc_below_floor": (dz["mean"] < dz.units).mean(),
                     "fc/actual_kept": z[~z.dropped]["mean"].sum() / z[~z.dropped].actual.sum(),
                     # WAPE if dropped weeks were scored on the floor (a lower bound on demand)
                     "wape_kept": (z[~z.dropped]["mean"] - z[~z.dropped].actual).abs().sum()
                     / z[~z.dropped].actual.sum(),
                     "wape_floor_incl": (z["mean"] - z.actual.fillna(z.units)).abs().sum()
                     / z.actual.fillna(z.units).sum(),
                     "underforecast_floor_units_share": (dz.units - dz["mean"]).clip(lower=0).sum()
                     / z.actual.fillna(z.units).sum()})
    return pd.DataFrame(rows)


cc = pd.concat([censor_check(ev, "ml", "current", "agent.duckdb", "2025-06-29", "eval2025"),
                censor_check(ev, "baseline_reselected", "baseline_reselected", "agent.duckdb",
                             "2025-06-29", "eval2025"),
                censor_check(se, "ml_A2", "baseline", "selection.duckdb", "2024-12-08", "selection")])
print(cc.round(3).to_string(index=False))
cc.to_csv(f"{OUT}/eval_censoring.csv", index=False)

# per-h WAPE: does error grow with horizon? (flat forecasts can still be fine if not)
for name, d, ms in [("eval2025", ev, ["ml", "current", "baseline_reselected"]),
                    ("selection", se, ["ml_A2", "baseline"])]:
    z = d[d.actual.notna() & d.model.isin(ms)]
    t = z.assign(e=(z["mean"] - z.actual).abs()).groupby(["model", "h"]).apply(
        lambda g: g.e.sum() / g.actual.sum(), include_groups=False).unstack("h")
    print(name, "WAPE by h (all tiers)\n", t.round(3).to_string())
    t.to_csv(f"{OUT}/wape_by_h_{name}.csv")
