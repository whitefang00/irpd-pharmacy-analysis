"""Within-window dispersion: is the realized week-to-week spread inside each 8-week
target window bigger than count noise around a flat mean? DIAGNOSTIC (2025 eval data).

Per (origin, drug) with >= 6 uncensored target weeks and window mean > 0:
  D_win  = sample var / sample mean of the actuals (index of dispersion; 1 = Poisson)
  D_hist = the drug's 52-week history dispersion (var/mean of the forecast, >= 1)
Also a lag-1 autocorrelation of the within-window deviations, pooled, against a
permutation null (shuffle weeks within each window), to see if deviations are smooth
(signal) or white (noise).
"""
import os
import numpy as np
import pandas as pd

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402

DATA = fp.DATA
OUT = fp.out("diagnostics")
d = pd.read_parquet(f"{OUT}/eval_predictions_combined.parquet")
ml = d[d.model == "ml"].copy()
ml["disp_row"] = ml["var"] / ml["mean"].where(ml["mean"] > 0)
eid_origins = pd.to_datetime(["2025-03-02", "2025-03-16", "2025-03-30", "2025-02-16", "2025-04-13"])
ml["eid_origin"] = ml.origin.isin(eid_origins)

g = ml.groupby(["origin", "drug_code"], observed=True)
w = g.agg(tier=("tier", "first"), n=("actual", "count"), m=("actual", "mean"),
          v=("actual", "var"), fm=("mean", "mean"), disp=("disp_row", "mean"),
          eid_any=("eid", "any")).reset_index()
w = w[(w.n >= 6) & (w.m > 0)]
w["D_win"] = w.v / w.m
w["D_ratio"] = w.v / (w.m * w.disp)
out = w.groupby(["tier", "eid_any"]).agg(paths=("D_win", "size"),
                                         med_D_win=("D_win", "median"),
                                         med_D_hist=("disp", "median"),
                                         med_var_over_NB=("D_ratio", "median"),
                                         share_D_win_lt2=("D_win", lambda s: (s < 2).mean()))
print(out.round(3).to_string())
out.to_csv(f"{OUT}/window_dispersion.csv")

# lag-1 autocorrelation of within-window deviations (scaled), pooled per tier
ml = ml.sort_values(["origin", "drug_code", "h"])
ml["dev"] = ml.actual - ml.groupby(["origin", "drug_code"], observed=True).actual.transform("mean")
ml["sd"] = ml.groupby(["origin", "drug_code"], observed=True).actual.transform("std")
ml["z"] = ml.dev / ml.sd.where(ml.sd > 0)
rng = np.random.default_rng(0)


def lag1(df):
    z = df.z.to_numpy()
    same = (df.drug_code.to_numpy()[1:] == df.drug_code.to_numpy()[:-1]) & \
           (df.origin.to_numpy()[1:] == df.origin.to_numpy()[:-1]) & \
           (np.diff(df.h.to_numpy()) == 1)
    a, b = z[:-1][same], z[1:][same]
    ok = ~np.isnan(a) & ~np.isnan(b)
    return np.corrcoef(a[ok], b[ok])[0, 1]


rows = []
for t, x in ml.groupby("tier", observed=True):
    x = x[x.z.notna()]
    obs = lag1(x)
    null = []
    for _ in range(50):
        y = x.copy()
        y["z"] = y.groupby(["origin", "drug_code"], observed=True).z.transform(
            lambda s: rng.permutation(s.to_numpy()))
        null.append(lag1(y))
    rows.append({"tier": t, "lag1_acf": obs, "null_mean": np.mean(null), "null_sd": np.std(null)})
r = pd.DataFrame(rows)
print(r.round(4).to_string(index=False))
r.to_csv(f"{OUT}/window_lag1_acf.csv", index=False)
