"""Flatness of the 8-week forecast path vs actual weekly variation, on the 2025 eval origins.

DIAGNOSTIC: uses the existing 2025 eval predictions (already inspected data).
Reads only data/ml/eval/predictions; writes to data/followup/diagnostics/.

For every (method, origin, drug) path of 8 horizons:
- path_cv: std over h of the forecast mean / mean over h
- act_cv: std over the observed (uncensored) target weeks / their mean
- noise_cv: the CV that pure Poisson noise around a flat mean would give
  (sqrt(mean)/mean), and NB noise using the method's own var/mean
- signal share: (sample var of actuals - expected noise var) / sample var,
  pooled per tier (method-of-moments).
"""
import glob
import os

import numpy as np
import pandas as pd

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402

DATA = fp.DATA
PRED = f"{DATA}/ml/eval/predictions"
OUT = fp.out("diagnostics")


def load(prefix, name):
    fs = sorted(glob.glob(f"{PRED}/{prefix}_*_v2.parquet"))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    if "seed" in d and (d.seed == -1).any():
        d = d[d.seed == -1]
    d["model"] = name
    return d


ml = load("36ad7c615b96", "ml")
old = load("eval-old", "current")
base = load("eval-baseline-g1", "baseline_reselected")
allp = pd.concat([ml, old, base], ignore_index=True)
allp.to_parquet(f"{OUT}/eval_predictions_combined.parquet")

allp["disp_row"] = allp["var"] / allp["mean"].where(allp["mean"] > 0)
g =allp.groupby(["model", "origin", "drug_code", "tier"], observed=True)
path = g["mean"].agg(["mean", "std", "min", "max"])
path.columns = ["f_mean", "f_std", "f_min", "f_max"]
act = g["actual"].agg(a_mean="mean", a_var="var", a_n="count")
disp = g["disp_row"].mean()
s = path.join(act)
s["disp"] = disp
s["path_cv"] = s.f_std / s.f_mean.where(s.f_mean > 0)
s["path_range"] = (s.f_max - s.f_min) / s.f_mean.where(s.f_mean > 0)
s["act_cv"] = np.sqrt(s.a_var) / s.a_mean.where(s.a_mean > 0)
s = s.reset_index()
s.to_parquet(f"{OUT}/path_stats.parquet")

# Pooled noise test per tier, uses the ML mean as the "local mean" and the
# realized window mean as an alternative (oracle flat mean = window mean).
ok = s[(s.a_n >= 6)]
res = []
for (m, t), x in ok.groupby(["model", "tier"]):
    tot = x.a_var.sum()
    pois = x.a_mean.sum()                     # Poisson var at the realized window mean
    nb = (x.a_mean * x.disp.fillna(1)).sum()  # NB var using the drug's history dispersion
    res.append({"model": m, "tier": t, "paths": len(x),
                "median_path_cv": x.path_cv.median(),
                "p90_path_cv": x.path_cv.quantile(0.9),
                "median_path_range": x.path_range.median(),
                "median_act_cv": x.act_cv.median(),
                "sum_actual_var": tot,
                "poisson_share_of_var": pois / tot,
                "nb_share_of_var": nb / tot,
                "signal_share_poisson": max(0, 1 - pois / tot),
                "signal_share_nb": max(0, 1 - nb / tot)})
res = pd.DataFrame(res)
res.to_csv(f"{OUT}/flatness_by_tier.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))
