"""Same diagnostics on the SELECTION period (11 origins, 7 Jan - 13 Oct 2024, targets
<= 8 Dec 2024): wave3 A2 predictions (3aafa0367da7 = the frozen feature set, Tweedie 1.3,
selection WAPE matches ablation A2) and the re-selected baseline (baseline-g1).
Also the level-response test, which needs history from selection.duckdb (read-only).
"""
import glob
import os
import warnings
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd

from irpd import forecast as fc

warnings.simplefilter("ignore")
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402

DATA = fp.DATA
OUT = fp.out("diagnostics")
P = f"{DATA}/ml/wave3/predictions"


def load(h, name):
    d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{P}/{h}_*"))], ignore_index=True)
    if (d.seed == -1).any():
        d = d[d.seed == -1]
    d["model"] = name
    return d


d = pd.concat([load("3aafa0367da7", "ml_A2"), load("baseline-g1", "baseline")], ignore_index=True)
d["disp"] = d["var"] / d["mean"].where(d["mean"] > 0)
d.to_parquet(f"{OUT}/selection_predictions_combined.parquet")

# 1. per-path flatness vs actual
g = d.groupby(["model", "origin", "drug_code"], observed=True)
s = g.agg(tier=("tier", "first"), fm=("mean", "mean"), fs=("mean", "std"), n=("actual", "count"),
          am=("actual", "mean"), av=("actual", "var"), disp=("disp", "mean"),
          eid=("eid", "any")).reset_index()
s["path_cv"] = s.fs / s.fm.where(s.fm > 0)
s["act_cv"] = np.sqrt(s.av) / s.am.where(s.am > 0)
ok = s[(s.n >= 6) & (s.am > 0)].copy()
ok["D_win"] = ok.av / ok.am
ok["var_over_nb"] = ok.av / (ok.am * ok.disp)
ok["poisson_cv"] = 1 / np.sqrt(ok.am)
t1 = ok.groupby(["model", "tier", "eid"]).agg(
    paths=("path_cv", "size"), med_path_cv=("path_cv", "median"), med_act_cv=("act_cv", "median"),
    med_poisson_cv=("poisson_cv", "median"), med_D_win=("D_win", "median"),
    med_D_hist=("disp", "median"), med_var_over_nb=("var_over_nb", "median"))
pd.set_option("display.width", 220)
print(t1.round(3).to_string())
t1.to_csv(f"{OUT}/selection_flatness.csv")

# 2. aggregate path per origin (store level, uncensored rows only)
x = d[d.actual.notna() & d.tier.isin(["fast", "regular", "slow"])]
agg = x.groupby(["origin", "h", "model"]).agg(f=("mean", "sum"), a=("actual", "sum")).reset_index()
w = agg.pivot_table(index=["origin", "h"], columns="model", values="f").join(
    agg[agg.model == "ml_A2"].set_index(["origin", "h"]).a).reset_index()
cv = w.groupby("origin").agg(cv_actual=("a", lambda v: v.std() / v.mean()),
                             cv_ml=("ml_A2", lambda v: v.std() / v.mean()),
                             cv_base=("baseline", lambda v: v.std() / v.mean()))
print(cv.round(3).to_string())
w.to_csv(f"{OUT}/selection_aggregate_paths.csv", index=False)
cv.to_csv(f"{OUT}/selection_aggregate_cv.csv")
# how much of the aggregate actual's weekly variation the ML path explains
w["a_dev"] = w.a / w.groupby("origin").a.transform("mean") - 1
w["m_dev"] = w.ml_A2 / w.groupby("origin").ml_A2.transform("mean") - 1
print("corr of within-origin deviations, store level (ML path vs actual):",
      round(np.corrcoef(w.a_dev, w.m_dev)[0, 1], 3))

# 3. level response: how forecasts and outcomes load on recent vs long means
con = duckdb.connect(f"{DATA}/warehouse/selection.duckdb", read_only=True)
hist = fc.load_history(con, date(2024, 12, 8))
y = np.where(hist["out"], np.nan, hist["y"])
weeks = list(hist["weeks"])
idx = {c: i for i, c in enumerate(hist["drugs"])}
feats = []
for o in sorted(d.origin.unique()):
    o = pd.Timestamp(o).date()
    n = sum(wk + timedelta(days=6) <= o for wk in weeks)
    yy = y[:, :n]
    f = {}
    for k in (4, 13, 52):
        f[f"m{k}"] = np.nanmean(yy[:, -k:], axis=1)
    f["ses"] = fc._ses(yy)
    df = pd.DataFrame(f)
    df["drug_code"] = hist["drugs"]
    df["origin"] = pd.Timestamp(o)
    feats.append(df)
feats = pd.concat(feats)
L = s.merge(feats, on=["drug_code", "origin"])
L = L[(L.n >= 6) & (L.m52 > 0) & L.m4.notna() & L.m13.notna()]
rows = []
for (m, t), z in L.groupby(["model", "tier"]):
    X = np.column_stack([z.m4, z.m13, z.m52]) / z.m52.to_numpy()[:, None]
    # relative-to-m52 regressions (so big drugs don't dominate), no intercept
    bf = np.linalg.lstsq(X, z.fm / z.m52, rcond=None)[0]
    ba = np.linalg.lstsq(X, z.am / z.m52, rcond=None)[0]
    xs = np.clip(z.m4 / z.m52 - 1, -1, 3)
    slope_f = np.polyfit(xs, z.fm / z.m52 - 1, 1)[0]
    slope_a = np.polyfit(xs, z.am / z.m52 - 1, 1)[0]
    rows.append({"model": m, "tier": t, "n": len(z),
                 "fc_w_m4": bf[0], "fc_w_m13": bf[1], "fc_w_m52": bf[2],
                 "act_w_m4": ba[0], "act_w_m13": ba[1], "act_w_m52": ba[2],
                 "slope_fc_on_recent_shift": slope_f, "slope_act_on_recent_shift": slope_a})
r = pd.DataFrame(rows)
print(r.round(3).to_string(index=False))
r.to_csv(f"{OUT}/selection_level_response.csv", index=False)
