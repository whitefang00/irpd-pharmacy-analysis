"""Proposed segmentation (as of 8 Dec 2024) and an exploratory check of simple flat
methods per segment on the 11 selection origins. Selection data only.

Segment labels use statistics computed to 8 Dec 2024, so slicing the 2024 origins
by them carries a little hindsight (labels, not forecasts). Exploratory.

Needs drug_stats.py and signal_decomp.py first. Writes to data/followup/demand/.
Run from the repo root:  uv run python eda/followup/scripts/demand/segment.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from irpd import forecast as fc
from irpd.ml import history, split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", "{:,.3f}".format)
OUT = fp.out("demand")
END = split.SELECTION_END
T = 101

A = pd.read_parquet(OUT / "active_stats.parquet")
SZ = pd.read_parquet(OUT / "size_factor.parquet")
A = A.join(SZ.size_factor)
A["break_big"] = (A.break_F > 15) & ((A.break_ratio > 1.5) | (A.break_ratio < 1 / 1.5))

ORDER = ["new", "fading", "censored", "bulk", "steady-stable", "steady-shifted",
         "intermittent", "sporadic"]


def seg(r):
    if r.first_week >= T - 26:
        return "new"
    if r.weeks_since_last_sale >= 13:
        return "fading"
    if r.cens_share >= 0.30:
        return "censored"
    if r.bulk_unit_share >= 0.20:
        return "bulk"
    if r.sb_class in ("smooth", "erratic") and r.mu52 >= 2:
        return "steady-shifted" if r.break_big else "steady-stable"
    if r.tier == "rare" or r.sb_class == "too few" or r.zero_share52 >= 0.85:
        return "sporadic"
    return "intermittent"


A["segment"] = pd.Categorical(A.apply(seg, axis=1), ORDER)
A[["segment"]].to_csv(OUT / "segments.csv")
A.to_parquet(OUT / "active_stats_seg.parquet")
TR, TU = A.rev52.sum(), A.units52.sum()
g = A.groupby("segment", observed=False)
tab = pd.DataFrame({
    "drugs": g.size(), "revenue %": 100 * g.rev52.sum() / TR, "units %": 100 * g.units52.sum() / TU,
    "median units/wk": g.mu52.median(), "median zero wks": g.zero_share52.median(),
    "median cens": g.cens_share.median(), "median var/mean": g.disp52.median(),
    "median size factor": g.size_factor.median(), "median acf1": g.acf1.median()})
print("== segments, drugs active in the 52 weeks to 8 Dec 2024")
print(tab)
tab.to_csv(OUT / "segments_summary.csv")
print("\nsegment x tier")
print(pd.crosstab(A.segment, A.tier).reindex(columns=["fast", "regular", "slow", "rare"]))
print("\nsegment x SB class")
print(pd.crosstab(A.segment, A.sb_class))

# ---- exploratory: flat methods by segment over 11 selection origins ----------
con = split.selection_con()
H = history.load(con, END)
DRUGS, Y = H["drugs"], H["y"]
YC = np.where(H["out"], np.nan, Y)
EIDX = np.load(OUT / "eid_index.npy")
segmap = A.segment.reindex(DRUGS).astype(object).to_numpy()
SF = A.size_factor.reindex(DRUGS).fillna(1).to_numpy()
METH = {
    "ses .05": lambda p: fc._ses(p, 0.05),
    "ses .15": lambda p: fc._ses(p, 0.15),
    "ses .30": lambda p: fc._ses(p, 0.30),
    "mean 13": lambda p: np.nanmean(p[:, -13:], axis=1),
    "mean 52": lambda p: np.nanmean(p[:, -52:], axis=1),
    "croston sba": lambda p: fc._croston_sba(p, 0.15),
    "tsb": lambda p: fc._tsb(p, 0.15, 0.15),
    "median 13": lambda p: np.nanmedian(p[:, -13:], axis=1),
}
acc = {}
for o in split.selection_origins():
    n = history.weeks_to(H, o)
    past = YC[:, :n]
    tgt = YC[:, n:n + 8]
    sold13 = np.nansum(Y[:, n - 13:n], axis=1) > 0
    fcs = {k: np.nan_to_num(f(past)) for k, f in METH.items()}
    for s in ORDER:
        rows = sold13 & (segmap == s)
        t = tgt[rows]
        m = ~np.isnan(t)
        d = acc.setdefault(s, {"act": 0.0, "noise": 0.0, "n": 0})
        d["act"] += np.nansum(t)
        d["n"] += int(m.sum())
        ybar = np.nanmean(t, axis=1)
        d["noise"] += np.nansum(np.where(m, (np.nan_to_num(ybar) * SF[rows])[:, None], 0))
        for k, f in fcs.items():
            e = np.where(m, t - f[rows][:, None], np.nan)
            d.setdefault(k + " abs", 0.0)
            d.setdefault(k + " sq", 0.0)
            d.setdefault(k + " sum", 0.0)
            d[k + " abs"] += np.nansum(np.abs(e))
            d[k + " sq"] += np.nansum(e ** 2)
            d[k + " sum"] += np.nansum(np.where(m, f[rows][:, None], 0))
wape, ratio, bias = {}, {}, {}
for s, d in acc.items():
    wape[s] = {k: d[k + " abs"] / d["act"] for k in METH}
    ratio[s] = {k: d[k + " sq"] / d["noise"] for k in METH}
    bias[s] = {k: d[k + " sum"] / d["act"] for k in METH}
print("\n== exploratory: WAPE of flat 8-week forecasts by segment (11 selection origins)")
print(pd.DataFrame(wape).T.reindex(ORDER))
print("\n== MSE / compound-Poisson noise floor (1.0 = nothing left to win)")
print(pd.DataFrame(ratio).T.reindex(ORDER))
print("\n== bias (forecast / actual)")
print(pd.DataFrame(bias).T.reindex(ORDER))
pd.DataFrame(wape).T.reindex(ORDER).to_csv(OUT / "segment_flat_wape.csv")
pd.DataFrame(ratio).T.reindex(ORDER).to_csv(OUT / "segment_mse_over_floor.csv")

# ---- lifecycle measured at each origin (no hindsight) -------------------------
print("\n== at-origin lifecycle: drugs with first sale < 26 weeks before origin vs older "
      "(sold in last 13 weeks), flat SES .15 / .30 / mean of weeks since first sale")
fs = H["first_sale"]
res = {}
for o in split.selection_origins():
    n = history.weeks_to(H, o)
    past, tgt = YC[:, :n], YC[:, n:n + 8]
    sold13 = np.nansum(Y[:, n - 13:n], axis=1) > 0
    age = n - fs
    for lab, rows in (("young <26w", sold13 & (fs >= 0) & (age < 26)),
                      ("young <13w", sold13 & (fs >= 0) & (age < 13)),
                      ("older", sold13 & (age >= 26))):
        d = res.setdefault(lab, {"act": 0, "n": 0})
        t = tgt[rows]
        d["act"] += np.nansum(t)
        d["n"] += rows.sum()
        for k, f in (("ses .15", fc._ses(past, .15)), ("ses .30", fc._ses(past, .30)),
                     ("ses .50", fc._ses(past, .50)), ("mean 4", np.nanmean(past[:, -4:], axis=1))):
            fr = np.nan_to_num(f[rows])[:, None]
            d[k + " abs"] = d.get(k + " abs", 0) + np.nansum(np.abs(t - fr))
            d[k + " sum"] = d.get(k + " sum", 0) + np.nansum(np.where(np.isnan(t), 0, fr))
for lab, d in res.items():
    print(f"  {lab:11s} drug-origins={d['n']:5d} " + "  ".join(
        f"{k}: WAPE {d[k+' abs']/d['act']:.3f} bias {d[k+' sum']/d['act']:.2f}"
        for k in ("ses .15", "ses .30", "ses .50", "mean 4")))
