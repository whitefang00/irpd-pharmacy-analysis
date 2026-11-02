"""Representative drugs: stats table (selection data, to 8 Dec 2024) and weekly plots.

The plots also show 2025 actuals and existing forecasts. Everything after
8 Dec 2024 is DIAGNOSTIC ONLY (that data has been inspected before).

Needs drug_stats.py, signal_decomp.py and segment.py first. Writes representative_stats.csv to
data/followup/demand/ and the figure to eda/followup/fig/demand/.
Run from the repo root:  uv run python eda/followup/scripts/demand/representative.py
"""
import glob
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from irpd import config, warehouse
from irpd.ml import history

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402

STATS = OUT = fp.out("demand")
FIG = fp.fig("demand")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)

PICKS = {
    "SW2106000809": "steady-stable",
    "SW2106000902": "steady-shifted",
    "OBT2108210081": "intermittent",
    "SW2106001787": "bulk",
    "OBT2306190002": "censored",
}
SEL_END = date(2024, 12, 8)

A = pd.read_parquet(STATS / "active_stats_seg.parquet")
R = pd.read_parquet(STATS / "oos_decomp_rows.parquet")
D = pd.read_parquet(STATS / "insample_decomp.parquet")
codes = list(PICKS)

# ---- stats table (selection data) ----
cols = ["drug_name", "tier", "sb_class", "segment", "rev52", "mu52", "zero_share52",
        "disp52", "size_factor", "cens_share", "shelf_week_share", "bulk_unit_share",
        "acf1", "acf4", "acf52", "half_ratio", "break_ratio", "break_pos", "trend_t",
        "eid2023", "eid2024", "adi", "cv2"]
S = A.loc[codes, cols].copy()
r = R[R.drug_code.isin(codes)].groupby("drug_code").agg(
    oos_mse_flat=("mse_flat", "mean"), oos_noise=("noise", "mean"), oos_n=("k", "size"))
S = S.join(r)
S["oos_mse_over_floor"] = S.oos_mse_flat / S.oos_noise
d = D.loc[[c for c in codes if c in D.index]]
S["insample_floor_share"] = d.floor / d.ss_tot
S["insample_trend_share"] = (d.ss_eid - d.ss_trend) / d.ss_tot
S["insample_eid_share"] = (d.ss_tot - d.ss_eid) / d.ss_tot

# ---- 2025 diagnostic: WAPE of ML (36ad7c615b96) and current baseline (eval-old) ----
P = []
for f in glob.glob(str(fp.ML / "eval/predictions/*.parquet")):
    x = pd.read_parquet(f)
    P.append(x[x.drug_code.isin(codes)])
P = pd.concat(P)
P["origin"] = pd.to_datetime(P.origin)
P = (P.groupby(["config_hash", "drug_code", "origin", "h"], as_index=False)
     .agg(mean=("mean", "mean"), actual=("actual", "first")))
for ch, lab in (("36ad7c615b96", "ml"), ("eval-old", "current"), ("eval-baseline-g1", "baseline_g1")):
    x = P[(P.config_hash == ch) & P.actual.notna()]
    w = x.groupby("drug_code").apply(lambda g: (g["mean"] - g.actual).abs().sum() / g.actual.sum())
    S[f"diag2025_wape_{lab}"] = w
    # how flat: range of h1..h8 means within an origin, relative to the mean
    fl = x.groupby(["drug_code", "origin"])["mean"].agg(lambda s: (s.max() - s.min()) / s.mean())
    S[f"diag2025_h_spread_{lab}"] = fl.groupby("drug_code").median()
S.to_csv(OUT / "representative_stats.csv")
with pd.option_context("display.float_format", "{:,.3f}".format):
    print(S.T)

# ---- full weekly history (diagnostic for 2025) ----
con = warehouse.as_of(date(2025, 12, 28), fp.WAREHOUSE / "irpd.duckdb")
H = history.load(con, date(2025, 12, 28), codes)
weeks = pd.to_datetime(pd.Series(H["weeks"]))
EID = [pd.Timestamp(e) for e in (date(2023, 4, 22), date(2024, 4, 10), date(2025, 3, 31))]

served = pd.read_csv(fp.WAREHOUSE / "ml_forecast.csv", parse_dates=["week_start"])
roll = pd.read_csv(fp.WAREHOUSE / "ml_rolling.csv", parse_dates=["origin", "week_start"])

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
C_ML, C_CUR, C_OUT, C_SHELF = "#2a78d6", "#eb6834", "#e34948", "#eda100"

fig, axes = plt.subplots(len(codes), 1, figsize=(13, 3.1 * len(codes)), sharex=True)
for ax, code in zip(axes, codes):
    i = list(H["drugs"]).index(code)
    y = H["y"][i]
    out = H["out"][i]
    shelf = H["shelfout_days"][i]
    for w, o, s in zip(weeks, out, shelf):
        if o:
            ax.axvspan(w, w + pd.Timedelta(days=7), color=C_OUT, alpha=0.18, lw=0)
        elif s > 0:
            ax.axvspan(w, w + pd.Timedelta(days=7), color=C_SHELF, alpha=0.18, lw=0)
    ax.step(weeks + pd.Timedelta(days=3.5), y, where="mid", color=INK, lw=1.2,
            label="weekly units")
    x = P[(P.config_hash == "36ad7c615b96") & (P.drug_code == code)]
    for k, (o, g) in enumerate(x.groupby("origin")):
        g = g.sort_values("h")
        ws = o + pd.to_timedelta(g.h * 7 - 3.5, unit="D")
        ax.plot(ws, g["mean"], color=C_ML, lw=1.6, label="ML eval paths (h1-8)" if k == 0 else None)
    x = P[(P.config_hash == "eval-old") & (P.drug_code == code)]
    for k, (o, g) in enumerate(x.groupby("origin")):
        g = g.sort_values("h")
        ws = o + pd.to_timedelta(g.h * 7 - 3.5, unit="D")
        ax.plot(ws, g["mean"], color=C_CUR, lw=1.6, label="current eval paths" if k == 0 else None)
    s = served[served.drug_code == code]
    ax.plot(s.week_start + pd.Timedelta(days=3.5), s["mean"], color=C_ML, lw=2, ls="--",
            label="ML served (from 30 Jun 2025)")
    # ml_rolling: one row per Sunday origin, value = 4-week total; plot weekly average
    rr = roll[roll.drug_code == code].sort_values("origin")
    ax.plot(rr.origin + pd.Timedelta(days=14), rr.current / 4, color=C_CUR, lw=1, ls=":",
            label="rolling current (4-wk total / 4)")
    ax.plot(rr.origin + pd.Timedelta(days=14), rr.ml / 4, color=C_ML, lw=1, ls=":",
            label="rolling ML (4-wk total / 4)")
    ax.axvline(pd.Timestamp(SEL_END), color=MUTED, lw=1)
    ax.text(pd.Timestamp(SEL_END), 0.97, "  selection end; right of here = diagnostic",
            transform=ax.get_xaxis_transform(), va="top", fontsize=8, color=MUTED)
    for e in EID:
        ax.axvline(e, color=MUTED, lw=0.8, ls="--")
    st = S.loc[code]
    ax.set_title(f"{code}  {st.drug_name}  [{PICKS[code]}; {st.tier}, {st.sb_class}]  "
                 f"Rp {st.rev52/1e6:.1f}M/yr, {st.mu52:.1f} units/wk", fontsize=10, loc="left",
                 color=INK)
    ax.set_ylabel("units / week", color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_ylim(bottom=0)
h, l = axes[0].get_legend_handles_labels()
h += [plt.Rectangle((0, 0), 1, 1, color=C_OUT, alpha=.3), plt.Rectangle((0, 0), 1, 1, color=C_SHELF, alpha=.3)]
l += ["censored week (stock-out, or 4+ shelf-out days)", "week with 1-3 shelf-out days"]
fig.legend(h, l, loc="upper center", ncol=4, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(FIG / "representative_drugs.png", dpi=130)
print("wrote", FIG / "representative_drugs.png")
