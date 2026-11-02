"""Figures for E0 (fig/harness) and E1 (fig/replan). Needs e0_checks.py and replan/e1.py first.

Run from the repo root:  uv run --with matplotlib python eda/followup/scripts/harness/figs.py
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402

H, E = fp.out("harness"), fp.out("replan")
HF, EF = fp.fig("harness"), fp.fig("replan")
C = {"static": "#4C72B0", "14d": "#DD8452", "7d": "#55A868"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# ---- E0: A/A and replay power ----
rep = pd.read_csv(H / "e0_replay_checks.csv")
r = rep[rep.metric == "net_out"].reset_index(drop=True)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
groups = [("aa_bracket", "A/A: matching start"), ("aa_offset", "A/A: re-plan phase"),
          ("noise", "noise s=0.1"), ("noise", "noise s=0.25"), ("scale", "means x0.9")]
y = 0; ticks = []
for kind, lab in groups:
    g = r[r.kind == kind]
    if kind == "noise":
        g = g[g.param.str.startswith("(0.1," if "0.1" in lab else "(0.25,")]
    for _, row in g.iterrows():
        col = "#C44E52" if row.excl_zero else "#555555"
        ax[0].plot([row.ratio_lo, row.ratio_hi], [y, y], color=col, lw=1.5)
        ax[0].plot(row.ratio, y, "o", color=col, ms=3)
        y += 1
    ticks.append((y - len(g) / 2 - 0.5, lab)); y += 2
ax[0].axvline(1, color="k", lw=0.8)
ax[0].set_yticks([t for t, _ in ticks], [l for _, l in ticks]); ax[0].invert_yaxis()
ax[0].set_xlabel("net stock-out lost revenue ratio vs clean (90% two-way CI)")
ax[0].set_title("Replay, H2 2024, rolling 14d, equal stock (red: CI excludes 1)")
f = pd.read_csv(H / "e0_forecast_power.csv")
f = f[(f.metric == "pinball_lc") & (f.tier == "pooled")].reset_index(drop=True)
y = 0; ticks = []
for kind, lab in groups[2:]:
    for boot, off, col in (("two-way", 0, "#DD8452"), ("drug-only", 0.35, "#4C72B0")):
        g = f[(f.kind == kind) & (f.boot == boot)]
        if kind == "noise":
            g = g[g.param.str.startswith("(0.1," if "0.1" in lab else "(0.25,")]
        for i, (_, row) in enumerate(g.iterrows()):
            ax[1].plot([100 * row.rel_lo, 100 * row.rel_hi], [y + i + off] * 2, color=col, lw=1.2,
                       label=boot if (i == 0 and kind == "noise" and "0.1" in lab) else None)
    ticks.append((y + len(g) / 2, lab)); y += len(g) + 3
ax[1].axvline(0, color="k", lw=0.8); ax[1].axvline(1, color="grey", lw=0.8, ls="--")
ax[1].set_yticks([t for t, _ in ticks], [l for _, l in ticks]); ax[1].invert_yaxis()
ax[1].set_xlabel("pooled pinball_lc, noisy - clean (% of clean, 90% CI)")
ax[1].set_title("Forecast score, 21 selection origins"); ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(HF / "power_calibration.png", dpi=150); plt.close(fig)

# ---- E1: frontiers ----
fr = pd.read_csv(E / "e1_frontier.csv")
mt = pd.read_csv(E / "e1_matched.csv")
for metric, lab in (("lost_net", "net lost revenue"), ("lost_gross", "gross lost revenue")):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=False)
    for j, w in enumerate(["H2-2024", "H1-2025", "H2-2025"]):
        for i, cap in enumerate([False, True]):
            a = axes[i, j]
            for cad in ("static", "14d", "7d"):
                g = fr[(fr.window == w) & (fr.floor_cap == cap) & (fr.cadence == cad) & (fr.start_offset == 0)]
                a.plot(g.stock_value / 1e6, g[metric] / 1e6, "-o", ms=3, color=C[cad], label=cad)
                k1 = g[g.k == 1.0]
                a.plot(k1.stock_value / 1e6, k1[metric] / 1e6, "s", ms=6, mfc="none", color=C[cad])
                m = mt[(mt.window == w) & (mt.floor_cap == cap) & (mt.cadence == cad) & (mt.start_offset == 0)]
                a.plot(m.stock_value / 1e6, m[metric] / 1e6, "*", ms=10, color=C[cad])
            a.set_title(f"{w}{' (diagnostic)' if w != 'H2-2024' else ''}, floor cap {'on' if cap else 'off'}")
            a.set_xlabel("average stock value (Rp M)"); a.set_ylabel(f"{lab} (Rp M)")
            if i == 0 and j == 0:
                a.legend(frameon=False)
    fig.suptitle(f"E1: {lab} vs stock value, k = 0.7..2.1 (square: k = 1; star: matched to pharmacist's stock)", fontsize=10)
    fig.tight_layout(); fig.savefig(EF / f"frontier_{metric}.png", dpi=150); plt.close(fig)

# ---- E1: matched bars split stock-out/shelf-out ----
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for i, cap in enumerate([False, True]):
    a = axes[i]
    m = mt[mt.floor_cap == cap].groupby(["window", "cadence"])[["net_out", "net_shelf"]].mean().reset_index()
    x = 0; xt = []
    for w in ["H2-2024", "H1-2025", "H2-2025"]:
        for cad in ("static", "14d", "7d"):
            row = m[(m.window == w) & (m.cadence == cad)].iloc[0]
            a.bar(x, row.net_out / 1e6, color=C[cad])
            a.bar(x, row.net_shelf / 1e6, bottom=row.net_out / 1e6, color=C[cad], alpha=0.4, hatch="//")
            x += 1
        xt.append((x - 2, w)); x += 1
    a.set_xticks([t for t, _ in xt], [l for _, l in xt]); a.set_ylabel("net lost revenue (Rp M)")
    a.set_title(f"At static's matched stock, mean of 2 start days; floor cap {'on' if cap else 'off'}\n(solid: stock-out, hatched: shelf-out; blue static, orange 14d, green 7d)", fontsize=8)
fig.tight_layout(); fig.savefig(EF / "matched_split.png", dpi=150); plt.close(fig)
print("figs done")
