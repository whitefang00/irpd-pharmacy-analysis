"""Figures for experiments.md -> eda/followup/fig/results/*.png.

Run from the repo root:  uv run --with matplotlib python eda/followup/scripts/results/figs.py [parts...]
"""
from __future__ import annotations

import json
import shutil
from datetime import date, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C
from common import fp
from irpd import forecast as fc

COL = {"actual": "#0b0b0b", "cur_control": "#2a78d6", "cur_eid": "#eb6834",
       "ml_control": "#1baf7a", "ml_eid": "#eda100", "ml_eid_censor": "#e87ba4",
       "ml_eid_offset": "#6250d6", "ml_censor": "#008300"}
LAB = {"cur_control": "current (control)", "cur_eid": "current + Eid fix",
       "ml_control": "ML (control)", "ml_eid": "ML + Eid fix", "ml_eid_censor": "ML + Eid + censor",
       "ml_eid_offset": "ML + Eid offset (POST HOC)", "ml_censor": "ML + censor"}
GOOD, BAD = "#0ca30c", "#d03b3b"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb"})


def save(fig, name):
    fig.savefig(C.FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------- store level ----------
def store_eid(sp, year, origins, eid_day, peak):
    arms = ["cur_control", "cur_eid", "ml_control", "ml_eid", "ml_eid_censor", "ml_eid_offset"]
    fig, axes = plt.subplots(1, len(origins), figsize=(4.2 * len(origins), 4.2), sharey=True)
    for ax, o in zip(axes, origins):
        g = sp[(sp.origin == o)]
        a = g[g.arm == "ml_control"].sort_values("week_start")
        x = pd.to_datetime(a.week_start)
        ax.plot(x, a.actual / 1e3, color=COL["actual"], lw=2.4, marker="o", ms=4, label="actual", zorder=5)
        for arm in arms:
            f = g[g.arm == arm].sort_values("week_start")
            ls = "--" if arm == "ml_eid_offset" else "-"
            ax.plot(pd.to_datetime(f.week_start), f.fc / 1e3, color=COL[arm], lw=1.6, ls=ls, label=LAB[arm])
        ax.axvline(pd.Timestamp(peak), color="#9a9890", lw=1, ls=":")
        ax.text(pd.Timestamp(peak), ax.get_ylim()[1] if False else 0.98, "actual peak week",
                transform=ax.get_xaxis_transform(), fontsize=8, color="#52514e", ha="center")
        ax.set_title(f"origin {o:%d %b %Y}", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("weekly units, thousands (fast+regular+slow)")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Eid {year} (Eid day {eid_day:%d %b}): store-level 8-week forecasts vs actual", fontsize=12)
    save(fig, f"store_eid_{year}.png")


def store_overview(sp):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    a = sp[(sp.arm == "ml_control") & (sp.h == 1)].sort_values("week_start")
    # actual series: h=1..2 rows of each origin cover every week once
    act = sp[(sp.arm == "ml_control") & (sp.h <= 2)].sort_values("week_start").drop_duplicates("week_start")
    act = act[act.n > 0]
    ax.plot(pd.to_datetime(act.week_start), act.actual / 1e3, color=COL["actual"], lw=1.8, label="actual")
    for i, o in enumerate(C.grid()[::3]):
        for arm in ("cur_control", "ml_control"):
            f = sp[(sp.arm == arm) & (sp.origin == o)].sort_values("week_start")
            f = f[pd.to_datetime(f.week_start) <= pd.Timestamp("2025-12-22")]
            ax.plot(pd.to_datetime(f.week_start), f.fc / 1e3, color=COL[arm], lw=1.2,
                    label=LAB[arm] if i == 0 else None)
    for e in (date(2024, 4, 10), date(2025, 3, 31)):
        ax.axvline(pd.Timestamp(e), color="#9a9890", ls=":", lw=1)
        ax.text(pd.Timestamp(e), 0.02, " Eid", transform=ax.get_xaxis_transform(), fontsize=8, color="#52514e")
    ax.axvspan(pd.Timestamp("2024-01-08"), pd.Timestamp("2024-12-08"), color="#cde2fb", alpha=0.35, lw=0)
    ax.text(pd.Timestamp("2024-01-15"), 0.93, "selection data", transform=ax.get_xaxis_transform(), fontsize=8)
    ax.text(pd.Timestamp("2025-01-15"), 0.93, "2025 diagnostic (already inspected)",
            transform=ax.get_xaxis_transform(), fontsize=8)
    ax.set_ylabel("weekly units, thousands")
    ax.set_title("Store-level weekly demand (fast+regular+slow) with 8-week forecasts from every 6th week")
    ax.legend(fontsize=8, loc="upper right")
    save(fig, "store_overview.png")


# ---------- representative drugs ----------
REP = [("SW2106000809", "Teamic 500mg", "steady-stable"), ("SW2106000902", "Zentra tab", "steady-shifted"),
       ("OBT2108210081", "Berotec 10mg", "intermittent"), ("SW2106001787", "Atorvastatin 10mg DM", "bulk"),
       ("OBT2306190002", "SGM Eksplor gain optigrow 400gr", "censored")]
REP_ORIGINS = [date(2024, 3, 17), date(2024, 8, 18), date(2025, 3, 2), date(2025, 8, 31)]


def drug_plot(code, name, seg):
    a = C.actuals()
    a = a[(a.drug_code == code) & (a.week_start >= date(2023, 7, 1))].sort_values("week_start")
    fig, ax = plt.subplots(figsize=(13, 4.2))
    x = pd.to_datetime(a.week_start)
    for ws, c in zip(a.week_start, a.censored):
        if c:
            ax.axvspan(pd.Timestamp(ws), pd.Timestamp(ws) + pd.Timedelta(days=7), color="#f6c9b8", lw=0, alpha=0.7)
    ax.plot(x, a.units, color=COL["actual"], lw=1.3, label="recorded units")
    ax.plot(x, a.uncensored_units, color=COL["actual"], lw=1, ls=":", label="uncensored estimate")
    for i, o in enumerate(REP_ORIGINS):
        for arm in ("cur_control", "ml_control"):
            p = C.raw(arm)
            p = p[(p.drug_code == code) & (p.origin == o)].sort_values("h")
            if p.empty:
                continue
            px = pd.to_datetime(p.week_start)
            q = fc.demand_quantile(p["mean"].to_numpy(), np.maximum(p["var"].to_numpy(), p["mean"].to_numpy()), 0.95)
            ax.fill_between(px, 0, q, color=COL[arm], alpha=0.07, lw=0)
            ax.plot(px, q, color=COL[arm], lw=0.8, ls="--")
            ax.plot(px, p["mean"], color=COL[arm], lw=2, label=f"{LAB[arm]}: mean (solid), weekly p95 (dashed)" if i == 0 else None)
        ax.axvline(pd.Timestamp(o), color="#9a9890", lw=0.8, ls=":")
        ax.text(pd.Timestamp(o), 0.97, f" {o:%d %b %y}", transform=ax.get_xaxis_transform(), fontsize=7.5, color="#52514e", va="top")
    ax.plot([], [], color="#f6c9b8", lw=8, label="week with a total stock-out day")
    ax.set_title(f"{name} ({code}), segment: {seg}. Forecasts from 4 origins (17 Mar 2024 and 2 Mar 2025 hold Eid)")
    ax.set_ylabel("units per week")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)
    save(fig, f"drug_{code}.png")


# ---------- E2 coverage ----------
def coverage():
    c = pd.read_csv(C.OUT / "e2_coverage_by_length.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    tcol = {"fast": "#2a78d6", "regular": "#eb6834", "slow": "#1baf7a"}
    for i, arm in enumerate(("ml_control", "cur_control")):
        for j, period in enumerate(("test", "diag")):
            ax = axes[i, j]
            ax.axhspan(0.935, 0.965, color="#cde2fb", alpha=0.5, lw=0)
            for t, col in tcol.items():
                for meth, ls, mk in (("plain", "--", "o"), ("rho", "-", "s")):
                    x = c[(c.arm == arm) & (c.period == period) & (c.tier == t) & (c.method == meth)].sort_values("weeks")
                    ax.plot(x.weeks, x["cov"], color=col, ls=ls, marker=mk, ms=5,
                            label=f"{t}, {'plain recipe' if meth == 'plain' else 'with rho factor'}")
            ax.set_title(f"{LAB[arm]}: {'Jul-Oct 2024 test origins' if period == 'test' else '2025 (diagnostic)'}", fontsize=10)
            ax.set_xticks([1, 2, 4, 8])
            if j == 0:
                ax.set_ylabel("p95 coverage of the w-week total")
            if i == 1:
                ax.set_xlabel("window length w (weeks from the origin)")
    axes[0, 1].legend(fontsize=7.5, loc="lower left")
    fig.suptitle("E2: p95 coverage before (dashed) and after (solid) the variance factor; band = target 0.935-0.965", fontsize=11)
    save(fig, "e2_coverage.png")


# ---------- E4 bias ----------
def bias_e4():
    b = pd.read_csv(C.OUT / "e4_bias.csv")
    d = pd.read_csv(C.OUT / "e4_diag_by_stockout_weeks.csv")
    segs = ["fast", "regular", "slow tier", "rare", "not censored", "censored (all tiers)"]
    arms = ["ml_control", "ml_censor", "ml_eid", "ml_eid_censor"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.4, 1.4, 1]})
    for ax, period in zip(axes[:2], ("sel", "diag")):
        x = np.arange(len(segs))
        wdt = 0.2
        for k, arm in enumerate(arms):
            v = b[(b.period == period) & (b.arm == arm)].set_index("segment").reindex(segs)["bias_gross"]
            ax.bar(x + (k - 1.5) * wdt, v, wdt * 0.92, color=COL[arm], label=LAB[arm])
        ax.axhspan(0.95, 1.05, color="#cde2fb", alpha=0.6, lw=0)
        ax.axhline(1, color="#52514e", lw=0.8)
        ax.set_xticks(x, [s.replace(" (all tiers)", "") for s in segs], rotation=20)
        ax.set_title("selection origins (gate)" if period == "sel" else "2025 (diagnostic)", fontsize=10)
        ax.set_ylabel("bias of 4-week totals (forecast / uncensored demand)")
    axes[0].legend(fontsize=8)
    ax = axes[2]
    for arm in ("ml_control", "ml_censor"):
        x = d[(d.arm == arm) & (d.censored_label)].sort_values("stockout_weeks_in_4")
        ax.plot(x.stockout_weeks_in_4, x.bias_gross, marker="o", color=COL[arm], label=LAB[arm])
    ax.set_yscale("log")
    ax.axhline(1, color="#52514e", lw=0.8)
    ax.set_xlabel("weeks of the 4 with a total stock-out day")
    ax.set_title("censored-labelled drugs, by stock-outs in the window\n(exploratory, selection)", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle("E4: bias by segment, control vs censoring-aware targets (band = 0.95-1.05)", fontsize=11)
    save(fig, "e4_bias.png")


# ---------- E5 ----------
def e5():
    b = pd.read_csv(C.OUT / "e5_diag_fading_by_stock.csv")
    r = pd.read_csv(C.OUT / "e5_replay.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axes[0]
    wins = ["no stock-out week", "1-3 stock-out weeks", "all 4 weeks out"]
    x = np.arange(len(wins))
    for k, (base, var) in enumerate([("ml_control", "base"), ("cur_control", "base"), ("ml_control", "fading->tsb")]):
        v = b[(b.base == base) & (b.variant == var)].set_index("window").reindex(wins)["bias"]
        col = COL[base] if var == "base" else "#6250d6"
        ax.bar(x + (k - 1) * 0.27, v, 0.25, color=col, label=f"{LAB[base]}" if var == "base" else "TSB (fading rule)")
    ax.set_yscale("log"); ax.axhline(1, color="#52514e", lw=0.8)
    ax.set_xticks(x, wins)
    ax.set_ylabel("bias of 4-week totals (log)")
    ax.set_title("Fading drugs: bias by stock status of the window (selection)", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    g = r.groupby(["base", "rule"]).agg(freed=("aff_stock_freed", "mean"), days=("extra_stockout_days_per_drug", "mean"),
                                         n=("affected_drugs", "mean")).reset_index()
    for _, row in g.iterrows():
        ax.scatter(row.freed * 100, row.days, s=80, color=COL[row.base], marker="o" if row.rule == "fading" else "s")
        ax.annotate(f"{LAB[row.base]}\n{row.rule} ({row.n:.0f} drugs)", (row.freed * 100, row.days), fontsize=8,
                    xytext=(6, 4), textcoords="offset points")
    ax.axvline(5, color=GOOD, ls="--", lw=1); ax.axhline(2, color=BAD, ls="--", lw=1)
    ax.text(5.3, 0.04, "need >= 5% freed", color=GOOD, fontsize=8, transform=ax.get_xaxis_transform())
    ax.text(-2.5, 2.15, "max +2 stock-out days", color=BAD, fontsize=8)
    ax.set_xlabel("stock value freed on affected drugs, %")
    ax.set_ylabel("extra stock-out days per affected drug")
    ax.set_title("Replay H2 2024 (mean of 2 starts), at the reference's k", fontsize=10)
    fig.suptitle("E5: lifecycle rules", fontsize=11)
    save(fig, "e5_lifecycle.png")


# ---------- E6 ----------
def e6():
    c = pd.read_csv(C.OUT / "e6_compare.csv")
    c = c[c.metric == "pinball_lc"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    tiers = ["pooled", "fast", "regular", "slow"]
    pairs = [("ml", "cur", "ML vs current", COL["ml_control"]), ("ens", "cur", "ensemble vs current", "#eda100"),
             ("ens", "ml", "ensemble vs ML", "#6250d6")]
    for ax, period in zip(axes, ("sel", "diag")):
        for k, (a, b_, lab, col) in enumerate(pairs):
            x = c[(c.period == period) & (c.arm == a) & (c.ref == b_)].set_index("tier").reindex(tiers)
            y = np.arange(len(tiers)) + (k - 1) * 0.22
            ax.errorbar(x.rel * 100, y, xerr=[(x.rel - x.rel_lo) * 100, (x.rel_hi - x.rel) * 100], fmt="o",
                        color=col, capsize=3, label=lab)
        ax.axvline(0, color="#52514e", lw=0.8)
        ax.axvline(1, color=BAD, lw=1, ls="--")
        ax.text(1.2, 3.45, "+1% NI margin", color=BAD, fontsize=8)
        ax.set_yticks(range(len(tiers)), tiers)
        ax.invert_yaxis()
        ax.set_xlabel("relative change in revenue-weighted pinball (%), 90% two-way CI")
        ax.set_title("selection origins (gate)" if period == "sel" else "2025 (diagnostic)", fontsize=10)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("E6: pinball(0.95) of lead + cycle totals, paired differences (lower is better)", fontsize=11)
    save(fig, "e6_pinball.png")


# ---------- summary of gates ----------
def summary():
    gates = json.loads((C.OUT / "gates.json").read_text())
    fig, ax = plt.subplots(figsize=(12, 0.36 * len(gates) + 1.2))
    for i, g in enumerate(gates):
        lo, hi = g["pass"]
        a0, a1 = g["axis"]
        v = g["value"]
        # signed distance from the nearest pass boundary, as a share of the row's plotted range
        out = (lo - v if v < lo else v - hi) if not (lo <= v <= hi) else -min(v - lo, hi - v)
        x = float(np.clip(out / (a1 - a0), -0.5, 0.95))
        ax.barh(i, x, color=GOOD if g["ok"] else BAD, height=0.6)
        ax.text(0.98, i, f"{g['value_txt']}  {'PASS' if g['ok'] else 'FAIL'}", va="center",
                fontsize=8.5, color="#0b0b0b")
    ax.axvline(0, color="#52514e", lw=1)
    ax.set_yticks(range(len(gates)), [g["name"] for g in gates], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.55, 1.6)
    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75])
    ax.set_xlabel("distance from the pass boundary, scaled to a plausible range "
                  "(left of 0 = inside the pass region, right = outside)")
    ax.set_title("Each pre-registered gate: inside (green) or outside (red) its pass region", fontsize=11)
    ax.grid(axis="y", visible=False)
    save(fig, "summary_gates.png")


def copy_e1():
    for src, dst in ((fp.fig("replan") / "frontier_lost_net.png", "e1_frontier_net.png"),
                     (fp.fig("replan") / "matched_split.png", "e1_matched_split.png"),
                     (fp.fig("harness") / "power_calibration.png", "e0_power_calibration.png")):
        shutil.copy(src, C.FIG / dst)
        print("copied", dst)


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["store", "drugs", "e2", "e4", "e5", "e6", "e1", "summary"]
    if "store" in what:
        sp = pd.read_parquet(C.OUT / "store_paths.parquet")
        store_overview(sp)
        store_eid(sp, 2024, [date(2024, 2, 18), date(2024, 3, 3), date(2024, 3, 17), date(2024, 3, 31)],
                  date(2024, 4, 10), date(2024, 4, 8))
        store_eid(sp, 2025, [date(2025, 2, 2), date(2025, 2, 16), date(2025, 3, 2), date(2025, 3, 16)],
                  date(2025, 3, 31), date(2025, 3, 24))
    if "drugs" in what:
        for code, name, seg in REP:
            drug_plot(code, name, seg)
    if "e2" in what:
        coverage()
    if "e4" in what:
        bias_e4()
    if "e5" in what:
        e5()
    if "e6" in what:
        e6()
    if "e1" in what:
        copy_e1()
    if "summary" in what:
        summary()
