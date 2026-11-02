"""Calibration of lead-time totals under the NB 'variance = mean x dispersion' recipe.

Reads existing out-of-sample prediction files only (no training). Two sets:
- selection period (2024 origins, wave3): baseline-g1 and an ML config (3aafa0367da7,
  LightGBM Tweedie 1.3) -- the clean set;
- 2025 eval origins: frozen ML (36ad7c615b96, seed average), eval-old (current
  forecaster) and baseline-g1 -- DIAGNOSTIC (2025 already inspected).

For each: coverage at 0.9/0.95 and pinball of 1-, 2-, 4-week totals by tier; the
ratio of the empirical variance of window-total errors to the model's summed
variance (how much the independence assumption understates spread); the lag-1..3
correlation of standardised weekly errors within (origin, drug); the dispersion
multiplier that would make 4-week p95 coverage hit 0.95 per tier; and how many
windows are dropped as censored.

Run from the repo root:  uv run python eda/followup/scripts/inventory/calibration.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from irpd import forecast as fc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402

PRED = fp.ML
OUT = fp.out("inventory")

SETS = {
    "sel:baseline-g1": (PRED / "wave3/predictions", "baseline-g1"),
    "sel:ml-3aafa0367da7": (PRED / "wave3/predictions", "3aafa0367da7"),
    "eval25:ml-frozen": (PRED / "eval/predictions", "36ad7c615b96"),
    "eval25:current(old)": (PRED / "eval/predictions", "eval-old"),
    "eval25:baseline-g1": (PRED / "eval/predictions", "eval-baseline-g1"),
}


def load(d, h):
    p = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(d / f"{h}_*")))])
    p = p[p.seed == (-1 if (p.seed == -1).any() else p.seed.min())]
    return p[p.tier.astype(str) != "none"].copy()


def totals(p, w, censored_as_recorded=False):
    p = p.copy()
    p["block"] = (p.h.astype(int) - 1) // w
    p["miss"] = p.actual.isna()
    g = p.groupby(["origin", "drug_code", "block"], sort=False)
    t = g.agg(tier=("tier", "first"), mean=("mean", "sum"), var=("var", "sum"),
              actual=("actual", "sum"), miss=("miss", "any"), n=("h", "size")).reset_index()
    t = t[t.n == w]
    return t


def pinball(y, q_hat, q):
    d = y - q_hat
    return np.where(d >= 0, q * d, (q - 1) * d)


rows, corr_rows, mult_rows = [], [], []
for name, (d, h) in SETS.items():
    p = load(d, h)
    p["tier"] = p.tier.astype(str)
    # within-origin correlation of standardised errors across h
    q = p[p.actual.notna()].copy()
    q["z"] = (q.actual - q["mean"]) / np.sqrt(np.maximum(q["var"], q["mean"]).clip(lower=1e-9))
    wide = q.pivot_table(index=["origin", "drug_code", "tier"], columns="h", values="z")
    for tier, g in wide.groupby(level="tier"):
        for lag in (1, 2, 3):
            a = np.concatenate([g[k].to_numpy() for k in range(1, 9 - lag)])
            b = np.concatenate([g[k + lag].to_numpy() for k in range(1, 9 - lag)])
            ok = ~np.isnan(a) & ~np.isnan(b)
            corr_rows.append({"set": name, "tier": tier, "lag": lag,
                              "corr": np.corrcoef(a[ok], b[ok])[0, 1], "n": ok.sum()})
    for w in (1, 2, 4):
        t = totals(p, w)
        n_all = len(t)
        clean = t[~t.miss]
        for tier, g in clean.groupby("tier"):
            m = g["mean"].to_numpy(float)
            v = np.maximum(g["var"].to_numpy(float), m)
            y = g.actual.to_numpy(float)
            err = y - m
            vr = (err ** 2).sum() / v.sum()  # empirical / model variance of the total
            row = {"set": name, "tier": tier, "window": w, "n": len(g),
                   "censored_share": t[t.tier == tier].miss.mean(),
                   "wape": np.abs(err).sum() / y.sum(), "bias": m.sum() / y.sum(),
                   "var_ratio": vr}
            for qq in (0.9, 0.95):
                qh = fc.demand_quantile(m, v, qq)
                row[f"cov{int(qq*100)}"] = (y <= qh).mean()
                row[f"pin{int(qq*100)}"] = pinball(y, qh, qq).sum() / y.sum()
            # censored windows scored on recorded sales (a floor on demand): upper bound
            gc = t[(t.tier == tier) & t.miss]
            rows.append(row)
            if w == 4:
                # dispersion multiplier needed for 4-week p95 coverage = 0.95
                for k in (1.0, 1.25, 1.5, 2.0, 3.0):
                    vk = np.maximum(v * k, m * 1.000001)
                    mult_rows.append({"set": name, "tier": tier, "mult": k,
                                      "cov95": (y <= fc.demand_quantile(m, vk, 0.95)).mean()})
    print("done", name, flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "calibration_windows.csv", index=False)
pd.DataFrame(corr_rows).to_csv(OUT / "error_autocorr.csv", index=False)
pd.DataFrame(mult_rows).to_csv(OUT / "dispersion_multiplier.csv", index=False)
with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
    print(res.to_string(index=False))
    print(pd.DataFrame(corr_rows).pivot_table(index=["set", "tier"], columns="lag", values="corr"))
    print(pd.DataFrame(mult_rows).pivot_table(index=["set", "tier"], columns="mult",
                                              values="cov95"))
