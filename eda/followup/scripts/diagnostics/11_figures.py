"""Two summary figures: (1) per-path forecast CV vs actual CV vs pure-noise CV on the
selection origins; (2) observed / compound-Poisson dispersion per drug, non-Eid blocks."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402

DATA = fp.DATA
OUT = fp.out("diagnostics")
FIG = fp.fig("diagnostics")
d = pd.read_parquet(f"{OUT}/selection_predictions_combined.parquet")
d = d[d.model == "ml_A2"]
g = d.groupby(["origin", "drug_code"], observed=True)
s = g.agg(tier=("tier", "first"), fm=("mean", "mean"), fs=("mean", "std"), n=("actual", "count"),
          am=("actual", "mean"), av=("actual", "var"), eid=("eid", "any")).reset_index()
s = s[(s.n >= 6) & (s.am > 0) & ~s.eid]
s["path_cv"] = s.fs / s.fm
s["act_cv"] = np.sqrt(s.av) / s.am
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, t in zip(axes, ["fast", "regular", "slow"]):
    z = s[s.tier == t]
    bins = np.logspace(-4, 1, 50)
    ax.hist(z.path_cv.clip(1e-4), bins=bins, alpha=.7, label="ML forecast path CV", color="#2a6fdb")
    ax.hist(z.act_cv.clip(1e-4), bins=bins, alpha=.6, label="actual CV (8 weeks)", color="#444")
    ax.set_xscale("log")
    ax.set_title(f"{t}: {len(z)} drug-origins, non-Eid windows", fontsize=9)
axes[0].legend(fontsize=8)
fig.suptitle("Selection origins 2024: variation across the 8 horizons, forecast vs realized", fontsize=10)
fig.tight_layout()
fig.savefig(f"{FIG}/path_cv_vs_actual_cv_selection.png", dpi=120)

c = pd.read_csv(f"{OUT}/compound_poisson_by_drug_noeid.csv")
fig, ax = plt.subplots(figsize=(6, 3.8))
for t, col in [("fast", "#2a6fdb"), ("regular", "#d9822b"), ("slow", "#555")]:
    z = c[c.tier == t]
    ax.hist(np.log2(z.ratio), bins=40, histtype="step", lw=1.5, label=f"{t} (median {z.ratio.median():.2f})", color=col)
ax.axvline(0, c="red", lw=.8)
ax.set_xlabel("log2( observed within-window dispersion / constant-rate compound-Poisson dispersion )")
ax.set_title("2024 non-Eid 8-week blocks: weekly variation is purchase noise", fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG}/oracle_compound_poisson_ratio.png", dpi=120)
print(s.groupby("tier").agg(path=("path_cv", "median"), act=("act_cv", "median")).round(3))
