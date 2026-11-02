"""Store-level (summed over drugs) forecast path vs actual per eval origin, around Eid
2025 (31 Mar). DIAGNOSTIC (2025 eval data). Rows with censored actuals are dropped for
every model alike, so sums compare like with like.
"""
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
d = pd.read_parquet(f"{OUT}/eval_predictions_combined.parquet")
d = d[d.actual.notna() & d.tier.isin(["fast", "regular", "slow"])]
d["target_week"] = d.origin + pd.to_timedelta(1 + 7 * (d.h - 1), unit="D")
agg = d.groupby(["origin", "h", "target_week", "model"]).agg(f=("mean", "sum"), a=("actual", "sum")).reset_index()
wide = agg.pivot_table(index=["origin", "h", "target_week"], columns="model", values="f").reset_index()
act = agg[agg.model == "ml"][["origin", "h", "a"]]
wide = wide.merge(act, on=["origin", "h"])
for m in ["ml", "current", "baseline_reselected"]:
    wide[f"{m}/actual"] = wide[m] / wide.a
wide.to_csv(f"{OUT}/aggregate_paths.csv", index=False)
pd.set_option("display.width", 220)
print(wide.round(3).to_string(index=False))

# CV across h of the aggregate actual vs aggregate forecasts per origin
cv = wide.groupby("origin").agg(**{f"cv_{c}": (c, lambda s: s.std() / s.mean()) for c in
                                  ["a", "ml", "current", "baseline_reselected"]})
print(cv.round(3).to_string())
cv.to_csv(f"{OUT}/aggregate_path_cv.csv")

fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharey=False)
for ax, (o, x) in zip(axes.ravel(), wide.groupby("origin")):
    ax.plot(x.target_week, x.a, "k-o", ms=3, label="actual (uncensored)")
    ax.plot(x.target_week, x.ml, "-", c="#2a6fdb", label="ML")
    ax.plot(x.target_week, x.current, "--", c="#d9822b", label="current SES+Eid")
    ax.plot(x.target_week, x.baseline_reselected, ":", c="#555", label="baseline mean8/SES")
    ax.axvline(pd.Timestamp("2025-03-31"), c="red", lw=0.8, alpha=0.5)
    ax.set_title(f"origin {o.date()}", fontsize=9)
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
axes[0, 0].legend(fontsize=7)
fig.suptitle("Units summed over fast+regular+slow drugs, 8-week paths, 2025 eval origins "
             "(red line = Eid 31 Mar 2025). Diagnostic.", fontsize=10)
fig.tight_layout()
fig.savefig(f"{FIG}/aggregate_paths_eid2025.png", dpi=120)
