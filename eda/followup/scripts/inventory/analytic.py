"""Analytic pieces: (1) elasticity of the NB reorder point to mean vs dispersion error;
(2) the service level the matched policy really runs at (k x q95 read back on the NB);
(3) figure of lost revenue vs forecast error from the sensitivity runs.

Needs sensitivity.py (both windows) first.
Run from the repo root:  uv run python eda/followup/scripts/inventory/analytic.py
"""
import pickle
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

from irpd import contest, replay
from irpd.agent.tools import plan

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402

OUT = fp.out("inventory")
FIG = fp.fig("inventory")


def nbq(m, d, q):
    if d <= 1.0001:
        return stats.poisson.ppf(q, m)
    n = m / (d - 1)
    p = 1 / d
    return stats.nbinom.ppf(q, n, p)


# (1) elasticities at typical lead+cycle totals
rows = []
for m in (2, 5, 20, 60):
    for d in (1.5, 3.0):
        base = nbq(m, d, 0.95)
        rows.append({"lead+cycle mean": m, "dispersion": d, "q95": base,
                     "q95 mean +10%": nbq(m * 1.1, d, 0.95),
                     "q95 disp +10%": nbq(m, d * 1.1, 0.95),
                     "q95 disp x1.6 (4-wk var ratio)": nbq(m, 1 + (d - 1) * 1.6 + 0.6, 0.95)
                     if False else nbq(m, d * 1.6, 0.95)})
el = pd.DataFrame(rows)
el.to_csv(OUT / "nb_elasticity.csv", index=False)
print(el.to_string(index=False))

# (2) effective service level of the matched policy
for label in ("H2-2024", "H1-2025"):
    h, b = pickle.loads((OUT / f"cache_{label}.pkl").read_bytes())
    sens = pd.read_csv(OUT / f"sensitivity_{label}.csv")
    k = sens[(sens.variant == "current forecaster") & (sens["mode"] == "matched")].scale.iloc[0]
    f = b.facts
    lead = f.lead_days.to_numpy(int)
    m_l, v_l = plan._window_sum(b.mean, lead), plan._window_sum(b.var, lead)
    pol = contest.to_policy({}, b)
    s = pol.s.to_numpy() * k
    eff = []
    for mi, vi, si, ti in zip(m_l, v_l, s, f.tier):
        if mi <= 0:
            continue
        d = max(vi / mi, 1.0001)
        eff.append((ti, stats.nbinom.cdf(si, mi / (d - 1), 1 / d)))
    e = pd.DataFrame(eff, columns=["tier", "service"])
    print(label, "scale", k, "effective lead-time service of matched s:",
          e.groupby("tier").service.describe()[["mean", "25%", "50%"]].round(4).to_dict("index"))

# (3) figure: matched and unmatched lost revenue vs half-year WAPE (mean-noise runs)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, label in zip(axes, ("H2-2024", "H1-2025")):
    s = pd.read_csv(OUT / f"sensitivity_{label}.csv")
    s = s[s.wape_halfyear.notna()]
    for mode, mk in (("matched", "o"), ("unmatched", "s")):
        g = s[s["mode"] == mode]
        ax.scatter(g.wape_halfyear, g.lost_revenue / 1e6, marker=mk, label=mode, alpha=.8)
        for _, r in g.iterrows():
            if r.variant.startswith(("current", "oracle")):
                ax.annotate(r.variant.replace("oracle mean: ", "oracle "),
                            (r.wape_halfyear, r.lost_revenue / 1e6), fontsize=7)
    ax.set_title(f"{label} replay (current forecaster, perturbed)")
    ax.set_xlabel("WAPE of policy-day forecast vs realised half-year demand")
    ax.set_ylabel("lost revenue, Rp M")
    ax.legend()
fig.tight_layout()
fig.savefig(FIG / "lost_revenue_vs_wape.png", dpi=130)
print("saved figure")
