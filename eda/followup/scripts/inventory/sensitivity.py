"""How much do mean error, dispersion error and the service level move replay outcomes?

Uses the project's own replay engine (contest.load_window, decide.baseline_basis,
contest.to_policy, contest.matched) on in-memory, read-only connections to the
warehouse. Writes only to data/followup/inventory/.

Window: H2 2024 (policy day 30 Jun 2024, replay 1 Jul - 31 Dec 2024) as the main
window (it sits almost wholly before the 8 Dec 2024 selection boundary); H1 2025
(policy day 31 Dec 2024) as a diagnostic second window. Demand is the uncensored
estimate, as in the ML replay.

Starting point: the current forecaster's basis (`decide.baseline_basis`). Then:
- oracle means: each drug's realised mean daily demand over (a) the first 28 days,
  (b) the whole half-year; dispersion recipe unchanged;
- mean noise: means x lognormal(-s^2/2, s) per drug, s in 0.1 / 0.25 / 0.5, 3 seeds;
- global bias: all means x 0.8 / 1.25;
- tier bias: fast drugs x 1.25 only;
- dispersion: x0.5, x2, Poisson (d = 1), one pooled d per tier;
- service level 0.8 / 0.9 / 0.99.
Each policy is matched to the pharmacist's stock value (as the evaluation does) and
also run unmatched (scale 1, what the app would actually recommend).

Run from the repo root:  uv run python eda/followup/scripts/inventory/sensitivity.py [H2-2024|H1-2025]
"""
import pickle
import sys
import time
from copy import copy
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from irpd import contest, holdout, replay
from irpd.ml import decide

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402

OUT = fp.out("inventory")
WINDOWS = {"H2-2024": (date(2024, 6, 30), date(2024, 12, 31)),
           "H1-2025": (date(2024, 12, 31), date(2025, 6, 30))}
label = sys.argv[1] if len(sys.argv) > 1 else "H2-2024"
day, end = WINDOWS[label]

cache = OUT / f"cache_{label}.pkl"
if cache.exists():
    h, b = pickle.loads(cache.read_bytes())
else:
    t = time.time()
    h = contest.load_window(day + timedelta(days=1), end)  # uncensored demand
    with holdout.agent_view(day) as con:
        b = decide.baseline_basis(con, day)
    cache.write_bytes(pickle.dumps((h, b)))
    print(f"loaded in {time.time() - t:.0f}s", flush=True)

f = b.facts
codes = list(f.index)
pos = {c: i for i, c in enumerate(h.drugs)}
hi = np.array([pos.get(c, -1) for c in codes])       # basis row -> holdout row
inh = hi >= 0
disp = np.where(b.mean > 0, b.var / np.where(b.mean > 0, b.mean, 1), 1.0)  # per drug-day


def with_mean(mean, d=None):
    d = disp if d is None else d
    nb = copy(b)
    nb.mean = mean
    nb.var = mean * d
    return nb


def realised(days):
    """Per basis drug, realised mean daily demand over the first `days` (all if None)."""
    r = np.zeros(len(codes))
    dem = h.demand if days is None else h.demand[:, :days]
    r[inh] = dem[hi[inh]].mean(axis=1)
    return r


def run(name, basis, knobs=None, **meta):
    pol = contest.to_policy(knobs or {}, basis)
    out = []
    for mode in ("matched", "unmatched"):
        if mode == "matched":
            m, k = replay.match_stock_value(pol, h)
        else:
            m, k = pol, 1.0
        r = replay.simulate(m, h)
        per = replay.metrics(r.floor, r.store, r.sold, h)
        s = replay.summarise(per)
        live_dem = (h.demand * h.live).sum()
        out.append({"window": label, "variant": name, "mode": mode, "scale": k,
                    "lost_revenue": s.lost_revenue,
                    "lost_rev_stockout": s.lost_revenue_stockout,
                    "lost_rev_shelfout": s.lost_revenue_shelfout,
                    "stockout_days": s.stockout_days, "shelfout_days": s.shelfout_days,
                    "stock_value": s.avg_stock_value,
                    "fill_rate": (r.sold * h.live).sum() / live_dem,
                    "drug_days_in_stock": 1 - s.stockout_share, **meta})
    print(out[0]["variant"], {k: round(v, 3) if isinstance(v, float) else v
                              for k, v in out[0].items() if k in ("scale", "lost_revenue",
                                                                  "stockout_days")},
          "| unmatched", round(out[1]["lost_revenue"]), round(out[1]["stock_value"]),
          flush=True)
    return out


def errs(mean):
    """WAPE of the policy-day forecast against realised demand: first 4 weeks, half-year.
    Scope: drugs in the replay, revenue-weighted version too."""
    m28 = mean[inh, :28].sum(1)
    y28 = h.demand[hi[inh], :28].sum(1)
    T = h.demand.shape[1]
    mT = mean[inh, :28].mean(1) * T  # the static policy extrapolates the first 4 weeks
    yT = h.demand[hi[inh]].sum(1)
    p = h.price[hi[inh]]
    return {"wape_4w": np.abs(m28 - y28).sum() / y28.sum(),
            "wape_halfyear": np.abs(mT - yT).sum() / yT.sum(),
            "rev_wape_halfyear": (p * np.abs(mT - yT)).sum() / (p * yT).sum()}


rows = []
base_mean = b.mean
rows += run("current forecaster", b, **errs(base_mean))
for nm, days in (("oracle mean: first 28 days", 28), ("oracle mean: whole half-year", None)):
    om = np.repeat(realised(days)[:, None], b.mean.shape[1], 1)
    om[~inh] = base_mean[~inh]
    rows += run(nm, with_mean(om), **errs(om))
for s in (0.1, 0.25, 0.5):
    for seed in range(3):
        rng = np.random.default_rng(seed)
        e = np.exp(rng.normal(-s * s / 2, s, len(codes)))[:, None]
        rows += run(f"mean noise s={s}", with_mean(base_mean * e), seed=seed,
                    **errs(base_mean * e))
for k in (0.8, 1.25):
    rows += run(f"all means x{k}", with_mean(base_mean * k), **errs(base_mean * k))
fast = (f.tier == "fast").to_numpy()[:, None]
rows += run("fast-tier means x1.25", with_mean(np.where(fast, base_mean * 1.25, base_mean)))
for k in (0.5, 2.0):
    rows += run(f"dispersion x{k}", with_mean(base_mean, np.maximum(disp * k, 1.0)))
rows += run("Poisson (d=1)", with_mean(base_mean, np.ones_like(disp)))
d0 = disp[:, 0]
pooled = pd.Series(d0, index=f.index).groupby(f.tier).transform("median").to_numpy()
rows += run("dispersion = tier median", with_mean(base_mean, pooled[:, None]))
for sl in (0.8, 0.9, 0.99):
    rows += run(f"service {sl}", b, knobs={"default": {"service_level": sl}})

res = pd.DataFrame(rows)
res.to_csv(OUT / f"sensitivity_{label}.csv", index=False)
with pd.option_context("display.width", 220, "display.float_format", "{:,.3f}".format):
    g = res.groupby(["mode", "variant"], sort=False).mean(numeric_only=True)
    print(g[["scale", "lost_revenue", "lost_rev_stockout", "lost_rev_shelfout",
             "stockout_days", "shelfout_days", "stock_value", "fill_rate", "wape_4w",
             "wape_halfyear"]].to_string())
