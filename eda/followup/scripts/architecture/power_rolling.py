"""EXPLORATORY power check: does a ROLLING (14-day re-plan), UNMATCHED replay detect forecast
quality differences better than the project's static matched replay?

Window H2 2024 (policy day 30 Jun 2024, before the 8 Dec 2024 selection boundary). Reuses the
current-forecaster bases cached by inventory/sensitivity.py and inventory/rolling_policy.py
(data/followup/inventory/*.pkl); nothing is trained.
The simulator is a copy of rolling_policy.py's simulate_tv (itself a time-varying copy of replay.simulate,
which reproduces the static result exactly).

Variants, all re-planned every 14 days, scale k = 1 (the policy the app would recommend):
- base: current forecaster
- level noise: each drug's mean x exp(N(-s^2/2, s)), one draw per drug held across refreshes
  (a persistent level error, like a real one), s = 0.1 (WAPE ~ +0.01) and 0.25 (~ +0.09), 5 seeds
- bias x0.9 on every mean
- variance x1.5 (proxy for the lead-time variance fix)
Per-drug paired bootstrap (decide.paired_bootstrap_replay, 90%) of each against base.

Needs inventory/sensitivity.py H2-2024 and inventory/rolling_policy.py first.
Run from the repo root:  uv run python eda/followup/scripts/architecture/power_rolling.py
"""
import pickle
import sys
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd

from irpd import contest, replay
from irpd.ml import decide

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402

INV = fp.out("inventory")
OUT = fp.out("architecture")
h, b0 = pickle.loads((INV / "cache_H2-2024.pkl").read_bytes())
bases = pickle.loads((INV / "rolling_bases_H2-2024.pkl").read_bytes())
REFRESH = 14
T = len(h.days)
refresh_idx = list(range(0, T, REFRESH))


def policies_over_time(make):
    n = len(h.drugs)
    S = {k: np.zeros((n, T)) for k in ("s", "S", "floor_min", "floor_max")}
    lead, prev = None, None
    for j, t in enumerate(refresh_idx):
        pol = make(j).set_index("drug_code")
        if prev is not None:
            pol = pd.concat([pol, prev[~prev.index.isin(pol.index)]])
        prev = pol
        a = replay.policy_arrays(pol, h)
        stop = refresh_idx[j + 1] if j + 1 < len(refresh_idx) else T
        for k in S:
            S[k][:, t:stop] = a[k][:, None]
        lead = a["lead"] if lead is None else lead
    S["lead"] = lead
    return S


def simulate_tv(p, k=1.0):
    n = len(h.drugs)
    floor, store = h.start_floor.astype(float).copy(), h.start_store.astype(float).copy()
    pipeline = np.zeros((n, T + p["lead"].max() + 1))
    out = {x: np.zeros((n, T)) for x in ("floor", "store", "sold")}
    rows = np.arange(n)
    for t, day in enumerate(h.days):
        arriving = pipeline[:, t]
        if day < h.store_opens:
            floor += arriving
        else:
            store += arriving
        fmin = p["floor_min"][:, t]
        fmax = np.where(np.isnan(p["floor_max"][:, t]), p["S"][:, t] * k, p["floor_max"][:, t])
        move = np.where((floor <= fmin) & (store > 0), np.clip(fmax - floor, 0, store), 0)
        floor, store = floor + move, store - move
        adj = h.adjust[:, t]
        ff = np.maximum(adj, -floor)
        floor = floor + ff
        store = np.maximum(store + (adj - ff), 0)
        sold = np.minimum(h.demand[:, t], np.maximum(floor, 0))
        floor -= sold
        position = floor + store + pipeline[:, t + 1:].sum(axis=1)
        s, S = p["s"][:, t] * k, p["S"][:, t] * k
        order = np.where(position <= s, np.maximum(S - position, 0), 0)
        pipeline[rows, t + p["lead"]] += order
        out["floor"][:, t], out["store"][:, t], out["sold"][:, t] = floor, store, sold
    return replay.metrics(out["floor"], out["store"], out["sold"], h)


def perturbed(b, mult_mean=None, mult_var=1.0):
    nb = copy(b)
    disp = np.where(b.mean > 0, b.var / np.where(b.mean > 0, b.mean, 1), 1.0)
    e = np.ones(len(b.mean)) if mult_mean is None else mult_mean
    nb.mean = b.mean * e[:, None]
    nb.var = nb.mean * disp * mult_var
    return nb


def drug_noise(sigma, seed):
    """persistent per-drug multiplier keyed by drug code, so it is the same at every refresh"""
    rng = np.random.default_rng(seed)
    codes = sorted(set().union(*[set(b.facts.index) for b in bases]))
    e = dict(zip(codes, np.exp(rng.normal(-sigma * sigma / 2, sigma, len(codes)))))
    return lambda b: np.array([e[c] for c in b.facts.index])


def run(name, make_basis):
    per = simulate_tv(policies_over_time(lambda j: contest.to_policy({}, make_basis(bases[j]))))
    return name, per


base_name, base = run("base", lambda b: b)
rows = []


def record(name, per, extra=None):
    s = replay.summarise(per)
    bt = decide.paired_bootstrap_replay(per, base)
    lr, so = bt.loc["lost_revenue"], bt.loc["stockout_days"]
    row = {"variant": name, "lost_revenue": s.lost_revenue, "lost_stockout": s.lost_revenue_stockout,
           "lost_shelfout": s.lost_revenue_shelfout, "stockout_days": s.stockout_days,
           "stock_value": s.avg_stock_value, "lost_ratio": lr.ratio,
           "lr_excl0": bool(lr.diff_lo > 0 or lr.diff_hi < 0), "so_excl0": bool(so.diff_lo > 0 or so.diff_hi < 0)}
    row.update(extra or {})
    rows.append(row)
    print(row, flush=True)


s0 = replay.summarise(base)
rows.append({"variant": "base", "lost_revenue": s0.lost_revenue, "lost_stockout": s0.lost_revenue_stockout,
             "lost_shelfout": s0.lost_revenue_shelfout, "stockout_days": s0.stockout_days,
             "stock_value": s0.avg_stock_value})
for sigma in (0.1, 0.25):
    for seed in range(5):
        f = drug_noise(sigma, 500 + seed)
        _, per = run("noise", lambda b, f=f: perturbed(b, f(b)))
        record(f"noise s={sigma}", per, {"sigma": sigma, "seed": seed})
_, per = run("bias", lambda b: perturbed(b, np.full(len(b.mean), 0.9)))
record("all means x0.9", per)
_, per = run("var", lambda b: perturbed(b, None, 1.5))
record("variance x1.5", per)

res = pd.DataFrame(rows)
res.to_csv(OUT / "power_rolling_unmatched_H2-2024.csv", index=False)
with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
    print(res.to_string(index=False))
    print(res.dropna(subset=["sigma"]).groupby("sigma")[["lost_ratio", "lr_excl0", "so_excl0"]].agg(["mean", "min", "max"]))
