"""Static vs rolling policy, and how noisy the replay's lost-revenue comparison is.

The project's replay sets (s, S) once on the policy day and holds them for six
months. Here the same engine is run with (s, S) refreshed every 14 days from the
current forecaster as of each refresh day (data up to that day only), matched to
the pharmacist's stock value by one store-wide scale, as in the project. Also an
oracle that knows the realised mean daily demand of the next 28 days at each refresh
(tests whether short-horizon accuracy is rewarded once the policy can react).

Part 2: the paired bootstrap over drugs (decide.paired_bootstrap_replay) applied to
the current forecaster vs. itself with small random mean noise (s = 0.1, WAPE +0.01),
to see whether the interval would declare a "winner" from noise alone.

Window: H2 2024 (policy day 30 Jun 2024). Read-only, in-memory connections.
Needs sensitivity.py H2-2024 first (it reuses that run's cached history and basis).
Run from the repo root:  uv run python eda/followup/scripts/inventory/rolling_policy.py
"""
import pickle
import sys
from copy import copy
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from irpd import contest, holdout, replay
from irpd.ml import decide

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402

OUT = fp.out("inventory")
label = "H2-2024"
h, b0 = pickle.loads((OUT / f"cache_{label}.pkl").read_bytes())
REFRESH = 14
T = len(h.days)
refresh_idx = list(range(0, T, REFRESH))

bases_cache = OUT / f"rolling_bases_{label}.pkl"
if bases_cache.exists():
    bases = pickle.loads(bases_cache.read_bytes())
else:
    bases = []
    for t in refresh_idx:
        day = h.days[t] - timedelta(days=1)
        with holdout.agent_view(day) as con:
            bases.append(decide.baseline_basis(con, day))
        print("basis", day, flush=True)
    bases_cache.write_bytes(pickle.dumps(bases))


def arrays(pol):
    return replay.policy_arrays(pol, h)


def policies_over_time(make):
    """s, S, floor arrays drugs x days from a policy per refresh."""
    n = len(h.drugs)
    S = {k: np.zeros((n, T)) for k in ("s", "S", "floor_min", "floor_max")}
    lead = None
    prev = None
    for j, t in enumerate(refresh_idx):
        pol = make(j).set_index("drug_code")
        if prev is not None:  # drugs that left the candidate list keep their last rule
            pol = pd.concat([pol, prev[~prev.index.isin(pol.index)]])
        prev = pol
        a = arrays(pol)
        stop = refresh_idx[j + 1] if j + 1 < len(refresh_idx) else T
        for k in S:
            S[k][:, t:stop] = a[k][:, None]
        lead = a["lead"] if lead is None else lead
    S["lead"] = lead
    return S


def simulate_tv(p, k):
    """replay.simulate with time-varying s, S (scaled by k) and floor levels."""
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
        fmin, fmax = p["floor_min"][:, t], np.where(np.isnan(p["floor_max"][:, t]),
                                                    p["S"][:, t] * k, p["floor_max"][:, t])
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
    return out


def matched_tv(p):
    target = float(replay.actual(h)["avg_stock_value"].sum())

    def val(k):
        r = simulate_tv(p, k)
        return float((np.maximum(r["floor"] + r["store"], 0) * h.price[:, None]).mean(1).sum())
    lo, hi = 0.0, 1.0
    while val(hi) < target:
        lo, hi = hi, hi * 2
    for _ in range(30):
        k = (lo + hi) / 2
        v = val(k)
        if abs(v / target - 1) < 0.005:
            break
        lo, hi = (k, hi) if v < target else (lo, k)
    r = simulate_tv(p, k)
    per = replay.metrics(r["floor"], r["store"], r["sold"], h)
    return per, k


pos = {c: i for i, c in enumerate(h.drugs)}


def oracle_basis(j):
    b = bases[j]
    codes = list(b.facts.index)
    t = refresh_idx[j]
    m = b.mean.copy()
    disp = np.where(b.mean > 0, b.var / np.where(b.mean > 0, b.mean, 1), 1.0)
    for i, c in enumerate(codes):
        if c in pos:
            m[i, :] = h.demand[pos[c], t:t + 28].mean()
    nb = copy(b)
    nb.mean, nb.var = m, m * disp
    return nb


rows, per_all = [], {}
variants = {
    "static, current forecaster": lambda j: contest.to_policy({}, bases[0]),
    "rolling 14d, current forecaster": lambda j: contest.to_policy({}, bases[j]),
    "rolling 14d, oracle next-28-day mean": lambda j: contest.to_policy({}, oracle_basis(j)),
}
for name, make in variants.items():
    p = policies_over_time(make)
    per, k = matched_tv(p)
    per_all[name] = per
    s = replay.summarise(per)
    rows.append({"variant": name, "scale": k, "lost_revenue": s.lost_revenue,
                 "lost_rev_stockout": s.lost_revenue_stockout,
                 "lost_rev_shelfout": s.lost_revenue_shelfout,
                 "stockout_days": s.stockout_days, "shelfout_days": s.shelfout_days,
                 "stock_value": s.avg_stock_value})
    print(rows[-1], flush=True)
res = pd.DataFrame(rows)
res.to_csv(OUT / f"rolling_{label}.csv", index=False)

# Part 2: bootstrap on pure noise (static policy, as in the project)
f = b0.facts
disp0 = np.where(b0.mean > 0, b0.var / np.where(b0.mean > 0, b0.mean, 1), 1.0)
base_pol = contest.to_policy({}, b0)
m, _ = replay.match_stock_value(base_pol, h)
r = replay.simulate(m, h)
per_base = replay.metrics(r.floor, r.store, r.sold, h)
boots = []
for sigma in (0.1, 0.25):
    for seed in range(10):
        rng = np.random.default_rng(100 + seed)
        e = np.exp(rng.normal(-sigma * sigma / 2, sigma, len(f)))[:, None]
        nb = copy(b0)
        nb.mean, nb.var = b0.mean * e, b0.mean * e * disp0
        mm, _ = replay.match_stock_value(contest.to_policy({}, nb), h)
        rr = replay.simulate(mm, h)
        per = replay.metrics(rr.floor, rr.store, rr.sold, h)
        bt = decide.paired_bootstrap_replay(per, per_base)
        lr = bt.loc["lost_revenue"]
        so = bt.loc["stockout_days"]
        boots.append({"sigma": sigma, "seed": seed, "lost_ratio": lr.ratio,
                      "diff_lo": lr.diff_lo, "diff_hi": lr.diff_hi,
                      "excludes_zero": (lr.diff_lo > 0) or (lr.diff_hi < 0),
                      "stockout_ratio": so.ratio,
                      "so_excludes_zero": (so.diff_lo > 0) or (so.diff_hi < 0)})
        print(boots[-1], flush=True)
bdf = pd.DataFrame(boots)
bdf.to_csv(OUT / f"bootstrap_noise_{label}.csv", index=False)
with pd.option_context("display.width", 200):
    print(res.to_string(index=False))
    print(bdf.groupby("sigma")[["lost_ratio", "excludes_zero", "stockout_ratio",
                                "so_excludes_zero"]].agg(["mean", "min", "max"]))
