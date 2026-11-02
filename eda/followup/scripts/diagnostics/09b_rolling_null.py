"""Null for 09_rolling: with iid weekly counts at a constant per-drug mean (NB with the
drug's own variance/mean over the same H2 weeks), what correlation between demeaned
past-4-week and next-4-week totals does the same design produce? DIAGNOSTIC.
"""
import os
import duckdb
import numpy as np
import pandas as pd

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402

DATA = fp.DATA
con = duckdb.connect(f"{DATA}/warehouse/irpd.duckdb", read_only=True)
wk = con.execute("SELECT drug_code, week_start, units FROM weekly_demand "
                 "WHERE week_start >= DATE '2025-06-02' AND week_start <= DATE '2025-12-22'").df()
p = wk.pivot(index="drug_code", columns="week_start", values="units").fillna(0)
p = p[p.mean(axis=1) >= 2]
Y = p.to_numpy(float)
rng = np.random.default_rng(0)


def stat(Y):
    T = Y.shape[1]
    s4 = np.stack([Y[:, i:i + 4].sum(1) for i in range(T - 3)], 1)  # 4-week totals from week i
    nxt = s4[:, 4:]           # origin t: weeks t..t+3
    prv = s4[:, :-4]          # weeks t-4..t-1
    a = nxt - nxt.mean(1, keepdims=True)
    b = prv - prv.mean(1, keepdims=True)
    return np.corrcoef(a.ravel(), b.ravel())[0, 1]


obs = stat(Y)
mu = Y.mean(1, keepdims=True)
var = Y.var(1, ddof=1, keepdims=True)
null = []
for _ in range(50):
    disp = np.maximum(var / mu, 1.0001)
    r = mu / (disp - 1)
    lam = rng.gamma(r * np.ones_like(Y), (disp - 1) * np.ones_like(Y))
    null.append(stat(rng.poisson(lam)))
print(f"drugs {len(Y)}, weeks {Y.shape[1]}: observed corr {obs:.3f}; iid-null {np.mean(null):.3f} +- {np.std(null):.3f}")
