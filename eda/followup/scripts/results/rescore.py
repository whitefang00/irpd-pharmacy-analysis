"""Re-score a plain (rho=None) Scores object with any per-window variance factor.

Used for the Eid-split rho arm (E2), which the harness's tier-only `rho` cannot
express. For a window with plain variance v = max(v_raw, m), the scaled variance
is max(v_raw * f, m); NB variances here are >= the mean, so this equals
max(v * f, m). Checked against score_arm(rho=...) in e2.py.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from irpd import forecast as fc
from irpd.evalx.scorer import Q, Scores, crps_nb, pinball, summarise


def rescore(s: Scores, f_l: np.ndarray, f_lc: np.ndarray, tag=None) -> Scores:
    w = s.windows.copy()
    for name, f in (("l", f_l), ("lc", f_lc)):
        m = w[f"m_{name}"].to_numpy(float)
        v = np.maximum(w[f"v_{name}"].to_numpy(float) * f, m)
        w[f"v_{name}"] = v
        q = fc.demand_quantile(m, v, Q)
        y = w[f"y_{name}"].to_numpy(float)
        w[f"q_{name}"] = q
        w[f"pin_{name}"] = pinball(y, q, Q)
        w[f"cov_{name}"] = (y <= q).astype(float)
        w[f"w_pin_{name}"] = w[f"pin_{name}"] * w["price"]
    w["crps_lc"] = crps_nb(w["m_lc"], w["v_lc"], w["y_lc"])
    w["w_crps_lc"] = w["crps_lc"] * w["price"]
    return Scores(w, s.weekly, summarise(w, s.weekly), tag)


def eid_mask(origin: pd.Series, days: np.ndarray, eids, before=30, after=28) -> np.ndarray:
    """True where the window [origin+1, origin+days] overlaps any [Eid-before, Eid+after]."""
    start = pd.to_datetime(origin) + pd.Timedelta(days=1)
    end = start + pd.to_timedelta(np.asarray(days) - 1, unit="D")
    hit = np.zeros(len(start), bool)
    for e in eids:
        lo, hi = pd.Timestamp(e - timedelta(days=before)), pd.Timestamp(e + timedelta(days=after))
        hit |= ((start <= hi) & (end >= lo)).to_numpy()
    return hit


def factor(days: np.ndarray, rho: np.ndarray) -> np.ndarray:
    return 1 + rho * (np.maximum(np.asarray(days, float) / 7, 1) - 1)
