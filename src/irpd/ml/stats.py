"""How sure a WAPE difference is: a paired bootstrap over drugs, and per-origin wins.

Both take two predictions tables, each already filtered to one method,
config, seed and period, and compare them on the rows they share
(origin, drug_code, h) with a known actual. Differences are A minus B, so a
negative difference means A has the lower WAPE.

**What the interval means.** `paired_bootstrap` resamples drugs, keeping all
of a drug's rows together, and recomputes both WAPEs on the same resample.
Its interval is variation across the observed mix of drugs only. Drugs share
store traffic and Eid shocks, and the interval says nothing about other
calendar periods. `origin_wins` shows the other axis: the difference at each
origin, and how many origins each side wins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["origin", "drug_code", "h"]


def _paired(pred_a: pd.DataFrame, pred_b: pd.DataFrame) -> pd.DataFrame:
    for name, p in (("pred_a", pred_a), ("pred_b", pred_b)):
        if p.duplicated(KEYS).any():
            raise ValueError(f"{name} has several rows per (origin, drug_code, h); "
                             "filter it to one method, config and seed")
    a = pred_a.dropna(subset=["actual"])[[*KEYS, "tier", "actual", "mean"]]
    b = pred_b.dropna(subset=["actual"])[[*KEYS, "mean"]]
    m = a.merge(b, on=KEYS, suffixes=("_a", "_b"), validate="one_to_one")
    if m.empty:
        raise ValueError("no shared rows with a known actual")
    m["err_a"] = (m["mean_a"] - m["actual"]).abs()
    m["err_b"] = (m["mean_b"] - m["actual"]).abs()
    m["tier"] = m["tier"].astype(str)
    return m


def _groups(m: pd.DataFrame, by: str | None):
    if by is None:
        yield "all", m
    else:
        yield from m.groupby(by, sort=True)


def paired_bootstrap(pred_a: pd.DataFrame, pred_b: pd.DataFrame, by: str | None = "tier",
                     n: int = 2000, seed: int = 0, level: float = 0.9) -> pd.DataFrame:
    """WAPE of A, of B, their difference and a `level` interval, per `by` group.

    Drugs are resampled with replacement within each group (a drug whose tier
    changes between origins counts once in each tier it was in). The interval
    is the percentile interval of the resampled differences: variation across
    drugs only, not across calendar periods.
    """
    m = _paired(pred_a, pred_b)
    rng = np.random.default_rng(seed)
    q = [(1 - level) / 2, 1 - (1 - level) / 2]
    out = []
    for key, g in _groups(m, by):
        d = g.groupby("drug_code")[["err_a", "err_b", "actual"]].sum()
        ea, eb, act = (d[c].to_numpy() for c in ("err_a", "err_b", "actual"))
        k = len(d)
        counts = rng.multinomial(k, np.full(k, 1 / k), size=n).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = (counts @ (ea - eb)) / (counts @ act)
        boot = boot[np.isfinite(boot)]
        lo, hi = np.quantile(boot, q) if len(boot) else (np.nan, np.nan)
        total = act.sum()
        out.append({by or "group": key, "drugs": k, "rows": len(g),
                    "wape_a": ea.sum() / total, "wape_b": eb.sum() / total,
                    "diff": (ea.sum() - eb.sum()) / total, "lo": lo, "hi": hi})
    return pd.DataFrame(out)


def origin_wins(pred_a: pd.DataFrame, pred_b: pd.DataFrame, by: str | None = "tier"
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-origin WAPE difference (A - B), and per group how many origins each wins.

    Returns (per_origin, summary). `per_origin` has one row per (group,
    origin); `summary` counts `a_wins` (A's WAPE lower), `b_wins` and `ties`.
    """
    m = _paired(pred_a, pred_b)
    rows = []
    for key, g in _groups(m, by):
        for origin, o in g.groupby("origin", sort=True):
            act = o["actual"].sum()
            wa, wb = o["err_a"].sum() / act, o["err_b"].sum() / act
            rows.append({by or "group": key, "origin": origin, "wape_a": wa, "wape_b": wb,
                         "diff": wa - wb})
    per = pd.DataFrame(rows)
    col = by or "group"
    summary = per.groupby(col, sort=True)["diff"].agg(
        origins="size", a_wins=lambda s: int((s < 0).sum()), b_wins=lambda s: int((s > 0).sum()),
        ties=lambda s: int((s == 0).sum()), mean_diff="mean").reset_index()
    return per, summary
