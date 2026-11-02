"""Own-demand features (d_): lags, means, intermittency, last year, censoring, shelf-outs.

Everything is computed from the history cut at the origin. Per (drug, origin):

- `d_lag_1..4`: units in the last four weeks.
- `d_mean_{4,8,13,26,52}`: mean weekly units over the window.
- `d_sale_share_{13,52}`: share of the window's observed weeks with a sale.
- `d_std_13`, `d_max_13`: spread and peak over 13 weeks.
- `d_since_sale`: weeks since the last week with a sale (0 = the last week).
- `d_since_first`: weeks since the first week with a sale.
- `d_lag_{1..4}_out`: the lag week had a total stock-out day (NaN where the
  stock ledger doesn't cover it).
- `d_out_{4,13}`: stock-out weeks in the window; `d_shelf_{4,13}`: shelf-out
  days in the window (NaN unless the ledger covers the whole window).

- `d_ewm_15`: exponentially smoothed level (alpha 0.15) over observed weeks,
  the level simple exponential smoothing would forecast. Added after the EDA,
  where SES at 0.15 was the best simple method.

Last-year values live in the `season` add-on group (EDA: no yearly pattern
beyond Eid, and they made forecasts worse).

Weeks before a drug's first sale are missing and skipped everywhere. The
censoring mode (`ctx.censor_mode`, plan 5.3) sets what a stock-out week
holds in the lag series behind the lags, means, shares, spread, peak and
last-year values:

- mask: missing, so skipped in means;
- raw: the recorded units;
- impute: the lost-demand estimate (`imputed_units`).
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd

from irpd import lost
from irpd.ml import panel as pn
from irpd.ml import schema, split
from irpd.tables.stock import spells_as_of

NEEDS_SQL = False  # only the impute mode needs a warehouse connection
LAGS = 4
MEANS = (4, 8, 13, 26, 52)
SHARES = (13, 52)
COUNTS = (4, 13)
EWM_ALPHA = 0.15


def imputed_units(ctx: schema.PanelContext) -> np.ndarray:
    """Weekly units with each stock-out day's sales replaced by lost.py's expected demand.

    Drugs x weeks, aligned with `ctx.history`. The estimator is the one in
    `lost.py`: the drug's units per unit of shop traffic over its last clean
    days before the spell started, times each day's traffic, over the first
    `lost.MAX_DAYS` days after the spell's first day (that day sold out
    partway, so its sales are kept). It is applied day by day rather than
    spread over a spell's total, so a day's value uses only the baseline
    before its spell and that day's traffic. A week's value therefore depends
    on nothing after the week, and cutting the series at an origin is the
    same as computing it from data cut there (leak test 1 checks this). Days
    whose spell has no baseline make the week NaN, as in the mask mode.
    """
    if not pn.has_sql(ctx):
        raise ValueError("the impute censoring mode needs a warehouse connection")
    h = ctx.history
    con = ctx.connection_at(ctx.data_end)
    p = lost.panel(con, ctx.data_end)
    daily = p.units.copy()
    sp = spells_as_of(con, ctx.data_end, "stockouts").df()
    sp = sp[sp.drug_code.isin(p.index)]
    if len(sp):
        j = sp.drug_code.map(p.index).to_numpy(int)
        s = np.array([p.day(d + timedelta(days=1))
                      for d in pd.to_datetime(sp.start).dt.date])
        count = np.clip(sp["days"].to_numpy(int) - 1, 0, lost.MAX_DAYS)
        _, r = lost.rates(p, j, s)
        for jj, ss, cc, rate in zip(j, s, count, r["units"], strict=True):
            daily[jj, ss:ss + cc] = rate * p.w[ss:ss + cc]
    weeks = h["weeks"]
    out = np.full(h["y"].shape, np.nan)
    if not len(weeks):
        return out
    rows = np.array([p.index.get(c, -1) for c in h["drugs"]])
    ok = rows >= 0
    for jw, w in enumerate(weeks):
        a = (w - p.days[0]).days
        if a < 0 or a + 7 > daily.shape[1]:
            continue
        out[ok, jw] = daily[rows[ok], a:a + 7].sum(axis=1)
    return out


def lag_series(h: dict, mode: str, imputed: np.ndarray | None = None) -> np.ndarray:
    """The weekly series behind lags and means, under a censoring mode."""
    if mode == "raw":
        return h["y"]
    if mode == "mask":
        return np.where(h["out"], np.nan, h["y"])
    if mode == "impute":
        if imputed is None:
            raise ValueError("impute needs the imputed series")
        return np.where(h["out"], imputed[:, :h["y"].shape[1]], h["y"])
    raise ValueError(f"unknown censor mode {mode!r}")


def _window(a: np.ndarray, n: int, w: int) -> np.ndarray:
    return a[:, max(0, n - w):n]


def _col(a: np.ndarray, i: int, fill=np.nan) -> np.ndarray:
    return a[:, i].astype(float) if 0 <= i < a.shape[1] else np.full(len(a), fill)


def _since(mask: np.ndarray, last: bool) -> np.ndarray:
    """Weeks from the last (or first) True column to the final column; NaN if none."""
    if not mask.shape[1]:
        return np.full(len(mask), np.nan)
    gap = (np.argmax(mask[:, ::-1], axis=1) if last
           else mask.shape[1] - 1 - np.argmax(mask, axis=1))
    return np.where(mask.any(axis=1), gap, np.nan)


def _origin_features(h: dict, s: np.ndarray, n: int) -> dict[str, np.ndarray]:
    """Per-drug features at one origin, from the cut history and lag series."""
    f: dict[str, np.ndarray] = {}
    known = h["stock_known"]
    out = h["out"].astype(float)
    for k in range(1, LAGS + 1):
        f[f"d_lag_{k}"] = _col(s, n - k)
    for w in MEANS:
        f[f"d_mean_{w}"] = np.nanmean(_window(s, n, w), axis=1)
    for w in SHARES:
        win = _window(s, n, w)
        f[f"d_sale_share_{w}"] = (win > 0).sum(axis=1) / (~np.isnan(win)).sum(axis=1)
    win = _window(s, n, 13)
    f["d_std_13"] = np.nanstd(win, axis=1)
    f["d_max_13"] = np.nanmax(win, axis=1) if win.shape[1] else np.full(len(s), np.nan)
    f["d_ewm_15"] = ewm_level(s[:, :n], EWM_ALPHA)
    sold = np.nan_to_num(h["y"]) > 0
    f["d_since_sale"] = _since(sold, last=True)
    f["d_since_first"] = _since(sold, last=False)
    for k in range(1, LAGS + 1):
        ok = 0 <= n - k < len(known) and known[n - k]
        f[f"d_lag_{k}_out"] = _col(out, n - k) if ok else np.full(len(s), np.nan)
    for w in COUNTS:
        full = n >= w and known[n - w:n].all()
        nan = np.full(len(s), np.nan)
        f[f"d_out_{w}"] = _window(out, n, w).sum(axis=1) if full else nan
        f[f"d_shelf_{w}"] = (_window(h["shelfout_days"].astype(float), n, w).sum(axis=1)
                             if full else nan)
    return f


def ewm_level(s: np.ndarray, alpha: float) -> np.ndarray:
    """Per row, the smoothed level after the last column; NaN weeks are skipped."""
    level = np.full(len(s), np.nan)
    for j in range(s.shape[1]):
        x = s[:, j]
        seen = ~np.isnan(x)
        start = seen & np.isnan(level)
        level = np.where(start, x, level)
        upd = seen & ~start
        level = np.where(upd, level + alpha * (x - level), level)
    return level


def build(ctx: schema.PanelContext) -> pd.DataFrame:
    imputed = imputed_units(ctx) if ctx.censor_mode == "impute" else None
    parts = []
    for origin in sorted(set(ctx.origins)):
        parts.append(build_origin(ctx, origin, imputed))
    return pn.finish(parts, by_h=True)


def build_origin(ctx: schema.PanelContext, origin: date,
                 imputed: np.ndarray | None = None) -> pd.DataFrame:
    rows = pn.active_rows(ctx, origin)
    full = ctx.history_at(origin)
    n = full["y"].shape[1]
    h = {**full, **{k: full[k][rows] for k in ("y", "out", "revenue", "shelfout_days")}}
    s = lag_series(h, ctx.censor_mode, None if imputed is None else imputed[rows])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per_origin = _origin_features(h, s, n)
    cols = {k: np.repeat(v[:, None], split.HORIZON, axis=1)
            for k, v in per_origin.items()}
    return pn.frame(ctx, origin, rows, cols, by_h=True)
