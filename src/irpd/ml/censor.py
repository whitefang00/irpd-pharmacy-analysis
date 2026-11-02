"""Censoring-aware training targets (experiment E4). Training only; scoring is elsewhere.

The panel drops a target week that had a total stock-out day or SHELF_CENSOR_DAYS
shelf-out days (`schema`: target NaN, `censored` True). Those are the weeks
demand ran high, so a model trained without them is biased low on drugs that
often run out. `censor_aware(panel, ctx, ...)` puts them back:

- **Stock-out weeks** (`target="impute"`): the week's units with each total
  stock-out day replaced by expected demand, the `lost.py` estimator applied
  day by day: the drug's units per unit of shop traffic over its last clean
  days before the spell started, times that day's traffic. As in `lost.py`,
  a spell's first day keeps its recorded sales (it sold out partway) and only
  its first `lost.MAX_DAYS` days after that are estimated. A week with any
  out day that can't be estimated (no baseline, or past `MAX_DAYS`, where the
  drug is more likely delisted than missed) stays NaN. This is the gross
  estimate: no demand is taken off alternatives.
- **Shelf-out weeks** (`shelf="flag"`): weeks dropped only for shelf-outs keep
  their recorded units as target, and every row gets `SHELF_FLAG`, the
  target week's shelf-out days (NaN before the stock ledger). The model
  learns how much a week with the floor empty sells; at prediction the flag
  is set to 0 (`for_prediction`), so the forecast is demand with the shelf
  stocked. With `shelf="mask"` they stay NaN, as now.

**No leak.** A week's estimate uses only days up to the week's end: the
baseline ends before its spell starts, traffic is the day's own, and
whether a day is past `MAX_DAYS` depends on the spell's start only. Spells
are read as they stood on `ctx.data_end`, and a spell still running later
has the same days up to any earlier week end. So a training row for origin O
(its target week ended by O) gets the same target from data cut at O as from
all the data (tests/test_ml_censor.py checks this). The shelf flag of a
training row is its own past target week's; prediction rows get 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from irpd import lost
from irpd.ml import panel as pn
from irpd.ml import schema
from irpd.tables.stock import spells_as_of

TARGETS = ("mask", "impute")
SHELF = ("mask", "flag")
SHELF_FLAG = "d_tgt_shelf"


def stockout_estimate(ctx: schema.PanelContext) -> tuple[np.ndarray, np.ndarray]:
    """Weekly units with stock-out days estimated, and whether every out day was.

    Two drugs x weeks arrays aligned with `ctx.history`: the units (NaN where
    the drug or week isn't in the daily data) and `ok` (False where a
    stock-out day in the week couldn't be estimated).
    """
    if not pn.has_sql(ctx):
        raise ValueError("censoring-aware targets need a warehouse connection")
    h = ctx.history
    con = ctx.connection_at(ctx.data_end)
    p = lost.panel(con, ctx.data_end)
    daily = p.units.copy()
    bad = np.zeros(daily.shape, dtype=bool)
    sp = spells_as_of(con, ctx.data_end, "stockouts").df()
    sp = sp[sp.drug_code.isin(p.index)]
    if len(sp):
        j = sp.drug_code.map(p.index).to_numpy(int)
        first = np.array([p.day(d) for d in pd.to_datetime(sp.start).dt.date])
        days = sp["days"].to_numpy(int)
        s = first + 1  # the first day sold out partway: its sales are real
        counted = np.clip(days - 1, 0, lost.MAX_DAYS)
        _, r = lost.rates(p, j, s)
        for jj, ss, cc, end, rate in zip(j, s, counted, first + days, r["units"],
                                         strict=True):
            if np.isfinite(rate):
                daily[jj, ss:ss + cc] = rate * p.w[ss:ss + cc]
            else:
                bad[jj, ss:ss + cc] = True
            bad[jj, ss + cc:end] = True  # past MAX_DAYS: not estimated
    weeks = h["weeks"]
    units = np.full(h["y"].shape, np.nan)
    ok = np.ones(h["y"].shape, dtype=bool)
    rows = np.array([p.index.get(c, -1) for c in h["drugs"]])
    have = rows >= 0
    for jw, w in enumerate(weeks):
        a = (w - p.days[0]).days
        if a < 0 or a + 7 > daily.shape[1]:
            continue
        units[have, jw] = daily[rows[have], a:a + 7].sum(axis=1)
        ok[have, jw] = ~bad[rows[have], a:a + 7].any(axis=1)
    return units, ok


def _cells(panel: pd.DataFrame, h: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """History (row, column) of each panel row's target week, and whether it is in it."""
    idx = {c: i for i, c in enumerate(h["drugs"])}
    col = {pd.Timestamp(w): j for j, w in enumerate(h["weeks"])}
    j = panel["drug_code"].map(idx).to_numpy()
    t = panel["target_week"].map(col)
    inside = t.notna().to_numpy() & panel["available"].to_numpy(bool)
    return j, t.fillna(0).to_numpy(int), inside


def censor_aware(panel: pd.DataFrame, ctx: schema.PanelContext, target: str = "impute",
                 shelf: str = "flag") -> pd.DataFrame:
    """A copy of `panel` with censored target weeks put back (see the module doc).

    `censored` keeps saying what happened in the week. The copy breaks the
    panel contract on purpose (censored rows with a target), so it is for
    fitting only; don't pass it to `schema.validate_panel`.
    """
    if target not in TARGETS or shelf not in SHELF:
        raise ValueError(f"target must be one of {TARGETS}, shelf one of {SHELF}")
    h = ctx.history
    out = panel.copy()
    j, t, inside = _cells(out, h)
    y = out["target"].to_numpy(np.float64).copy()
    cens = out["censored"].to_numpy(bool) & inside
    stock = np.zeros(len(out), dtype=bool)
    stock[inside] = h["stockout"][j[inside], t[inside]]
    if target == "impute":
        units, ok = stockout_estimate(ctx)
        pick = cens & stock
        est = np.full(len(out), np.nan)
        est[pick] = np.where(ok[j[pick], t[pick]], units[j[pick], t[pick]], np.nan)
        y[pick] = est[pick]
    if shelf == "flag":
        only_shelf = cens & ~stock
        y[only_shelf] = h["y"][j[only_shelf], t[only_shelf]]
        flag = np.full(len(out), np.nan)
        known = inside & np.isin(t, np.flatnonzero(h["stock_known"]))
        flag[known] = h["shelfout_days"][j[known], t[known]]
        out[SHELF_FLAG] = flag.astype(np.float32)
    out["target"] = np.where(np.isfinite(y) & (y >= 0), y, np.nan).astype(np.float32)
    return out


def for_prediction(rows: pd.DataFrame) -> pd.DataFrame:
    """Rows to predict on: a stocked shelf (SHELF_FLAG 0) where the flag is used."""
    if SHELF_FLAG in rows.columns:
        rows = rows.assign(**{SHELF_FLAG: np.float32(0)})
    return rows
