"""Weekly history per drug, aligned arrays every model and feature builder shares.

`load(con, until)` extends `forecast.load_history` with the series feature
builders would otherwise each compute their own way:

- `stock_known`: per week, True from the first full Monday-start week on or
  after `config.STOCK_START` (Mon 2 Oct 2023). Before that the ledger can't
  show stock-outs, so `out` is set False there and targets are kept.
- `revenue`: rupiah per drug and week (NaN where `y` is NaN).
- `shelfout_days`: days in the week inside a shelf-out spell (floor empty,
  store stocked), as the spells stood on `until`.
- `first_sale`: index of each drug's first week with a sale (-1 if none).
- `stockout`: the week had a total stock-out day (`out` before shelf-outs
  are added to it), for censoring-aware targets and segment labels.

`out` also marks weeks with SHELF_CENSOR_DAYS or more shelf-out days. The EDA
found such weeks sell 1-11% of normal when the floor is empty all week, so
they hide demand much like a stock-out does. This applies to every model,
baselines included, since they all read this history.

`cut(h, origin)` gives the same dict cut to the weeks that ended by
`origin`, *including* the daily store `lines`, so nothing (such as Eid
factors learned from `lines`) can see past the origin.
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd

from irpd import config
from irpd import forecast as fc

SHELF_CENSOR_DAYS = 4


def first_stock_week() -> date:
    """Monday of the first week fully covered by the stock ledger."""
    s = config.STOCK_START
    return s + timedelta(days=(7 - s.weekday()) % 7)


def _d(day: date) -> str:
    return f"DATE '{day.isoformat()}'"


def load(con: duckdb.DuckDBPyConnection, until: date,
         drug_codes: list[str] | None = None) -> dict:
    h = fc.load_history(con, until, drug_codes)
    weeks, drugs = h["weeks"], h["drugs"]
    known = np.array([w >= first_stock_week() for w in weeks], dtype=bool)
    h["stock_known"] = known
    h["out"] = h["out"] & known[None, :]

    idx = {c: i for i, c in enumerate(drugs)}
    col = {w: j for j, w in enumerate(weeks)}
    rev = np.full(h["y"].shape, np.nan)
    wk = con.execute(f"""
        SELECT drug_code, week_start, revenue::DOUBLE AS revenue FROM weekly_demand
        WHERE week_end <= {_d(until)}""").df()
    wk["week_start"] = pd.to_datetime(wk.week_start).dt.date
    wk = wk[wk.drug_code.isin(idx) & wk.week_start.isin(col)]
    rev[wk.drug_code.map(idx).to_numpy(), wk.week_start.map(col).to_numpy()] = wk.revenue
    rev[np.isnan(h["y"])] = np.nan
    h["revenue"] = rev

    shelf = np.zeros(h["y"].shape, dtype=np.int16)
    if len(weeks):
        last = weeks[-1] + timedelta(days=6)
        sp = con.execute(f"""
            SELECT drug_code, start, least(stop, {_d(last)}) AS stop FROM shelfouts
            WHERE start <= {_d(last)}""").df()
        for code, a, b in zip(sp.drug_code, pd.to_datetime(sp.start).dt.date,
                              pd.to_datetime(sp.stop).dt.date, strict=True):
            if code not in idx:
                continue
            d = a
            while d <= b:
                j = col.get(d - timedelta(days=d.weekday()))
                if j is not None:
                    shelf[idx[code], j] += 1
                d += timedelta(days=1)
    h["shelfout_days"] = shelf
    h["stockout"] = h["out"].copy()  # a total stock-out day only, before shelf-outs join in
    h["out"] = h["out"] | ((shelf >= SHELF_CENSOR_DAYS) & known[None, :])

    h["first_sale"] = _first_sale(h["y"])
    return h


def _first_sale(y: np.ndarray) -> np.ndarray:
    """Index of each row's first week with a sale, -1 if none (works with no weeks)."""
    sold = np.nan_to_num(y) > 0
    if sold.shape[1] == 0:
        return np.full(len(y), -1)
    return np.where(sold.any(axis=1), sold.argmax(axis=1), -1)


def weeks_to(h: dict, origin: date) -> int:
    """How many history weeks ended by `origin`."""
    return int(sum(w + timedelta(days=6) <= origin for w in h["weeks"]))


def cut(h: dict, origin: date) -> dict:
    """The history as it stood on `origin`: weekly arrays and daily lines both cut."""
    n = weeks_to(h, origin)
    out = dict(h)
    for k in ("y", "out", "revenue", "shelfout_days", "stockout"):
        if k in h:
            out[k] = h[k][:, :n]
    out["weeks"], out["stock_known"] = h["weeks"][:n], h["stock_known"][:n]
    out["lines"] = h["lines"][h["lines"].index <= origin]
    if "store_units" in h:
        out["store_units"] = h["store_units"][h["store_units"].index <= origin]
    out["first_sale"] = _first_sale(out["y"])
    return out


def target_scale(h: dict, origin: date) -> np.ndarray:
    """Scaled-target denominator per drug on `origin` (schema.SCALE_WEEKS / SCALE_FLOOR)."""
    from irpd.ml.schema import SCALE_FLOOR, SCALE_WEEKS

    c = cut(h, origin)
    y = np.where(c["out"], np.nan, c["y"])
    scale = np.full(len(y), np.nan)
    for w in SCALE_WEEKS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows
            m = np.nanmean(y[:, -w:], axis=1) if y.shape[1] else np.full(len(y), np.nan)
        scale = np.where(np.isnan(scale) | (scale <= 0), m, scale)
    return np.where(np.isnan(scale) | (scale <= 0), SCALE_FLOOR, scale).astype(np.float32)


def mase_scale(h: dict, origin: date) -> np.ndarray:
    """In-sample naive MAE per drug, as `forecast.backtest` computes it (NaN if none)."""
    c = cut(h, origin)
    y = np.where(c["out"], np.nan, c["y"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.abs(np.diff(y, axis=1)), axis=1).astype(np.float32)


def active(h: dict, origin: date, weeks: int) -> np.ndarray:
    """Drugs with a sale in the `weeks` weeks before `origin`."""
    c = cut(h, origin)
    return np.nansum(c["y"][:, -weeks:], axis=1) > 0
