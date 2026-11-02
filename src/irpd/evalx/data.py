"""Inputs for the scorer built from the warehouse: daily truth, labels, current preds.

The forecast agent writes the shared files (`preds/<arm>.parquet`,
`actuals.parquet`, `labels.parquet`). These helpers build the same shapes for
the current forecaster directly, so the harness can be checked and calibrated
on its own, and give the daily truth that exact lead-time windows need.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from irpd import config, contest, lost, replay
from irpd import forecast as fc
from irpd.evalx.rolling import view

PRICE_DAYS = 182


def daily_truth(first: date, last: date, as_of: date | None = None) -> pd.DataFrame:
    """drug_code, date, units, uncensored_units, stockout for every drug, first..last.

    `as_of` (default `last`) is the day the connection stands on; the
    uncensored estimate uses spells known by then and switch rates from spells
    before `first`.
    """
    as_of = as_of or last
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    with view(as_of) as con:
        drugs = [r[0] for r in con.execute(
            "SELECT DISTINCT drug_code FROM daily_demand ORDER BY 1").fetchall()]
        unc = lost.uncensored_units(con, drugs, days)
        rec = replay.recorded_demand(con, drugs, days)
        out = con.execute(f"""
            SELECT drug_code, date, closing <= 0 AS stockout FROM daily_stock
            WHERE location = 'total' AND date BETWEEN DATE '{first}' AND DATE '{last}'
              AND date >= DATE '{config.STOCK_START}'""").df()
    so = replay._matrix(out.assign(stockout=out.stockout.astype(float)), "stockout",
                        drugs, days) > 0
    return pd.DataFrame({"drug_code": np.repeat(drugs, len(days)),
                         "date": np.tile(np.array(days, dtype=object), len(drugs)),
                         "units": rec.ravel(), "uncensored_units": unc.ravel(),
                         "stockout": so.ravel()})


def labels_at(con, origin: date, codes: list[str]) -> pd.DataFrame:
    """lead_days and trailing-182-day average price per drug, as of `origin`."""
    contest.ensure_baseline(con, origin)
    lead = replay.default_lead(con).reindex(codes).fillna(replay.LEAD_FALLBACK)
    since = origin - timedelta(days=PRICE_DAYS - 1)
    pr = con.execute(f"""
        SELECT drug_code, sum(revenue) / nullif(sum(units), 0) AS price FROM daily_demand
        WHERE units > 0 AND date BETWEEN DATE '{since}' AND DATE '{origin}'
        GROUP BY 1""").df().set_index("drug_code")["price"]
    allp = con.execute(f"""
        SELECT drug_code, sum(revenue) / nullif(sum(units), 0) AS price FROM daily_demand
        WHERE units > 0 AND date <= DATE '{origin}' GROUP BY 1""").df() \
        .set_index("drug_code")["price"]
    price = pr.reindex(codes).fillna(allp.reindex(codes)).fillna(0)
    return pd.DataFrame({"origin": origin, "drug_code": codes,
                         "lead_days": lead.to_numpy(int), "price": price.to_numpy(float)})


def current_preds(origins: list[date]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The current forecaster at each origin, in the arm format, and its labels.

    Every drug with a tier at the origin (sold by then), h = 1..8, variance =
    mean x the drug's dispersion, as `forecast.forecast` gives it.
    """
    preds, labels = [], []
    for o in origins:
        with view(o) as con:
            hist = fc.load_history(con, o, None)
            tier_of = fc.tiers(con, o)
            tier = np.array([tier_of.get(c, "none") for c in hist["drugs"]], dtype=object)
            method = np.array([fc.METHOD_BY_TIER.get(t, "snaive") for t in tier], dtype=object)
            p = fc.predict(hist, method, 8)
            keep = tier != "none"
            codes = list(hist["drugs"][keep])
            n = len(codes)
            preds.append(pd.DataFrame({
                "origin": o, "drug_code": np.repeat(codes, 8), "h": np.tile(np.arange(1, 9), n),
                "week_start": np.tile(np.array(p["future"], dtype=object), n),
                "mean": p["mean"][keep].ravel(), "var": p["var"][keep].ravel(),
                "tier": np.repeat(tier[keep], 8)}))
            if p["future"][0] != o + timedelta(days=1):
                raise ValueError(f"origin {o}: first forecast week is {p['future'][0]}")
            labels.append(labels_at(con, o, codes).assign(tier=tier[keep]))
    return pd.concat(preds, ignore_index=True), pd.concat(labels, ignore_index=True)
