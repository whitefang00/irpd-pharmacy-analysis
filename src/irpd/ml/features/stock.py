"""Stock features (k_, ablation step A4 only; never used by the replay forecaster).

- `k_on_hand`: total stock (floor + store) closing on the origin day, from
  `daily_stock` as it stood on the origin.
- `k_days_cover`: on-hand stock (negatives taken as 0) over the mean units
  sold per day in the COVER_DAYS days up to the origin. Both come from the
  stock ledger, so they share its unit. The mean is over the days the ledger
  has rows for the drug in that window; NaN when nothing sold.

Both are NaN where the ledger has no row for the drug on the origin day.
See plan 5.4 for why these are an ablation only.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from irpd.ml import history as hist
from irpd.ml import schema

NEEDS_SQL = True  # reads daily_stock

COVER_DAYS = 91
COLUMNS = ["k_on_hand", "k_days_cover"]


def _d(day: date) -> str:
    return f"DATE '{day.isoformat()}'"


def _origin(ctx: schema.PanelContext, origin: date) -> pd.DataFrame:
    con = ctx.connection_at(origin)
    first = origin - timedelta(days=COVER_DAYS - 1)
    return con.execute(f"""
        SELECT drug_code,
               max(closing) FILTER (WHERE date = {_d(origin)})::DOUBLE AS on_hand,
               sum(sold)::DOUBLE / count(*) AS per_day
        FROM daily_stock
        WHERE location = 'total' AND date BETWEEN {_d(first)} AND {_d(origin)}
        GROUP BY drug_code""").df()


def build(ctx: schema.PanelContext) -> pd.DataFrame:
    """Stock features indexed by (drug_code, origin)."""
    frames = []
    for origin in ctx.origins:
        s = _origin(ctx, origin).set_index("drug_code").reindex(ctx.drugs)
        on_hand = s["on_hand"].to_numpy(float)
        per_day = s["per_day"].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            cover = np.where(per_day > 0, np.clip(on_hand, 0, None) / per_day, np.nan)
        cover = np.where(np.isnan(on_hand), np.nan, cover)
        df = pd.DataFrame({"drug_code": ctx.drugs, "origin": pd.Timestamp(origin),
                           "k_on_hand": on_hand, "k_days_cover": cover})
        # Only drugs with panel rows at this origin (none that first sell later).
        frames.append(df.iloc[np.flatnonzero(
            hist.active(ctx.history, origin, schema.ACTIVE_WEEKS))])
    out = (pd.concat(frames, ignore_index=True) if frames
           else pd.DataFrame(columns=["drug_code", "origin", *COLUMNS]))
    out["origin"] = out["origin"].astype("datetime64[s]")
    out[COLUMNS] = out[COLUMNS].astype("float32")
    return out.set_index(["drug_code", "origin"])[COLUMNS]
