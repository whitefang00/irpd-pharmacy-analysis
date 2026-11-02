"""Seasonal add-on features (y_): week of year and last year's sales.

Not in any ablation step. The EDA found no yearly pattern beyond Eid, last
year's sales made forecasts worse, and with about a year of training data the
week of year doubles as a clock (models used it to pull growing drugs back to
last year's levels). They are tried once as an add-on (plan section 8).

Per (drug, origin, h):

- `y_week_of_year`: ISO week number of the target week.
- `y_ly`: units in the target week a year back (52 weeks before it, always
  before the origin); `y_ly_mean3`: mean of it and its neighbours. Both use
  the censoring mode's lag series, like the demand group.
"""

from __future__ import annotations

import warnings
from datetime import date

import numpy as np
import pandas as pd

from irpd.ml import panel as pn
from irpd.ml import schema, split
from irpd.ml.features import demand

NEEDS_SQL = False
YEAR = 52


def _last_year(s: np.ndarray, n: int) -> dict[str, np.ndarray]:
    ly = np.full((len(s), split.HORIZON), np.nan)
    ly3 = np.full_like(ly, np.nan)
    for hh in range(1, split.HORIZON + 1):
        t = n + hh - 1 - YEAR  # always < n: known at the origin
        if 0 <= t < n:
            ly[:, hh - 1] = s[:, t]
        idx = [i for i in (t - 1, t, t + 1) if 0 <= i < n]
        if idx:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ly3[:, hh - 1] = np.nanmean(s[:, idx], axis=1)
    return {"y_ly": ly, "y_ly_mean3": ly3}


def build_origin(ctx: schema.PanelContext, origin: date,
                 imputed: np.ndarray | None = None) -> pd.DataFrame:
    rows = pn.active_rows(ctx, origin)
    full = ctx.history_at(origin)
    n = full["y"].shape[1]
    h = {**full, **{k: full[k][rows] for k in ("y", "out")}}
    s = demand.lag_series(h, ctx.censor_mode, None if imputed is None else imputed[rows])
    weeks = [schema.target_week(origin, k) for k in range(1, split.HORIZON + 1)]
    woy = np.array([w.isocalendar().week for w in weeks], float)
    cols = {"y_week_of_year": np.tile(woy, (len(rows), 1)), **_last_year(s, n)}
    return pn.frame(ctx, origin, rows, cols, by_h=True)


def build(ctx: schema.PanelContext) -> pd.DataFrame:
    imputed = demand.imputed_units(ctx) if ctx.censor_mode == "impute" else None
    parts = [build_origin(ctx, o, imputed) for o in sorted(set(ctx.origins))]
    return pn.finish(parts, by_h=True)
