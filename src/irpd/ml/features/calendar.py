"""Calendar features of the target week (c_), per (drug, origin, h).

The calendar is known in advance, so these describe the target week itself.
The Eid uplift is the one exception that is learned: it comes from the Eids
seen by the origin only.

- `c_eid_days`: days from the target week's Thursday to the nearest Eid
  (positive: Eid is still ahead). By default the nearest in
  `forecast.EID_LEGACY`, the list the ML system was frozen with.
- `c_eid`: the target week overlaps an Eid window (-7..+28 days).
- `c_ramadan`: share of the target week's days in Ramadan, taken as the
  RAMADAN_DAYS days before Eid.
- `c_eid_uplift`: the store-wide Eid factor for the target week, as the
  baseline forecaster uses it: `forecast.eid_factors` on the store's daily
  sale lines up to the origin, then `forecast.eid_index` (1 outside Eid).

Experiment E3 switches (`ctx.options`, defaults in OPTIONS = the frozen
behaviour):

- `eid_list`: 'legacy' (Eids to 2025) or 'full' (`forecast.EID`, with 2026-27)
  for `c_eid_days`.
- `eid_days_cap`: None, or a number of days beyond which `c_eid_days` is
  missing. Far from Eid it only marks the position in the year, and the
  model fitted one-off store dips to it (Jan 2024).
- `eid_day_counts`: add how many of the target week's 7 days fall in each
  band around the nearest Eid: `c_eid_pre7` (-7..-1), `c_eid_closed` (Eid
  day: the store closed on Eid day in 2023-2025 and traded through every
  cuti bersama day, so only Eid day counts as closed), `c_eid_post7`
  (+1..+7) and `c_eid_post14` (+8..+14). Where the Monday weeks cut the Eid
  days moves each year, and a single Thursday distance can't say which week
  holds the pre-Eid peak.
- `eid_uplift`: 'lines' or 'units' (`forecast.eid_profile`): the uplift from
  store units, unshrunk. Lines rate the week after Eid above the week
  before; units rate it the other way, as the 2025 sales did.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from irpd import forecast as fc
from irpd.ml import panel as pn
from irpd.ml import schema, split

NEEDS_SQL = False
RAMADAN_DAYS = 30
OPTIONS = {"eid_list": "legacy", "eid_days_cap": None, "eid_day_counts": False,
           "eid_uplift": "lines"}
EID_LISTS = ("legacy", "full")
# Bands of days relative to Eid for the day counts, inclusive.
DAY_BANDS = {"c_eid_pre7": (-7, -1), "c_eid_closed": (0, 0), "c_eid_post7": (1, 7),
             "c_eid_post14": (8, 14)}


def options(ctx: schema.PanelContext) -> dict:
    """The calendar switches in force: OPTIONS overridden by `ctx.options`."""
    opts = {**OPTIONS, **{k: v for k, v in ctx.options.items() if k in OPTIONS}}
    if opts["eid_list"] not in EID_LISTS:
        raise ValueError(f"eid_list must be one of {EID_LISTS}")
    if opts["eid_uplift"] not in fc.EID_PROFILES:
        raise ValueError(f"eid_uplift must be one of {fc.EID_PROFILES}")
    return opts


def eid_list(name: str) -> list[date]:
    return list(fc.EID_LEGACY) if name == "legacy" else list(fc.EID)


def eid_days(week: date, eids: list[date] | None = None) -> int:
    mid = week + timedelta(days=3)
    return min(((e - mid).days for e in (eids or fc.EID_LEGACY)), key=abs)


def capped_eid_days(week: date, eids: list[date], cap: float | None) -> float:
    """`eid_days`, or NaN when it is further than `cap` days from Eid."""
    d = eid_days(week, eids)
    return float(d) if cap is None or abs(d) <= cap else np.nan


def day_counts(week: date, eids: list[date] | None = None) -> dict[str, int]:
    """Days of the Monday-start `week` in each DAY_BANDS band around an Eid."""
    eids = eids or fc.EID
    out = dict.fromkeys(DAY_BANDS, 0)
    for i in range(7):
        d = week + timedelta(days=i)
        for e in eids:
            k = (d - e).days
            for name, (lo, hi) in DAY_BANDS.items():
                if lo <= k <= hi:
                    out[name] += 1
    return out


def ramadan_share(week: date) -> float:
    days = [week + timedelta(days=i) for i in range(7)]
    return sum(any(e - timedelta(days=RAMADAN_DAYS) <= d < e for e in fc.EID)
               for d in days) / 7


def origin_columns(ctx: schema.PanelContext, origin: date) -> dict[str, np.ndarray]:
    """HORIZON values per feature for one origin (the same for every drug)."""
    opts = options(ctx)
    eids = eid_list(opts["eid_list"])
    weeks = [schema.target_week(origin, h) for h in range(1, split.HORIZON + 1)]
    factors = fc.eid_profile(ctx.history_at(origin), opts["eid_uplift"])
    cols = {
        "c_eid_days": np.array([capped_eid_days(w, eids, opts["eid_days_cap"])
                                for w in weeks], float),
        "c_eid": np.array([pn.eid_week(w) for w in weeks], float),
        "c_ramadan": np.array([ramadan_share(w) for w in weeks], float),
        "c_eid_uplift": fc.eid_index(weeks, factors),
    }
    if opts["eid_day_counts"]:
        counts = [day_counts(w, eids) for w in weeks]
        for name in DAY_BANDS:
            cols[name] = np.array([c[name] for c in counts], float)
    return cols


def build(ctx: schema.PanelContext) -> pd.DataFrame:
    parts = []
    for origin in sorted(set(ctx.origins)):
        rows = pn.active_rows(ctx, origin)
        cols = {k: np.tile(v, (len(rows), 1)) for k, v in origin_columns(ctx, origin).items()}
        parts.append(pn.frame(ctx, origin, rows, cols, by_h=True))
    return pn.finish(parts, by_h=True)
