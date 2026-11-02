"""Store traffic up to the origin (s_), per (drug, origin).

Traffic is the store's daily sale lines (`history["lines"]`, cut at the
origin). Target-week traffic is never used: it isn't known at the origin.

- `s_lines_{4,13}`: mean weekly sale lines over the last 4 and 13 weeks.
- `s_trend`: `s_lines_4 / s_lines_13 - 1`.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from irpd.ml import panel as pn
from irpd.ml import schema

NEEDS_SQL = False
WINDOWS = (4, 13)


def weekly_lines(lines: pd.Series, origin: date, weeks: int) -> float:
    """Mean weekly sale lines over the `weeks` weeks ending on `origin` (closed days 0)."""
    start = origin - timedelta(weeks=weeks) + timedelta(days=1)
    if not len(lines) or lines.index.min() > start:
        return np.nan
    inside = lines[(lines.index >= start) & (lines.index <= origin)]
    return float(inside.sum()) / weeks


def build(ctx: schema.PanelContext) -> pd.DataFrame:
    parts = []
    for origin in sorted(set(ctx.origins)):
        lines = ctx.history_at(origin)["lines"]
        v = {f"s_lines_{w}": weekly_lines(lines, origin, w) for w in WINDOWS}
        v["s_trend"] = (v["s_lines_4"] / v["s_lines_13"] - 1
                        if v["s_lines_13"] and np.isfinite(v["s_lines_13"]) else np.nan)
        rows = pn.active_rows(ctx, origin)
        parts.append(pn.frame(ctx, origin, rows,
                              {k: np.full(len(rows), x, float) for k, x in v.items()}))
    return pn.finish(parts, by_h=False)
