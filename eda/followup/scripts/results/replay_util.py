"""Replay helpers for E5/E6 that the harness doesn't expose (harness code unchanged).

`make_plan_hook` is `irpd.evalx.rolling.make_plan` with a hook that edits each
re-plan day's policy (E5's sporadic min-stock rule and fading stop flag).
"""
from __future__ import annotations

from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd

import common as C
from irpd import config, contest, replay
from irpd.evalx import rolling as R


def make_plan_hook(source, w, cadence=14, hook=None):
    pols = []
    for t in R.replan_days(w, cadence):
        day = w.h.days[t] - timedelta(days=1)
        pol = contest.to_policy({}, source.basis(day))
        if hook is not None:
            pol = hook(pol, day)
        pols.append((t, pol))
    return replay.schedule(pols, w.h)


def max_single_sale(day: date) -> pd.Series:
    """Largest single sales line (qty) per drug in the 52 weeks up to `day`."""
    def make():
        with R.view(day) as con:
            return con.execute(f"""
                SELECT drug_code, max(qty) AS mx FROM sales_lines
                WHERE NOT is_service AND date > DATE '{day - timedelta(days=364)}'
                  AND date <= DATE '{day}' GROUP BY 1""").df()
    return R._cached(f"maxsale_{day}", make).set_index("drug_code")["mx"]


def labels_at(day: date) -> pd.DataFrame:
    """Labels at the latest grid origin on or before `day`."""
    lb = C.labels()
    o = max(x for x in lb["origin"].unique() if x <= day)
    return lb[lb["origin"] == o].set_index("drug_code")


def per_drug(out: R.Outcome) -> pd.DataFrame:
    return out.periods.groupby("drug_code")[["stock_value", "lost_net", "lost_gross", "net_out",
                                             "stockout_days"]].sum()
