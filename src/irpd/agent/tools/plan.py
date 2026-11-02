"""Demand forecasts and reorder plans built on them.

`reorder_rows` gives the full plan as a DataFrame and `to_policy` turns it into
the policy format `irpd.replay.simulate` scores, so the agent's plan can be
replayed against the pharmacist's.
"""

import math
from datetime import date, timedelta
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from irpd import forecast as fc
from irpd.agent.registry import tool
from irpd.agent.tools.common import as_of, clean, recent, records, resolve_many, supply
from irpd.agent.tools.drugs import DRUG_ARG
from irpd.agent.tools.stock import TIERS

MAX_FORECAST_DRUGS = 10
MAX_PLAN_ROWS = 25
CYCLE_DAYS = 14      # days between orders that S covers beyond the lead time
FLOOR_FILL_DAYS = 7  # floor_max = floor_min + this many days of mean demand
STALE_DAYS = 90      # no sale for this long: likely delisted, so nothing is ordered now
POLICY_COLUMNS = ["drug_code", "s", "S", "lead_days", "floor_min", "floor_max"]

DRUG_LIST = {"type": "array", "items": DRUG_ARG, "description": "Drug codes or names."}


FORECASTER_ARG = {"type": "string", "enum": list(fc.FORECASTERS),
                  "description": "Which forecast to plan from: 'current' (the standard "
                                 "method, the default) or 'ml' (the ML model)."}


def _ml_or_none(con: duckdb.DuckDBPyConnection, codes: list[str], day: date,
                horizon: int) -> pd.DataFrame | None:
    try:
        return fc.forecast(con, codes, day, horizon, forecaster="ml")
    except FileNotFoundError:
        return None


@tool("forecast",
      "Weekly demand forecast per drug in its base unit, for Monday-start weeks after the "
      "current day, from two forecasters side by side. 'current' (the standard method): "
      "smoothed level with an Eid uplift for drugs that sell most weeks, same week last "
      "year for rare ones; stock-out weeks are left out of the fit. 'ml': a LightGBM model "
      "trained on every drug's sales history, stock-outs, price, store traffic and the "
      "Eid calendar (method 'ml'; drugs it can't forecast show the current method). Each "
      "gives mean, median (p50) and a high case (p90, 9 weeks in 10 sell no more), plus "
      f"the horizon total. Up to {MAX_FORECAST_DRUGS} drugs.",
      {"type": "object", "additionalProperties": False, "required": ["drugs", "horizon_weeks"],
       "properties": {"drugs": DRUG_LIST,
                      "horizon_weeks": {"type": "integer", "minimum": 1, "maximum": 26,
                                        "description": "Weeks ahead, 1-26."}}})
def forecast(con: duckdb.DuckDBPyConnection, drugs: list[str],
             horizon_weeks: int) -> dict[str, Any]:
    if len(drugs) > MAX_FORECAST_DRUGS:
        raise ValueError(f"at most {MAX_FORECAST_DRUGS} drugs per call")
    horizon_weeks = min(max(int(horizon_weeks), 1), 26)
    day = as_of(con)
    names = resolve_many(con, drugs)
    units = dict(con.execute("SELECT drug_code, stock_unit FROM drugs").fetchall())
    runs = {"current": fc.forecast(con, list(names), day, horizon_weeks),
            "ml": _ml_or_none(con, list(names), day, horizon_weeks)}
    out = []
    for code in names:
        row = {"drug_code": code, "drug_name": names[code], "unit": units.get(code)}
        for name, f in runs.items():
            if f is None:
                row[name] = None
                continue
            g = f[f.drug_code == code]
            row["tier"] = g.tier.iloc[0]
            total_p90 = fc.demand_quantile(np.array([g["mean"].sum()]),
                                           np.array([g["var"].sum()]), 0.9)[0]
            row[name] = {"method": g.method.iloc[0], "total_mean": round(g["mean"].sum(), 1),
                         "total_p90": total_p90,
                         "weeks": clean(g[["week_start", "mean", "p50", "p90"]]
                                        .assign(mean=g["mean"].round(1)).to_dict("records"))}
        out.append(row)
    res = {"as_of": day.isoformat(), "forecasts": clean(out)}
    if runs["ml"] is None:
        res["note"] = ("No ML forecasts are saved for today, so only the current method "
                       "is shown.")
    return res


def _daily(f: pd.DataFrame, codes: list[str], day: date, n_days: int) -> tuple[np.ndarray, ...]:
    """Daily mean and variance for days day+1 .. day+n_days, each week's spread evenly."""
    weeks = sorted(f.week_start.unique())
    mean = f.pivot(index="drug_code", columns="week_start", values="mean").reindex(codes)
    var = f.pivot(index="drug_code", columns="week_start", values="var").reindex(codes)
    # The first forecast week can start after tomorrow if the last week of data was
    # cut short; those days get the first week's rate.
    skip = (day + timedelta(days=1) - weeks[0]).days

    def spread(x: pd.DataFrame) -> np.ndarray:
        x = np.repeat(x[weeks].fillna(0).to_numpy() / 7, 7, axis=1)
        x = np.concatenate([np.repeat(x[:, :1], max(-skip, 0), axis=1), x], axis=1)
        return x[:, max(skip, 0):max(skip, 0) + n_days]

    return spread(mean), spread(var)


def _window_sum(x: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Per row, the sum of its first `days[i]` columns."""
    c = np.concatenate([np.zeros((len(x), 1)), np.cumsum(x, axis=1)], axis=1)
    return c[np.arange(len(x)), days]


def reorder_rows(con: duckdb.DuckDBPyConnection, codes: list[str], day: date,
                 service_level: float = 0.95, lead_days: int | None = None,
                 forecaster: str = "current") -> pd.DataFrame:
    """An (s, S) plan per drug from the forecast distribution.

    s = the service-level quantile of demand over the lead time; S = the same
    quantile over the lead time plus CYCLE_DAYS (at least s + 1 when demand is
    expected). Order now = S - total stock if total stock <= s. Nothing is known
    to be on order (the data has no purchase orders), so stock is the position.
    A drug with no sale in STALE_DAYS keeps its s and S but orders nothing now:
    the forecaster skips stock-out weeks, so a drug left unstocked for months
    still carries its old level.
    floor_min = the quantile of one day's demand, floor_max = floor_min plus
    FLOOR_FILL_DAYS of mean demand. `forecaster` picks the forecast
    (`forecast.FORECASTERS`).
    """
    sup = supply(con, codes)
    lead = (pd.Series(int(lead_days), index=codes) if lead_days
            else sup.lead_days.reindex(codes)).to_numpy(int)
    horizon = math.ceil((lead.max() + CYCLE_DAYS + 7) / 7)
    f = fc.forecast(con, codes, day, horizon, forecaster=forecaster)
    mean, var = _daily(f, codes, day, lead.max() + CYCLE_DAYS)
    q = lambda m, v: fc.demand_quantile(m, np.maximum(v, m), service_level)  # noqa: E731
    m_l, v_l = _window_sum(mean, lead), _window_sum(var, lead)
    m_s, v_s = _window_sum(mean, lead + CYCLE_DAYS), _window_sum(var, lead + CYCLE_DAYS)
    s = q(m_l, v_l)
    S = np.maximum(q(m_s, v_s), np.where(m_s > 0, s + 1, s))
    floor_min = q(mean[:, 0], var[:, 0])
    stock = recent(con, day, codes).reindex(codes)
    total = stock.total.to_numpy(float)  # NaN for the few drugs sold with no stock ledger
    # Those few are named from their sale lines.
    names = dict(con.execute("""
        SELECT drug_code, any_value(drug_name) AS drug_name FROM sales_lines GROUP BY 1
        UNION ALL BY NAME SELECT drug_code, drug_name FROM drugs""").fetchall())
    last_sale = dict(con.execute("SELECT drug_code, max(date) FROM daily_demand "
                                 "WHERE units > 0 GROUP BY 1").fetchall())
    idle = np.array([(day - last_sale[c]).days if c in last_sale else np.nan for c in codes])
    stale = ~(idle < STALE_DAYS)
    tier = f.groupby("drug_code").tier.first().reindex(codes)
    return pd.DataFrame({
        "drug_code": codes, "drug_name": [names.get(c) for c in codes], "tier": tier.to_numpy(),
        "s": s, "S": S, "lead_days": lead,
        "floor_min": floor_min,
        "floor_max": floor_min + np.ceil(mean[:, :FLOOR_FILL_DAYS].sum(axis=1)),
        "stock_total": total,
        "mean_demand_lead": m_l.round(1),
        "days_since_last_sale": idle,
        "order_now": np.where((total <= s) & ~stale, S - total, 0),
        "main_supplier": sup.main_supplier.reindex(codes).to_numpy(),
    })


def to_policy(plan_rows: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
    """A reorder plan (from `reorder_rows` or the tool's rows) as a replay policy."""
    return pd.DataFrame(plan_rows)[POLICY_COLUMNS].reset_index(drop=True)


@tool("reorder_plan",
      "Reorder plan per drug from the demand forecast: reorder point s (order when total "
      "stock falls to s or below), order-up-to level S, and how much to order now. s covers "
      "lead-time demand at the service level; S covers lead time plus 14 days. Lead time "
      "defaults to the main supplier's delivery cadence (1-14 days). Also floor_min / "
      "floor_max: refill the shop floor up to floor_max when it holds floor_min or less. "
      "All in each drug's base unit. Give drugs, or a tier for all its drugs; drugs to "
      f"order now come first, up to {MAX_PLAN_ROWS} rows. forecaster picks the demand "
      "forecast: 'current' (standard method) unless the owner asks for the ML model.",
      {"type": "object", "additionalProperties": False,
       "required": ["drugs", "tier", "service_level", "lead_days", "forecaster"],
       "properties": {
           "drugs": {**DRUG_LIST, "type": ["array", "null"],
                     "description": "Drug codes or names, or null to use tier."},
           "tier": {"type": ["string", "null"], "enum": [*TIERS, None],
                    "description": "Plan every drug in this demand tier (when drugs is null)."},
           "service_level": {"type": "number", "minimum": 0.5, "maximum": 0.999,
                             "description": "Chance a lead time's demand is covered, e.g. 0.95."},
           "lead_days": {"type": ["integer", "null"], "minimum": 1, "maximum": 60,
                         "description": "Days from order to delivery; null for the default."},
           "forecaster": FORECASTER_ARG}})
def reorder_plan(con: duckdb.DuckDBPyConnection, drugs: list[str] | None, tier: str | None,
                 service_level: float, lead_days: int | None,
                 forecaster: str = "current") -> dict[str, Any]:
    day = as_of(con)
    if drugs:
        codes = list(resolve_many(con, drugs))
    elif tier:
        codes = [r[0] for r in con.execute(
            "SELECT drug_code FROM demand_tiers WHERE tier = $1 ORDER BY 1", [tier]).fetchall()]
    else:
        raise ValueError("give drugs or a tier")
    if not 0.5 <= service_level < 1:
        raise ValueError("service_level must be between 0.5 and 0.999")
    # The schema's bounds aren't enforced by the API, and a negative lead time
    # would index the forecast from the wrong end.
    if lead_days is not None and not 1 <= lead_days <= 60:
        raise ValueError("lead_days must be between 1 and 60, or null")
    if forecaster not in fc.FORECASTERS:
        raise ValueError(f"forecaster must be one of {fc.FORECASTERS}")
    plan = reorder_rows(con, codes, day, service_level, lead_days, forecaster)
    plan = plan.assign(_order=plan.order_now > 0,
                       _cover=plan.stock_total / (plan.mean_demand_lead / plan.lead_days))
    plan = plan.sort_values(["_order", "_cover"], ascending=[False, True])
    r = records(plan.drop(columns=["_order", "_cover"]), MAX_PLAN_ROWS)
    return {"as_of": day.isoformat(), "service_level": service_level, "forecaster": forecaster,
            "drugs_planned": len(plan), "drugs_to_order_now": int((plan.order_now > 0).sum()),
            "plan": r["rows"], "plan_left_out": r["rows_left_out"],
            "note": "Units are each drug's base unit. Stock on order is unknown, so the "
                    "current total stock is taken as the stock position. Drugs with no "
                    f"sale in {STALE_DAYS}+ days order nothing now (likely delisted)."}
