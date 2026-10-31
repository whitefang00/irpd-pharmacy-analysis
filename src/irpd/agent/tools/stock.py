"""Stock now, stock-out history with lost demand, and what to move to the shop floor."""

import math
from datetime import date, timedelta
from typing import Any

import duckdb
import pandas as pd

from irpd import config, forecast, lost
from irpd.agent.registry import tool
from irpd.agent.tools.common import (
    RECENT_DAYS,
    as_of,
    clean,
    parse_day,
    recent,
    records,
    resolve,
    resolve_many,
    supply,
    unit_of,
)
from irpd.agent.tools.drugs import DRUG_ARG, one_drug

TIERS = ["fast", "regular", "slow", "rare"]
MAX_SPELLS = 15
MAX_DRUGS = 10
MAX_TRANSFERS = 30
FLOOR_LOW_DAYS = 2      # the floor is low below this many days of floor sales
FLOOR_TARGET_DAYS = 7   # a transfer fills the floor to this many days of sales
FORECAST_WEEKS = 4      # forecast window behind "cover at the forecast rate"


def _cover(stock: float, rate: float) -> float | None:
    return stock / rate if rate and rate > 0 else None


@tool("stock_position",
      "Stock of one drug at close on the current day: shop floor, upstairs store and "
      "total, in the drug's base unit; whether it is out of stock (total at 0 or below) or "
      "shelf-out (floor at 0 or below, store has stock) now; last delivery and main "
      "supplier. cover_days = total stock / average daily units sold over the last 28 days "
      "(the standard definition). Two variants: cover_days_at_in_stock_rate divides by the "
      "rate on days it was in stock only, cover_days_at_forecast by the next 4 weeks' "
      "forecast daily mean.",
      one_drug("Drug code or name."))
def stock_position(con: duckdb.DuckDBPyConnection, drug: str) -> dict[str, Any]:
    code, name = resolve(con, drug)
    day = as_of(con)
    r = recent(con, day, [code]).iloc[0]
    sup = supply(con, [code]).iloc[0]
    f = forecast.forecast(con, [code], day, FORECAST_WEEKS)
    f_rate = f["mean"].sum() / (7 * FORECAST_WEEKS)
    running = {t: con.execute(f"SELECT max(start) FROM {t} WHERE drug_code = $1 AND stop = $2",
                              [code, day]).fetchone()[0] for t in ("stockouts", "shelfouts")}
    return clean({
        "drug_code": code, "drug_name": name, "as_of": day, "unit": unit_of(con, code),
        "stock_floor": r.floor, "stock_store": r.store, "stock_total": r.total,
        "out_of_stock": bool(r.total <= 0), "out_of_stock_since": running["stockouts"],
        "shelf_out": bool(r.floor <= 0 < r.store), "shelf_out_since": running["shelfouts"],
        "units_sold_last_28d": r.units_sold,
        "avg_daily_units_last_28d": r.units_sold / RECENT_DAYS,
        "cover_days": _cover(r.total, r.units_sold / RECENT_DAYS),
        "in_stock_days_last_28d": r.in_stock_days,
        "cover_days_at_in_stock_rate": _cover(r.total, r.rate),
        "forecast_avg_daily_units_next_4w": f_rate,
        "cover_days_at_forecast": _cover(r.total, f_rate),
        "forecast_method": f["method"].iloc[0],
        "last_delivery": sup.last_delivery, "last_delivery_units": sup.last_delivery_units,
        "last_delivery_suppliers": sup.last_delivery_suppliers,
        "days_since_last_delivery": (day - sup.last_delivery.date()).days
        if pd.notna(sup.last_delivery) else None,
        "main_supplier": sup.main_supplier,
        "main_supplier_cadence_days": sup.main_supplier_cadence_days,
    })


LOST_NOTE = (
    "A stock-out day is a day the total stock (floor + store) closed at 0 or below; a "
    "shelf-out day, the floor at 0 or below while the store had stock; both counted from "
    "2023-10-01. days_in_period counts only the period's days. Lost demand is an estimate "
    "for each whole spell (even the part outside the period), over its first 28 days: "
    "expected sales (the drug's normal share of shop traffic) minus recorded sales. On "
    "shelf-outs it can be negative when sales held up. 'switched' is the part estimated to "
    "have gone to a same-ingredient alternative; 'truly_lost' is the rest, an upper bound "
    "since switching to other ingredients isn't seen. Units are the drug's base units; "
    "money in rupiah at selling price.")


def _totals(sp: pd.DataFrame, with_units: bool) -> dict[str, Any]:
    out = {}
    for kind in ("stockout", "shelfout"):
        k = sp[sp.kind == kind]
        t = {"spells": len(k), "days_in_period": int(k.days_in_period.sum()),
             "lost_revenue_rp": k.lost_revenue.sum(),
             "switched_revenue_rp": k.switched_revenue.sum(),
             "truly_lost_revenue_rp": k.truly_lost_revenue.sum()}
        if with_units:
            t |= {"lost_units": k.lost_units.sum(), "switched_units": k.switched_units.sum()}
        out[kind] = t
    return out


@tool("stockout_history",
      "Stock-out spells (total stock closed at 0 or below) and shelf-out spells (floor at 0 "
      "or below while the store had stock) that overlap a period, with days out in the "
      "period and estimated lost demand, switched to alternatives vs truly lost. For one "
      "drug, lists its spells (newest first) with how each ended; for a tier or the whole "
      "shop (drug and tier both null), totals plus the drugs with the most stock-out days "
      "and the most lost revenue. Counts from 2023-10-01.",
      {"type": "object", "additionalProperties": False,
       "required": ["drug", "tier", "start", "end"],
       "properties": {
           "drug": {**DRUG_ARG, "type": ["string", "null"],
                    "description": "Drug code or name, or null for a tier or the whole shop."},
           "tier": {"type": ["string", "null"], "enum": [*TIERS, None],
                    "description": "Demand tier, or null."},
           "start": {"type": ["string", "null"],
                     "description": "First day, YYYY-MM-DD. Null: 365 days before today."},
           "end": {"type": ["string", "null"],
                   "description": "Last day, YYYY-MM-DD. Null: today."}}})
def stockout_history(con: duckdb.DuckDBPyConnection, drug: str | None, tier: str | None,
                     start: str | None, end: str | None) -> dict[str, Any]:
    day = as_of(con)
    p0 = max(parse_day(start, day - timedelta(days=364)), config.STOCK_START)
    p1 = min(parse_day(end, day), day)
    sp = lost.spells(con, day)
    sp = sp[(sp.stop >= p0) & (sp.start <= p1)]
    sp = sp.assign(days_in_period=[(min(b, p1) - max(a, p0)).days + 1
                                   for a, b in zip(sp.start, sp.stop, strict=True)])
    head = {"as_of": day.isoformat(), "period_start": p0.isoformat(),
            "period_end": p1.isoformat()}
    if drug:
        code, name = resolve(con, drug)
        sp = sp[sp.drug_code == code].sort_values("start", ascending=False)
        ended = con.execute("SELECT start, ended_by FROM stockouts WHERE drug_code = $1",
                            [code]).df()
        ended["start"] = pd.to_datetime(ended.start).dt.date
        sp = sp.merge(ended.assign(kind="stockout"), on=["kind", "start"], how="left")
        rows = sp.assign(ongoing=sp.stop == day)[
            ["kind", "start", "stop", "days", "days_in_period", "ongoing", "ended_by",
             "lost_units", "lost_lo", "lost_hi", "lost_revenue", "switched_units",
             "truly_lost_revenue"]].rename(columns={
                 "lost_lo": "lost_units_low_80", "lost_hi": "lost_units_high_80",
                 "lost_revenue": "lost_revenue_rp", "truly_lost_revenue": "truly_lost_revenue_rp"})
        r = records(rows, MAX_SPELLS)
        return {**head, "drug_code": code, "drug_name": name, "unit": unit_of(con, code),
                **clean(_totals(sp, with_units=True)),
                "spells": r["rows"], "spells_left_out": r["rows_left_out"], "note": LOST_NOTE}
    if tier:
        sp = sp[sp.tier == tier]
    by = (sp.assign(stockout_days=sp.days_in_period.where(sp.kind == "stockout", 0),
                    shelfout_days=sp.days_in_period.where(sp.kind == "shelfout", 0),
                    lost_revenue_rp=sp.lost_revenue,
                    truly_lost_revenue_rp=sp.truly_lost_revenue)
            .groupby("drug_code")[["stockout_days", "shelfout_days", "lost_revenue_rp",
                                   "truly_lost_revenue_rp"]].sum())
    names = con.execute("""
        SELECT drug_code, drug_name,
               (SELECT max(date) FROM daily_demand x WHERE x.drug_code = d.drug_code) AS last_sale
        FROM drugs d""").df().set_index("drug_code")
    by = names.join(by, how="inner").reset_index()

    def top(col: str) -> list[dict[str, Any]]:
        return records(by.sort_values([col, "drug_name"], ascending=[False, True]),
                       MAX_DRUGS)["rows"]

    return {**head, "tier": tier or "all", "drugs_with_spells": len(by),
            **clean(_totals(sp, with_units=False)),
            "top_by_stockout_days": top("stockout_days"),
            "top_by_lost_revenue": top("lost_revenue_rp"), "note": LOST_NOTE}


def transfer_rows(con: duckdb.DuckDBPyConnection, day: date,
                  codes: list[str] | None = None) -> pd.DataFrame:
    """Drugs whose floor holds under FLOOR_LOW_DAYS of floor sales while the store has
    stock, and how much to bring down to reach FLOOR_TARGET_DAYS (capped by the store)."""
    r = recent(con, day, codes)
    r = r[(r.store > 0) & (r.floor_rate > 0)]
    r = r[r.floor < FLOOR_LOW_DAYS * r.floor_rate].copy()
    target = (FLOOR_TARGET_DAYS * r.floor_rate).apply(math.ceil)
    r["move_units"] = (target - r.floor.clip(lower=0)).clip(upper=r.store)
    r["floor_cover_days"] = r.floor.clip(lower=0) / r.floor_rate
    r = r[r.move_units > 0].sort_values(["floor_cover_days", "floor_rate"],
                                        ascending=[True, False])
    names = con.execute("SELECT drug_code, drug_name, stock_unit AS unit FROM drugs").df()
    return (r.reset_index().merge(names, on="drug_code", how="left")
             [["drug_code", "drug_name", "unit", "floor", "store", "floor_rate",
               "floor_cover_days", "move_units"]]
             .rename(columns={"floor": "stock_floor", "store": "stock_store",
                              "floor_rate": "floor_daily_units"}))


@tool("transfer_plan",
      "Drugs whose shop floor is empty or low (under 2 days of floor sales) while the "
      "upstairs store has stock, with how many base units to bring down to fill the floor "
      "to 7 days of sales (never more than the store holds). Floor sales rate is units "
      "sold from the floor per day the floor had stock, over the last 28 days. Emptiest "
      "floors first.",
      {"type": "object", "additionalProperties": False, "required": ["drugs"],
       "properties": {"drugs": {"type": ["array", "null"], "items": DRUG_ARG,
                                "description": "Codes or names, or null for every drug."}}})
def transfer_plan(con: duckdb.DuckDBPyConnection, drugs: list[str] | None) -> dict[str, Any]:
    day = as_of(con)
    names = resolve_many(con, drugs) if drugs else None
    rows = transfer_rows(con, day, list(names) if names else None)
    r = records(rows, MAX_TRANSFERS)
    shelf_out = recent(con, day, list(names) if names else None).query("floor <= 0 < store")
    return {"as_of": day.isoformat(), "drugs_to_move": len(rows),
            "shelf_out_now": len(shelf_out),
            "transfers": r["rows"], "transfers_left_out": r["rows_left_out"],
            "note": "shelf_out_now counts every drug whose floor closed at 0 or below "
                    "while the store had stock, including ones with no floor sales in the "
                    f"last {RECENT_DAYS} days, which get no transfer. Units are each "
                    "drug's base unit."}
