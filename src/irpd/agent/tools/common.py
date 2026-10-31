"""Helpers the tools share: the connection's day, drug lookup, JSON-safe values."""

import difflib
import math
import re
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

import duckdb
import pandas as pd

RECENT_DAYS = 28  # the "recent sales rate" window, as in the evals' days of cover
MAX_CANDIDATES = 8
MATCH_SCORE = 0.75  # a name match needs at least this score ...
MATCH_GAP = 0.1     # ... and to beat the runner-up by this much


def d(day: date) -> str:
    return f"DATE '{day.isoformat()}'"


def as_of(con: duckdb.DuckDBPyConnection) -> date:
    """The day the connection stands on: the last day `daily_stock` shows.

    `daily_stock` has a row for every drug and day up to the end of the data,
    so through `warehouse.as_of(day)` its last date is `day`.
    """
    return con.execute("SELECT max(date) FROM daily_stock").fetchone()[0]


def clean(v: Any) -> Any:
    """A value as plain JSON: dates as ISO text, NaN as None, floats rounded."""
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [clean(x) for x in v]
    if v is None or v is pd.NaT:
        return None
    if isinstance(v, pd.Timestamp):
        return v.date().isoformat()
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, Decimal):
        v = float(v)
    if hasattr(v, "item"):  # numpy scalars
        v = v.item()
    if isinstance(v, float):
        if not math.isfinite(v):
            return None
        return int(v) if v.is_integer() and abs(v) < 2**53 else round(v, 2)
    return v


def records(df: pd.DataFrame, limit: int) -> dict[str, Any]:
    """The first `limit` rows as dicts, and how many were left out."""
    return {"rows": clean(df.head(limit).to_dict("records")),
            "rows_left_out": max(len(df) - limit, 0)}


def parse_day(text: str | None, default: date) -> date:
    return date.fromisoformat(text) if text else default


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def _score(query: str, name: str) -> float:
    """How well `name` matches `query`, 0-1: the share of query words that start a
    word of the name (so "zetamol 500" fits "Zetamol tb 500mg"), or the plain
    string similarity if that is higher."""
    q, n = _norm(query), _norm(name)
    words = n.split()
    hits = sum(any(w.startswith(t) for w in words) for t in q.split())
    by_words = hits / max(len(q.split()), 1)
    # A full-word match on a longer name ranks below one on a shorter name.
    by_words -= 0.01 * max(len(words) - len(q.split()), 0)
    return max(by_words, difflib.SequenceMatcher(None, q, n).ratio())


def candidates(con: duckdb.DuckDBPyConnection, query: str,
               limit: int = MAX_CANDIDATES) -> pd.DataFrame:
    """Drugs known on the connection's day, best name matches first."""
    df = con.execute("SELECT drug_code, drug_name, stock_unit, last_seen FROM drugs").df()
    df["score"] = [_score(query, n) for n in df.drug_name]
    return df.sort_values(["score", "last_seen", "drug_name"],
                          ascending=[False, False, True]).head(limit).reset_index(drop=True)


def resolve(con: duckdb.DuckDBPyConnection, drug: str) -> tuple[str, str]:
    """(drug_code, drug_name) for a code or a name. Raises LookupError, listing
    candidates, if the name matches nothing or more than one drug."""
    drug = drug.strip()
    exact = con.execute(
        "SELECT drug_code, drug_name FROM drugs WHERE drug_code = $1 OR lower(drug_name) = $2 "
        "ORDER BY drug_code = $1 DESC, last_seen DESC", [drug, drug.lower()]).fetchall()
    if exact:
        return exact[0]
    c = candidates(con, drug)
    clear = len(c) == 1 or (len(c) > 1 and c.score[0] - c.score[1] >= MATCH_GAP)
    if len(c) and c.score[0] >= MATCH_SCORE and clear:
        return c.drug_code[0], c.drug_name[0]
    shown = "; ".join(f"{r.drug_name} ({r.drug_code})" for r in c.itertuples() if r.score > 0.4)
    raise LookupError(f"no single drug matches {drug!r}. Candidates: {shown or 'none'}. "
                      "Call again with a drug_code, or use find_drug.")


def resolve_many(con: duckdb.DuckDBPyConnection, drugs: list[str]) -> dict[str, str]:
    """drug_code -> drug_name for several codes or names, in the order given."""
    return dict(resolve(con, x) for x in drugs)


def unit_of(con: duckdb.DuckDBPyConnection, code: str) -> str | None:
    row = con.execute("SELECT stock_unit FROM drugs WHERE drug_code = $1", [code]).fetchone()
    return row[0] if row else None


def recent(con: duckdb.DuckDBPyConnection, day: date, codes: list[str] | None = None,
           days: int = RECENT_DAYS) -> pd.DataFrame:
    """Per drug on `day`: floor, store and total stock at close, and over the last
    `days` days the units sold, in-stock days (stock > 0 at close, or a sale) and
    the daily rate on in-stock days, for the whole pharmacy and for the floor."""
    where = "" if codes is None else "AND drug_code IN (SELECT unnest($codes))"
    in_stock = "(closing > 0 OR sold > 0)"
    df = con.execute(f"""
        SELECT drug_code,
               max(closing) FILTER (WHERE location = 'floor' AND date = {d(day)}) AS floor,
               max(closing) FILTER (WHERE location = 'store' AND date = {d(day)}) AS store,
               max(closing) FILTER (WHERE location = 'total' AND date = {d(day)}) AS total,
               sum(sold) FILTER (WHERE location = 'total') AS units_sold,
               count(*) FILTER (WHERE location = 'total' AND {in_stock}) AS in_stock_days,
               sum(sold) FILTER (WHERE location = 'floor') AS floor_units_sold,
               count(*) FILTER (WHERE location = 'floor' AND {in_stock}) AS floor_days
        FROM daily_stock
        WHERE date BETWEEN {d(day - timedelta(days=days - 1))} AND {d(day)} {where}
        GROUP BY drug_code""", {"codes": codes} if codes is not None else {}).df()
    df["rate"] = df.units_sold / df.in_stock_days.where(df.in_stock_days > 0)
    df["floor_rate"] = df.floor_units_sold / df.floor_days.where(df.floor_days > 0)
    return df.set_index("drug_code")


LEAD_CLIP = (1, 14)  # as `irpd.replay.default_lead`
LEAD_FALLBACK = 7


def supply(con: duckdb.DuckDBPyConnection, codes: list[str]) -> pd.DataFrame:
    """Per drug: last delivery (day, units, suppliers), main supplier, that supplier's
    delivery cadence and the default lead time.

    The main supplier and cadence follow `pharmacist_baseline` and
    `supplier_cadence`, but are computed from `purchases` up to the day, since
    those tables are hidden before the cutoff. Lead time is the cadence clipped
    to 1-14 days, as `replay.default_lead`, else 7.
    """
    df = con.execute("""
        WITH sup AS (
            SELECT drug_code, supplier, count(DISTINCT date) AS days, max(date) AS last
            FROM purchases GROUP BY ALL
        ), main AS (
            SELECT drug_code, list(supplier ORDER BY days DESC, last DESC, supplier)[1]
                       AS main_supplier
            FROM sup GROUP BY drug_code
        ), gaps AS (
            SELECT supplier, date - lag(date) OVER (PARTITION BY supplier ORDER BY date) AS gap
            FROM (SELECT DISTINCT supplier, date FROM purchases)
        ), cadence AS (
            SELECT supplier, median(gap) AS cadence_days FROM gaps GROUP BY supplier
        ), last AS (
            SELECT drug_code, date AS last_delivery, sum(qty) AS last_delivery_units,
                   string_agg(DISTINCT supplier, ', ') AS last_delivery_suppliers
            FROM purchases GROUP BY drug_code, date
            QUALIFY date = max(date) OVER (PARTITION BY drug_code)
        )
        SELECT c.drug_code, last.* EXCLUDE (drug_code), main.main_supplier,
               cadence.cadence_days AS main_supplier_cadence_days
        FROM (SELECT unnest($codes) AS drug_code) c
        LEFT JOIN last USING (drug_code) LEFT JOIN main USING (drug_code)
        LEFT JOIN cadence ON cadence.supplier = main.main_supplier""", {"codes": codes}).df()
    df["lead_days"] = (df.main_supplier_cadence_days.clip(*LEAD_CLIP).round()
                       .fillna(LEAD_FALLBACK).astype(int))
    return df.set_index("drug_code")
