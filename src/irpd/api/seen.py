"""What the agent saw: queries over `holdout.agent_view()`, data up to the cutoff.

Every function here takes the agent's connection and nothing else. This
module must never import or open the sealed holdout; `irpd.api.sealed` is the
only place in the app that does (`tests/test_api.py` checks).
"""

from datetime import date
from decimal import Decimal

import duckdb

from irpd import config


def _rows(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> list[dict]:
    cur = con.execute(sql, params or [])
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, (_json(v) for v in row), strict=True)) for row in cur.fetchall()]


def _json(v):
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    return v


def search(con: duckdb.DuckDBPyConnection, q: str = "", limit: int = 20) -> list[dict]:
    """Drugs whose name or code contains `q`, by revenue up to the cutoff."""
    return _drug_rows(con, "d.drug_name ILIKE '%' || ? || '%' OR d.drug_code ILIKE ? || '%'",
                      [q, q, limit])


def search_one(con: duckdb.DuckDBPyConnection, code: str) -> dict | None:
    rows = _drug_rows(con, "d.drug_code = ?", [code, 1])
    return rows[0] if rows else None


def _drug_rows(con: duckdb.DuckDBPyConnection, where: str, params: list) -> list[dict]:
    return _rows(con, f"""
        WITH rev AS (SELECT drug_code, sum(revenue) AS revenue FROM daily_demand GROUP BY 1)
        SELECT d.drug_code AS code, d.drug_name AS name, d.category,
               coalesce(t.tier, 'none') AS tier, coalesce(r.revenue, 0)::BIGINT AS revenue,
               nullif(m.ingredients, '') AS ingredients, nullif(m.strength, '') AS strength,
               nullif(m.form, '') AS form, nullif(m.kind, '') AS kind,
               coalesce(m.mapped AND m.ingredients <> '', false) AS mapped
        FROM drugs d
        LEFT JOIN rev r USING (drug_code)
        LEFT JOIN demand_tiers t USING (drug_code)
        LEFT JOIN ingredient_map m USING (drug_code)
        WHERE {where}
        ORDER BY revenue DESC, name
        LIMIT ?""", params)


def drug(con: duckdb.DuckDBPyConnection, code: str) -> dict | None:
    """Everything the drug page shows, as the agent saw it at the cutoff."""
    found = search_one(con, code)
    if found is None:
        return None
    p = [code]
    return {
        "drug": found,
        "as_of": config.CUTOFF.isoformat(),
        "stock_start": config.STOCK_START.isoformat(),
        "stock": _rows(con, """
            SELECT date,
                   max(closing) FILTER (WHERE location = 'floor') AS floor,
                   max(closing) FILTER (WHERE location = 'store') AS store,
                   max(closing) FILTER (WHERE location = 'total') AS total
            FROM daily_stock WHERE drug_code = ? GROUP BY date ORDER BY date""", p),
        "sales": _rows(con, """
            SELECT date, units, revenue::BIGINT AS revenue FROM daily_demand
            WHERE drug_code = ? ORDER BY date""", p),
        "stockouts": _rows(con, """
            SELECT start, stop, days, ended_by FROM stockouts
            WHERE drug_code = ? ORDER BY start""", p),
        "shelfouts": _rows(con, """
            SELECT start, stop, days FROM shelfouts WHERE drug_code = ? ORDER BY start""", p),
        "purchases": _rows(con, """
            SELECT date, qty, supplier, location FROM purchases
            WHERE drug_code = ? ORDER BY date, row_number""", p),
        "prices": _rows(con, """
            SELECT unit, price, valid_from, valid_to FROM prices
            WHERE drug_code = ? ORDER BY unit, valid_from""", p),
        "alternatives": alternatives(con, code),
    }


def alternatives(con: duckdb.DuckDBPyConnection, code: str) -> list[dict]:
    """Other drugs with the same ingredient set, same strength and form first."""
    return _rows(con, """
        WITH me AS (SELECT * FROM ingredient_map WHERE drug_code = ? AND ingredients <> ''),
             rev AS (SELECT drug_code, sum(revenue) AS revenue FROM daily_demand GROUP BY 1)
        SELECT m.drug_code AS code, d.drug_name AS name, coalesce(t.tier, 'none') AS tier,
               coalesce(r.revenue, 0)::BIGINT AS revenue, m.strength, m.form,
               replace(lower(m.strength), ' ', '') = replace(lower(me.strength), ' ', '')
                   AND (m.form = me.form OR '' IN (m.form, me.form)) AS same_strength_form,
               s.closing AS stock_at_cutoff
        FROM ingredient_map m
        JOIN me ON m.ingredients = me.ingredients AND m.drug_code <> me.drug_code
        JOIN drugs d ON d.drug_code = m.drug_code
        LEFT JOIN rev r ON r.drug_code = m.drug_code
        LEFT JOIN demand_tiers t ON t.drug_code = m.drug_code
        LEFT JOIN daily_stock s ON s.drug_code = m.drug_code AND s.location = 'total'
             AND s.date = (SELECT max(date) FROM daily_stock)
        WHERE coalesce(r.revenue, 0) > 0 OR s.closing > 0  -- skip ones never sold nor held
        ORDER BY same_strength_form DESC, revenue DESC
        LIMIT 20""", [code])


def overview(con: duckdb.DuckDBPyConnection) -> dict:
    """Headline numbers up to the cutoff."""
    rev = _rows(con, """
        SELECT year(date) AS year, sum(revenue)::BIGINT AS revenue,
               count(DISTINCT drug_code) AS drugs_sold
        FROM daily_demand GROUP BY 1 ORDER BY 1""")
    # Days out inside each drug's selling window, from STOCK_START, by tier.
    tiers = _rows(con, f"""
        WITH w AS (
            SELECT drug_code, tier, greatest(first_sale, DATE '{config.STOCK_START}') AS a,
                   last_sale AS b FROM demand_tiers),
        days AS (
            SELECT w.tier, s.drug_code, s.date,
                   max(closing) FILTER (WHERE location = 'floor') AS floor,
                   max(closing) FILTER (WHERE location = 'store') AS store,
                   max(closing) FILTER (WHERE location = 'total') AS total
            FROM daily_stock s JOIN w ON s.drug_code = w.drug_code AND s.date BETWEEN w.a AND w.b
            GROUP BY ALL)
        SELECT tier, count(DISTINCT drug_code) AS drugs, count(*) AS drug_days,
               count(*) FILTER (WHERE total <= 0) AS stockout_days,
               count(*) FILTER (WHERE total > 0 AND floor <= 0 AND store > 0) AS shelfout_days
        FROM days GROUP BY tier
        ORDER BY list_position(['fast', 'regular', 'slow', 'rare'], tier)""")
    for t in tiers:
        t["stockout_share"] = t["stockout_days"] / t["drug_days"] if t["drug_days"] else 0
        t["shelfout_share"] = t["shelfout_days"] / t["drug_days"] if t["drug_days"] else 0
    mapped = con.execute("""
        SELECT sum(d.revenue) FILTER (WHERE m.mapped AND m.ingredients <> '') / sum(d.revenue)
        FROM daily_demand d LEFT JOIN ingredient_map m USING (drug_code)""").fetchone()[0]
    counts = con.execute("""
        SELECT (SELECT count(*) FROM drugs),
               (SELECT count(DISTINCT drug_code) FROM daily_demand),
               (SELECT count(*) FROM demand_tiers WHERE tier IN ('fast', 'regular'))""").fetchone()
    first, last = con.execute("SELECT min(date), max(date) FROM daily_demand").fetchone()
    return {
        "as_of": config.CUTOFF.isoformat(),
        "period": {"first": _json(first), "last": _json(last)},
        "revenue_total": sum(r["revenue"] for r in rev),
        "revenue_by_year": rev,
        "drugs": {"listed": counts[0], "sold": counts[1], "fast_or_regular": counts[2]},
        "tiers": tiers,
        "revenue_share_mapped": float(mapped or 0),
    }
