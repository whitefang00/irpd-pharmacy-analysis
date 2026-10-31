"""`run_sql`: read-only SQL on the agent's view of the warehouse."""

import threading
from typing import Any

import duckdb

from irpd.agent.registry import tool
from irpd.agent.tools.common import clean

MAX_ROWS = 50
TIMEOUT_SECONDS = 20

TABLES = """\
Tables (DuckDB SQL; each shows only what was known on the current day):
- drugs(drug_code, drug_name, category, stock_unit, first_seen, last_seen)
- sales_lines(row_number, date, ts, invoice, cashier, drug_code, drug_name, category,
  is_service, qty, unit, unit_price, discount, revenue, payment_type): one row per receipt
  line; qty in the sales unit (Tab, Strip, Box...); revenue in rupiah.
- daily_demand(date, drug_code, units, lines, revenue): units sold per drug per day in stock
  base units (rows only on days with a sale). weekly_demand(week_start, week_end, drug_code,
  units, lines, revenue, days_open): Monday-start weeks, zeros included.
- daily_stock(date, drug_code, location, closing, sold): end-of-day stock in base units,
  location 'floor' (shop), 'store' (upstairs) or 'total'; every day for every drug.
- stockouts(drug_code, start, stop, days, ended_by): spells of total stock <= 0, from
  2023-10-01. shelfouts(drug_code, start, stop, days): floor <= 0 while store > 0.
- purchases(row_number, date, drug_code, supplier, document, location, qty, stock_before):
  delivery lines. transfers(date, drug_code, from_location, to_location, qty, document,
  paired). stock_takes(date, drug_code, location, adjustment, stock_before, stock_after, kind).
- prices(drug_code, unit, price, valid_from, valid_to, lines, prev_price, likely_unit_change):
  list price in rupiah per sales unit. pack_sizes(drug_code, sales_unit, base_per_unit,
  base_unit, matched_lines).
- demand_tiers(drug_code, tier, sale_days_per_year, adi, cv2, sb_class, revenue_share,
  sale_days, first_sale, last_sale): tier is fast, regular, slow or rare.
- ingredient_map(drug_code, drug_name, ingredients, strength, form, kind, mapped,
  source_model): active ingredients joined by ' + ', '' when unknown.
- stock_flags(drug_code, date, flag, detail): known data problems.
There are no costs, only selling prices."""


def check(query: str) -> None:
    """Raise ValueError unless `query` is one SELECT (or WITH ... SELECT) statement
    that reads only the agent's own views."""
    try:
        statements = duckdb.extract_statements(query)
    except duckdb.Error as e:
        raise ValueError(f"could not parse the query: {e}") from None
    if len(statements) != 1:
        raise ValueError("send exactly one statement")
    if statements[0].type != duckdb.StatementType.SELECT:
        raise ValueError(f"only SELECT queries are allowed, not {statements[0].type.name}")
    # The views read from the attached agent file `wh`, which holds every row up to
    # the cutoff; querying it directly would skip the as-of filter.
    for name in duckdb.get_table_names(query, qualified=True):
        if name.split(".")[0].strip('"').lower() == "wh":
            raise ValueError("query the views by their plain names, not the 'wh' catalog")


@tool("run_sql",
      f"Run one read-only SELECT (or WITH ... SELECT) query and get up to {MAX_ROWS} rows "
      "back. Use it for anything the other tools don't answer. Aggregate in SQL rather "
      "than fetching raw rows.\n" + TABLES,
      {"type": "object", "additionalProperties": False, "required": ["query"],
       "properties": {"query": {"type": "string", "description": "One DuckDB SELECT query."}}})
def run_sql(con: duckdb.DuckDBPyConnection, query: str) -> dict[str, Any]:
    check(query)
    timer = threading.Timer(TIMEOUT_SECONDS, con.interrupt)
    timer.start()
    try:
        cur = con.execute(query)
        columns = [c[0] for c in cur.description]
        rows = cur.fetchmany(MAX_ROWS)
        # Count the rest in chunks: `SELECT * FROM daily_stock` as Python tuples
        # would take over a gigabyte.
        left_out = 0
        while chunk := cur.fetchmany(10_000):
            left_out += len(chunk)
    except duckdb.InterruptException:
        raise TimeoutError(f"query ran over {TIMEOUT_SECONDS}s; simplify it") from None
    finally:
        timer.cancel()
    return {"columns": columns, "rows": clean(rows), "rows_left_out": left_out}
