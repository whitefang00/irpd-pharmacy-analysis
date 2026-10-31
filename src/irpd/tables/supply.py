"""Supply side: deliveries, transfers, stock-takes, and the pharmacist's policy.

Built straight from `stock_moves`:

- `purchases`: one row per delivery line, with the total stock (floor + store)
  on hand just before the delivery arrived. Purchase invoices that were later
  deleted in the POS ("telah di hapus", booked as a purchase with stock going
  out) are cancelled against the line they reversed and left out.
- `transfers`: one row per move between the shop floor and the upstairs
  store, pairing the out row with the in row. The 'test' rows of July 2023
  are real transfers and are included.
- `stock_takes`: count corrections per drug, location and day. `kind` marks
  the two mass events apart from routine counts.
- `pharmacist_baseline` and `supplier_cadence`: the pharmacist's observed
  reorder policy, computed only from moves up to `config.CUTOFF`. They are
  what the agent is scored against on the holdout, so they must not see it.
"""

from datetime import date

import duckdb

from irpd import config

TABLES = {
    "purchases": "date",
    "transfers": "date",
    "stock_takes": "date",
    "pharmacist_baseline": None,  # computed through config.CUTOFF
    "supplier_cadence": None,  # computed through config.CUTOFF
}
DEPENDS: list[str] = []

# The two mass stock-take events (see FINDINGS.md). Everything else is routine.
STORE_SETUP = (date(2023, 7, 1), date(2023, 9, 30))
FULL_COUNT = (date(2025, 8, 1), date(2025, 8, 31))

# Days of sales history used for the sales rate behind "days of cover".
RATE_DAYS = 28


def _sql_date(d: date) -> str:
    return f"DATE '{d.isoformat()}'"


def _purchases_sql(moves: str) -> str:
    """SELECT for delivery lines with stock_before, reading only from `moves`."""
    return f"""
    WITH bal AS (
        -- Each location's balance just before every row: the closing of its
        -- latest earlier row, or the opening it started the export with.
        SELECT m.*,
               coalesce(last_value(CASE WHEN location = 'floor' THEN closing END IGNORE NULLS)
                            OVER before_row,
                        first_value(CASE WHEN location = 'floor' THEN opening END IGNORE NULLS)
                            OVER whole_drug, 0)
             + coalesce(last_value(CASE WHEN location = 'store' THEN closing END IGNORE NULLS)
                            OVER before_row,
                        first_value(CASE WHEN location = 'store' THEN opening END IGNORE NULLS)
                            OVER whole_drug, 0) AS total_before
        FROM {moves} m
        WINDOW before_row AS (PARTITION BY drug_code ORDER BY row_number
                              ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
               whole_drug AS (PARTITION BY drug_code ORDER BY row_number
                              ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
    ),
    lines AS (
        SELECT *,
               -- Lines of one invoice arrive together: all get the stock
               -- before its first line.
               first_value(total_before) OVER (PARTITION BY drug_code, document
                                               ORDER BY row_number) AS stock_before,
               row_number() OVER (PARTITION BY drug_code, document, location, qty_in
                                  ORDER BY row_number DESC) AS latest
        FROM bal WHERE move_type = 'purchase' AND qty_in > 0
    ),
    deleted AS (
        SELECT drug_code, document, location, qty_out AS qty_in, count(*) AS n
        FROM {moves} WHERE move_type = 'purchase' AND qty_out > 0
        GROUP BY ALL
    )
    SELECT l.row_number, l.date, l.drug_code, l.supplier, l.document, l.location,
           l.qty_in AS qty, l.stock_before
    FROM lines l LEFT JOIN deleted d USING (drug_code, document, location, qty_in)
    WHERE l.latest > coalesce(d.n, 0)  -- a deletion cancels the latest matching line
    """


def _build_purchases(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"CREATE OR REPLACE TABLE purchases AS {_purchases_sql('stock_moves')} "
                "ORDER BY row_number")


def _build_transfers(con: duckdb.DuckDBPyConnection) -> None:
    # Within a document and drug, the k-th row out of one location pairs with
    # the k-th row into the other. A row with no partner is kept, unpaired.
    con.execute("""
        CREATE OR REPLACE TABLE transfers AS
        WITH t AS (
            SELECT *, row_number() OVER (PARTITION BY document, drug_code, location,
                                                      qty_out > 0 ORDER BY row_number) AS k
            FROM stock_moves WHERE move_type IN ('transfer', 'test')
        ),
        o AS (SELECT * FROM t WHERE qty_out > 0),
        i AS (SELECT * FROM t WHERE qty_in > 0)
        SELECT coalesce(o.date, i.date) AS date,
               coalesce(o.drug_code, i.drug_code) AS drug_code,
               coalesce(o.location, CASE i.location WHEN 'floor' THEN 'store' ELSE 'floor' END)
                   AS from_location,
               coalesce(i.location, CASE o.location WHEN 'floor' THEN 'store' ELSE 'floor' END)
                   AS to_location,
               coalesce(o.qty_out, i.qty_in) AS qty,
               coalesce(o.document, i.document) AS document,
               o.row_number IS NOT NULL AND i.row_number IS NOT NULL
                   AND o.qty_out = i.qty_in AS paired
        FROM o FULL JOIN i
          ON o.document = i.document AND o.drug_code = i.drug_code
         AND o.location <> i.location AND o.k = i.k
        ORDER BY coalesce(o.row_number, i.row_number)""")


def _build_stock_takes(con: duckdb.DuckDBPyConnection) -> None:
    # Several rows in one day are usually one count entered in steps, so sum
    # them. adjustment > 0 means the count found more than the books said.
    kind = f"""CASE
        WHEN date BETWEEN {_sql_date(STORE_SETUP[0])} AND {_sql_date(STORE_SETUP[1])}
            THEN 'store_setup'
        WHEN date BETWEEN {_sql_date(FULL_COUNT[0])} AND {_sql_date(FULL_COUNT[1])}
            THEN 'full_count'
        ELSE 'routine' END"""
    con.execute(f"""
        CREATE OR REPLACE TABLE stock_takes AS
        SELECT date, drug_code, location,
               sum(qty_in - qty_out)           AS adjustment,
               arg_min(opening, row_number)    AS stock_before,
               arg_max(closing, row_number)    AS stock_after,
               {kind}                          AS kind
        FROM stock_moves
        WHERE move_type IN ('stock_take', 'stock_take_import')
        GROUP BY date, drug_code, location
        ORDER BY date, drug_code, location""")


def build_baseline(con: duckdb.DuckDBPyConnection, day: date = config.CUTOFF) -> None:
    """The baseline and cadence from `stock_moves` up to `day`.

    The build runs it at the cutoff. `contest.ensure_baseline` also runs it on
    an as-of connection before the cutoff, where the stored tables are hidden.
    """
    cutoff = _sql_date(day)
    start = _sql_date(config.STOCK_START)
    # Everything below reads only `seen`, the ledger up to `day`.
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW seen AS
        SELECT * FROM stock_moves WHERE date <= {cutoff}""")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE seen_purchases AS {_purchases_sql('seen')}""")

    # One delivery = one drug on one day (floor and store lines, any invoices).
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE deliveries AS
        WITH daily_sales AS (
            SELECT drug_code, date, sum(qty_out) AS units
            FROM seen WHERE move_type = 'sale' GROUP BY ALL
        ),
        d AS (
            SELECT drug_code, date, sum(qty) AS qty,
                   arg_min(stock_before, row_number) AS stock_before
            FROM seen_purchases GROUP BY ALL
        )
        SELECT d.*,
               d.date - lag(d.date) OVER (PARTITION BY d.drug_code ORDER BY d.date) AS gap,
               (SELECT coalesce(sum(s.units), 0) FROM daily_sales s
                 WHERE s.drug_code = d.drug_code
                   AND s.date BETWEEN d.date - {RATE_DAYS} AND d.date - 1) / {RATE_DAYS}
                   AS rate
        FROM d""")

    # Stock on arrival is only trusted from STOCK_START (opening balances were
    # wrong before the 2023 counts). Cover needs a non-zero recent sales rate.
    con.execute(f"""
        CREATE OR REPLACE TABLE pharmacist_baseline AS
        WITH sup AS (
            SELECT drug_code, supplier, count(DISTINCT date) AS days, max(date) AS last
            FROM seen_purchases GROUP BY ALL
        ),
        top AS (
            SELECT drug_code, count(*) AS n_suppliers,
                   list(supplier ORDER BY days DESC, last DESC, supplier)[1:3] AS main_suppliers
            FROM sup GROUP BY drug_code
        )
        SELECT drug_code,
               count(*)                                   AS deliveries,
               max(date)                                  AS last_delivery,
               median(gap)                                AS median_gap_days,
               median(qty)                                AS median_order_qty,
               median(stock_before / rate) FILTER (WHERE date >= {start} AND rate > 0)
                                                          AS median_cover_days,
               avg((stock_before <= 0)::INT) FILTER (WHERE date >= {start})
                                                          AS share_at_zero,
               any_value(top.n_suppliers)                 AS n_suppliers,
               any_value(top.main_suppliers)              AS main_suppliers
        FROM deliveries JOIN top USING (drug_code)
        GROUP BY drug_code
        ORDER BY drug_code""")

    # One supplier delivery = one supplier on one day.
    con.execute("""
        CREATE OR REPLACE TABLE supplier_cadence AS
        WITH days AS (
            SELECT supplier, date, count(DISTINCT drug_code) AS drugs
            FROM seen_purchases GROUP BY ALL
        ),
        g AS (
            SELECT *, date - lag(date) OVER (PARTITION BY supplier ORDER BY date) AS gap
            FROM days
        ),
        -- The most common weekday; a tie goes to the earlier one in the week.
        wd AS (
            SELECT supplier, list(d ORDER BY n DESC, dow)[1] AS main_weekday
            FROM (SELECT supplier, dayname(date) AS d, isodow(date) AS dow, count(*) AS n
                  FROM days GROUP BY ALL)
            GROUP BY supplier
        )
        SELECT supplier,
               count(*)              AS deliveries,
               median(gap)           AS median_gap_days,
               any_value(main_weekday) AS main_weekday,
               max(date)             AS last_delivery
        FROM g JOIN wd USING (supplier) GROUP BY supplier
        ORDER BY deliveries DESC, supplier""")
    con.execute("DROP TABLE deliveries; DROP TABLE seen_purchases; DROP VIEW seen")


def as_of_views(day: date) -> dict[str, str | None]:
    """For `warehouse.as_of`.

    Purchases are recomputed from the ledger up to `day`: a delivery line
    only drops out once its deletion is on the books (a few came days later).
    The baseline and cadence are computed through the cutoff, so they are
    hidden on earlier days rather than shown with later knowledge.
    """
    out: dict[str, str | None] = {"purchases": f"{_purchases_sql('stock_moves')} "
                                               "ORDER BY row_number"}
    if day < config.CUTOFF:
        out |= {"pharmacist_baseline": None, "supplier_cadence": None}
    return out


def build(con: duckdb.DuckDBPyConnection) -> None:
    _build_purchases(con)
    _build_transfers(con)
    _build_stock_takes(con)
    build_baseline(con)
