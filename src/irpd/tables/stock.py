"""Daily stock per drug and location, and stock-out and shelf-out spells.

- `daily_stock`: end-of-day stock per drug for 'floor', 'store' and 'total'
  (floor + store), every day from the drug's first movement to LAST_DAY.
  End of day is the closing balance of the day's last ledger row by
  row_number (not by timestamp), carried forward over days with no movement.
  A location holds 0 before its first row for that drug, even when that row
  opens with stock: back-filling it would leak a later row into earlier days
  (such drugs are flagged `unbooked_opening`). `sold` is the units sold that
  day, from the ledger's sale rows, in the stock unit.
- `stockouts`: spells of consecutive days where total stock ends <= 0.
  `ended_by` is the move type that brought it back above zero ('unbooked' if
  no inflow did), or NULL if the spell is still open on the last day.
- `shelfouts`: spells where the floor ends <= 0 while the store ends > 0.
- `stock_flags`: known data problems (`huge_opening`, `negative_close`,
  `unbooked_opening`), each dated the day it could first be known.

Spells only count days from STOCK_START, so a spell starting that day may
have begun earlier. Spell columns are `start` and `stop` (`end` is an SQL
keyword).

The stored spells carry their final `stop`, `days` and `ended_by`, which
can lie after a given day. `warehouse.as_of` therefore rebuilds both tables
with `spells_as_of()`'s query (see `as_of_views`), so a spell still running on
the day stops there with `ended_by` NULL.
"""

from datetime import date, timedelta

import duckdb

from irpd import config

TABLES = {"daily_stock": "date", "stockouts": "start", "shelfouts": "start",
          "stock_flags": "date"}
DEPENDS = ["stock_moves"]

# Moves that shift stock between floor and store; they cancel out at the total.
INTERNAL = ("transfer", "test")


def _d(day: date) -> str:
    return f"DATE '{day.isoformat()}'"


def _spells_sql(table: str, day: date) -> str:
    """SQL for the spells in `table`, using only rows dated up to `day`."""
    if table == "stockouts":
        out_days = f"""
            SELECT drug_code, date FROM daily_stock
            WHERE location = 'total' AND closing <= 0
              AND date BETWEEN {_d(config.STOCK_START)} AND {_d(day)}"""
    elif table == "shelfouts":
        out_days = f"""
            SELECT drug_code, date FROM daily_stock
            WHERE location IN ('floor', 'store')
              AND date BETWEEN {_d(config.STOCK_START)} AND {_d(day)}
            GROUP BY drug_code, date
            HAVING max(closing) FILTER (WHERE location = 'floor') <= 0
               AND max(closing) FILTER (WHERE location = 'store') > 0"""
    else:
        raise ValueError(f"unknown spell table {table!r}")

    # Consecutive days share the same date minus their rank.
    spells = f"""
        SELECT drug_code, min(date) AS start, max(date) AS stop, count(*)::INTEGER AS days
        FROM (SELECT drug_code, date,
                     date - row_number() OVER (PARTITION BY drug_code ORDER BY date)::INTEGER
                         AS grp
              FROM ({out_days}))
        GROUP BY drug_code, grp"""
    if table == "shelfouts":
        return f"{spells} ORDER BY drug_code, start"

    # The day after a spell, the largest inflow that isn't an internal move
    # (those cancel at the total) is what restocked it. 'unbooked' means stock
    # came back with no inflow, as when a location's first row opens with stock.
    internal = ", ".join(f"'{t}'" for t in INTERNAL)
    return f"""
        WITH s AS ({spells})
        SELECT s.*,
               CASE WHEN s.stop < {_d(day)} THEN coalesce(e.ended_by, 'unbooked') END
                   AS ended_by
        FROM s LEFT JOIN (
            SELECT s.drug_code, s.start,
                   first(m.move_type ORDER BY m.move_type IN ({internal}), m.qty_in DESC)
                       AS ended_by
            FROM s JOIN stock_moves m
              ON m.drug_code = s.drug_code AND m.date = s.stop + 1 AND m.qty_in > 0
            WHERE m.date <= {_d(day)}
            GROUP BY ALL) e USING (drug_code, start)
        ORDER BY drug_code, start"""


def spells_as_of(con: duckdb.DuckDBPyConnection, day: date,
                 table: str = "stockouts") -> duckdb.DuckDBPyRelation:
    """Stock-out or shelf-out spells as they looked at the end of `day`.

    A spell still running on `day` stops there, with `ended_by` NULL. Works on
    a full warehouse connection or on one from `warehouse.as_of`.
    """
    return con.sql(_spells_sql(table, day))


def as_of_views(day: date) -> dict[str, str]:
    """For `warehouse.as_of`: spells recomputed from daily stock up to `day`."""
    day = min(day, config.LAST_DAY)  # the data ends there; nothing after it is an inflow
    return {t: _spells_sql(t, day) for t in ("stockouts", "shelfouts")}


def build(con: duckdb.DuckDBPyConnection) -> None:
    last = _d(config.LAST_DAY)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _eod AS
        SELECT drug_code, location, date,
               arg_max(closing, row_number)::INTEGER AS closing,
               coalesce(sum(qty_out) FILTER (WHERE move_type = 'sale'), 0)::INTEGER AS sold
        FROM stock_moves GROUP BY ALL;

        CREATE OR REPLACE TEMP TABLE _loc AS
        WITH days AS (
            SELECT drug_code,
                   unnest(generate_series(min(date)::TIMESTAMP, {last}::TIMESTAMP,
                                          INTERVAL 1 DAY))::DATE AS date
            FROM stock_moves GROUP BY drug_code)
        SELECT d.date, d.drug_code, l.location,
               coalesce(last_value(e.closing IGNORE NULLS) OVER (
                   PARTITION BY d.drug_code, l.location ORDER BY d.date), 0) AS closing,
               coalesce(e.sold, 0) AS sold
        FROM days d
        CROSS JOIN (VALUES ('floor'), ('store')) l(location)
        LEFT JOIN _eod e USING (drug_code, location, date);

        CREATE OR REPLACE TABLE daily_stock AS
        SELECT * FROM (
            SELECT * FROM _loc
            UNION ALL
            SELECT date, drug_code, 'total', sum(closing)::INTEGER, sum(sold)::INTEGER
            FROM _loc GROUP BY date, drug_code)
        ORDER BY drug_code, location, date;
        """)
    con.execute(f"CREATE OR REPLACE TABLE stockouts AS {_spells_sql('stockouts', config.LAST_DAY)}")
    con.execute(f"CREATE OR REPLACE TABLE shelfouts AS {_spells_sql('shelfouts', config.LAST_DAY)}")
    _build_flags(con)
    con.execute("DROP TABLE _eod; DROP TABLE _loc")


def _build_flags(con: duckdb.DuckDBPyConnection) -> None:
    """Each flag is dated the day it could first be known, so as_of stays honest."""
    start = _d(config.STOCK_START)
    before = _d(config.STOCK_START - timedelta(days=1))
    pre_days = (config.STOCK_START - config.FIRST_DAY).days
    con.execute(f"""
        CREATE OR REPLACE TABLE stock_flags AS
        -- Years of stock on the books before STOCK_START, at the sales rate
        -- of the same months (EDA: >= 500 units and > 3 years).
        WITH pre AS (
            SELECT drug_code, max(closing) AS peak, arg_max(date, closing) AS peak_day,
                   sum(sold) AS sold
            FROM daily_stock WHERE location = 'total' AND date < {start}
            GROUP BY drug_code),
        counts AS (
            SELECT drug_code, sum(qty_out - qty_in) AS cut
            FROM stock_moves
            WHERE move_type IN ('stock_take', 'stock_take_import')
              AND date BETWEEN DATE '2023-07-01' AND {before}
            GROUP BY drug_code)
        SELECT drug_code, {before} AS date, 'huge_opening' AS flag,
               format('peak {{:,}} units on {{}} = {{}} of sales; Jul-Sep 2023 counts cut {{:,}}',
                      peak, peak_day,
                      CASE WHEN sold > 0
                           THEN round(peak / (sold / {pre_days / 365.25}), 1) || ' years'
                           ELSE 'no' END,
                      coalesce(cut, 0)) AS detail
        FROM pre LEFT JOIN counts USING (drug_code)
        WHERE peak >= 500 AND (sold = 0 OR peak > 3 * sold / {pre_days / 365.25})

        UNION ALL
        -- A location that ends the day below zero (usually a sale rung up
        -- before the stock was booked in, not fixed the same day).
        SELECT drug_code, date, 'negative_close',
               format('{{}} closed at {{}}', location, closing)
        FROM _eod WHERE closing < 0

        UNION ALL
        -- A location's first row opens with stock, so it held stock before
        -- any row shows it. daily_stock counts 0 there until that row.
        SELECT drug_code, date, 'unbooked_opening',
               format('{{}} first row opens at {{}}; earlier stock not counted', location, o)
        FROM (SELECT drug_code, location, min(date) AS date,
                     arg_min(opening, row_number) AS o
              FROM stock_moves GROUP BY drug_code, location)
        WHERE o <> 0 AND date >= {start}
        ORDER BY drug_code, flag, date""")
