"""Demand, prices and pack sizes.

- `pack_sizes`: how many stock base units one sales unit holds (Strip -> 6 Tab).
- `daily_demand`: units sold per drug per day, in stock base units.
- `weekly_demand`: the same per Monday-start week, with every week filled in.
- `prices`: list-price history per drug and sales unit.
- `demand_tiers`: fast / regular / slow / rare and the Syntetos-Boylan class,
  from data up to `config.CUTOFF` only.

Services (`is_service`) carry no stock and are left out of every table here.
"""

from datetime import date

import duckdb

from irpd import config

TABLES = {
    "pack_sizes": None,
    "daily_demand": "date",
    # A week only shows once it is over, so an as_of day mid-week never sees
    # sales from later that week.
    "weekly_demand": "week_end",
    "prices": "valid_from",
    # Built from data up to config.CUTOFF only, so it is safe at any as_of day
    # after the cutoff, and it is the only tier table the agent gets.
    "demand_tiers": None,
}
DEPENDS: list[str] = []

# A price level seen on at most this many sale lines, with the same price on
# both sides (A-B-A), is a one-off line price, not a price change.
BLIP_MAX_LINES = 2

# Tiers by sale-days per year inside a drug's selling window (first to last sale).
FAST, REGULAR, SLOW = 156, 52, 12
# Syntetos-Boylan cut-offs on weekly demand.
ADI_CUT, CV2_CUT = 1.32, 0.49


def build(con: duckdb.DuckDBPyConnection) -> None:
    _pack_sizes(con)
    _daily_demand(con)
    _weekly_demand(con)
    _prices(con)
    con.execute(f"CREATE OR REPLACE TABLE demand_tiers AS {tiers_sql(config.CUTOFF)}")


def as_of_views(day: date) -> dict[str, str]:
    """For `warehouse.as_of`: tables rebuilt from the sales and ledger up to `day`.

    Pack sizes and prices are learned from all lines, so the stored tables
    would show units, prices and line counts from later days. Tiers are frozen
    at the cutoff; before it they are recomputed up to the day.
    """
    return {"pack_sizes": PACK_SIZES_SQL, "prices": prices_sql(),
            "demand_tiers": tiers_sql(min(day, config.CUTOFF))}


def _pack_sizes(con: duckdb.DuckDBPyConnection) -> None:
    """Learn base units per sales unit from sale lines matched to their ledger row.

    A sale line and the ledger's sale row share the invoice number and drug.
    The ratio ledger qty_out / sales qty is the pack size. Invoices that sell
    the drug in two units, or that later had a return, are skipped. The most
    common ratio wins (a handful of odd lines disagree). A drug-unit with no
    usable match gets 1 if it is the ledger's own unit, else NULL. The few
    drugs that never appear in the ledger keep their sales unit as the base.
    """
    con.execute(f"CREATE OR REPLACE TABLE pack_sizes AS {PACK_SIZES_SQL}")


# Reads only sales_lines, stock_moves and drugs, so `as_of_views` can rerun it.
PACK_SIZES_SQL = """
    WITH s AS (
        SELECT invoice, drug_code, any_value(unit) AS unit, sum(qty) AS qty
        FROM sales_lines WHERE NOT is_service
        GROUP BY invoice, drug_code HAVING count(DISTINCT unit) = 1
    ), k AS (
        SELECT document AS invoice, drug_code, sum(qty_out) AS qty
        FROM stock_moves WHERE move_type = 'sale' GROUP BY ALL
    ), returned AS (
        SELECT DISTINCT document AS invoice, drug_code FROM stock_moves
        WHERE move_type = 'return'
    ), ratios AS (
        SELECT s.drug_code, s.unit, k.qty / s.qty AS ratio
        FROM s JOIN k USING (invoice, drug_code)
        ANTI JOIN returned USING (invoice, drug_code)
        WHERE s.qty > 0
    ), learned AS (
        SELECT drug_code, unit, mode(ratio) AS ratio, count(*) AS lines
        FROM ratios GROUP BY ALL
    ), pairs AS (
        SELECT DISTINCT drug_code, unit FROM sales_lines WHERE NOT is_service
    )
    SELECT p.drug_code, p.unit AS sales_unit,
           coalesce(l.ratio, CASE WHEN p.unit = d.stock_unit OR d.drug_code IS NULL
                                  THEN 1 END) AS base_per_unit,
           coalesce(d.stock_unit, p.unit) AS base_unit,
           coalesce(l.lines, 0) AS matched_lines
    FROM pairs p
    LEFT JOIN learned l USING (drug_code, unit)
    LEFT JOIN drugs d USING (drug_code)
    ORDER BY drug_code, sales_unit"""


def _daily_demand(con: duckdb.DuckDBPyConnection) -> None:
    """Units, sale lines and revenue per drug per day. Only days with a sale have a row.

    Units come from the ledger's sale rows (qty_out, in base units), joined per
    invoice and drug. The few sale lines with no ledger row fall back to
    sales qty x pack size. Customer returns (ledger 'return' rows) are not
    netted off, because the sales file keeps those lines and their revenue
    too; they come to about 0.1% of units.
    """
    con.execute("""
        CREATE OR REPLACE TABLE daily_demand AS
        WITH s AS (
            SELECT l.date, l.invoice, l.drug_code, count(*) AS lines,
                   sum(l.revenue) AS revenue, sum(l.qty * p.base_per_unit) AS units_from_sales
            FROM sales_lines l
            LEFT JOIN pack_sizes p ON p.drug_code = l.drug_code AND p.sales_unit = l.unit
            WHERE NOT l.is_service GROUP BY ALL
        ), k AS (
            SELECT date, document AS invoice, drug_code, sum(qty_out) AS units
            FROM stock_moves WHERE move_type = 'sale' GROUP BY ALL
        )
        SELECT date, drug_code,
               sum(coalesce(k.units, s.units_from_sales)) AS units,
               sum(s.lines)::INT AS lines,
               sum(s.revenue) AS revenue
        FROM s FULL JOIN k USING (date, invoice, drug_code)
        GROUP BY ALL
        ORDER BY date, drug_code""")


def _weekly_demand(con: duckdb.DuckDBPyConnection) -> None:
    """Demand per drug per Monday-start week, zeros included.

    Every week from a drug's first sale to the end of the data has a row.
    `days_open` counts days the shop traded (any sale line); the shop is closed
    on a few days and the last week of the data is partial. `week_end` is the
    week's last day, or the last day of the data for the final week.
    """
    con.execute("""
        CREATE OR REPLACE TABLE weekly_demand AS
        WITH open_days AS (SELECT DISTINCT date FROM sales_lines),
        weeks AS (
            SELECT date_trunc('week', date)::DATE AS week_start, count(*) AS days_open
            FROM open_days GROUP BY 1
        ), last_day AS (SELECT max(date) AS d FROM open_days),
        firsts AS (
            SELECT drug_code, date_trunc('week', min(date))::DATE AS first_week
            FROM daily_demand GROUP BY 1
        ), per_week AS (
            SELECT date_trunc('week', date)::DATE AS week_start, drug_code,
                   sum(units) AS units, sum(lines) AS lines, sum(revenue) AS revenue
            FROM daily_demand GROUP BY ALL
        )
        SELECT w.week_start, least(w.week_start + 6, l.d) AS week_end, f.drug_code,
               coalesce(p.units, 0) AS units,
               coalesce(p.lines, 0)::INT AS lines,
               coalesce(p.revenue, 0) AS revenue,
               w.days_open::INT AS days_open
        FROM firsts f
        JOIN weeks w ON w.week_start >= f.first_week
        CROSS JOIN last_day l
        LEFT JOIN per_week p USING (week_start, drug_code)
        ORDER BY week_start, drug_code""")


def _prices(con: duckdb.DuckDBPyConnection) -> None:
    """List-price spells per drug and sales unit.

    Each day's price is the most common unit price that day (ties: the lower).
    Days in a row at one price make a spell. One-off A-B-A blips (a middle
    spell of at most BLIP_MAX_LINES lines) are dropped and the spells rebuilt,
    so A-B-A becomes one spell at A. `valid_to` is the day before the next
    spell starts, NULL for the latest price. Spotting a blip needs the price
    after it, and `valid_to` and `lines` come from later days, so
    `warehouse.as_of` reruns `prices_sql()` on the sales it can see.

    `likely_unit_change` marks a jump to 2x or more, or to half or less, of the
    previous price: more likely the unit was redefined (a Tab priced as a strip)
    than a real repricing.
    """
    con.execute(f"CREATE OR REPLACE TABLE prices AS {prices_sql()}")


def prices_sql() -> str:
    """The query behind `prices`, reading only `sales_lines`."""
    def spells(src: str) -> str:
        return f"""
            WITH flagged AS (
                SELECT *, (price IS DISTINCT FROM lag(price) OVER w)::INT AS new_spell
                FROM {src} WINDOW w AS (PARTITION BY drug_code, unit ORDER BY date)
            ), numbered AS (
                SELECT *, sum(new_spell) OVER (PARTITION BY drug_code, unit ORDER BY date)
                          AS spell
                FROM flagged
            ), s AS (
                SELECT drug_code, unit, spell, any_value(price) AS price,
                       min(date) AS start_date, max(date) AS end_date, sum(lines) AS lines
                FROM numbered GROUP BY ALL
            )
            SELECT *, lag(price) OVER w AS prev_price, lead(price) OVER w AS next_price,
                   lead(start_date) OVER w AS next_start
            FROM s WINDOW w AS (PARTITION BY drug_code, unit ORDER BY spell)"""

    return f"""
        WITH daily_price AS (
            SELECT drug_code, unit, date,
                   arg_max(unit_price, n * 1e12 - unit_price) AS price,
                   sum(n) AS lines
            FROM (SELECT drug_code, unit, date, unit_price, count(*) AS n
                  FROM sales_lines WHERE NOT is_service GROUP BY ALL)
            GROUP BY ALL
        ), blip_days AS (
            SELECT d.drug_code, d.unit, d.date
            FROM daily_price d JOIN ({spells('daily_price')}) s
              ON d.drug_code = s.drug_code AND d.unit = s.unit
             AND d.date BETWEEN s.start_date AND s.end_date
            WHERE s.lines <= {BLIP_MAX_LINES} AND s.prev_price = s.next_price
        ), kept AS (
            SELECT * FROM daily_price ANTI JOIN blip_days USING (drug_code, unit, date)
        )
        SELECT drug_code, unit, price,
               start_date AS valid_from,
               next_start - 1 AS valid_to,
               lines,
               prev_price,
               coalesce(price >= 2 * prev_price OR price <= 0.5 * prev_price, false)
                   AS likely_unit_change
        FROM ({spells('kept')})
        ORDER BY drug_code, unit, valid_from"""


def tiers_sql(until: date) -> str:
    """A query for demand tiers and demand classes, using daily_demand up to `until`.

    Tier, as in eda/stock.py: sale-days per year inside the drug's selling
    window (first to last sale up to `until`). Fast >= 156, regular >= 52,
    slow >= 12, else rare; under 28 days of window or 6 sale days is rare too.

    ADI and CV² are on weekly units over the weeks of the same window:
    ADI = weeks / weeks with a sale, CV² = (sd / mean)² of the non-zero weeks.
    Fewer than 3 non-zero weeks is 'too few'. `revenue_share` is the drug's
    share of all revenue up to `until`.
    """
    d = f"DATE '{until.isoformat()}'"
    return f"""
        WITH dd AS (SELECT * FROM daily_demand WHERE date <= {d} AND units > 0),
        span AS (
            SELECT drug_code, min(date) AS first_sale, max(date) AS last_sale,
                   count(*) AS sale_days, sum(revenue) AS revenue
            FROM dd GROUP BY 1
        ), wk AS (
            SELECT drug_code, date_trunc('week', date) AS week, sum(units) AS units
            FROM dd GROUP BY ALL
        ), sb AS (
            SELECT w.drug_code,
                   (date_diff('week', date_trunc('week', any_value(s.first_sale)),
                              date_trunc('week', any_value(s.last_sale))) + 1)
                       / count(*) AS adi,
                   CASE WHEN count(*) >= 2 THEN power(stddev_samp(units) / avg(units), 2)
                        END AS cv2,
                   count(*) AS sale_weeks
            FROM wk w JOIN span s USING (drug_code) GROUP BY 1
        ), t AS (
            SELECT s.*, last_sale - first_sale + 1 AS live_days,
                   sale_days * 365.0 / (last_sale - first_sale + 1) AS sale_days_per_year
            FROM span s
        )
        SELECT drug_code,
               CASE WHEN live_days < 28 OR sale_days < 6 THEN 'rare'
                    WHEN sale_days_per_year >= {FAST} THEN 'fast'
                    WHEN sale_days_per_year >= {REGULAR} THEN 'regular'
                    WHEN sale_days_per_year >= {SLOW} THEN 'slow'
                    ELSE 'rare' END AS tier,
               sale_days_per_year, adi, cv2,
               CASE WHEN sale_weeks < 3 THEN 'too few'
                    WHEN adi < {ADI_CUT} AND cv2 < {CV2_CUT} THEN 'smooth'
                    WHEN adi < {ADI_CUT} THEN 'erratic'
                    WHEN cv2 < {CV2_CUT} THEN 'intermittent'
                    ELSE 'lumpy' END AS sb_class,
               revenue / sum(revenue) OVER () AS revenue_share,
               sale_days, first_sale, last_sale
        FROM t JOIN sb USING (drug_code)
        ORDER BY drug_code"""
