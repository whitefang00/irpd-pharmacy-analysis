"""The eval question bank: questions a pharmacy owner would ask, with gold answers.

Each question has a gold function that computes its answer with plain SQL on a
connection from `warehouse.as_of(config.CUTOFF)`, after `prepare()` has added
the clipped spell tables. Gold answers never use data after the cutoff. They
also avoid the columns that would carry later facts if `as_of` failed to
rebuild them:

- stock-out and shelf-out spells are recomputed up to the cutoff
  (`stockout_spells`, `shelfout_spells`), not read from `stockouts`/`shelfouts`;
- price questions only use `valid_from`, never `valid_to`.

Words used in the questions:

- stock-out day: the drug's total stock (shop floor + upstairs store) closed
  the day at zero or below. Counted from 1 Oct 2023 (`config.STOCK_START`).
- shelf-out day: the floor closed at zero or below while the store held stock.
- units are the stock ledger's base units (tablets, bottles, ...).
- days of cover: stock at close on 30 Jun 2025 / average daily units sold over
  the 28 days 3–30 Jun 2025.
- revenue is the sales file's line subtotal in rupiah, services included
  unless a question says otherwise.

Kinds of answer: number, date, yes_no, label (one name), list (drug or
supplier names; `top` lists are scored on overlap with the top k, `set`
lists on Jaccard), and cant_tell (the honest answer is that the data can't
say). See `irpd.evals.scorer`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

import duckdb

from irpd import config
from irpd.tables import stock

Con = duckdb.DuckDBPyConnection

H1_24 = "date BETWEEN DATE '2024-01-01' AND DATE '2024-06-30'"
Y24 = "date BETWEEN DATE '2024-01-01' AND DATE '2024-12-31'"
H1_25 = "date BETWEEN DATE '2025-01-01' AND DATE '2025-06-30'"
CUT = f"DATE '{config.CUTOFF.isoformat()}'"
FAST_REG = "('fast', 'regular')"


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    text: str
    kind: str  # number | date | yes_no | label | list | cant_tell
    tools: tuple[str, ...]
    gold: Callable[[Con], Any] | None = None
    tol: float = 0.0  # number: absolute tolerance; date: days
    rel: float = 0.0  # number: relative tolerance (passes if within tol or rel)
    list_mode: str = "top"  # list: 'top' (top-k overlap) or 'set' (Jaccard)
    needs: str | None = None  # e.g. 'ingredient_map': skipped until it exists


QUESTIONS: list[Question] = []


def q(id: str, category: str, text: str, kind: str, tools: list[str], **kw: Any):
    """Register a question whose gold answer is the decorated function."""
    def add(fn: Callable[[Con], Any]) -> Callable[[Con], Any]:
        QUESTIONS.append(Question(id, category, text, kind, tuple(tools), fn, **kw))
        return fn
    return add


def cant_tell(id: str, text: str, tools: list[str]) -> None:
    QUESTIONS.append(Question(id, "cant_tell", text, "cant_tell", tuple(tools)))


def prepare(con: Con) -> None:
    """Add spell tables recomputed up to the cutoff, so no spell shows a later stop."""
    stock.spells_as_of(con, config.CUTOFF, "stockouts").to_table("stockout_spells")
    stock.spells_as_of(con, config.CUTOFF, "shelfouts").to_table("shelfout_spells")


def plain(v: Any) -> Any:
    """Plain Python values: floats rounded, dates as ISO strings."""
    if v is None:
        return None
    if isinstance(v, date):
        return v.isoformat()
    if hasattr(v, "item"):  # numpy / pandas scalars
        v = v.item()
    if isinstance(v, float):
        return round(v, 4)
    return v


def one(con: Con, sql: str, *params: Any) -> Any:
    row = con.execute(sql, list(params)).fetchone()
    return plain(row[0]) if row else None


def names(con: Con, sql: str, *params: Any) -> list[str]:
    return [r[0] for r in con.execute(sql, list(params)).fetchall()]


def pct(part: Any, whole: Any) -> float | None:
    return None if not whole or part is None else round(100 * part / whole, 4)


def code(con: Con, name: str) -> str | None:
    return one(con, "SELECT drug_code FROM drugs WHERE drug_name = ?", name)


# Stock-out days per drug in a period, from daily stock (never from spells).
def _out_days(where: str) -> str:
    return f"""
        SELECT drug_code, count(*) AS out_days FROM daily_stock
        WHERE location = 'total' AND closing <= 0 AND {where}
          AND date >= DATE '{config.STOCK_START.isoformat()}'
        GROUP BY drug_code"""


# Shelf-out days (drug, date) in a period, from daily stock.
def _shelf_days(where: str) -> str:
    return f"""
        SELECT drug_code, date FROM daily_stock
        WHERE location IN ('floor', 'store') AND {where}
        GROUP BY drug_code, date
        HAVING max(closing) FILTER (WHERE location = 'floor') <= 0
           AND max(closing) FILTER (WHERE location = 'store') > 0"""

# Days of cover at the cutoff, per drug with stock and recent sales.
_COVER = f"""
    SELECT drug_code,
           max(closing) FILTER (WHERE date = {CUT}) AS stock,
           sum(sold) / 28 AS rate,
           max(closing) FILTER (WHERE date = {CUT}) / (sum(sold) / 28) AS cover
    FROM daily_stock
    WHERE location = 'total' AND date BETWEEN {CUT} - 27 AND {CUT}
    GROUP BY drug_code
    HAVING max(closing) FILTER (WHERE date = {CUT}) > 0 AND sum(sold) > 0"""


# Revenue

@q("rev_2024_total", "revenue",
   "What was our total sales revenue in 2024 (1 Jan–31 Dec), in rupiah, "
   "services included?",
   "number", ["run_sql"], rel=0.005)
def _(con):
    return one(con, f"SELECT sum(revenue) FROM sales_lines WHERE {Y24}")


@q("rev_h1_change", "revenue",
   "By what percentage did revenue in Jan–Jun 2025 change compared with Jan–Jun 2024? "
   "Give the change in percent (negative if it fell).",
   "number", ["run_sql"], tol=0.5)
def _(con):
    now = one(con, f"SELECT sum(revenue) FROM sales_lines WHERE {H1_25}")
    then = one(con, f"SELECT sum(revenue) FROM sales_lines WHERE {H1_24}")
    return round(100 * (now / then - 1), 4) if now and then else None


@q("rev_top10_drugs_h1_2025", "revenue",
   "Which 10 products brought in the most revenue in Jan–Jun 2025? Leave out "
   "services; break ties by name.",
   "list", ["run_sql"])
def _(con):
    return names(con, f"""
        SELECT drug_name FROM sales_lines WHERE NOT is_service AND {H1_25}
        GROUP BY drug_name ORDER BY sum(revenue) DESC, drug_name LIMIT 10""")


@q("rev_patent_share_2024", "revenue",
   "What percentage of 2024 revenue came from the 'Patent' category?",
   "number", ["run_sql"], tol=0.5)
def _(con):
    r = con.execute(f"""SELECT sum(revenue) FILTER (WHERE category = 'Patent'), sum(revenue)
                        FROM sales_lines WHERE {Y24}""").fetchone()
    return pct(*r)


@q("rev_peak_hour_h1_2025", "revenue",
   "In Jan–Jun 2025, which hour of the day brought in the most revenue? Answer with "
   "the hour it starts, 0–23 (e.g. 14 for 14:00–14:59).",
   "number", ["run_sql"])
def _(con):
    return one(con, f"""SELECT hour(ts) FROM sales_lines WHERE {H1_25}
                        GROUP BY 1 ORDER BY sum(revenue) DESC, 1 LIMIT 1""")


@q("rev_evening_share_2024", "revenue",
   "What percentage of 2024 revenue was rung up between 16:00 and 20:59?",
   "number", ["run_sql"], tol=0.5)
def _(con):
    r = con.execute(f"""SELECT sum(revenue) FILTER (WHERE hour(ts) BETWEEN 16 AND 20),
                               sum(revenue) FROM sales_lines WHERE {Y24}""").fetchone()
    return pct(*r)


@q("rev_flazen_mar_2025", "revenue",
   "How much revenue did Flazen 50 bring in during March 2025, in rupiah?",
   "number", ["run_sql"], rel=0.005)
def _(con):
    return one(con, """SELECT coalesce(sum(revenue), 0) FROM sales_lines
                       WHERE drug_name = 'Flazen 50'
                         AND date BETWEEN DATE '2025-03-01' AND DATE '2025-03-31'""")


@q("rev_best_weekday_2024", "revenue",
   "In 2024, which day of the week had the highest average revenue per trading day? "
   "A trading day is a day with at least one sale.",
   "label", ["run_sql"])
def _(con):
    return one(con, f"""SELECT dayname(date) FROM sales_lines WHERE {Y24}
                        GROUP BY 1 ORDER BY sum(revenue) / count(DISTINCT date) DESC LIMIT 1""")


@q("rev_top5_growth", "multi_step",
   "Which 5 products grew their revenue the most in rupiah from Jan–Jun 2024 to "
   "Jan–Jun 2025? A product with no sales in a period counts as zero. Leave out "
   "services; break ties by name.",
   "list", ["run_sql"])
def _(con):
    return names(con, f"""
        SELECT drug_name FROM sales_lines
        WHERE NOT is_service AND ({H1_24} OR {H1_25})
        GROUP BY drug_name
        ORDER BY coalesce(sum(revenue) FILTER (WHERE {H1_25}), 0)
               - coalesce(sum(revenue) FILTER (WHERE {H1_24}), 0) DESC, drug_name
        LIMIT 5""")


# Demand tiers

@q("tier_count_fast", "demand_tiers",
   "How many products are in the fast demand tier (sold on 156+ days a year), "
   "using sales up to 30 Jun 2025?",
   "number", ["run_sql"])
def _(con):
    return one(con, "SELECT count(*) FROM demand_tiers WHERE tier = 'fast'")


@q("tier_fast_share", "demand_tiers",
   "What percentage of all revenue up to 30 Jun 2025 came from fast-tier products?",
   "number", ["run_sql"], tol=0.5)
def _(con):
    return one(con, "SELECT 100 * sum(revenue_share) FROM demand_tiers WHERE tier = 'fast'")


@q("tier_zeniflox", "demand_tiers",
   "Which demand tier (fast, regular, slow or rare) is Zeniflox 500 in?",
   "label", ["run_sql"])
def _(con):
    return one(con, """SELECT tier FROM demand_tiers JOIN drugs USING (drug_code)
                       WHERE drug_name = 'Zeniflox 500'""")


@q("tier_top5_sale_days", "demand_tiers",
   "Which 5 products sold on the most different days between Jan 2023 and 30 Jun 2025? "
   "Break ties by name.",
   "list", ["run_sql"])
def _(con):
    return names(con, """
        SELECT drug_name FROM demand_tiers JOIN drugs USING (drug_code)
        ORDER BY sale_days DESC, drug_name LIMIT 5""")


# Stock-outs

@q("so_spells_h1_2025", "stockouts",
   "How many stock-out spells (runs of consecutive days with total stock at zero) "
   "started in Jan–Jun 2025 on fast or regular products?",
   "number", ["stockout_history"])
def _(con):
    return one(con, f"""
        SELECT count(*) FROM stockout_spells JOIN demand_tiers USING (drug_code)
        WHERE tier IN {FAST_REG} AND start BETWEEN DATE '2025-01-01' AND {CUT}""")


@q("so_top5_fast_days", "stockouts",
   "Among fast-tier products still selling in June 2025 (at least one sale 1–30 Jun), "
   "which 5 had the most stock-out days in Jan–Jun 2025? Break ties by name.",
   "list", ["stockout_history"])
def _(con):
    return names(con, f"""
        SELECT drug_name FROM ({_out_days(H1_25)})
        JOIN demand_tiers USING (drug_code) JOIN drugs USING (drug_code)
        WHERE tier = 'fast' AND last_sale >= DATE '2025-06-01'
        ORDER BY out_days DESC, drug_name LIMIT 5""")


@q("so_flazen_any_h1", "stockouts",
   "Did Flazen 50 run out of stock (total stock at zero at close) on any day in "
   "Jan–Jun 2025?",
   "yes_no", ["stockout_history"])
def _(con):
    c = code(con, "Flazen 50")
    return bool(one(con, f"SELECT count(*) FROM ({_out_days(H1_25)}) WHERE drug_code = ?", c))


@q("so_amlodipine_any_h1", "stockouts",
   "Did Amlodipine 10mg Novel run out of stock (total stock at zero at close) on any "
   "day in Jan–Jun 2025?",
   "yes_no", ["stockout_history"])
def _(con):
    c = code(con, "Amlodipine 10mg Novel")
    return bool(one(con, f"SELECT count(*) FROM ({_out_days(H1_25)}) WHERE drug_code = ?", c))


@q("so_amlodipine_last", "stockouts",
   "When did the most recent stock-out of Amlodipine 10mg Novel start, up to 30 Jun 2025?",
   "date", ["stockout_history"])
def _(con):
    return one(con, """SELECT max(start) FROM stockout_spells JOIN drugs USING (drug_code)
                       WHERE drug_name = 'Amlodipine 10mg Novel'""")


@q("so_out_at_cutoff_fast", "stockouts",
   "Which fast-tier products that sold in June 2025 were out of stock (total stock "
   "at zero) at close on 30 Jun 2025?",
   "list", ["stock_position", "stockout_history"], list_mode="set")
def _(con):
    return names(con, f"""
        SELECT drug_name FROM daily_stock JOIN demand_tiers USING (drug_code)
        JOIN drugs USING (drug_code)
        WHERE location = 'total' AND date = {CUT} AND closing <= 0 AND tier = 'fast'
          AND drug_code IN (SELECT drug_code FROM daily_demand
                            WHERE date >= DATE '2025-06-01' AND units > 0)
        ORDER BY drug_name""")


@q("so_longest_2024", "stockouts",
   "Among fast and regular products, which one had the longest single stock-out "
   "spell that both started and ended in 2024? Break ties by name.",
   "label", ["stockout_history"])
def _(con):
    return one(con, f"""
        SELECT drug_name FROM stockout_spells JOIN demand_tiers USING (drug_code)
        JOIN drugs USING (drug_code)
        WHERE tier IN {FAST_REG} AND start >= DATE '2024-01-01'
          AND stop <= DATE '2024-12-31'
        ORDER BY days DESC, drug_name LIMIT 1""")


@q("so_ended_by_purchase_2024", "stockouts",
   "Of all stock-out spells that ended in 2024 (any product), what percentage ended "
   "with a purchase delivery?",
   "number", ["stockout_history"], tol=1.0)
def _(con):
    r = con.execute("""
        SELECT count(*) FILTER (WHERE ended_by = 'purchase'), count(*) FROM stockout_spells
        WHERE stop BETWEEN DATE '2024-01-01' AND DATE '2024-12-31'""").fetchone()
    return pct(*r)


# Shelf-outs

@q("sh_fast_days_h1", "shelfouts",
   "How many shelf-out days (shop floor empty while the upstairs store had stock) did "
   "fast-tier products have in Jan–Jun 2025, one per product per day?",
   "number", ["stockout_history", "run_sql"], rel=0.02)
def _(con):
    return one(con, f"""SELECT count(*) FROM ({_shelf_days(H1_25)})
                        JOIN demand_tiers USING (drug_code) WHERE tier = 'fast'""")


@q("sh_count_at_cutoff", "shelfouts",
   "At close on 30 Jun 2025, how many products had an empty shop floor while the "
   "upstairs store still had stock?",
   "number", ["stock_position", "transfer_plan"])
def _(con):
    return one(con, f"SELECT count(*) FROM ({_shelf_days(f'date = {CUT}')})")


@q("sh_move_down", "shelfouts",
   "Which fast or regular products should be brought down from the store right now: "
   "floor empty but store holding stock at close on 30 Jun 2025?",
   "list", ["transfer_plan", "stock_position"], list_mode="set")
def _(con):
    return names(con, f"""
        SELECT drug_name FROM ({_shelf_days(f'date = {CUT}')})
        JOIN demand_tiers USING (drug_code) JOIN drugs USING (drug_code)
        WHERE tier IN {FAST_REG} ORDER BY drug_name""")


# Stock position at the cutoff

@q("pos_teamic_total", "stock_position",
   "How many units (tablets) of Teamic 500mg did we hold at close on 30 Jun 2025, "
   "floor and store together?",
   "number", ["stock_position"])
def _(con):
    return one(con, f"""SELECT closing FROM daily_stock JOIN drugs USING (drug_code)
                        WHERE drug_name = 'Teamic 500mg' AND location = 'total'
                          AND date = {CUT}""")


@q("pos_flazen_store", "stock_position",
   "How many units of Flazen 50 were in the upstairs store (not the shop floor) at "
   "close on 30 Jun 2025?",
   "number", ["stock_position"])
def _(con):
    return one(con, f"""SELECT closing FROM daily_stock JOIN drugs USING (drug_code)
                        WHERE drug_name = 'Flazen 50' AND location = 'store'
                          AND date = {CUT}""")


@q("pos_teamic_cover", "stock_position",
   "How many days of cover did Teamic 500mg have at close on 30 Jun 2025, at its "
   "average daily sales over 3–30 Jun 2025?",
   "number", ["stock_position"], rel=0.05)
def _(con):
    return one(con, f"SELECT cover FROM ({_COVER}) JOIN drugs USING (drug_code) "
                    "WHERE drug_name = 'Teamic 500mg'")


@q("pos_low_cover_fast", "stock_position",
   "Which 5 fast-tier products had the fewest days of cover at close on 30 Jun 2025 "
   "(average daily sales over 3–30 Jun)? Only count products with stock and sales in "
   "that window; break ties by name.",
   "list", ["stock_position", "reorder_plan"])
def _(con):
    return names(con, f"""
        SELECT drug_name FROM ({_COVER}) JOIN demand_tiers USING (drug_code)
        JOIN drugs USING (drug_code)
        WHERE tier = 'fast' ORDER BY cover, drug_name LIMIT 5""")


@q("pos_over_90_fast", "stock_position",
   "How many fast-tier products had more than 90 days of cover at close on 30 Jun 2025 "
   "(average daily sales over 3–30 Jun; products with stock and sales in that window)?",
   "number", ["stock_position"], tol=1)
def _(con):
    return one(con, f"""SELECT count(*) FROM ({_COVER}) JOIN demand_tiers USING (drug_code)
                        WHERE tier = 'fast' AND cover > 90""")


@q("pos_value_fast_regular", "multi_step",
   "What was the total value of stock held at close on 30 Jun 2025 for fast and regular "
   "products, in rupiah at selling price? Value each product at its median selling "
   "price per base unit on Jan–Jun 2025 sale lines; leave out products with no sales "
   "then and count negative stock as zero.",
   "number", ["stock_position", "price_history", "run_sql"], rel=0.03)
def _(con):
    return one(con, f"""
        WITH price AS (
            SELECT l.drug_code, median(l.unit_price / p.base_per_unit) AS per_unit
            FROM sales_lines l JOIN pack_sizes p
              ON p.drug_code = l.drug_code AND p.sales_unit = l.unit
            WHERE NOT l.is_service AND p.base_per_unit > 0 AND l.{H1_25}
            GROUP BY 1)
        SELECT sum(greatest(s.closing, 0) * price.per_unit)
        FROM daily_stock s JOIN price USING (drug_code) JOIN demand_tiers USING (drug_code)
        WHERE s.location = 'total' AND s.date = {CUT} AND tier IN {FAST_REG}""")


# Purchases, transfers and stock-takes

@q("pur_flazen_units_2024", "purchases",
   "How many units (tablets) of Flazen 50 were delivered in 2024?",
   "number", ["run_sql"])
def _(con):
    return one(con, f"""SELECT coalesce(sum(qty), 0) FROM purchases JOIN drugs USING (drug_code)
                        WHERE drug_name = 'Flazen 50' AND {Y24}""")


@q("pur_teamic_days_h1", "purchases",
   "On how many different days was Teamic 500mg delivered in Jan–Jun 2025?",
   "number", ["run_sql"])
def _(con):
    return one(con, f"""SELECT count(DISTINCT date) FROM purchases JOIN drugs USING (drug_code)
                        WHERE drug_name = 'Teamic 500mg' AND {H1_25}""")


@q("pur_flazen_last", "purchases",
   "When was Flazen 50 last delivered, up to 30 Jun 2025?",
   "date", ["run_sql", "stock_position"])
def _(con):
    return one(con, """SELECT max(date) FROM purchases JOIN drugs USING (drug_code)
                       WHERE drug_name = 'Flazen 50'""")


@q("pur_flazen_cover", "purchases",
   "When a delivery of Flazen 50 arrives, how many days of stock do we typically still "
   "have? Give the median over deliveries from 1 Oct 2023 to 30 Jun 2025, with cover "
   "measured as stock just before the delivery / average daily sales of the 28 days "
   "before it.",
   "number", ["run_sql", "reorder_plan"], rel=0.05)
def _(con):
    return one(con, """SELECT median_cover_days FROM pharmacist_baseline JOIN drugs
                       USING (drug_code) WHERE drug_name = 'Flazen 50'""")


@q("tr_days_down_h1", "transfers",
   "On how many days in Jan–Jun 2025 was any stock moved from the upstairs store down "
   "to the shop floor?",
   "number", ["run_sql"])
def _(con):
    return one(con, f"""SELECT count(DISTINCT date) FROM transfers
                        WHERE from_location = 'store' AND to_location = 'floor' AND {H1_25}""")


@q("tr_top5_2024", "transfers",
   "Which 5 products were brought down from the store to the shop floor on the most "
   "days in 2024? Break ties by name.",
   "list", ["run_sql"])
def _(con):
    return names(con, f"""
        SELECT drug_name FROM transfers JOIN drugs USING (drug_code)
        WHERE from_location = 'store' AND to_location = 'floor' AND {Y24}
        GROUP BY drug_name ORDER BY count(DISTINCT date) DESC, drug_name LIMIT 5""")


@q("st_routine_2024", "transfers",
   "In 2024, how many routine stock counts changed the books? Count each product, "
   "location and day once, and only counts that found a difference.",
   "number", ["run_sql"])
def _(con):
    return one(con, f"""SELECT count(*) FROM stock_takes
                        WHERE kind = 'routine' AND adjustment <> 0 AND {Y24}""")


# Suppliers

@q("sup_most_days", "suppliers",
   "Which supplier delivered to us on the most days, up to 30 Jun 2025?",
   "label", ["run_sql"])
def _(con):
    return one(con, """SELECT supplier FROM purchases GROUP BY 1
                       ORDER BY count(DISTINCT date) DESC, supplier LIMIT 1""")


@q("sup_count_h1", "suppliers",
   "How many different suppliers delivered to us in Jan–Jun 2025?",
   "number", ["run_sql"])
def _(con):
    return one(con, f"SELECT count(DISTINCT supplier) FROM purchases WHERE {H1_25}")


@q("sup_main_flazen", "suppliers",
   "Which supplier do we mostly get Flazen 50 from (most delivery days, up to 30 Jun 2025)?",
   "label", ["run_sql", "reorder_plan"])
def _(con):
    return one(con, """SELECT supplier FROM purchases JOIN drugs USING (drug_code)
                       WHERE drug_name = 'Flazen 50' GROUP BY 1
                       ORDER BY count(DISTINCT date) DESC, max(date) DESC, supplier LIMIT 1""")


@q("sup_top5_range_2024", "suppliers",
   "Which 5 suppliers delivered the widest range of products (most different products) "
   "in 2024? Break ties by name.",
   "list", ["run_sql"])
def _(con):
    return names(con, f"""SELECT supplier FROM purchases WHERE {Y24} GROUP BY 1
                          ORDER BY count(DISTINCT drug_code) DESC, supplier LIMIT 5""")


@q("sup_enseval_gap", "suppliers",
   "How many days typically pass between deliveries from Enseval Putra Megatrading.PT? "
   "Give the median gap between its delivery days, up to 30 Jun 2025.",
   "number", ["run_sql", "reorder_plan"], tol=0.5)
def _(con):
    return one(con, """
        SELECT median(gap) FROM (
            SELECT date - lag(date) OVER (ORDER BY date) AS gap
            FROM (SELECT DISTINCT date FROM purchases
                  WHERE supplier = 'Enseval Putra Megatrading.PT'))""")


@q("sup_main_weekday", "suppliers",
   "On which day of the week do the most supplier deliveries arrive (counting each "
   "supplier once per day), up to 30 Jun 2025?",
   "label", ["run_sql"])
def _(con):
    return one(con, """SELECT dayname(date) FROM (SELECT DISTINCT supplier, date FROM purchases)
                       GROUP BY 1 ORDER BY count(*) DESC LIMIT 1""")


# Prices
# A price change is a new spell in the price history (most common daily price
# per drug and sales unit, one-off blips dropped), leaving out jumps to 2x or
# half that look like a unit redefinition.

_CHANGES = """SELECT * FROM prices
              WHERE prev_price IS NOT NULL AND NOT likely_unit_change AND {where}"""


@q("pr_changes_2024", "prices",
   "How many price changes were there in 2024, across all products and sales units? "
   "Ignore one-off blips and jumps to double or half (likely unit changes).",
   "number", ["price_history", "run_sql"], rel=0.02)
def _(con):
    where = "valid_from BETWEEN DATE '2024-01-01' AND DATE '2024-12-31'"
    return one(con, f"SELECT count(*) FROM ({_CHANGES.format(where=where)})")


@q("pr_flazen_mar1", "prices",
   "What was the selling price of one tablet (Tab) of Flazen 50 on 1 Mar 2025, in rupiah?",
   "number", ["price_history"])
def _(con):
    return one(con, """
        SELECT price FROM prices JOIN drugs USING (drug_code)
        WHERE drug_name = 'Flazen 50' AND unit = 'Tab' AND valid_from <= DATE '2025-03-01'
        ORDER BY valid_from DESC LIMIT 1""")


@q("pr_teamic_changes", "prices",
   "How many times did the price per tablet (Tab) of Teamic 500mg change between "
   "Jan 2023 and 30 Jun 2025 (no blips, no likely unit changes)?",
   "number", ["price_history"])
def _(con):
    c = code(con, "Teamic 500mg")
    where = f"drug_code = '{c}' AND unit = 'Tab'"
    return one(con, f"SELECT count(*) FROM ({_CHANGES.format(where=where)})")


@q("pr_top_rise_fast_2024", "prices",
   "Among fast-tier products, which one had the biggest single price rise in percent "
   "in 2024 (no blips, no likely unit changes)? Break ties by name.",
   "label", ["price_history", "run_sql"])
def _(con):
    where = "valid_from BETWEEN DATE '2024-01-01' AND DATE '2024-12-31'"
    return one(con, f"""
        SELECT drug_name FROM ({_CHANGES.format(where=where)})
        JOIN demand_tiers USING (drug_code) JOIN drugs USING (drug_code)
        WHERE tier = 'fast' ORDER BY price / prev_price DESC, drug_name LIMIT 1""")


# Multi-step

@q("ms_supplier_most_out_days", "multi_step",
   "Group fast and regular products by their main supplier (the one that delivered "
   "them on the most days up to 30 Jun 2025; ties go to the most recent, then by name). "
   "Which supplier's products had the most stock-out days in Jan–Jun 2025?",
   "label", ["stockout_history", "run_sql"])
def _(con):
    return one(con, f"""
        WITH sup AS (
            SELECT drug_code, supplier, count(DISTINCT date) AS days, max(date) AS last
            FROM purchases GROUP BY ALL),
        main AS (
            SELECT drug_code, supplier FROM sup
            QUALIFY row_number() OVER (PARTITION BY drug_code
                                       ORDER BY days DESC, last DESC, supplier) = 1)
        SELECT supplier FROM ({_out_days(H1_25)}) JOIN main USING (drug_code)
        JOIN demand_tiers USING (drug_code) WHERE tier IN {FAST_REG}
        GROUP BY 1 ORDER BY sum(out_days) DESC, supplier LIMIT 1""")


@q("ms_category_most_out_days", "multi_step",
   "Among fast and regular products, which product category had the most stock-out "
   "days in Jan–Jun 2025? Leave out products with no category ('-').",
   "label", ["stockout_history", "run_sql"])
def _(con):
    return one(con, f"""
        SELECT category FROM ({_out_days(H1_25)}) JOIN demand_tiers USING (drug_code)
        JOIN drugs USING (drug_code) WHERE tier IN {FAST_REG} AND category <> '-'
        GROUP BY 1 ORDER BY sum(out_days) DESC, category LIMIT 1""")


@q("ms_out_recent_delivery", "multi_step",
   "Of the fast and regular products that were out of stock at close on 30 Jun 2025, "
   "how many had a delivery in the 30 days before (1–30 Jun 2025)?",
   "number", ["stock_position", "run_sql"])
def _(con):
    return one(con, f"""
        SELECT count(*) FROM daily_stock s JOIN demand_tiers USING (drug_code)
        WHERE s.location = 'total' AND s.date = {CUT} AND s.closing <= 0
          AND tier IN {FAST_REG}
          AND drug_code IN (SELECT drug_code FROM purchases
                            WHERE date BETWEEN {CUT} - 29 AND {CUT})""")


@q("ms_flazen_weeks", "multi_step",
   "If Flazen 50 keeps selling at its average weekly rate of the last 8 full weeks "
   "before the cutoff (Mon 5 May–Sun 29 Jun 2025), how many weeks will the stock it held "
   "at close on 30 Jun 2025 last?",
   "number", ["stock_position", "forecast"], rel=0.05)
def _(con):
    c = code(con, "Flazen 50")
    rate = one(con, """SELECT sum(units) / 8 FROM weekly_demand WHERE drug_code = ?
                       AND week_start BETWEEN DATE '2025-05-05' AND DATE '2025-06-23'""", c)
    held = one(con, f"""SELECT closing FROM daily_stock WHERE drug_code = ?
                        AND location = 'total' AND date = {CUT}""", c)
    return round(held / rate, 4) if rate and held is not None else None


# The data can't tell us

cant_tell("ct_margin_flazen",
          "What profit margin did we make on Flazen 50 in 2024?", ["run_sql"])
cant_tell("ct_most_profitable_category",
          "Which product category makes us the most profit?", ["run_sql"])
cant_tell("ct_lead_time_enseval",
          "How many days after we place an order does Enseval Putra Megatrading.PT "
          "deliver it?", ["run_sql", "reorder_plan"])
cant_tell("ct_expiry_bufamoxy",
          "How many units of Bufamoxy 500 will expire before the end of 2025?",
          ["stock_position", "run_sql"])
cant_tell("ct_repeat_customers",
          "How many of our customers in Jan–Jun 2025 were repeat customers?", ["run_sql"])


# Ingredients (need the ingredient map)
# These read a table `ingredient_map` (drug_code, ingredients, ...) where
# `ingredients` is the sorted ingredient list joined with ' + ', empty if unknown.

_SAME = """
    SELECT m.drug_code FROM ingredient_map m
    WHERE m.ingredients <> '' AND m.ingredients =
          (SELECT ingredients FROM ingredient_map JOIN drugs USING (drug_code)
           WHERE drug_name = ?)
      AND m.drug_code <> (SELECT drug_code FROM drugs WHERE drug_name = ?)"""


@q("ing_zetamol_alts", "ingredients",
   "Which other products have exactly the same active ingredients as Zetamol tb 500mg "
   "(any strength or form) and sold at least once in Jan–Jun 2025?",
   "list", ["alternatives"], list_mode="set", needs="ingredient_map")
def _(con):
    return names(con, f"""
        SELECT DISTINCT drug_name FROM sales_lines
        WHERE {H1_25} AND drug_code IN ({_SAME}) ORDER BY 1""", *["Zetamol tb 500mg"] * 2)


@q("ing_bufamoxy_alts_in_stock", "ingredients",
   "If Bufamoxy 500 ran out, how many other products with exactly the same active "
   "ingredients had stock at close on 30 Jun 2025?",
   "number", ["alternatives", "stock_position"], needs="ingredient_map")
def _(con):
    return one(con, f"""
        SELECT count(*) FROM daily_stock
        WHERE location = 'total' AND date = {CUT} AND closing > 0
          AND drug_code IN ({_SAME})""", *["Bufamoxy 500"] * 2)


@q("ing_paracetamol_rev_h1", "ingredients",
   "How much revenue in Jan–Jun 2025 came from products whose only active ingredient "
   "is paracetamol, in rupiah?",
   "number", ["alternatives", "run_sql"], rel=0.01, needs="ingredient_map")
def _(con):
    return one(con, f"""
        SELECT coalesce(sum(revenue), 0) FROM sales_lines JOIN ingredient_map USING (drug_code)
        WHERE ingredients = 'paracetamol' AND {H1_25}""")


@q("ing_top_single_2024", "ingredients",
   "Among single-ingredient products, which active ingredient brought in the most "
   "revenue in 2024?",
   "label", ["alternatives", "run_sql"], needs="ingredient_map")
def _(con):
    return one(con, f"""
        SELECT ingredients FROM sales_lines JOIN ingredient_map USING (drug_code)
        WHERE ingredients <> '' AND ingredients NOT LIKE '%+%' AND {Y24}
        GROUP BY 1 ORDER BY sum(revenue) DESC, 1 LIMIT 1""")
