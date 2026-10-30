"""EDA route 2: stock ledger and stock-outs.

Rebuilds end-of-day stock per drug and location from the raw ledger, finds
stock-out spells, and describes how the pharmacist restocks: supplier
cadence, stock left when a delivery lands, order sizes, stock-takes and
transfers between the sales floor and the upstairs store. Also explains the
negative and the huge balances. Prints everything; writes nothing.

Run from anywhere:
    uvx --with duckdb --with pandas python -I eda/stock.py [data_dir]
"""

import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "data" / "raw" / "irpd"
STOCK_RAW = DATA / "raw_dataset" / "stock_raw.csv"
SALES_PROC = DATA / "processed_dataset" / "sales_processed.csv"
STOCK_PROC = DATA / "processed_dataset" / "stock_processed.csv"

END = "2025-12-31"
FLOOR, STORE = "Apotek", "Gudang Obat 1"
OUT_OF_SCOPE = "('Alkes', 'Baby', 'Minuman')"  # dropped by the processed files

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 45)
pd.set_option("display.max_rows", 80)

con = duckdb.connect()  # in-memory


def q(sql):
    return con.execute(sql).df()


def one(sql):
    return con.execute(sql).fetchone()[0]


def show(title, sql):
    print(f"\n-- {title}")
    print(q(sql).to_string(index=False))


def header(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def pct(a, b):
    return f"{100 * a / b:.2f}%" if b else "n/a"


t0 = time.time()

# ---------------------------------------------------------------------------
# Load the raw ledger. Quantities are "1.234 Tab" (dot = thousands, no
# decimals in this file). TransactionDescription is free text; we classify it
# into movement kinds. Everything that isn't a sale, purchase, return,
# stock-take or master-data row is an inter-location transfer ("mutasi",
# "mts", "m", ... and a few "test" entries that also moved stock).
# ---------------------------------------------------------------------------


def qty(col):
    return f"replace(split_part({col}, ' ', 1), '.', '')::BIGINT"


con.execute(f"""
CREATE TABLE led AS
SELECT RowNumber::INT AS rn,
       strptime(DateTime, '%d %b %Y %H:%M:%S') AS ts,
       strptime(DateTime, '%d %b %Y %H:%M:%S')::DATE AS d,
       Warehouse AS wh, DrugCode AS code, DrugName AS drug,
       DocumentNumber AS doc, TransactionDescription AS descr,
       {qty('InitialStock')} AS s0, {qty('StockIn')} AS qin,
       {qty('StockOut')} AS qout, {qty('RemainingStock')} AS s1,
       split_part(RemainingStock, ' ', 2) AS unit,
       Category AS cat,
       CASE WHEN descr ILIKE 'Penjualan%'                 THEN 'sale'
            WHEN descr ILIKE 'Retur Penjualan%'           THEN 'sale_return'
            WHEN descr ILIKE 'Retur Pembelian%'           THEN 'supplier_return'
            WHEN descr ILIKE 'Pembelian%'                 THEN 'purchase'
            WHEN descr ILIKE '%import%stokopname%'        THEN 'stocktake_import'
            WHEN descr ILIKE '%stokopname%'               THEN 'stocktake'
            WHEN descr ILIKE 'Import Data Obat%'          THEN 'import'
            WHEN descr ILIKE 'Tambah Master%'             THEN 'new_item'
            ELSE 'transfer' END AS kind,
       trim(regexp_extract(descr, 'Pembelian Dari (.*) dengan no faktur', 1)) AS supplier,
       (descr ILIKE 'tes%') AS test_entry
FROM read_csv('{STOCK_RAW}', all_varchar = true)
""")

# One category and one unit per drug code (checked below), so a lookup table.
con.execute("""
CREATE TABLE drugs AS
SELECT code, any_value(drug) AS drug, any_value(cat) AS cat,
       any_value(unit) AS unit, count(DISTINCT cat) AS n_cat,
       count(DISTINCT unit) AS n_unit
FROM led GROUP BY code
""")

# Selling price per drug: median unit price over all sale lines (processed
# sales only cover the in-scope categories). Used to value stock at selling
# price, since the data has no costs.
con.execute(f"""
CREATE TABLE price AS
SELECT DrugCode AS code, median(UnitPrice) AS price,
       median(UnitPrice) FILTER (WHERE Year = 2025) AS price_2025
FROM read_csv('{SALES_PROC}') GROUP BY 1
""")

header("0. Ledger as loaded")
show("Movement kinds by location", """
SELECT kind, wh, count(*) AS rows, sum(qin) AS units_in, sum(qout) AS units_out,
       min(d) AS first, max(d) AS last
FROM led GROUP BY ALL ORDER BY kind, wh""")
print(f"\ndrugs: {one('SELECT count(*) FROM drugs')}, "
      f"with >1 unit: {one('SELECT count(*) FROM drugs WHERE n_unit > 1')}, "
      f">1 category: {one('SELECT count(*) FROM drugs WHERE n_cat > 1')}")
print(f"test-entry transfers: {one('SELECT count(*) FROM led WHERE test_entry')} rows, "
      f"{one('SELECT sum(qin) FROM led WHERE test_entry')} units moved")

# ---------------------------------------------------------------------------
# 1. Rebuild end-of-day stock and check it reconciles.
# ---------------------------------------------------------------------------
header("1. End-of-day stock: rebuild and reconcile")

n = one("SELECT count(*) FROM led")
bad_row = one("SELECT count(*) FROM led WHERE s0 + qin - qout <> s1")
con.execute("""
CREATE TABLE led_seq AS
SELECT *, lag(s1) OVER w AS prev_s1, lag(ts) OVER w AS prev_ts,
       row_number() OVER w AS k
FROM led WINDOW w AS (PARTITION BY code, wh ORDER BY rn)
""")
bad_chain = one("SELECT count(*) FROM led_seq WHERE k > 1 AND s0 <> prev_s1")
back_ts = one("SELECT count(*) FROM led_seq WHERE ts < prev_ts")
print(f"opening + in - out = closing fails on {bad_row} of {n} rows")
print(f"opening = previous closing (per drug, location, by RowNumber) fails on {bad_chain} rows")
print(f"timestamps going backwards within a drug and location: {back_ts}")
show("Where the chain breaks (row kind, previous row kind, same second as previous?)", """
SELECT kind, lag_kind, (ts = prev_ts) AS same_second, count(*) AS rows, count(DISTINCT code) AS drugs
FROM (SELECT *, lag(kind) OVER (PARTITION BY code, wh ORDER BY rn) AS lag_kind FROM led_seq)
WHERE k > 1 AND s0 <> prev_s1 GROUP BY ALL ORDER BY rows DESC""")

# Transfers come in pairs: same document and drug, out of one location, in to
# the other. Check every pair balances.
show("Transfer pairing (per document and drug)", f"""
WITH t AS (
  SELECT doc, code,
         sum(qout) FILTER (WHERE wh = '{FLOOR}') AS floor_out,
         sum(qin)  FILTER (WHERE wh = '{FLOOR}') AS floor_in,
         sum(qout) FILTER (WHERE wh = '{STORE}') AS store_out,
         sum(qin)  FILTER (WHERE wh = '{STORE}') AS store_in
  FROM led WHERE kind = 'transfer' GROUP BY ALL)
SELECT count(*) AS pairs,
       sum((coalesce(floor_out,0) = coalesce(store_in,0)
            AND coalesce(store_out,0) = coalesce(floor_in,0))::INT) AS balanced,
       sum(coalesce(store_out,0) + coalesce(floor_out,0)) AS units_moved,
       sum(coalesce(store_out,0)) AS store_to_floor,
       sum(coalesce(floor_out,0)) AS floor_to_store
FROM t""")

# End-of-day stock per drug and location: the closing balance of the last row
# that day, carried forward over days with no movement. A series starts on
# its first movement day, or on the location's first day if it already held
# stock then (opening balance carried from before the export).
con.execute(f"""
CREATE TABLE eod_rows AS
SELECT code, wh, d, arg_max(s1, rn) AS stock FROM led GROUP BY ALL;

CREATE TABLE series AS
SELECT code, wh, s0 AS opening,
       CASE WHEN s0 <> 0 THEN (SELECT min(d) FROM led l2 WHERE l2.wh = f.wh)
            ELSE d END AS start
FROM led_seq f WHERE k = 1;

CREATE TABLE eod_wh AS
SELECT s.code, s.wh, c.d,
       coalesce(last_value(e.stock IGNORE NULLS) OVER
                (PARTITION BY s.code, s.wh ORDER BY c.d), s.opening) AS stock
FROM series s
JOIN (SELECT range::DATE AS d FROM range(DATE '2023-01-02', DATE '{END}' + 1, INTERVAL 1 DAY)) c
  ON c.d >= s.start
LEFT JOIN eod_rows e ON e.code = s.code AND e.wh = s.wh AND e.d = c.d;
""")

# Per drug per day: total, floor, store, units sold (ledger sales).
con.execute(f"""
CREATE TABLE eod AS
SELECT w.code, w.d, sum(stock) AS stock,
       sum(stock) FILTER (WHERE wh = '{FLOOR}') AS floor,
       coalesce(sum(stock) FILTER (WHERE wh = '{STORE}'), 0) AS store,
       coalesce(any_value(s.sold), 0) AS sold
FROM eod_wh w
LEFT JOIN (SELECT code, d, sum(qout) AS sold FROM led WHERE kind = 'sale' GROUP BY ALL) s
  ON s.code = w.code AND s.d = w.d
GROUP BY w.code, w.d
""")

# Check 1: day-on-day change in total stock equals that day's net movements
# (transfers cancel out at the total).
show("Total stock: day-on-day change vs net movements", """
WITH mv AS (SELECT code, d, sum(qin - qout) AS net FROM led GROUP BY ALL),
x AS (SELECT e.code, e.d, e.stock - lag(e.stock, 1, 0) OVER (PARTITION BY e.code ORDER BY e.d)
             AS change, coalesce(mv.net, 0) AS net,
             row_number() OVER (PARTITION BY e.code ORDER BY e.d) AS k
      FROM eod e LEFT JOIN mv USING (code, d))
SELECT count(*) AS drug_days, sum((change <> net)::INT) AS mismatches
FROM x WHERE k > 1""")

# Check 2: ledger sale units vs the sales file, per drug-day.
show("Ledger sale units vs sales_processed Quantity, per drug-day", f"""
WITH s AS (SELECT DrugCode AS code, Date::DATE AS d, sum(Quantity) AS q
           FROM read_csv('{SALES_PROC}') GROUP BY ALL),
l AS (SELECT code, d, sum(qout) AS q FROM led WHERE kind = 'sale' GROUP BY ALL)
SELECT count(*) AS drug_days_in_both,
       sum((round(s.q) = l.q)::INT) AS equal,
       sum(s.q) AS sales_units, sum(l.q) AS ledger_units
FROM s JOIN l USING (code, d)""")
# The gap is a unit problem: some sale lines are rung up in a bigger pack
# (Strip, Box) than the stock unit, while the ledger books base units.
show("Sale lines whose sales Unit differs from the drug's stock unit", f"""
SELECT count(*) AS lines, count(DISTINCT s.DrugCode) AS drugs,
       sum(Quantity) AS sales_qty, any_value(s.Unit || ' vs ' || d.unit) AS example
FROM read_csv('{SALES_PROC}') s JOIN drugs d ON d.code = s.DrugCode
WHERE s.Unit <> d.unit""")

# Check 3: the processed stock file has the same rows but no location, so its
# last balance of the day is one location's balance, not the total.
show("Processed file's last balance of day vs rebuilt total (drugs that used the store)", f"""
WITH p AS (SELECT DrugCode AS code, Date::DATE AS d, arg_max(RemainingStock, RowNumber) AS bal
           FROM read_csv('{STOCK_PROC}') GROUP BY ALL),
st AS (SELECT DISTINCT code FROM led WHERE wh = '{STORE}')
SELECT count(*) AS drug_days, sum((p.bal <> e.stock)::INT) AS differ,
       round(100.0 * sum((p.bal <> e.stock)::INT) / count(*), 1) AS pct_differ
FROM p JOIN eod e USING (code, d) WHERE code IN (SELECT code FROM st)""")
print(f"\nrebuilt: {one('SELECT count(*) FROM eod_wh')} drug-location-days, "
      f"{one('SELECT count(*) FROM eod')} drug-days")

# ---------------------------------------------------------------------------
# Demand tiers. A drug's live window runs from its first to its last sale in
# the ledger. Tier = sale-days per year inside that window. Stock-out shares
# below count only days inside the live window, so drugs that were dropped
# or not yet listed don't count as "out". Scope: the processed categories.
# ---------------------------------------------------------------------------
con.execute(f"""
CREATE TABLE tiers AS
WITH s AS (SELECT code, min(d) AS first_sale, max(d) AS last_sale,
                  count(DISTINCT d) AS sale_days, sum(qout) AS units
           FROM led WHERE kind = 'sale' GROUP BY code)
SELECT s.*, (last_sale - first_sale + 1) AS live_days,
       sale_days * 365.0 / (last_sale - first_sale + 1) AS sale_days_per_year,
       CASE WHEN live_days < 28 OR sale_days < 6 THEN '4 rare'
            WHEN sale_days_per_year >= 156 THEN '1 fast'
            WHEN sale_days_per_year >= 52 THEN '2 regular'
            WHEN sale_days_per_year >= 12 THEN '3 slow'
            ELSE '4 rare' END AS tier
FROM s JOIN drugs USING (code) WHERE drugs.cat NOT IN {OUT_OF_SCOPE}
""")
con.execute("""
CREATE TABLE live AS
SELECT e.*, t.tier FROM eod e JOIN tiers t USING (code)
WHERE e.d BETWEEN t.first_sale AND t.last_sale
""")

header("2. Stock-outs")
show("Demand tiers (sale-days per year in live window: fast >=156, regular 52-155, slow 12-51)", """
SELECT tier, count(*) AS drugs, sum(units) AS units_sold,
       round(100.0 * sum(units) / sum(sum(units)) OVER (), 1) AS pct_units,
       round(median(live_days)) AS median_live_days
FROM tiers GROUP BY tier ORDER BY tier""")

show("Share of drug-days with end-of-day stock <= 0", f"""
SELECT 'all drug-days, all drugs (calendar fill)' AS basis, count(*) AS drug_days,
       round(100.0 * avg((stock <= 0)::INT), 1) AS pct_out FROM eod
UNION ALL
SELECT 'live window, in scope, tier ' || tier, count(*),
       round(100.0 * avg((stock <= 0)::INT), 1) FROM live GROUP BY tier
ORDER BY basis""")

# Spells: runs of consecutive days at <= 0 total stock, over the whole series.
con.execute(f"""
CREATE TABLE spells AS
WITH x AS (SELECT code, d, stock, sold,
                  d - CAST(row_number() OVER (PARTITION BY code ORDER BY d) AS INT) AS grp
           FROM eod WHERE stock <= 0),
sp AS (SELECT code, min(d) AS start, max(d) AS stop, count(*) AS days,
              sum(sold) - arg_min(sold, d) AS sold_after_day1,
              min(stock) AS min_stock
       FROM x GROUP BY code, grp)
SELECT sp.*, t.tier,
       sp.stop = DATE '{END}' AS censored,
       (SELECT coalesce(sum(qout), 0) FROM led l WHERE l.kind = 'sale' AND l.code = sp.code
          AND l.d BETWEEN sp.start - 60 AND sp.start - 1) AS sold_60d_before,
       sp.stop < t.last_sale AS sells_after
FROM sp JOIN tiers t USING (code)
""")
# What ends a spell: the restocking movement on the first day back above
# zero. At the total level transfers cancel, so it's a purchase, stock-take,
# customer return or master-data import. The largest inflow that day wins.
con.execute("""
CREATE TABLE spell_end AS
SELECT s.code, s.start, arg_max(l.kind, l.qin) AS ended_by
FROM spells s JOIN led l ON l.code = s.code AND l.d = s.stop + 1 AND l.qin > 0
     AND l.kind <> 'transfer'
WHERE NOT s.censored GROUP BY ALL
""")
# A "real" stock-out: the drug sold in the 60 days before, and sold again
# after (so it was not simply dropped from the range).
show("Spell counts (in scope)", """
SELECT tier, count(*) AS spells,
       sum(censored::INT) AS still_out_at_end,
       sum((sold_60d_before > 0 AND sells_after)::INT) AS real_spells,
       sum(days) AS spell_days
FROM spells GROUP BY tier ORDER BY tier""")

show("Real spells: length in days (fast + regular drugs)", """
SELECT tier, count(*) AS spells,
       quantile_cont(days, 0.25) AS p25, median(days) AS p50,
       quantile_cont(days, 0.75) AS p75, quantile_cont(days, 0.9) AS p90,
       round(100.0 * avg((days = 1)::INT), 1) AS pct_1day,
       round(100.0 * avg((days <= 7)::INT), 1) AS pct_le7,
       round(100.0 * avg((days > 30)::INT), 1) AS pct_gt30
FROM spells WHERE sold_60d_before > 0 AND sells_after AND tier IN ('1 fast', '2 regular')
GROUP BY tier ORDER BY tier""")

show("What ends a spell (real spells, all tiers; total stock)", """
SELECT coalesce(e.ended_by, '(no inflow next day)') AS ended_by, count(*) AS spells,
       round(100.0 * count(*) / sum(count(*)) OVER (), 1) AS pct
FROM spells s LEFT JOIN spell_end e USING (code, start)
WHERE s.sold_60d_before > 0 AND s.sells_after GROUP BY 1 ORDER BY 2 DESC""")

show("Sales booked on days 2+ of a spell (stock already <= 0 the night before)", """
SELECT tier, sum(days) AS spell_days, sum(sold_after_day1) AS units_sold_days_2_plus,
       sum((sold_after_day1 > 0)::INT) AS spells_with_such_sales
FROM spells GROUP BY tier ORDER BY tier""")

show("Most stock-out days, fast + regular drugs (live window)", """
SELECT l.code, d.drug, d.cat, l.tier, count(*) AS live_days,
       sum((stock <= 0)::INT) AS out_days,
       round(100.0 * avg((stock <= 0)::INT), 1) AS pct_out,
       (SELECT count(*) FROM spells s WHERE s.code = l.code
          AND s.sold_60d_before > 0 AND s.sells_after) AS real_spells
FROM live l JOIN drugs d USING (code)
WHERE l.tier IN ('1 fast', '2 regular')
GROUP BY ALL ORDER BY out_days DESC LIMIT 15""")

show("Stock-out share by category (fast + regular, live window)", """
SELECT d.cat, count(DISTINCT code) AS drugs, count(*) AS live_days,
       round(100.0 * avg((stock <= 0)::INT), 1) AS pct_out
FROM live JOIN drugs d USING (code) WHERE tier IN ('1 fast', '2 regular')
GROUP BY 1 ORDER BY pct_out DESC""")

show("Stock-out share by year (fast + regular, live window)", """
SELECT year(d) AS year, count(*) AS live_days,
       round(100.0 * avg((stock <= 0)::INT), 1) AS pct_out,
       round(100.0 * avg((stock <= 0)::INT) FILTER (WHERE tier = '1 fast'), 1) AS pct_out_fast
FROM live WHERE tier IN ('1 fast', '2 regular') GROUP BY 1 ORDER BY 1""")

# Shelf-outs: floor empty while the store still has stock (store in use from
# July 2023). A lost sale from bad replenishment, not bad ordering.
show("Floor <= 0 while store > 0 (fast + regular, live window, from Jul 2023)", """
SELECT tier, count(*) AS live_days,
       sum((floor <= 0 AND store > 0)::INT) AS shelf_out_days,
       sum((stock <= 0)::INT) AS total_out_days
FROM live WHERE d >= DATE '2023-07-07' AND tier IN ('1 fast', '2 regular')
GROUP BY tier ORDER BY tier""")

# Do shelf-outs lose sales? Compare units sold the day after the floor ended
# empty (store stocked) with the drug's mean on days after a stocked floor.
show("Sales the day after a shelf-out vs after a stocked floor (fast + regular, from Jul 2023)", """
WITH x AS (SELECT code, tier, sold,
                  lag(floor) OVER (PARTITION BY code ORDER BY d) AS floor_prev,
                  lag(store) OVER (PARTITION BY code ORDER BY d) AS store_prev,
                  lag(stock) OVER (PARTITION BY code ORDER BY d) AS stock_prev
           FROM live WHERE d >= DATE '2023-07-07' AND tier IN ('1 fast', '2 regular')),
b AS (SELECT code, avg(sold) FILTER (WHERE floor_prev > 0) AS base FROM x GROUP BY code)
SELECT CASE WHEN stock_prev <= 0 THEN 'total out' WHEN floor_prev <= 0 THEN 'shelf-out (store stocked)'
            ELSE 'floor stocked' END AS state_prev_night,
       count(*) AS drug_days, round(sum(sold) / sum(base), 2) AS sold_vs_drug_normal
FROM x JOIN b USING (code) WHERE floor_prev IS NOT NULL GROUP BY 1 ORDER BY 1""")

# Rough size of the hidden demand: spell days x the drug's pre-spell rate.
# Not an estimate yet (pillar 2 does that), just the order of magnitude.
show("Rough hidden demand in real spells: days x mean daily sales in the 60 days before", """
SELECT s.tier, count(*) AS spells, round(sum(s.days * s.sold_60d_before / 60.0)) AS approx_units,
       round(sum(s.days * s.sold_60d_before / 60.0 * p.price) / 1e6) AS approx_rp_million
FROM spells s LEFT JOIN price p USING (code)
WHERE s.sold_60d_before > 0 AND s.sells_after GROUP BY 1 ORDER BY 1""")

# ---------------------------------------------------------------------------
# 3. The pharmacist's behaviour: deliveries, reorder points, order sizes.
# ---------------------------------------------------------------------------
header("3. Purchases and the pharmacist's reorder behaviour")

# Running total stock (both locations) after every row, so we know the stock
# on hand just before each delivery line.
con.execute("""
CREATE TABLE run AS
SELECT l.*, o.opening_total + sum(qin - qout) OVER (PARTITION BY code ORDER BY rn) AS tot_after
FROM led l JOIN (SELECT code, sum(opening) AS opening_total FROM series GROUP BY 1) o USING (code)
""")
show("Running total after the day's last row vs rebuilt end-of-day total", """
WITH r AS (SELECT code, d, arg_max(tot_after, rn) AS tot FROM run GROUP BY ALL)
SELECT count(*) AS drug_days, sum((r.tot <> e.stock)::INT) AS mismatches
FROM r JOIN eod e USING (code, d)""")

# One delivery = supplier + document. One purchase event = drug + day.
con.execute("""
CREATE TABLE buys AS
SELECT code, d, string_agg(DISTINCT supplier, ' | ') AS supplier, sum(qin) AS qty,
       arg_min(tot_after - qin, rn) AS stock_before
FROM run WHERE kind = 'purchase' AND qin > 0 GROUP BY code, d
""")
con.execute("""
CREATE TABLE buys_x AS
SELECT b.*, t.tier,
       (SELECT coalesce(sum(sold), 0) FROM eod e WHERE e.code = b.code
          AND e.d BETWEEN b.d - 28 AND b.d - 1) / 28.0 AS rate28,
       b.d - lag(b.d) OVER (PARTITION BY b.code ORDER BY b.d) AS gap
FROM buys b JOIN tiers t USING (code)
""")

show("Suppliers: delivery cadence (top 15 by delivery days)", """
WITH dd AS (SELECT supplier, d, count(DISTINCT doc) AS invoices, count(*) AS lines
            FROM led WHERE kind = 'purchase' AND qin > 0 GROUP BY ALL),
g AS (SELECT *, d - lag(d) OVER (PARTITION BY supplier ORDER BY d) AS gap FROM dd)
SELECT supplier, count(*) AS delivery_days, sum(lines) AS lines,
       round(count(*) / ((max(d) - min(d) + 1) / 7.0), 2) AS days_per_week,
       median(gap) AS median_gap_days, round(avg(lines), 1) AS lines_per_day,
       min(d) AS first, max(d) AS last
FROM g GROUP BY supplier ORDER BY delivery_days DESC LIMIT 15""")
n_sup = one("SELECT count(DISTINCT supplier) FROM led WHERE kind = 'purchase'")
n_lines = one("SELECT count(*) FROM led WHERE kind = 'purchase' AND qin > 0")
n_days = one("SELECT count(DISTINCT d) FROM led WHERE kind = 'purchase'")
print(f"\nsuppliers: {n_sup}; purchase lines: {n_lines}; days with any delivery: {n_days}")

show("Suppliers per drug (purchased drugs, by tier)", """
WITH s AS (SELECT code, count(DISTINCT supplier) AS n FROM led
           WHERE kind = 'purchase' AND qin > 0 GROUP BY code)
SELECT tier, count(*) AS drugs, median(n) AS median_suppliers,
       round(100.0 * avg((n = 1)::INT), 1) AS pct_single_source
FROM s JOIN tiers USING (code) GROUP BY tier ORDER BY tier""")

show("Purchase events per drug: how often, how much (by tier)", """
SELECT tier, count(*) AS purchase_days, count(DISTINCT code) AS drugs,
       median(gap) AS median_days_between, median(qty) AS median_qty,
       round(median(qty / nullif(rate28, 0)), 1) AS median_order_days_of_sales
FROM buys_x GROUP BY tier ORDER BY tier""")

# Implied reorder point: stock on hand when the delivery lands, in days of
# recent sales. (No order dates exist, so this is stock at arrival, which is
# the reorder point minus demand over the lead time.)
show("Stock on hand when a delivery lands, in days of cover (28-day sales rate)", """
SELECT tier, count(*) AS purchase_days,
       round(100.0 * avg((stock_before <= 0)::INT), 1) AS pct_arrive_at_zero,
       round(quantile_cont(stock_before / rate28, 0.25), 1) AS p25_cover,
       round(median(stock_before / rate28), 1) AS p50_cover,
       round(quantile_cont(stock_before / rate28, 0.75), 1) AS p75_cover,
       round(100.0 * avg((stock_before / rate28 < 3)::INT), 1) AS pct_lt3d,
       round(100.0 * avg((stock_before / rate28 > 60)::INT), 1) AS pct_gt60d
FROM buys_x WHERE rate28 > 0 GROUP BY tier ORDER BY tier""")

show("Per drug (fast tier): median stock at arrival in units and days of cover", """
SELECT round(quantile_cont(m_units, 0.25)) AS p25_units, round(median(m_units)) AS p50_units,
       round(quantile_cont(m_units, 0.75)) AS p75_units,
       round(quantile_cont(m_cover, 0.25), 1) AS p25_cover, round(median(m_cover), 1) AS p50_cover,
       round(quantile_cont(m_cover, 0.75), 1) AS p75_cover, count(*) AS drugs
FROM (SELECT code, median(stock_before) AS m_units, median(stock_before / rate28) AS m_cover
      FROM buys_x WHERE tier = '1 fast' AND rate28 > 0 GROUP BY code HAVING count(*) >= 5)""")

show("Deliveries by weekday", """
SELECT dayname(d) AS weekday, isodow(d) AS dow, count(DISTINCT (supplier, d)) AS supplier_days,
       count(*) AS lines
FROM led WHERE kind = 'purchase' AND qin > 0 GROUP BY ALL ORDER BY dow""")

show("Which location receives deliveries, by year", f"""
SELECT year(d) AS year, round(100.0 * avg((wh = '{STORE}')::INT), 1) AS pct_lines_to_store,
       count(*) AS lines, sum(qin) AS units
FROM led WHERE kind = 'purchase' AND qin > 0 GROUP BY 1 ORDER BY 1""")

# ---------------------------------------------------------------------------
# 4. Stock-takes and transfers.
# ---------------------------------------------------------------------------
header("4. Stock-takes (shrinkage and gains) and transfers")

# Two mass counts dominate: July-September 2023, when the upstairs store was
# set up and stock was moved by counting it out of the floor and into the
# store, and August 2025, a full count of every drug (inside the holdout).
con.execute(f"""
CREATE TABLE takes AS
SELECT l.*, coalesce(p.price, 0) AS price,
       CASE WHEN d BETWEEN DATE '2023-07-01' AND DATE '2023-09-30' THEN '2023 Q3 store setup'
            WHEN d BETWEEN DATE '2025-08-01' AND DATE '2025-08-31' THEN '2025-08 full count'
            ELSE 'routine' END AS period
FROM led l LEFT JOIN price p USING (code)
WHERE kind IN ('stocktake', 'stocktake_import')
""")
show("Stock-take rows by period and location", """
SELECT period, wh, kind, count(*) AS rows, count(DISTINCT code) AS drugs,
       sum((qin = 0 AND qout = 0)::INT) AS no_change, sum(qin) AS units_up,
       sum(qout) AS units_down, round(sum((qin - qout) * price) / 1e6, 1) AS net_rp_million
FROM takes GROUP BY ALL ORDER BY period, wh, kind""")

show("Store setup: stock-take moves floor -> store, and the drugs with the biggest cuts", f"""
SELECT code, drug, sum(qout) FILTER (WHERE wh = '{FLOOR}') AS floor_down,
       sum(qin) FILTER (WHERE wh = '{STORE}') AS store_up,
       sum(qin - qout) AS net
FROM takes WHERE period = '2023 Q3 store setup' GROUP BY ALL ORDER BY floor_down DESC NULLS LAST LIMIT 8""")

# Routine counts: real adjustments, per drug and day (several rows in a
# second are one count entered in steps).
show("Routine stock-takes: adjustments per drug-day", """
WITH a AS (SELECT code, d, sum(qin - qout) AS adj, arg_min(s0, rn) AS before,
                  any_value(price) AS price
           FROM takes WHERE period = 'routine' GROUP BY ALL)
SELECT count(*) AS drug_days, sum((adj > 0)::INT) AS gains, sum((adj < 0)::INT) AS losses,
       sum((adj = 0)::INT) AS no_change,
       median(abs(adj)) FILTER (WHERE adj <> 0) AS median_abs_adj,
       round(median(abs(adj) / nullif(abs(before), 0)) FILTER (WHERE adj <> 0), 2) AS median_rel_adj,
       sum(adj) FILTER (WHERE adj > 0) AS units_gained, sum(adj) FILTER (WHERE adj < 0) AS units_lost,
       round(sum(adj * price) FILTER (WHERE adj > 0) / 1e6, 1) AS gained_rp_m,
       round(sum(adj * price) FILTER (WHERE adj < 0) / 1e6, 1) AS lost_rp_m,
       sum((before <= 0 AND adj > 0)::INT) AS gains_from_zero_or_neg
FROM a""")

show("Routine stock-takes by year: net value at selling price vs revenue", f"""
WITH a AS (SELECT year(d) AS year, sum((qin - qout) * price) AS net FROM takes
           WHERE period = 'routine' GROUP BY 1),
r AS (SELECT Year AS year, sum(Subtotal) AS rev FROM read_csv('{SALES_PROC}') GROUP BY 1)
SELECT year, round(net / 1e6, 1) AS net_rp_million, round(rev / 1e9, 2) AS revenue_rp_bn,
       round(100 * net / rev, 2) AS pct_of_revenue
FROM a JOIN r USING (year) ORDER BY year""")

show("Transfers by quarter: store -> floor vs floor -> store, store share of stock", f"""
WITH t AS (SELECT date_trunc('quarter', d) AS q,
                  sum(qin) FILTER (WHERE wh = '{FLOOR}') AS store_to_floor,
                  sum(qin) FILTER (WHERE wh = '{STORE}') AS floor_to_store,
                  count(DISTINCT doc) AS docs, sum(test_entry::INT) AS test_rows
           FROM led WHERE kind = 'transfer' GROUP BY 1),
s AS (SELECT date_trunc('quarter', d) AS q, sum(greatest(store, 0)) / count(DISTINCT d) AS store_units,
             sum(greatest(stock, 0)) / count(DISTINCT d) AS total_units
      FROM eod WHERE code NOT IN (SELECT code FROM drugs WHERE cat IN {OUT_OF_SCOPE}) GROUP BY 1)
SELECT strftime(s.q, '%Y-Q') || quarter(s.q) AS quarter, t.docs, t.store_to_floor, t.floor_to_store,
       t.test_rows, round(100 * store_units / total_units, 1) AS pct_stock_in_store
FROM s LEFT JOIN t USING (q) ORDER BY s.q""")

show("Transfer lead: days a fast/regular drug's floor sat empty while the store had stock (spells)", f"""
WITH x AS (SELECT code, d, d - CAST(row_number() OVER (PARTITION BY code ORDER BY d) AS INT) AS grp
           FROM live WHERE floor <= 0 AND store > 0 AND tier IN ('1 fast', '2 regular')),
sp AS (SELECT code, count(*) AS days FROM x GROUP BY code, grp)
SELECT count(*) AS spells, median(days) AS median_days, quantile_cont(days, 0.9) AS p90_days,
       sum(days) AS total_days FROM sp""")

# ---------------------------------------------------------------------------
# 5. Negative stock and huge balances.
# ---------------------------------------------------------------------------
header("5. Negative stock and huge balances")

show("Rows that leave a location negative, by kind", """
SELECT kind, (s0 < 0) AS already_negative, count(*) AS rows, count(DISTINCT code) AS drugs,
       median(s1) AS median_balance
FROM led WHERE s1 < 0 GROUP BY ALL ORDER BY rows DESC""")

# Most negatives are intraday: a sale is rung up at zero on the floor and the
# matching transfer or stock-take is booked minutes later.
show("Sales that take a location negative: next inflow and how soon", """
SELECT coalesce(f.kind, '(none)') AS next_inflow, count(*) AS sales,
       median(datediff('minute', n.ts, f.ts)) AS median_minutes_later
FROM (SELECT rn, ts, code, wh FROM led WHERE s1 < 0 AND kind = 'sale' AND s0 >= 0) n
LEFT JOIN LATERAL (SELECT kind, ts FROM led l2 WHERE l2.code = n.code AND l2.wh = n.wh
                   AND l2.rn > n.rn AND l2.qin > 0 ORDER BY rn LIMIT 1) f ON true
GROUP BY 1 ORDER BY 2 DESC""")
show("Drug-location-days that dip negative vs still negative at close", """
WITH n AS (SELECT DISTINCT code, wh, d FROM led WHERE s1 < 0)
SELECT count(*) AS dip_days, sum((e.stock < 0)::INT) AS negative_at_close
FROM n JOIN eod_rows e USING (code, wh, d)""")

con.execute("""
CREATE TABLE neg AS
WITH x AS (SELECT code, d, stock, d - CAST(row_number() OVER (PARTITION BY code ORDER BY d) AS INT) AS grp
           FROM eod WHERE stock < 0)
SELECT code, min(d) AS start, max(d) AS stop, count(*) AS days, min(stock) AS depth
FROM x GROUP BY code, grp
""")
show("Negative total-stock spells: length and what clears them", """
WITH e AS (SELECT n.code, n.start, arg_max(l.kind, l.qin) AS cleared_by
           FROM neg n JOIN led l ON l.code = n.code AND l.d = n.stop + 1 AND l.qin > 0
           GROUP BY ALL)
SELECT coalesce(cleared_by, '(still negative / no inflow)') AS cleared_by, count(*) AS spells,
       median(days) AS median_days, quantile_cont(days, 0.9) AS p90_days, median(depth) AS median_depth
FROM neg LEFT JOIN e USING (code, start) GROUP BY 1 ORDER BY 2 DESC""")
print(f"\nnegative total-stock drug-days: {one('SELECT sum(days) FROM neg')} "
      f"({pct(one('SELECT sum(days) FROM neg'), one('SELECT count(*) FROM eod'))} of all), "
      f"drugs ever negative: {one('SELECT count(DISTINCT code) FROM neg')}")
open_neg = one(f"SELECT count(*) FROM neg WHERE stop = DATE '{END}'")
print(f"negative spells still open on {END}: {open_neg}")

# Huge balances: stock far above what the drug ever sells.
show("Largest balances vs sales: years of cover at the peak", """
WITH m AS (SELECT code, max(stock) AS peak, arg_max(d, stock) AS peak_day,
                  max(stock) FILTER (WHERE d >= DATE '2023-10-01') AS peak_after_sep23,
                  any_value(stock) FILTER (WHERE d = DATE '2025-06-30') AS on_30jun25
           FROM eod GROUP BY code)
SELECT m.code, d.drug, d.cat, d.unit, m.peak, m.peak_day, m.peak_after_sep23, m.on_30jun25,
       t.units / 3 AS sold_per_year, round(m.peak / (t.units / 3.0), 1) AS years_cover
FROM m JOIN drugs d USING (code) LEFT JOIN tiers t USING (code)
ORDER BY m.peak DESC LIMIT 10""")
show("Drugs with > 3 years of sales on hand at some point (>= 500 units)", """
WITH m AS (SELECT code, max(stock) AS peak,
                  max(stock) FILTER (WHERE d >= DATE '2023-10-01') AS peak_late FROM eod GROUP BY code)
SELECT count(*) AS drugs_ever,
       sum((peak_late > 3 * coalesce(t.units, 0) / 3.0)::INT) AS still_after_sep23
FROM m LEFT JOIN tiers t USING (code)
WHERE peak >= 500 AND peak > 3 * coalesce(t.units, 0) / 3.0""")

# ---------------------------------------------------------------------------
# 6. Baseline numbers for the replay: stock held at selling price.
# ---------------------------------------------------------------------------
header("6. Stock held at selling price (replay baseline)")
show("Mean daily stock value, negatives clipped to 0, in scope, by half-year", """
SELECT year(e.d) || '-H' || (CASE WHEN month(e.d) <= 6 THEN 1 ELSE 2 END) AS half,
       round(sum(greatest(e.stock, 0) * p.price) / count(DISTINCT e.d) / 1e6, 1) AS all_rp_m,
       round(sum(greatest(e.stock, 0) * p.price) FILTER (WHERE t.tier IN ('1 fast', '2 regular'))
             / count(DISTINCT e.d) / 1e6, 1) AS fast_regular_rp_m,
       round(sum(greatest(e.stock, 0)) / count(DISTINCT e.d)) AS all_units
FROM eod e JOIN tiers t USING (code) JOIN price p USING (code)
GROUP BY 1 ORDER BY 1""")
show("Fast + regular drugs, 2025 H2 (the holdout): days out, stock value", """
SELECT t.tier, count(DISTINCT e.code) AS drugs,
       round(100.0 * avg((e.stock <= 0)::INT), 1) AS pct_days_out,
       round(sum(greatest(e.stock, 0) * p.price) / count(DISTINCT e.d) / 1e6, 1) AS mean_value_rp_m,
       round(sum(e.sold * p.price) / 1e6) AS revenue_rp_m_approx
FROM eod e JOIN tiers t USING (code) JOIN price p USING (code)
WHERE e.d >= DATE '2025-07-01' AND t.tier IN ('1 fast', '2 regular') AND t.last_sale >= DATE '2025-07-01'
GROUP BY 1 ORDER BY 1""")

print(f"\ndone in {time.time() - t0:.0f}s")
