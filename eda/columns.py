"""EDA route 1: columns and quality.

Profiles every column in the four IRPD files, checks what the authors'
processing changed, tests the keys used to join sales to stock, and lists
anomalies. Prints everything; writes nothing.

Run from anywhere:
    uvx --with duckdb --with pandas python -I eda/columns.py [data_dir]
"""

import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "data" / "raw" / "irpd"

FILES = {
    "sales_raw": DATA / "raw_dataset" / "sales_raw.csv",
    "stock_raw": DATA / "raw_dataset" / "stock_raw.csv",
    "sales_proc": DATA / "processed_dataset" / "sales_processed.csv",
    "stock_proc": DATA / "processed_dataset" / "stock_processed.csv",
}

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 70)
pd.set_option("display.max_rows", 120)

con = duckdb.connect()


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


# Load each file twice: as plain text (to see values exactly as written)
# and with DuckDB's type detection (to see what a naive load would give).
t0 = time.time()
for name, path in FILES.items():
    if not path.exists():
        sys.exit(f"missing {path}")
    src = str(path).replace("'", "''")
    con.execute(f"CREATE TABLE {name}_txt AS SELECT * FROM read_csv('{src}', all_varchar=true)")
    con.execute(f"CREATE TABLE {name} AS SELECT * FROM read_csv('{src}')")

# Raw stock with quantities parsed: "9.999 Tab" -> 9999 and unit "Tab".
# Dots are thousand separators; no quantity in the file has a comma.
# Built from the typed table so RowNumber is an integer (text sorts wrong).
# Sale and return rows are found by their description, not by document
# prefix: some supplier invoice numbers also start with "PJ".
con.execute("""
CREATE TABLE stock_raw_p AS
SELECT *,
  strptime(DateTime, '%d %b %Y %H:%M:%S') AS ts,
  CASE WHEN TransactionDescription LIKE 'Penjualan%' THEN 'sale'
       WHEN TransactionDescription LIKE 'Retur%' THEN 'return' ELSE 'other' END AS kind,
  CAST(replace(split_part(InitialStock, ' ', 1), '.', '') AS BIGINT)   AS init_q,
  CAST(replace(split_part(StockIn, ' ', 1), '.', '') AS BIGINT)        AS in_q,
  CAST(replace(split_part(StockOut, ' ', 1), '.', '') AS BIGINT)       AS out_q,
  CAST(replace(split_part(RemainingStock, ' ', 1), '.', '') AS BIGINT) AS rem_q,
  split_part(InitialStock, ' ', 2)   AS unit_init,
  split_part(StockIn, ' ', 2)        AS unit_in,
  split_part(StockOut, ' ', 2)       AS unit_out,
  split_part(RemainingStock, ' ', 2) AS unit_rem
FROM stock_raw
""")
con.execute("""
CREATE TABLE sales_raw_p AS
SELECT *, strptime(DateTime, '%d %b %Y %H:%M:%S') AS ts FROM sales_raw
""")
print(f"Loaded 4 files from {DATA} in {time.time() - t0:.1f}s")

header("1. Column profiles")
# fill = not null, not empty. '-' is the POS placeholder for "none", so it is
# counted separately: a column can be 100% filled and still mostly '-'.
for name in FILES:
    n = one(f"SELECT count(*) FROM {name}")
    types = dict(q(f"DESCRIBE {name}")[["column_name", "column_type"]].values)
    rows = []
    for col, typ in types.items():
        c = f'"{col}"'
        filled, dash, distinct = con.execute(f"""
            SELECT count(*) FILTER (WHERE trim({c}) <> ''),
                   count(*) FILTER (WHERE trim({c}) = '-'),
                   count(DISTINCT {c})
            FROM {name}_txt""").fetchone()
        top = con.execute(f"""
            SELECT {c}, count(*) k FROM {name}_txt GROUP BY 1 ORDER BY k DESC LIMIT 3
        """).fetchall()
        top_s = "; ".join(f"{str(v)[:22]} ({100 * k / n:.0f}%)" for v, k in top)
        if col in ("PatientID", "DoctorID", "SalespersonID"):
            top_s = "(not shown: person codes)"
        rows.append({
            "column": col, "type": typ, "fill": pct(filled, n),
            "dash": pct(dash, n) if dash else "",
            "distinct": distinct, "const": "CONST" if distinct <= 1 else "",
            "top values": top_s,
        })
    print(f"\n-- {name}: {n:,} rows x {len(types)} columns")
    print(pd.DataFrame(rows).to_string(index=False))

show("Value sets of the small categorical columns (raw)", """
SELECT 'sales.Category' col, Category v, count(*) n FROM sales_raw GROUP BY 2
UNION ALL SELECT 'sales.DrugClass', DrugClass, count(*) FROM sales_raw GROUP BY 2
UNION ALL SELECT 'sales.PaymentType', PaymentType, count(*) FROM sales_raw GROUP BY 2
UNION ALL SELECT 'sales.PriceTier', PriceTier, count(*) FROM sales_raw GROUP BY 2
UNION ALL SELECT 'sales.Unit', Unit, count(*) FROM sales_raw GROUP BY 2
UNION ALL SELECT 'stock.Warehouse', Warehouse, count(*) FROM stock_raw GROUP BY 2
UNION ALL SELECT 'stock.DrugType', DrugType, count(*) FROM stock_raw GROUP BY 2
UNION ALL SELECT 'stock.Unit', unit_init, count(*) FROM stock_raw_p GROUP BY 2
ORDER BY col, n DESC
""")

show("Raw sales Shift is '<date>-<shift>-<cashier>'; shift label casing", """
SELECT split_part(Shift, '-', 2) AS shift_label, count(*) n,
       count(*) FILTER (WHERE split_part(Shift, '-', 3) <> CashierID) cashier_mismatch,
       count(*) FILTER (WHERE split_part(Shift, '-', 1) <> strftime(ts, '%Y%m%d')) date_mismatch
FROM sales_raw_p GROUP BY 1 ORDER BY n DESC
""")

header("2. What processing changed")
for kind in ("sales", "stock"):
    raw, proc = f"{kind}_raw", f"{kind}_proc"
    rc = set(q(f"DESCRIBE {raw}").column_name)
    pc = set(q(f"DESCRIBE {proc}").column_name)
    print(f"\n-- {kind}: columns only in raw:       {sorted(rc - pc)}")
    print(f"-- {kind}: columns only in processed: {sorted(pc - rc)}")
    show(f"{kind}: raw rows by Category, and how many survive into processed", f"""
    SELECT r.Category, count(*) raw_rows, count(p.RowNumber) kept,
           count(*) - count(p.RowNumber) dropped
    FROM {raw} r LEFT JOIN {proc} p USING (RowNumber)
    GROUP BY 1 ORDER BY raw_rows DESC""")
    show(f"{kind}: RowNumber as a key", f"""
    SELECT (SELECT count(*) - count(DISTINCT RowNumber) FROM {raw})  raw_dup_rownum,
           (SELECT max(RowNumber) - min(RowNumber) + 1 - count(*) FROM {raw}) raw_gaps,
           (SELECT count(*) - count(DISTINCT RowNumber) FROM {proc}) proc_dup_rownum,
           (SELECT count(*) FROM {proc} p ANTI JOIN {raw} r USING (RowNumber)) proc_not_in_raw""")

# Shared columns must carry the same value on matched rows.
for kind, cols in {
    "sales": ["InvoiceNumber", "CashierID", "DrugCode", "DrugName", "Category", "DrugClass",
              "Manufacturer", "Quantity", "Unit", "UnitPrice", "DiscountAmount", "ServiceFee",
              "PackagingFee", "Subtotal", "PaymentType", "PrintCount", "TotalPrice"],
    "stock": ["DocumentNumber", "DrugCode", "DrugName", "Category", "DrugClass", "DrugType",
              "Status"],
}.items():
    diffs = " UNION ALL ".join(
        f"SELECT '{c}' col, count(*) FILTER (WHERE r.\"{c}\" IS DISTINCT FROM p.\"{c}\") n_diff "
        f"FROM {kind}_raw_txt r JOIN {kind}_proc_txt p USING (RowNumber)" for c in cols)
    df = q(diffs)
    bad = df[df.n_diff > 0]
    print(f"\n-- {kind}: shared columns identical on all matched rows except: "
          f"{'none' if bad.empty else bad.to_dict('records')}")

show("sales: derived columns vs raw (mismatch counts on matched rows)", """
SELECT
  count(*) FILTER (WHERE p.Date <> CAST(r.ts AS DATE))                      date_diff,
  count(*) FILTER (WHERE p.Time <> CAST(r.ts AS TIME))                      time_diff,
  count(*) FILTER (WHERE p.Hour <> hour(r.ts))                              hour_diff,
  count(*) FILTER (WHERE p.ShiftPeriod <> upper(split_part(r.Shift, '-', 2))) shift_diff,
  count(*) FILTER (WHERE p.PriceTier <> CAST(right(r.PriceTier, 1) AS INT)) tier_diff,
  count(*) FILTER (WHERE p.ExpiryDate IS DISTINCT FROM
        try_cast(regexp_extract(r.BatchAndExpiryDate, 'ED : ([0-9-]+)', 1) AS DATE)) expiry_diff
FROM sales_raw_p r JOIN sales_proc p USING (RowNumber)
""")
# A line can draw from several batches: "B1 (ED : d1), B2 (ED : d2)".
# Processed keeps only the first batch and its expiry.
show("sales: batch field holding more than one batch (processed keeps the first)", """
SELECT count(*) FILTER (WHERE len(regexp_extract_all(BatchAndExpiryDate, '\\(ED')) > 1)
         multi_batch_lines,
       count(*) FILTER (WHERE BatchAndExpiryDate IS NULL) no_batch_field,
       count(*) FILTER (WHERE trim(BatchAndExpiryDate) LIKE 'SW (ED%') placeholder_batch_SW,
       count(*) FILTER (WHERE trim(BatchAndExpiryDate) LIKE 'SW (ED%'
                          AND BatchAndExpiryDate LIKE '%-01-01)%') SW_with_jan1_expiry
FROM sales_raw
""")

show("stock: parsed raw quantities vs processed (mismatch counts on matched rows)", """
SELECT
  count(*) FILTER (WHERE p.InitialStock <> r.init_q)   init_diff,
  count(*) FILTER (WHERE p.StockIn <> r.in_q)          in_diff,
  count(*) FILTER (WHERE p.StockOut <> r.out_q)        out_diff,
  count(*) FILTER (WHERE p.RemainingStock <> r.rem_q)  rem_diff,
  count(*) FILTER (WHERE p.Unit <> r.unit_init)        unit_diff,
  count(*) FILTER (WHERE r.unit_init <> r.unit_in OR r.unit_in <> r.unit_out
                      OR r.unit_out <> r.unit_rem)     raw_units_disagree_in_row,
  count(*) FILTER (WHERE p.Date <> CAST(r.ts AS DATE) OR p.Time <> CAST(r.ts AS TIME)) ts_diff
FROM stock_raw_p r JOIN stock_proc p USING (RowNumber)
""")

show("stock: how free-text TransactionDescription became TransactionType", """
SELECT CASE WHEN p.TransactionType LIKE 'Pembelian%' THEN 'Pembelian (supplier)'
            ELSE p.TransactionType END AS TransactionType,
       count(*) n, count(DISTINCT r.TransactionDescription) raw_variants,
       count(DISTINCT regexp_extract(r.DocumentNumber, '^[A-Za-z]*', 0)) doc_prefixes
FROM stock_raw r JOIN stock_proc p USING (RowNumber)
GROUP BY 1 ORDER BY n DESC
""")
print("\n-- stock: distinct purchase suppliers:",
      one("SELECT count(DISTINCT TransactionType) FROM stock_proc "
          "WHERE TransactionType LIKE 'Pembelian%'"))

header("3. Keys and joins")
show("Sales InvoiceNumber: format and grain", """
SELECT count(*) lines, count(DISTINCT InvoiceNumber) invoices,
       count(*) FILTER (WHERE NOT regexp_full_match(InvoiceNumber, 'PJ[0-9]{10}')) odd_format_PJM,
       count(*) FILTER (WHERE regexp_full_match(InvoiceNumber, 'PJ[0-9]{10}')
                          AND substr(InvoiceNumber, 3, 6) <> strftime(ts, '%y%m%d'))
         date_not_in_number,
       (SELECT count(*) FROM (SELECT InvoiceNumber FROM sales_raw_p GROUP BY 1
          HAVING count(DISTINCT ts) > 1)) invoices_with_2plus_timestamps,
       (SELECT count(*) FROM (SELECT InvoiceNumber FROM sales_raw_p GROUP BY 1
          HAVING count(DISTINCT CashierID) > 1)) invoices_with_2plus_cashiers,
       (SELECT count(*) FROM (SELECT InvoiceNumber, DrugCode FROM sales_raw GROUP BY 1, 2
          HAVING count(*) > 1)) invoice_drug_pairs_repeated
FROM sales_raw_p
""")
show("Shift field date differs from the sale date (shift left open, or stale)", """
SELECT split_part(Shift, '-', 1) shift_date, count(*) lines,
       min(CAST(ts AS DATE)) first_sale, max(CAST(ts AS DATE)) last_sale
FROM sales_raw_p WHERE split_part(Shift, '-', 1) <> strftime(ts, '%Y%m%d')
GROUP BY 1 ORDER BY lines DESC
""")
show("ShiftPeriod is the shift label, not the clock: hours seen per label", """
SELECT ShiftPeriod, min(Hour) min_hour, max(Hour) max_hour,
       count(*) FILTER (WHERE Hour < 12) before_noon, count(*) FILTER (WHERE Hour >= 16) after_4pm
FROM sales_proc GROUP BY 1
""")

show("Stock row kinds (raw), by description; note 'PJ' document prefixes on purchases", """
SELECT kind, count(*) n, count(*) FILTER (WHERE DocumentNumber LIKE 'PJ%') doc_starts_PJ,
       count(*) FILTER (WHERE Warehouse = 'Gudang Obat 1') in_gudang
FROM stock_raw_p GROUP BY 1 ORDER BY n DESC
""")
show("Stock DocumentNumber prefix by processed TransactionType (top 12)", """
SELECT regexp_extract(r.DocumentNumber, '^[A-Za-z]*', 0) prefix,
       CASE WHEN p.TransactionType LIKE 'Pembelian%' THEN 'Pembelian'
            ELSE p.TransactionType END ttype,
       count(*) n
FROM stock_raw r LEFT JOIN stock_proc p USING (RowNumber)
GROUP BY 1, 2 ORDER BY n DESC LIMIT 12
""")

# Line-level join: sales lines <-> stock sale rows on (invoice, drug).
con.execute("""
CREATE TABLE s_lines AS
SELECT InvoiceNumber inv, DrugCode, any_value(Unit) unit, count(DISTINCT Unit) n_units,
       sum(Quantity) qty, any_value(Category) cat
FROM sales_raw GROUP BY 1, 2
""")
con.execute("""
CREATE TABLE k_lines AS
SELECT DocumentNumber inv, DrugCode, any_value(unit_out) unit, sum(out_q) qty,
       bool_or(Warehouse = 'Gudang Obat 1') gudang
FROM stock_raw_p WHERE kind = 'sale' GROUP BY 1, 2
""")
con.execute("""
CREATE TABLE r_lines AS
SELECT DocumentNumber inv, DrugCode, sum(in_q) qty
FROM stock_raw_p WHERE kind = 'return' GROUP BY 1, 2
""")
show("Sales lines vs stock sale rows on (InvoiceNumber = DocumentNumber, DrugCode), raw", """
SELECT count(*) FILTER (WHERE s.inv IS NOT NULL AND k.inv IS NOT NULL) AS n_matched,
       count(*) FILTER (WHERE k.inv IS NULL) sales_only,
       count(*) FILTER (WHERE s.inv IS NULL) stock_only,
       count(*) FILTER (WHERE s.unit = k.unit AND s.qty = k.qty) same_unit_same_qty,
       count(*) FILTER (WHERE s.unit = k.unit AND s.qty <> k.qty) same_unit_qty_differs,
       count(*) FILTER (WHERE s.unit <> k.unit) sales_in_pack_unit,
       count(*) FILTER (WHERE k.gudang) stock_side_in_gudang
FROM s_lines s FULL JOIN k_lines k USING (inv, DrugCode)
""")
show("Sales-only lines: by category (Jasa = services, never stocked)", """
SELECT s.cat, count(*) n, sum(s.qty) units
FROM s_lines s ANTI JOIN k_lines k USING (inv, DrugCode)
GROUP BY 1 ORDER BY n DESC LIMIT 8
""")
show("Stock-only sale rows (sold per ledger, no sales line)", """
SELECT year(r.ts) yr, r.Category, count(*) n, sum(r.out_q) units_out
FROM stock_raw_p r ANTI JOIN s_lines s ON s.inv = r.DocumentNumber AND s.DrugCode = r.DrugCode
WHERE r.kind = 'sale' GROUP BY 1, 2 ORDER BY n DESC LIMIT 8
""")
show("Returns: ledger 'Retur' rows tied to a sales invoice; the sales file keeps the line", """
SELECT count(*) return_lines, sum(x.qty) units_returned,
       count(s.inv) matched_to_sales_line,
       sum(CASE WHEN s.unit = k.unit THEN x.qty END) units_returned_same_unit
FROM r_lines x LEFT JOIN s_lines s USING (inv, DrugCode) LEFT JOIN k_lines k USING (inv, DrugCode)
""")
# Sales Quantity is in the unit sold (Strip, Box) while the ledger is in the
# base unit (Tab, Sachet). The ratio gives the pack size.
show("Pack sizes implied by sales unit vs ledger unit", """
WITH m AS (SELECT DrugCode, s.unit su, k.unit ku, k.qty / s.qty f
           FROM s_lines s JOIN k_lines k USING (inv, DrugCode)
           LEFT JOIN r_lines x USING (inv, DrugCode)
           WHERE s.unit <> k.unit AND s.n_units = 1 AND x.inv IS NULL)
SELECT count(DISTINCT DrugCode) codes, count(DISTINCT (DrugCode, su)) code_unit_pairs,
       count(*) lines,
       (SELECT count(*) FROM (SELECT DrugCode, su FROM m GROUP BY 1, 2
                              HAVING count(DISTINCT f) > 1))
         pairs_with_varying_factor
FROM m
""")
show("Most common pack conversions", """
SELECT s.unit sales_unit, k.unit ledger_unit, count(*) lines, median(k.qty / s.qty) median_factor
FROM s_lines s JOIN k_lines k USING (inv, DrugCode)
WHERE s.unit <> k.unit AND s.n_units = 1 GROUP BY 1, 2 ORDER BY lines DESC LIMIT 8
""")
show("Units: sales Quantity total vs ledger sale StockOut total", """
SELECT (SELECT sum(Quantity) FROM sales_raw) sales_qty_raw,
       (SELECT sum(out_q) FROM stock_raw_p WHERE kind = 'sale') ledger_sale_out_raw,
       (SELECT sum(Quantity) FROM sales_proc) sales_qty_proc,
       (SELECT sum(StockOut) FROM stock_proc WHERE TransactionType = 'Penjualan')
         ledger_sale_out_proc
""")

show("DrugCode coverage across files", """
WITH s AS (SELECT DISTINCT DrugCode FROM sales_raw), k AS (SELECT DISTINCT DrugCode FROM stock_raw)
SELECT (SELECT count(*) FROM s) sales_codes, (SELECT count(*) FROM k) stock_codes,
       (SELECT count(*) FROM s ANTI JOIN k USING (DrugCode)) sales_not_in_stock,
       (SELECT count(*) FROM k ANTI JOIN s USING (DrugCode)) stock_never_sold,
       (SELECT count(DISTINCT DrugCode) FROM sales_proc) sales_proc_codes,
       (SELECT count(DISTINCT DrugCode) FROM stock_proc) stock_proc_codes
""")
show("DrugCode formats (stock raw)", """
SELECT regexp_replace(DrugCode, '[0-9]', '9', 'g') pattern, count(DISTINCT DrugCode) codes
FROM stock_raw GROUP BY 1 ORDER BY codes DESC
""")
show("Sales codes missing from the stock ledger", """
SELECT Category, count(DISTINCT DrugCode) codes, count(*) lines
FROM sales_raw s ANTI JOIN (SELECT DISTINCT DrugCode FROM stock_raw) k USING (DrugCode)
GROUP BY 1 ORDER BY lines DESC
""")

# How many codes carry more than one value of an attribute, within and across files.
attrs = []
for src, cols in {"sales_raw": ["DrugName", "Unit", "Category", "DrugClass", "Manufacturer"],
                  "stock_raw": ["DrugName", "Category", "DrugClass", "DrugType"]}.items():
    for c in cols:
        attrs.append(f"""SELECT '{src}' file, '{c}' attr,
            count(*) FILTER (WHERE k > 1) codes_with_2plus, count(*) codes
            FROM (SELECT DrugCode, count(DISTINCT "{c}") k FROM {src} GROUP BY 1)""")
attrs.append("""SELECT 'stock_raw', 'Unit', count(*) FILTER (WHERE k > 1), count(*)
    FROM (SELECT DrugCode, count(DISTINCT unit_init) k FROM stock_raw_p GROUP BY 1)""")
show("Attributes that change within a DrugCode", " UNION ALL ".join(attrs))

# Compare each code's most common value of an attribute in sales vs stock.
con.execute("""
CREATE TABLE s_mode AS SELECT DrugCode, mode(DrugName) AS nm, mode(Unit) unit, mode(Category) cat,
    mode(DrugClass) cls FROM sales_raw GROUP BY 1""")
con.execute("""
CREATE TABLE k_mode AS SELECT DrugCode, mode(DrugName) AS nm, mode(unit_init) unit,
    mode(Category) cat,
    mode(DrugClass) cls FROM stock_raw_p GROUP BY 1""")
show("Sales vs stock: most common attribute per shared DrugCode disagrees", """
SELECT count(*) shared_codes,
       count(*) FILTER (WHERE s.nm <> k.nm) name_diff,
       count(*) FILTER (WHERE s.unit <> k.unit) unit_diff,
       count(*) FILTER (WHERE s.cat <> k.cat) category_diff,
       count(*) FILTER (WHERE s.cls <> k.cls) class_diff
FROM s_mode s JOIN k_mode k USING (DrugCode)
""")
show("Examples: sales unit differs from stock unit for the same code", """
SELECT DrugCode, s.nm AS drug_name, s.unit sales_unit, k.unit stock_unit
FROM s_mode s JOIN k_mode k USING (DrugCode) WHERE s.unit <> k.unit ORDER BY DrugCode LIMIT 8
""")
show("Same DrugName under several DrugCodes", """
SELECT DrugName, count(DISTINCT DrugCode) codes, string_agg(DISTINCT DrugCode, ', ') drug_codes
FROM sales_raw GROUP BY 1 HAVING codes > 1
""")
show("Drug names of 3 characters or fewer (unusable for ingredient mapping)", """
SELECT DrugCode, DrugName, Category, mode(Unit) unit, count(*) lines, sum(Quantity) units,
       sum(Subtotal) revenue
FROM sales_raw WHERE length(DrugName) <= 3 GROUP BY ALL ORDER BY lines DESC
""")

header("4. Anomalies")
for name, excl in [("sales_raw", "RowNumber"), ("stock_raw", "RowNumber")]:
    cols = [c for c in q(f"DESCRIBE {name}").column_name if c != excl]
    collist = ", ".join(f'"{c}"' for c in cols)
    dup = one(f"SELECT coalesce(sum(k - 1), 0) "
              f"FROM (SELECT count(*) k FROM {name} GROUP BY {collist})")
    print(f"\n-- {name}: rows identical to another row on every column but RowNumber: {dup:,}")
show("Sales: the same drug + batch scanned twice on one invoice (looser duplicate)", """
SELECT count(*) AS n_groups, sum(k) AS rows_in_groups FROM (
  SELECT count(*) k FROM sales_raw
  GROUP BY InvoiceNumber, DrugCode, BatchAndExpiryDate HAVING count(*) > 1)
""")

show("Sales: quantity, price and money oddities (raw)", """
SELECT count(*) FILTER (WHERE Quantity <= 0)                    qty_le_0,
       count(*) FILTER (WHERE Quantity <> floor(Quantity))      qty_fractional,
       max(Quantity)                                            qty_max,
       count(*) FILTER (WHERE UnitPrice = 0)                    price_0,
       count(*) FILTER (WHERE UnitPrice < 100 AND UnitPrice > 0) price_under_100,
       max(UnitPrice)                                           price_max,
       count(*) FILTER (WHERE Subtotal <= 0)                    subtotal_le_0,
       count(*) FILTER (WHERE Subtotal <> Quantity * UnitPrice - DiscountAmount
                                + ServiceFee + PackagingFee) subtotal_formula_off,
       count(*) FILTER (WHERE DiscountAmount > 0)               has_discount,
       count(*) FILTER (WHERE ServiceFee > 0 OR PackagingFee > 0) has_fee
FROM sales_raw
""")
show("Sales: TotalPrice = Subtotal less the invoice-level discount, per line", """
SELECT count(*) FILTER (WHERE TotalPrice = Subtotal) total_eq_subtotal,
       count(*) FILTER (WHERE TotalPrice <> Subtotal) total_ne_subtotal,
       count(*) FILTER (WHERE TotalPrice <> Subtotal AND InvoiceDiscountPct = '0%')
         ne_without_invoice_discount,
       sum(Subtotal) - sum(TotalPrice) rupiah_gap
FROM sales_raw
""")
show("Sales: most extreme prices (top 8 by UnitPrice)", """
SELECT DrugCode, DrugName, Category, Unit, Quantity, UnitPrice, Subtotal
FROM sales_raw ORDER BY UnitPrice DESC LIMIT 8
""")
show("Sales: UnitPrice spread max/min > 10, per code vs per (code, unit)", """
SELECT (SELECT count(*) FROM (SELECT DrugCode FROM sales_raw GROUP BY 1
          HAVING max(UnitPrice) > 10 * min(UnitPrice))) codes,
       (SELECT count(*) FROM (SELECT DrugCode FROM sales_raw GROUP BY 1
          HAVING max(UnitPrice) > 10 * min(UnitPrice) AND count(DISTINCT Unit) > 1))
         codes_sold_in_2plus_units,
       (SELECT count(*) FROM (SELECT DrugCode, Unit FROM sales_raw GROUP BY 1, 2
          HAVING max(UnitPrice) > 10 * min(UnitPrice))) code_unit_pairs
""")
show("Sales: dates, hours and expiry", """
SELECT min(ts) first_ts, max(ts) last_ts,
       count(DISTINCT CAST(ts AS DATE)) AS n_days,
       min(hour(ts)) min_hour, max(hour(ts)) max_hour,
       count(*) FILTER (WHERE hour(ts) < 7 OR hour(ts) >= 22) outside_7_22,
       count(*) FILTER (WHERE p.ExpiryDate < p.Date) sold_after_expiry,
       count(*) FILTER (WHERE p.ExpiryDate < p.Date AND p.BatchNumber <> 'SW')
         after_expiry_real_batch,
       count(*) FILTER (WHERE p.ExpiryDate IS NULL) no_expiry,
       count(*) FILTER (WHERE year(p.ExpiryDate) > 2040) expiry_after_2040
FROM sales_raw_p r LEFT JOIN sales_proc p USING (RowNumber)
""")
show("Sales: lines by hour", """
SELECT hour(ts) h, count(*) n FROM sales_raw_p GROUP BY 1 ORDER BY 1
""")
show("Sales: RowNumber order vs time order", """
SELECT count(*) FILTER (WHERE ts < prev) steps_back_in_time, max(prev - ts) biggest_step_back FROM
  (SELECT ts, lag(ts) OVER (ORDER BY RowNumber) prev FROM sales_raw_p)
""")

show("Stock: dates and hours", """
SELECT min(ts) first_ts, max(ts) last_ts, count(DISTINCT CAST(ts AS DATE)) AS n_days,
       count(*) FILTER (WHERE hour(ts) < 7 OR hour(ts) >= 22) outside_7_22
FROM stock_raw_p
""")
show("Stock: negative and huge balances (raw, per row)", """
SELECT count(*) FILTER (WHERE rem_q < 0) rem_negative,
       count(DISTINCT DrugCode) FILTER (WHERE rem_q < 0) codes_negative,
       min(rem_q) min_rem,
       count(*) FILTER (WHERE in_q < 0 OR out_q < 0) negative_in_or_out,
       count(*) FILTER (WHERE in_q > 0 AND out_q > 0) in_and_out_same_row,
       count(*) FILTER (WHERE in_q = 0 AND out_q = 0) no_movement,
       count(*) FILTER (WHERE init_q + in_q - out_q <> rem_q) balance_broken,
       count(*) FILTER (WHERE rem_q >= 10000) rem_ge_10k,
       count(DISTINCT DrugCode) FILTER (WHERE rem_q >= 10000) codes_ge_10k,
       max(rem_q) max_rem
FROM stock_raw_p
""")
show("Stock: biggest balances", """
SELECT DrugCode, DrugName, unit_init unit, max(rem_q) max_rem, count(*) rows_ge_10k,
       mode(TransactionDescription) typical_descr
FROM stock_raw_p WHERE rem_q >= 10000 GROUP BY ALL ORDER BY max_rem DESC LIMIT 8
""")
show("Stock: biggest single movements", """
SELECT DrugCode, DrugName, unit_init unit, in_q, out_q, left(TransactionDescription, 50) descr
FROM stock_raw_p ORDER BY greatest(in_q, out_q) DESC LIMIT 8
""")
show("Stock: running balance across rows, per (DrugCode, Warehouse) and per DrugCode only", """
WITH w AS (SELECT init_q,
                  lag(rem_q) OVER (PARTITION BY DrugCode, Warehouse ORDER BY RowNumber) prev_wh,
                  lag(rem_q) OVER (PARTITION BY DrugCode ORDER BY RowNumber) prev_code
           FROM stock_raw_p)
SELECT count(*) FILTER (WHERE prev_wh IS NOT NULL AND init_q <> prev_wh) breaks_per_location,
       count(*) FILTER (WHERE prev_code IS NOT NULL AND init_q <> prev_code) breaks_per_code_only
FROM w
""")
show("Stock: where the per-location chain breaks (by transaction kind)", """
WITH w AS (SELECT *, lag(rem_q) OVER (PARTITION BY DrugCode, Warehouse ORDER BY RowNumber) prev,
                  lag(ts) OVER (PARTITION BY DrugCode, Warehouse ORDER BY RowNumber) prev_ts
           FROM stock_raw_p)
SELECT CASE WHEN TransactionDescription ILIKE '%stokopname%' THEN 'stock-take' ELSE kind END kind,
       count(*) breaks, count(*) FILTER (WHERE ts = prev_ts) same_second_as_previous_row,
       count(DISTINCT DrugCode) codes
FROM w WHERE prev IS NOT NULL AND init_q <> prev GROUP BY 1 ORDER BY breaks DESC
""")
show("Stock: processed file balance chain (locations merged)", """
SELECT count(*) FILTER (WHERE prev IS NOT NULL AND InitialStock <> prev) breaks,
       count(*) FILTER (WHERE prev IS NOT NULL AND InitialStock <> prev
                          AND TransactionType = 'Mutasi') breaks_on_mutasi
FROM (SELECT InitialStock, TransactionType,
             lag(RemainingStock) OVER (PARTITION BY DrugCode ORDER BY RowNumber) prev
      FROM stock_proc)
""")
show("Stock: 'Test Entry' rows", """
SELECT count(*) n, count(DISTINCT p.DrugCode) codes,
       min(p.Date) AS first_date, max(p.Date) AS last_date,
       sum(r.in_q) units_in, sum(r.out_q) units_out,
       mode(left(r.DocumentNumber, 2)) doc_prefix, mode(r.TransactionDescription) typical_descr
FROM stock_proc p JOIN stock_raw_p r USING (RowNumber) WHERE p.TransactionType = 'Test Entry'
""")
show("Stock: system/setup rows (bulk imports; mostly no movement)", """
SELECT TransactionType, count(*) n, min(Date) AS first_date, max(Date) AS last_date,
       sum(StockIn) units_in, sum(StockOut) units_out
FROM stock_proc
WHERE TransactionType IN ('Import Data Obat', 'Tambah Master Obat', 'Import Stokopname')
GROUP BY 1
""")
show("Odd units: units used by fewer than 1,000 rows (raw)", """
SELECT 'sales' f, Unit, count(*) n, count(DISTINCT DrugCode) codes
FROM sales_raw GROUP BY 2 HAVING n < 1000
UNION ALL
SELECT 'stock', unit_init, count(*), count(DISTINCT DrugCode)
FROM stock_raw_p GROUP BY 2 HAVING count(*) < 1000
ORDER BY f, n
""")

print(f"\nDone in {time.time() - t0:.1f}s")
