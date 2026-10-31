"""Load the IRPD files into the two base tables everything else builds on.

- `sales_lines`: one row per receipt line, from the processed sales file.
- `stock_moves`: one row per stock movement, from the processed stock file,
  plus the storage location from the raw file. The processed file drops it,
  which merges the shop floor and the upstairs store and breaks the running
  balance on transfers. Rows match one to one on RowNumber.

The processed files are the raw files minus three categories (medical
equipment, baby, drinks), with numbers and dates already parsed.
"""

from pathlib import Path

import duckdb

# Indonesian transaction types in the stock file -> short English names.
MOVE_TYPES = {
    "Penjualan": "sale",
    "Pembelian": "purchase",
    "Mutasi": "transfer",
    "Stokopname": "stock_take",
    "Import Stokopname": "stock_take_import",
    "Retur": "return",
    "Import Data Obat": "import",
    "Tambah Master Obat": "new_item",
    "Test Entry": "test",
}
LOCATIONS = {"Apotek": "floor", "Gudang Obat 1": "store"}


def _csv(path: Path, **opts: str) -> str:
    extra = "".join(f", {k}={v}" for k, v in opts.items())
    return f"read_csv_auto('{path}', sample_size=-1{extra})"


def load(con: duckdb.DuckDBPyConnection, data: Path) -> None:
    sales = data / "processed_dataset/sales_processed.csv"
    stock = data / "processed_dataset/stock_processed.csv"
    stock_raw = data / "raw_dataset/stock_raw.csv"

    con.execute(f"""
        CREATE OR REPLACE TABLE sales_lines AS
        SELECT
            RowNumber            AS row_number,
            Date                 AS date,
            Date + Time          AS ts,
            InvoiceNumber        AS invoice,
            CashierID            AS cashier,
            DrugCode             AS drug_code,
            DrugName             AS drug_name,
            Category             AS category,
            Category = 'Jasa'    AS is_service,
            Quantity             AS qty,
            Unit                 AS unit,
            UnitPrice            AS unit_price,
            DiscountAmount       AS discount,
            Subtotal             AS revenue,
            PaymentType          AS payment_type
        FROM {_csv(sales)}
        ORDER BY row_number""")

    types = " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in MOVE_TYPES.items())
    places = " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in LOCATIONS.items())
    con.execute(f"""
        CREATE OR REPLACE TABLE stock_moves AS
        SELECT
            p.RowNumber                                  AS row_number,
            p.Date                                       AS date,
            p.Date + p.Time                              AS ts,
            CASE r.Warehouse {places} END                AS location,
            p.DrugCode                                   AS drug_code,
            p.DrugName                                   AS drug_name,
            p.Category                                   AS category,
            p.DocumentNumber                             AS document,
            CASE split_part(p.TransactionType, ' (', 1) {types} END AS move_type,
            nullif(regexp_extract(p.TransactionType, '\\((.*)\\)$', 1), '') AS supplier,
            p.InitialStock                               AS opening,
            p.StockIn                                    AS qty_in,
            p.StockOut                                   AS qty_out,
            p.RemainingStock                             AS closing,
            p.Unit                                       AS unit
        FROM {_csv(stock)} p
        JOIN {_csv(stock_raw, all_varchar="true")} r
          ON r.RowNumber::BIGINT = p.RowNumber AND r.DrugCode = p.DrugCode
        ORDER BY row_number""")

    con.execute(f"CREATE OR REPLACE TABLE drugs AS {DRUGS_SQL}")


# One row per drug: stable attributes (one name, category and unit per code).
# `warehouse.as_of` reruns it on the filtered ledger, so `last_seen` and drugs
# that first appear later never leak.
DRUGS_SQL = """
    SELECT drug_code,
           any_value(drug_name) AS drug_name,
           any_value(category)  AS category,
           any_value(unit)      AS stock_unit,
           min(date)            AS first_seen,
           max(date)            AS last_seen
    FROM stock_moves
    GROUP BY drug_code
    ORDER BY drug_code"""
