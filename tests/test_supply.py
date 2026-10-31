from datetime import date
from pathlib import Path

import duckdb
import pytest

from irpd import config, warehouse
from irpd.load import load
from irpd.tables import supply

# Two drugs. A1: deliveries either side of the cutoff, a transfer, a sale, an
# invoice deleted before the cutoff (PB3) and one deleted after it (PB1).
# B1: the July 2023 'test' transfer and a store-setup stock-take.
FIXTURES = Path(__file__).parent / "fixtures/supply"


@pytest.fixture(scope="module")
def db(tmp_path_factory) -> Path:
    return warehouse.build(FIXTURES, tmp_path_factory.mktemp("wh") / "supply.duckdb")


def rows(db, sql):
    with warehouse.as_of(date(2099, 1, 1), db) as con:
        return con.execute(sql).fetchall()


def test_purchases_drop_deleted_invoices(db):
    got = rows(db, "SELECT document, qty FROM purchases ORDER BY row_number")
    assert got == [("PB2", 20), ("PB2", 5), ("PB4", 100)]


def test_purchase_stock_before_is_floor_plus_store(db):
    # PB2: floor 2 + store 10, and both invoice lines get the stock before the first.
    # PB4: floor 27 + store 7 (after the store count took 3 off).
    got = rows(db, "SELECT document, stock_before FROM purchases ORDER BY row_number")
    assert got == [("PB2", 12), ("PB2", 12), ("PB4", 34)]


def test_transfers_pair_out_and_in_rows(db):
    got = rows(db, """
        SELECT date, drug_code, from_location, to_location, qty, paired
        FROM transfers ORDER BY date""")
    assert got == [
        (date(2023, 7, 6), "B1", "floor", "store", 1, True),  # a 'test' row, still a move
        (date(2025, 5, 1), "A1", "floor", "store", 10, True),
    ]


def test_stock_take_adjustment_sign_and_kind(db):
    got = rows(db, """
        SELECT date, drug_code, location, adjustment, kind
        FROM stock_takes ORDER BY date, location""")
    assert got == [
        (date(2023, 8, 1), "B1", "floor", -2, "store_setup"),
        (date(2023, 8, 1), "B1", "store", 2, "store_setup"),
        (date(2025, 5, 25), "A1", "store", -3, "routine"),
        (date(2025, 8, 10), "A1", "floor", 5, "full_count"),
    ]


def test_baseline_values(db):
    got = rows(db, """
        SELECT drug_code, deliveries, last_delivery, median_gap_days, median_order_qty,
               median_cover_days, share_at_zero, n_suppliers, main_suppliers
        FROM pharmacist_baseline""")
    # PB1 (0 on hand, no sales yet, so no cover) and PB2 (12 on hand, 28 sold
    # in the last 28 days: 12 days of cover). PB1 still counts: it was only
    # deleted after the cutoff. PB3 was deleted before it, PB4 is after it.
    assert got == [("A1", 2, date(2025, 5, 20), 19, 32.5, 12.0, 0.5, 2, ["Sup Y", "Sup X"])]
    assert rows(db, "SELECT supplier, deliveries FROM supplier_cadence ORDER BY supplier") == [
        ("Sup X", 1), ("Sup Y", 1)]


def test_baseline_ignores_everything_after_cutoff():
    def baseline(con):
        supply.build(con)
        return [con.execute(f"SELECT * FROM {t} ORDER BY ALL").fetchall()
                for t in ("pharmacist_baseline", "supplier_cadence")]

    con = duckdb.connect()
    load(con, FIXTURES)
    full = baseline(con)
    con.execute(f"DELETE FROM stock_moves WHERE date > DATE '{config.CUTOFF}'")
    assert baseline(con) == full


def test_baseline_tables_have_no_date_column():
    assert supply.TABLES["pharmacist_baseline"] is None
    assert supply.TABLES["supplier_cadence"] is None


TIERS = """
    WITH s AS (SELECT drug_code, min(date) AS f, max(date) AS l, count(DISTINCT date) AS n
               FROM stock_moves WHERE move_type = 'sale' AND date <= DATE '{cutoff}'
               GROUP BY 1)
    SELECT drug_code,
           CASE WHEN l - f + 1 < 28 OR n < 6 THEN 'rare'
                WHEN n * 365.0 / (l - f + 1) >= 156 THEN 'fast'
                WHEN n * 365.0 / (l - f + 1) >= 52 THEN 'regular'
                ELSE 'slow' END AS tier
    FROM s"""


@pytest.mark.data
def test_real_purchases_and_transfers(real_db):
    # 46,601 delivery lines, minus 53 cancelled by a later deletion.
    assert rows(real_db, "SELECT count(*) FROM purchases") == [(46_548,)]
    # Every transfer pairs up except one store row with no partner.
    assert rows(real_db, "SELECT count(*) FILTER (WHERE NOT paired) FROM transfers") == [(1,)]
    assert rows(real_db, "SELECT count(*) FROM transfers") == [(34_321,)]


@pytest.mark.data
def test_real_stock_take_kinds(real_db):
    got = dict(rows(real_db, "SELECT kind, sum(adjustment) FROM stock_takes GROUP BY kind"))
    assert set(got) == {"routine", "store_setup", "full_count"}
    assert got["store_setup"] < -700_000  # the floor was counted down into the store
    assert abs(got["routine"]) < 50_000  # routine counts barely move stock


@pytest.mark.data
def test_real_baseline_matches_eda(real_db):
    tiers = TIERS.format(cutoff=config.CUTOFF)
    (cover, zero, gap), = rows(real_db, f"""
        SELECT median(median_cover_days), avg(share_at_zero), median(median_gap_days)
        FROM pharmacist_baseline JOIN ({tiers}) USING (drug_code)
        WHERE tier = 'fast' AND deliveries >= 5""")
    assert 14 <= cover <= 24  # EDA: about 18 days of cover when a delivery lands
    assert zero < 0.05  # EDA: only 2% arrive at zero
    assert 10 <= gap <= 21  # EDA: reordered about every 14 days
    (last,), = rows(real_db, "SELECT max(last_delivery) FROM pharmacist_baseline")
    assert last <= config.CUTOFF
