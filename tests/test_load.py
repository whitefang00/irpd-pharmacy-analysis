from datetime import date

import pytest

from irpd import warehouse


def rows(db, sql):
    with warehouse.as_of(date(2099, 1, 1), db) as con:
        return con.execute(sql).fetchall()


def test_base_tables_have_every_row(fixture_db):
    assert rows(fixture_db, "SELECT count(*) FROM sales_lines") == [(5,)]
    assert rows(fixture_db, "SELECT count(*) FROM stock_moves") == [(8,)]
    assert rows(fixture_db, "SELECT count(*) FROM drugs") == [(2,)]


def test_moves_get_location_type_and_supplier(fixture_db):
    got = rows(fixture_db, """
        SELECT row_number, location, move_type, supplier FROM stock_moves ORDER BY row_number""")
    assert got[0] == (1, "floor", "purchase", "Antar Mitra Sembada")
    assert got[2] == (3, "store", "transfer", None)
    assert got[7] == (8, "floor", "stock_take", None)
    assert all(r[1] and r[2] for r in got)


def test_ledger_chains_per_location(fixture_db):
    # Opening equals the previous closing for the same drug and location.
    broken = rows(fixture_db, """
        SELECT count(*) FROM (
            SELECT opening, lag(closing) OVER (PARTITION BY drug_code, location
                                               ORDER BY row_number) AS prev
            FROM stock_moves) WHERE prev IS NOT NULL AND opening <> prev""")
    assert broken == [(0,)]


def test_services_are_flagged(fixture_db):
    assert rows(fixture_db, "SELECT drug_code FROM sales_lines WHERE is_service") == [("S1",)]


@pytest.mark.data
def test_real_data_loads_completely(real_db):
    assert rows(real_db, "SELECT count(*) FROM sales_lines") == [(627_389,)]
    assert rows(real_db, "SELECT count(*) FROM stock_moves") == [(775_560,)]
    assert rows(real_db, """
        SELECT count(*) FROM stock_moves WHERE location IS NULL OR move_type IS NULL""") == [(0,)]
