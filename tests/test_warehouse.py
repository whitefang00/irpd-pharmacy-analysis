from datetime import date

from irpd import warehouse


def test_as_of_hides_later_rows(fixture_db):
    with warehouse.as_of(date(2024, 1, 2), fixture_db) as con:
        assert con.execute("SELECT max(date) FROM sales_lines").fetchone()[0] == date(2024, 1, 2)
        assert con.execute("SELECT count(*) FROM stock_moves").fetchone()[0] == 5


def test_as_of_is_read_only(fixture_db):
    with warehouse.as_of(date(2024, 1, 2), fixture_db) as con:
        try:
            con.execute("DELETE FROM wh.sales_lines")
        except Exception:
            pass
        else:
            raise AssertionError("the warehouse should be read-only through as_of")


def test_every_table_module_declares_its_contract():
    for m in warehouse.table_modules():
        assert isinstance(m.TABLES, dict) and isinstance(m.DEPENDS, list)
        assert callable(m.build)
