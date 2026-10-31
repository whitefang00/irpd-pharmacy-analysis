from datetime import date
from pathlib import Path

import pytest

from irpd import config, warehouse
from irpd.tables.stock import spells_as_of

FIXTURES = Path(__file__).parent / "fixtures/stock"
EVERYTHING = date(2099, 1, 1)


@pytest.fixture(scope="module")
def stock_db(tmp_path_factory) -> Path:
    """A warehouse built from tests/fixtures/stock (four drugs, see the comments below)."""
    return warehouse.build(FIXTURES, tmp_path_factory.mktemp("wh") / "stock.duckdb")


def rows(db, sql, day=EVERYTHING):
    with warehouse.as_of(day, db) as con:
        return con.execute(sql).fetchall()


def stock(db, drug, day):
    """{location: closing} for one drug on one day."""
    return dict(rows(db, f"""
        SELECT location, closing FROM daily_stock
        WHERE drug_code = '{drug}' AND date = DATE '{day}'"""))


# A1: bought 20 on the floor, moved 15 upstairs (1 Mar 2024), sold the floor
# empty (2 Mar), moved the 15 back down (4 Mar), sold out (5 Mar), restocked by
# a purchase (8 Mar), sold out again for good (9 Mar).

def test_series_runs_from_first_movement_to_last_day(stock_db):
    first, last, n = rows(stock_db, """
        SELECT min(date), max(date), count(*) FROM daily_stock
        WHERE drug_code = 'A1' AND location = 'total'""")[0]
    assert (first, last) == (date(2024, 3, 1), config.LAST_DAY)
    assert n == (last - first).days + 1


def test_carry_forward_over_days_without_movement(stock_db):
    assert stock(stock_db, "A1", "2024-03-03") == {"floor": 0, "store": 15, "total": 15}
    assert stock(stock_db, "A1", "2024-12-31") == {"floor": 0, "store": 0, "total": 0}


def test_total_is_floor_plus_store(stock_db):
    assert rows(stock_db, """
        SELECT count(*) FROM daily_stock t
        JOIN daily_stock f ON f.drug_code = t.drug_code AND f.date = t.date
         AND f.location = 'floor'
        JOIN daily_stock s ON s.drug_code = t.drug_code AND s.date = t.date
         AND s.location = 'store'
        WHERE t.location = 'total' AND t.closing <> f.closing + s.closing""") == [(0,)]


def test_transfer_moves_stock_but_not_the_total(stock_db):
    assert stock(stock_db, "A1", "2024-03-01") == {"floor": 5, "store": 15, "total": 20}
    assert stock(stock_db, "A1", "2024-03-04") == {"floor": 15, "store": 0, "total": 15}


def test_end_of_day_follows_row_number_not_timestamp(stock_db):
    # On 8 Mar the last row booked is timestamped before the one booked just
    # before it; the day closes at that last row's 5, not 6.
    assert stock(stock_db, "A1", "2024-03-08")["total"] == 5


def test_units_sold_come_from_sale_rows(stock_db):
    assert rows(stock_db, """
        SELECT date, sold FROM daily_stock
        WHERE drug_code = 'A1' AND location = 'total' AND sold > 0 ORDER BY date""") == [
        (date(2024, 3, 2), 5), (date(2024, 3, 5), 15), (date(2024, 3, 8), 5), (date(2024, 3, 9), 5)]


def test_stockout_spells_and_what_ended_them(stock_db):
    got = rows(stock_db, "SELECT * FROM stockouts ORDER BY drug_code, start")
    assert got == [
        ("A1", date(2024, 3, 5), date(2024, 3, 7), 3, "purchase"),
        ("A1", date(2024, 3, 9), config.LAST_DAY, (config.LAST_DAY - date(2024, 3, 9)).days + 1,
         None),
        # B1 sold before stock was booked, closed at -3, and a count fixed it.
        ("B1", date(2024, 3, 2), date(2024, 3, 2), 1, "stock_take"),
        # D1 was out from 20 Sep 2023, but spells only count from STOCK_START.
        ("D1", config.STOCK_START, date(2023, 10, 2), 2, "purchase"),
    ]


def test_shelfout_is_empty_floor_with_stock_upstairs(stock_db):
    assert rows(stock_db, "SELECT * FROM shelfouts") == [
        ("A1", date(2024, 3, 2), date(2024, 3, 3), 2)]


def test_flags(stock_db):
    got = {(d, f): detail for d, _, f, detail in rows(stock_db, "SELECT * FROM stock_flags")}
    assert set(got) == {("B1", "negative_close"), ("C1", "unbooked_opening"),
                        ("D1", "huge_opening")}
    assert got["B1", "negative_close"] == "floor closed at -3"
    assert "1,000 units" in got["D1", "huge_opening"]
    # C1's first row opens at 7 and nothing earlier is invented.
    assert rows(stock_db, "SELECT min(date) FROM daily_stock WHERE drug_code = 'C1'") == [
        (date(2024, 3, 5),)]


def test_as_of_views_stop_a_running_spell_on_the_day(stock_db):
    # The stored spell runs to 7 Mar and ended with a purchase; as of 6 Mar
    # neither is known yet.
    got = rows(stock_db, "SELECT stop, days, ended_by FROM stockouts WHERE drug_code = 'A1'",
               date(2024, 3, 6))
    assert got == [(date(2024, 3, 6), 2, None)]


def test_spells_as_of_does_not_leak(stock_db):
    day = date(2024, 3, 6)
    with warehouse.as_of(day, stock_db) as con:
        out = spells_as_of(con, day).fetchall()
        shelf = spells_as_of(con, day, "shelfouts").fetchall()
        later = rows(stock_db, "SELECT count(*) FROM daily_stock WHERE date > DATE '2024-03-06'",
                     day)
    assert later == [(0,)]
    assert ("A1", date(2024, 3, 5), day, 2, None) in out
    assert ("B1", date(2024, 3, 2), date(2024, 3, 2), 1, "stock_take") in out
    assert shelf == [("A1", date(2024, 3, 2), date(2024, 3, 3), 2)]


def test_spells_as_of_last_day_matches_stored_tables(stock_db):
    with warehouse.as_of(config.LAST_DAY, stock_db) as con:
        for table in ("stockouts", "shelfouts"):
            stored = con.execute(f"SELECT * FROM {table} ORDER BY ALL").fetchall()
            assert sorted(spells_as_of(con, config.LAST_DAY, table).fetchall()) == stored


def test_spell_columns_work_in_plain_sql(stock_db):
    assert rows(stock_db, "SELECT max(stop), sum(days) FROM shelfouts") == [
        (date(2024, 3, 3), 2)]


# Real data: numbers from eda/stock.py and FINDINGS.md.

TIERS = """
    CREATE TEMP TABLE tiers AS
    WITH s AS (SELECT drug_code, min(date) AS first_sale, max(date) AS last_sale,
                      count(DISTINCT date) AS n, max(date) - min(date) + 1 AS live
               FROM stock_moves WHERE move_type = 'sale' GROUP BY drug_code)
    SELECT *, CASE WHEN live < 28 OR n < 6 THEN 'rare'
                   WHEN n * 365.0 / live >= 156 THEN 'fast'
                   WHEN n * 365.0 / live >= 52 THEN 'regular'
                   WHEN n * 365.0 / live >= 12 THEN 'slow'
                   ELSE 'rare' END AS tier
    FROM s"""


@pytest.fixture(scope="module")
def real(real_db):
    with warehouse.as_of(EVERYTHING, real_db) as con:
        con.execute(TIERS)
        yield con


@pytest.mark.data
def test_real_ledger_chains_per_location(real):
    broken = real.execute("""
        SELECT count(*) FROM (
            SELECT opening, lag(closing) OVER (PARTITION BY drug_code, location
                                               ORDER BY row_number) AS prev
            FROM stock_moves) WHERE prev IS NOT NULL AND opening <> prev""").fetchall()[0][0]
    assert broken < 200  # EDA: 188 on the raw file, nearly all same-second stock-takes


@pytest.mark.data
def test_real_daily_stock_shape(real):
    drugs, n, bad = real.execute("""
        SELECT count(DISTINCT drug_code), count(*),
               count(*) FILTER (WHERE location = 'total' AND closing <> f + s)
        FROM (SELECT *, sum(closing) FILTER (WHERE location = 'floor')
                            OVER (PARTITION BY drug_code, date) AS f,
                        sum(closing) FILTER (WHERE location = 'store')
                            OVER (PARTITION BY drug_code, date) AS s
              FROM daily_stock)""").fetchall()[0]
    assert drugs == 4051 and bad == 0
    assert 9_000_000 < n < 13_000_000  # ~4K drugs x ~900 days x 3 locations


@pytest.mark.data
def test_real_out_of_stock_share_by_tier(real):
    got = dict(real.execute("""
        SELECT tier, avg((closing <= 0)::INT) FROM daily_stock JOIN tiers USING (drug_code)
        WHERE location = 'total' AND date BETWEEN first_sale AND last_sale
        GROUP BY tier""").fetchall())
    # EDA: fast 0.8%, regular 3.1%, slow 8.7%, rare 20.9%.
    assert 0.005 < got["fast"] < 0.012
    assert 0.025 < got["regular"] < 0.04
    assert 0.07 < got["slow"] < 0.10


@pytest.mark.data
def test_real_spells_end_with_deliveries(real):
    got = dict(real.execute("""
        SELECT ended_by, count(*) FROM stockouts JOIN tiers USING (drug_code)
        WHERE start > first_sale AND stop < last_sale GROUP BY ALL""").fetchall())
    assert got["purchase"] / sum(got.values()) > 0.9  # EDA: 94%
    median_fast = real.execute("""
        SELECT median(days) FROM stockouts JOIN tiers USING (drug_code)
        WHERE tier = 'fast' AND start > first_sale AND stop < last_sale""").fetchall()[0][0]
    assert median_fast == 2


@pytest.mark.data
def test_real_shelfouts_beat_stockouts_on_fast_drugs(real):
    # Spell days inside each fast drug's selling period.
    shelf, out = (real.execute(f"""
        SELECT coalesce(sum(least(stop, last_sale) - greatest(start, first_sale) + 1), 0)
        FROM {table} JOIN tiers USING (drug_code)
        WHERE tier = 'fast' AND start <= last_sale AND stop >= first_sale""").fetchall()[0][0]
        for table in ("shelfouts", "stockouts"))
    assert shelf > out  # EDA: 2,577 shelf-out days vs 1,864 out (from Jul 2023)


@pytest.mark.data
def test_real_flags_catch_known_problems(real):
    huge = {r[0] for r in real.execute("""
        SELECT d.drug_name FROM stock_flags f JOIN drugs d USING (drug_code)
        WHERE flag = 'huge_opening'""").fetchall()}
    assert {"Ambroxol tab 30mg Nova", "De", "Allopurinol 100mg Nova"} <= huge
    neg = real.execute(
        "SELECT count(*) FROM stock_flags WHERE flag = 'negative_close'").fetchall()[0][0]
    assert 20 <= neg <= 40  # EDA: 30 drug-location-days, all categories


@pytest.mark.data
def test_real_spells_as_of_cutoff(real_db):
    day = config.CUTOFF
    with warehouse.as_of(day, real_db) as con:
        recomputed = sorted(spells_as_of(con, day).fetchall())
        # Truncating the stored spells at the cutoff gives the same answer,
        # except ended_by, which is only known once the spell ends.
        truncated = sorted(con.execute(f"""
            SELECT drug_code, start, least(stop, DATE '{day}'),
                   (least(stop, DATE '{day}') - start + 1)::INTEGER,
                   CASE WHEN stop < DATE '{day}' THEN ended_by END
            FROM stockouts""").fetchall())
    assert recomputed == truncated
    assert max(r[2] for r in recomputed) == day
