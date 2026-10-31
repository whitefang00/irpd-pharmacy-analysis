from datetime import date, timedelta

import duckdb
import pytest

from irpd import config, warehouse
from irpd.tables import sales

MON = date(2024, 1, 1)  # a Monday


def day(n: int) -> date:
    return MON + timedelta(days=n)


def make_db(lines, moves=()):
    """An in-memory DB with tiny base tables, then this module's tables built on top.

    lines: (date, invoice, drug_code, qty, unit, unit_price), revenue = qty x price.
    moves: (date, document, drug_code, move_type, qty_out, unit).
    """
    con = duckdb.connect()
    con.execute("""CREATE TABLE sales_lines (date DATE, invoice VARCHAR, drug_code VARCHAR,
        is_service BOOLEAN, qty DOUBLE, unit VARCHAR, unit_price BIGINT, revenue BIGINT)""")
    con.execute("""CREATE TABLE stock_moves (date DATE, document VARCHAR, drug_code VARCHAR,
        move_type VARCHAR, qty_in BIGINT, qty_out BIGINT, unit VARCHAR)""")
    for d, inv, code, qty, unit, price in lines:
        con.execute("INSERT INTO sales_lines VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [d, inv, code, code.startswith("S"), qty, unit, price, int(qty * price)])
    for d, doc, code, kind, out, unit in moves:
        con.execute("INSERT INTO stock_moves VALUES (?, ?, ?, ?, 0, ?, ?)",
                    [d, doc, code, kind, out, unit])
    con.execute("""CREATE TABLE drugs AS SELECT drug_code, any_value(unit) AS stock_unit
                   FROM stock_moves GROUP BY 1""")
    sales.build(con)
    return con


def rows(con, sql):
    return con.execute(sql).fetchall()


# Drug A: ledger counts tablets, sold by the strip (6 Tab) and by the tablet.
# S1 is a service.
STRIP_AND_TAB = make_db(
    lines=[(day(0), "PJ1", "A", 2, "Strip", 6000), (day(0), "PJ2", "A", 3, "Tab", 1000),
           (day(0), "PJ3", "S1", 1, "Kali", 5000), (day(1), "PJ4", "A", 1, "Strip", 6000)],
    moves=[(day(0), "PJ1", "A", "sale", 12, "Tab"), (day(0), "PJ2", "A", "sale", 3, "Tab"),
           (day(1), "PJ4", "A", "sale", 6, "Tab")],
)


def test_pack_sizes_learn_strip_and_tab():
    assert rows(STRIP_AND_TAB, """
        SELECT sales_unit, base_per_unit, base_unit FROM pack_sizes
        WHERE drug_code = 'A' ORDER BY 1""") == [("Strip", 6, "Tab"), ("Tab", 1, "Tab")]
    assert rows(STRIP_AND_TAB, "SELECT count(*) FROM pack_sizes WHERE drug_code = 'S1'") == [(0,)]


def test_daily_demand_is_in_base_units_without_services():
    assert rows(STRIP_AND_TAB, "SELECT * FROM daily_demand ORDER BY date") == [
        (day(0), "A", 15, 2, 15000), (day(1), "A", 6, 1, 6000)]


def test_sale_without_ledger_row_falls_back_to_pack_size():
    con = make_db(
        lines=[(day(0), "PJ1", "A", 1, "Strip", 6000), (day(1), "PJ2", "A", 2, "Strip", 6000)],
        moves=[(day(0), "PJ1", "A", "sale", 6, "Tab")])
    assert rows(con, "SELECT date, units FROM daily_demand ORDER BY 1") == [
        (day(0), 6), (day(1), 12)]


def test_weekly_demand_fills_zero_weeks_and_counts_open_days():
    # Week 1: open Mon-Sat (closed Sunday). Week 2: nobody buys A. Week 3: Mon-Tue only.
    lines = [(day(i), f"B{i}", "B", 1, "Tab", 100) for i in range(6)]
    lines += [(day(7 + i), f"C{i}", "B", 1, "Tab", 100) for i in range(7)]
    lines += [(day(14), "A1", "A", 2, "Tab", 100), (day(15), "C9", "B", 1, "Tab", 100)]
    lines += [(day(2), "A0", "A", 4, "Tab", 100)]
    moves = [(d, inv, code, "sale", qty, "Tab") for d, inv, code, qty, _, _ in lines]
    con = make_db(lines, moves)
    assert rows(con, """
        SELECT week_start, week_end, units, lines, days_open FROM weekly_demand
        WHERE drug_code = 'A' ORDER BY 1""") == [
        (day(0), day(6), 4, 1, 6),
        (day(7), day(13), 0, 0, 7),
        (day(14), day(15), 2, 1, 2),  # the last week is partial
    ]


def test_weekly_demand_hides_a_week_until_it_ends(tmp_path):
    lines = [(day(i), f"B{i}", "B", 1, "Tab", 100) for i in range(14)]
    con = make_db(lines, [(d, inv, c, "sale", q, "Tab") for d, inv, c, q, _, _ in lines])
    db = tmp_path / "w.duckdb"
    con.execute(f"ATTACH '{db}' AS out")
    con.execute("CREATE TABLE out.weekly_demand AS SELECT * FROM weekly_demand")
    con.execute("DETACH out")
    with warehouse.as_of(day(9), db) as view:  # a Wednesday
        assert view.execute("SELECT max(week_start) FROM weekly_demand").fetchall() == [(day(0),)]


def test_prices_drop_blips_and_flag_unit_changes():
    # 100, 100, one line at 120, 100, then 250: the 120 is a blip, 250 a 2x+ jump.
    prices = [100, 100, 120, 100, 250, 250]
    con = make_db([(day(i), f"P{i}", "A", 1, "Tab", p) for i, p in enumerate(prices)])
    assert rows(con, """
        SELECT price, valid_from, valid_to, lines, likely_unit_change FROM prices
        ORDER BY valid_from""") == [
        (100, day(0), day(3), 3, False),
        (250, day(4), None, 2, True),
    ]


def test_prices_keep_a_real_change():
    prices = [100, 100, 110, 110, 110]
    con = make_db([(day(i), f"P{i}", "A", 1, "Tab", p) for i, p in enumerate(prices)])
    assert rows(con, "SELECT price, prev_price FROM prices ORDER BY valid_from") == [
        (100, None), (110, 100)]


def test_tiers_only_use_data_up_to_the_cutoff():
    # A sells weekly before the cutoff and daily after it; B only sells after it.
    start = config.CUTOFF - timedelta(days=364)
    lines = [(start + timedelta(days=7 * i), f"A{i}", "A", 1, "Tab", 100) for i in range(53)]
    lines += [(config.CUTOFF + timedelta(days=i), f"X{i}", c, 1, "Tab", 100)
              for i in range(1, 100) for c in "AB"]
    con = make_db(lines, [(d, inv, c, "sale", q, "Tab") for d, inv, c, q, _, _ in lines])
    got = rows(con, "SELECT drug_code, tier, sale_days, last_sale, sb_class FROM demand_tiers")
    assert got == [("A", "regular", 53, config.CUTOFF, "smooth")]
    assert rows(con, "SELECT revenue_share FROM demand_tiers") == [(1.0,)]


def test_tables_build_from_the_shared_fixture(fixture_db):
    with warehouse.as_of(date(2099, 1, 1), fixture_db) as con:
        assert con.execute("""SELECT base_per_unit FROM pack_sizes
            WHERE drug_code = 'D2' AND sales_unit = 'Strip'""").fetchall() == [(10,)]
        assert con.execute("""SELECT sum(units) FROM daily_demand
            WHERE drug_code = 'D2'""").fetchall() == [(10,)]


# Real data: roughly the EDA's numbers.


def real(db, sql):
    with warehouse.as_of(config.LAST_DAY, db) as con:
        return con.execute(sql).fetchall()


@pytest.mark.data
def test_real_pack_sizes(real_db):
    [(pack, base_not_1, unknown)] = real(real_db, """
        SELECT count(*) FILTER (WHERE sales_unit <> base_unit),
               count(*) FILTER (WHERE sales_unit = base_unit AND base_per_unit <> 1),
               count(*) FILTER (WHERE base_per_unit IS NULL)
        FROM pack_sizes""")
    assert 180 <= pack <= 210  # EDA: 207 on the raw file, which has 3 more categories
    assert base_not_1 == 0 and unknown == 0


@pytest.mark.data
def test_real_daily_demand_matches_ledger_and_revenue(real_db):
    [(units, ledger, rev, sales_rev)] = real(real_db, """
        SELECT (SELECT sum(units) FROM daily_demand),
               (SELECT sum(qty_out) FROM stock_moves WHERE move_type = 'sale'),
               (SELECT sum(revenue) FROM daily_demand),
               (SELECT sum(revenue) FROM sales_lines WHERE NOT is_service)""")
    assert ledger <= units < ledger * 1.001  # plus a few sale lines with no ledger row
    assert rev == sales_rev


@pytest.mark.data
def test_real_price_changes(real_db):
    [(changes,)] = real(real_db, "SELECT count(*) FROM prices WHERE prev_price IS NOT NULL")
    assert abs(changes - 2552) < 30  # EDA: 2,552 real changes


@pytest.mark.data
def test_real_tiers_match_the_eda_on_full_data(real_db):
    # The EDA tiered on all three years; the table itself stops at the cutoff.
    with duckdb.connect(str(real_db), read_only=True) as con:
        got = dict(con.execute(f"""
            SELECT tier, count(*) FROM ({sales.tiers_sql(config.LAST_DAY)})
            WHERE drug_code IN (SELECT drug_code FROM drugs)  -- EDA used ledger drugs only
            GROUP BY 1""").fetchall())
    assert got == {"fast": 242, "regular": 661, "slow": 1097, "rare": 956}


@pytest.mark.data
def test_real_tiers_stop_at_the_cutoff(real_db):
    [(last, smooth_share)] = real(real_db, """
        SELECT max(last_sale), sum(revenue_share) FILTER (WHERE sb_class = 'smooth')
        FROM demand_tiers""")
    assert last <= config.CUTOFF
    assert 0.5 < smooth_share < 0.7  # EDA: smooth weekly drugs make about 60% of revenue
