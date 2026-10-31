"""`warehouse.as_of(day)` must show nothing that wasn't known at the end of `day`.

The check: build a second warehouse from the fixture files cut at `day`, and
require every row an as_of view shows to also be there in the cut build. A
row that only the full build produces (a spell's later stop, a price's later
valid_to, a purchase line not yet deleted, a drug first seen later) is a leak.
The cut build may show more, e.g. a week that hasn't ended yet.
"""

import csv
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import pytest

from irpd import config, warehouse

FIXTURES = Path(__file__).parent / "fixtures"
# leaks/: L1 sells at 100, once at 120 (a blip, only known once 100 is back),
# shelf-out 2-3 Mar, a Strip first sold 5 Mar, PB2 delivered 5 Mar and deleted
# 7 Mar, out of stock 8-9 Mar until a purchase on 10 Mar. L2 first moves 9 Mar.
# Each supplier delivers on one weekday only, so `mode(weekday)` has no ties.
SETS = ["leaks", "stock", "supply", "irpd"]


def cut(src: Path, dst: Path, day: date) -> Path:
    """Copy a fixture set keeping only rows dated up to `day`."""
    kept: set[str] = set()
    for name, date_col in [("processed_dataset/stock_processed.csv", "Date"),
                           ("processed_dataset/sales_processed.csv", "Date"),
                           ("raw_dataset/stock_raw.csv", None)]:
        with open(src / name) as f:
            r = csv.DictReader(f)
            fields, rows = r.fieldnames, list(r)
        if date_col:
            rows = [x for x in rows if date.fromisoformat(x[date_col]) <= day]
            if name.startswith("processed_dataset/stock"):
                kept = {x["RowNumber"] for x in rows}
        else:
            rows = [x for x in rows if x["RowNumber"] in kept]
        (dst / name).parent.mkdir(parents=True, exist_ok=True)
        with open(dst / name, "w", newline="") as f:
            w = csv.DictWriter(f, fields)
            w.writeheader()
            w.writerows(rows)
    return dst


def days_in(src: Path) -> list[date]:
    """Every day with a stock row, the day before each, and the cutoff, from the
    first day both files have a row (an empty CSV won't load)."""
    first = []
    for name in ("stock", "sales"):
        with open(src / f"processed_dataset/{name}_processed.csv") as f:
            dates = {date.fromisoformat(r["Date"]) for r in csv.DictReader(f)}
        first.append(min(dates))
        if name == "stock":
            days = dates | {d - timedelta(days=1) for d in dates} | {config.CUTOFF}
    return sorted(d for d in days if d >= max(first))


# A cut build's data ends on the cut day, so its last week_end is earlier.
# That is the cut build's artefact, not information, so it isn't compared.
SKIP_COLUMNS = {"weekly_demand": "EXCLUDE (week_end)"}


def snapshot(day: date, db: Path) -> dict[str, Counter]:
    def hashable(row):
        return tuple(tuple(v) if isinstance(v, list) else v for v in row)

    with warehouse.as_of(day, db) as con:
        names = [r[0] for r in con.execute(
            "SELECT view_name FROM duckdb_views() WHERE NOT internal").fetchall()]
        return {n: Counter(hashable(r) for r in con.execute(
                    f"SELECT * {SKIP_COLUMNS.get(n, '')} FROM {n}").fetchall())
                for n in names}


@pytest.mark.parametrize("fixture_set", SETS)
def test_as_of_shows_only_what_a_build_cut_at_that_day_shows(fixture_set, tmp_path):
    src = FIXTURES / fixture_set
    full = warehouse.build(src, tmp_path / "full.duckdb")
    for day in days_in(src):
        part = warehouse.build(cut(src, tmp_path / f"cut{day}", day), tmp_path / f"{day}.duckdb")
        seen, allowed = snapshot(day, full), snapshot(day, part)
        assert set(seen) == set(allowed), day
        for name in seen:
            leaked = seen[name] - allowed[name]
            assert not leaked, f"{fixture_set} {name} as of {day} leaks {dict(leaked)}"


@pytest.fixture(scope="module")
def leaks_db(tmp_path_factory) -> Path:
    return warehouse.build(FIXTURES / "leaks", tmp_path_factory.mktemp("wh") / "leaks.duckdb")


def one(db, day, sql):
    with warehouse.as_of(day, db) as con:
        return con.execute(sql).fetchall()


def test_a_price_blip_is_the_current_price_until_the_old_price_returns(leaks_db):
    sql = "SELECT price, valid_from, valid_to FROM prices WHERE unit = 'Tab' ORDER BY 2"
    assert one(leaks_db, date(2025, 3, 2), sql) == [
        (100, date(2025, 3, 1), date(2025, 3, 1)), (120, date(2025, 3, 2), None)]
    assert one(leaks_db, date(2025, 3, 4), sql) == [(100, date(2025, 3, 1), None)]


def test_running_spells_stop_on_the_day(leaks_db):
    assert one(leaks_db, date(2025, 3, 9), "SELECT * FROM stockouts") == [
        ("L1", date(2025, 3, 8), date(2025, 3, 9), 2, None)]
    assert one(leaks_db, date(2025, 3, 2), "SELECT * FROM shelfouts") == [
        ("L1", date(2025, 3, 2), date(2025, 3, 2), 1)]


def test_a_purchase_shows_until_its_deletion_is_booked(leaks_db):
    sql = "SELECT document FROM purchases ORDER BY row_number"
    assert one(leaks_db, date(2025, 3, 6), sql) == [("PB1",), ("PB2",), ("PB3",)]
    assert one(leaks_db, date(2025, 3, 7), sql) == [("PB1",), ("PB3",)]


def test_drugs_and_pack_sizes_only_know_the_past(leaks_db):
    assert one(leaks_db, date(2025, 3, 4), "SELECT drug_code, last_seen FROM drugs") == [
        ("L1", date(2025, 3, 4))]
    assert one(leaks_db, date(2025, 3, 4), "SELECT sales_unit FROM pack_sizes") == [("Tab",)]


def test_baseline_is_hidden_before_the_cutoff(leaks_db):
    with warehouse.as_of(config.CUTOFF - timedelta(days=1), leaks_db) as con:
        views = {r[0] for r in con.execute("SELECT view_name FROM duckdb_views()").fetchall()}
    assert "pharmacist_baseline" not in views and "supplier_cadence" not in views


def test_rebuilt_views_match_the_stored_tables_on_the_last_day(leaks_db):
    with warehouse.as_of(config.LAST_DAY, leaks_db) as con:
        for name in warehouse.rebuilt_as_of(config.LAST_DAY):
            assert sorted(con.execute(f"SELECT * FROM {name}").fetchall()) == sorted(
                con.execute(f"SELECT * FROM wh.{name}").fetchall()), name


@pytest.mark.data
def test_real_as_of_cutoff_shows_nothing_later(real_db):
    with warehouse.as_of(config.CUTOFF, real_db) as con:
        for name, col in warehouse.date_columns().items():
            if col is None:
                continue
            latest = con.execute(f"SELECT max({col}) FROM {name}").fetchone()[0]
            assert latest is None or latest <= config.CUTOFF, name
        for name, col in [("stockouts", "stop"), ("drugs", "last_seen"),
                          ("prices", "valid_to")]:
            latest = con.execute(f"SELECT max({col}) FROM {name}").fetchone()[0]
            assert latest <= config.CUTOFF, name
