import csv
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from irpd import config, holdout, policies, replay, warehouse

D0 = date(2025, 7, 1)


def make(demand, start_floor=0, start_store=0, rate=None, adjust=None, store_opens=None,
         actual_floor=None, actual_store=None, live=None, price=1000.0) -> replay.Holdout:
    """A one-drug holdout ('A') from a list of daily demand."""
    demand = np.array([demand], float)
    T = demand.shape[1]
    zeros = np.zeros((1, T))
    return replay.Holdout(
        drugs=["A"], days=[D0 + timedelta(days=i) for i in range(T)],
        price=np.array([price]), lead=pd.Series({"A": 2.0}),
        start_floor=np.array([start_floor], float), start_store=np.array([start_store], float),
        demand=demand, adjust=zeros if adjust is None else np.array([adjust], float),
        store_opens=store_opens or D0 + timedelta(days=T),
        rate=np.full((1, T), 2.0) if rate is None else np.array([rate], float),
        actual_floor=zeros if actual_floor is None else np.array([actual_floor], float),
        actual_store=zeros if actual_store is None else np.array([actual_store], float),
        actual_sold=zeros, live=np.ones((1, T), bool) if live is None else np.array([live]))


def policy(**cols) -> pd.DataFrame:
    return pd.DataFrame({"drug_code": ["A"], **{k: [v] for k, v in cols.items()}})


def test_sales_are_capped_by_floor_stock():
    run = replay.simulate(policy(s=-1, S=0), make([3, 3, 3], start_floor=5))
    assert run.sold.tolist() == [[3, 2, 0]]
    assert run.unmet.tolist() == [[0, 1, 3]]
    assert run.floor.tolist() == [[2, 0, 0]]


def test_orders_up_to_S_and_arrive_after_the_lead_time():
    # Ends day 0 at 4 <= s=4: order 10 - 4 = 6, arriving 2 days later.
    run = replay.simulate(policy(s=4, S=10, lead_days=2), make([1, 1, 1, 1], start_floor=5))
    assert run.ordered.tolist() == [[6, 0, 0, 0]]  # the order counts toward the position
    assert run.floor.tolist() == [[4, 3, 8, 7]]


def test_lead_time_defaults_to_the_supplier_cadence():
    run = replay.simulate(policy(s=4, S=10), make([1, 1, 1, 1], start_floor=5))  # lead 2
    assert run.floor.tolist() == [[4, 3, 8, 7]]


def test_deliveries_go_to_the_store_once_it_reopens_and_the_floor_is_topped_up():
    h = make([2, 2, 2, 2, 2], start_floor=4, store_opens=D0 + timedelta(days=2))
    run = replay.simulate(policy(s=2, S=10, lead_days=2, floor_min=0, floor_max=3), h)
    # Day 0 ends at floor 2: order 8 for day 2, which lands in the store, and
    # 3 go down that morning. Day 3 starts with 1 on the floor, above
    # floor_min, so nothing moves and a sale is lost; day 4 tops up again.
    assert run.floor.tolist() == [[2, 0, 1, 0, 1]]
    assert run.store.tolist() == [[0, 0, 5, 5, 2]]
    assert run.unmet.tolist() == [[0, 0, 0, 1, 0]]


def test_stock_take_corrections_come_off_the_floor_first_and_never_below_zero():
    h = make([0, 0], start_floor=3, start_store=5, adjust=[-4, -10])
    run = replay.simulate(policy(s=-1, S=0), h)
    assert run.floor.tolist() == [[0, 0]]
    assert run.store.tolist() == [[4, 0]]


def test_a_policy_must_cover_every_drug():
    with pytest.raises(ValueError, match="no row for 1 drugs"):
        replay.simulate(pd.DataFrame({"drug_code": ["B"], "s": [1], "S": [2]}), make([1]))


def test_metrics_count_out_days_and_lost_units_from_the_rate_before_the_spell():
    # Total out on days 1-2 (rate at spell start 4), shelf-out day 4, nothing
    # counted on day 5 (after the last sale).
    h = make([0] * 6, rate=[1, 4, 9, 9, 3, 3], live=[1, 1, 1, 1, 1, 0])
    floor = np.array([[5, 0, 0, 5, 0, 0]], float)
    store = np.array([[0, 0, 0, 0, 2, 2]], float)
    sold = np.array([[0, 1, 0, 0, 1, 0]], float)
    m = replay.metrics(floor, store, sold, h).iloc[0]
    assert (m.days, m.stockout_days, m.shelfout_days) == (5, 2, 1)
    assert m.lost_units_stockout == (4 - 1) + 4
    assert m.lost_units_shelfout == 3 - 1
    assert m.lost_revenue_stockout == 7000
    assert m.avg_stock_value == pytest.approx(1000 * (5 + 5 + 2 + 2) / 6)


def test_long_spells_are_reported_apart():
    h = make([0] * 20)
    out = np.array([[0] * 3 + [5] * 2 + [0] * 15], float)  # one 3-day, one 15-day spell
    m = replay.metrics(out, np.zeros_like(out), np.zeros_like(out), h).iloc[0]
    assert (m.stockout_days, m.stockout_days_long) == (18, 15)


def test_trailing_rate_skips_days_out_of_stock():
    units = np.array([[2, 0, 4, 9]], float)
    in_stock = np.array([[1, 0, 1, 1]], bool)
    assert replay._trailing_rate(units, in_stock).tolist() == [[0, 2, 2, 3]]


def test_match_stock_value_hits_the_target():
    h = make([1] * 30, start_floor=10)
    p = policy(s=5, S=20, lead_days=1)
    matched, k = replay.match_stock_value(p, h, target=20_000)
    assert replay.stock_value(matched, h) == pytest.approx(20_000, rel=0.01)
    assert matched["S"].iloc[0] == pytest.approx(20 * k)


def write_fixture(root: Path) -> Path:
    """Drug A sells 2 a day from April to mid-November 2025 off a big
    delivery. In H2 a delivery goes to the store on 20 Jul (the store reopens)
    and the August count takes 10 off. Drug B only sells before the cutoff."""
    moves, sales = [], []
    floor, store = 0, 0

    def move(day, doc, kind, drug, loc, qin, qout):
        nonlocal floor, store
        before = floor if loc == "Apotek" else store
        after = before + qin - qout
        if drug == "A":
            if loc == "Apotek":
                floor = after
            else:
                store = after
        moves.append((day, doc, kind, drug, loc, before, qin, qout, after))

    move(date(2025, 4, 1), "PB1", "Pembelian (Sup X)", "A", "Apotek", 600, 0)
    move(date(2025, 4, 1), "PB2", "Pembelian (Sup X)", "B", "Apotek", 50, 0)
    day, i = date(2025, 4, 1), 0
    while day <= date(2025, 11, 15):
        i += 1
        if day == date(2025, 7, 20):
            move(day, "PB3", "Pembelian (Sup X)", "A", "Gudang Obat 1", 100, 0)
        if day == date(2025, 8, 5):
            move(day, "SO1", "Stokopname", "A", "Apotek", 0, 10)
        if floor >= 2:
            move(day, f"PJ{i}", "Penjualan", "A", "Apotek", 0, 2)
            sales.append((day, f"PJ{i}", "A", 2))
        if day < date(2025, 5, 1) and day.day % 3 == 0:
            moves.append((day, f"PK{i}", "Penjualan", "B", "Apotek", 0, 0, 1, 0))
            sales.append((day, f"PK{i}", "B", 1))
        day += timedelta(days=1)

    proc, raw = root / "processed_dataset", root / "raw_dataset"
    proc.mkdir(parents=True), raw.mkdir(parents=True)
    with open(proc / "stock_processed.csv", "w", newline="") as f, \
         open(raw / "stock_raw.csv", "w", newline="") as g:
        w, r = csv.writer(f), csv.writer(g)
        w.writerow("RowNumber,Date,Time,DocumentNumber,TransactionType,DrugCode,DrugName,"
                   "Category,InitialStock,StockIn,StockOut,RemainingStock,Unit".split(","))
        r.writerow(["RowNumber", "DateTime", "Warehouse", "DrugCode"])
        for n, (d, doc, kind, drug, loc, o, qi, qo, c) in enumerate(moves, 1):
            w.writerow([n, d, "09:00:00", doc, kind, drug, f"Drug {drug}", "Generik",
                        o, qi, qo, c, "Tab"])
            r.writerow([n, d.strftime("%d %b %Y 09:00:00"), loc, drug])
    with open(proc / "sales_processed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow("RowNumber,Date,Time,InvoiceNumber,CashierID,DrugCode,DrugName,Category,"
                   "Quantity,Unit,UnitPrice,DiscountAmount,Subtotal,PaymentType".split(","))
        for n, (d, inv, drug, q) in enumerate(sales, 1):
            w.writerow([n, d, "09:00:00", inv, "C1", drug, f"Drug {drug}", "Generik",
                        q, "Tab", 500, 0, q * 500, "TUNAI"])
    return root


@pytest.fixture(scope="module")
def sealed(tmp_path_factory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("replay")
    wh = warehouse.build(write_fixture(tmp / "data"), tmp / "wh.duckdb")
    return holdout.seal(wh, tmp / "agent.duckdb", tmp / "holdout.duckdb")


def test_the_agent_file_holds_nothing_after_the_cutoff(sealed):
    agent, _ = sealed
    with duckdb.connect(str(agent), read_only=True) as con:
        tables = [r[0] for r in con.execute("SELECT table_name FROM duckdb_tables()").fetchall()]
        assert "daily_stock" in tables and "pharmacist_baseline" in tables
        for name, col in warehouse.date_columns().items():
            if col and name in tables:
                latest = con.execute(f"SELECT max({col}) FROM {name}").fetchone()[0]
                assert latest is None or latest <= config.CUTOFF, name


def test_the_agent_view_cannot_open_the_holdout(sealed):
    agent, hold = sealed
    with holdout.agent_view(config.CUTOFF, agent) as con:
        assert con.execute("SELECT max(date) FROM wh.daily_stock").fetchone()[0] \
            == config.CUTOFF  # even behind the views
        for sql in (f"ATTACH '{hold}' AS h", f"SELECT * FROM '{hold}'",
                    "SET enable_external_access = true"):
            with pytest.raises(duckdb.Error):
                con.execute(sql)
        dbs = {r[0] for r in con.execute("SELECT database_name FROM duckdb_databases()")
               .fetchall()}
        assert not any("holdout" in d for d in dbs)


def test_the_holdout_file_holds_only_the_holdout(sealed):
    _, hold = sealed
    with holdout.open_sealed(hold) as con:
        for name, col in holdout.SEALED_TABLES.items():
            first = con.execute(f"SELECT min({col}) FROM {name}").fetchone()[0]
            assert first is None or first > config.CUTOFF, name
        assert con.execute("SELECT sum(units) FROM daily_demand").fetchone()[0] > 0


@pytest.fixture(scope="module")
def loaded(sealed) -> replay.Holdout:
    return replay.load(*sealed)


def test_load_reads_scope_start_stock_and_events(loaded):
    h = loaded
    assert h.drugs == ["A"]  # B is a candidate but never sold after the cutoff
    assert h.store_opens == date(2025, 7, 20)
    assert h.adjust.sum() == -10 and h.adjust[0, h.days.index(date(2025, 8, 5))] == -10
    assert h.start_floor[0] == 600 - 2 * 91 and h.start_store[0] == 0
    assert h.price.tolist() == [500]
    assert h.rate[0, 0] == 2
    last = h.days.index(date(2025, 11, 15))
    assert h.live[0, last] and not h.live[0, last + 1]


def test_reference_policies_cover_the_candidates_and_replay(sealed, loaded):
    with holdout.agent_view(config.CUTOFF, sealed[0]) as con:
        m, t = policies.mimic(con), policies.textbook(con)
        assert replay.candidates(con) == ["A", "B"]
    assert m.drug_code.tolist() == t.drug_code.tolist() == ["A", "B"]
    got = replay.score({"mimic": m, "textbook": t}, loaded)
    assert got.loc["pharmacist (actual)", "units_sold"] == loaded.actual_sold.sum()
    # A steady 2 a day with stock always on hand: no policy should run out.
    assert got.loc["textbook", "stockout_days"] == 0


def test_reference_policies_ignore_rows_after_their_day(sealed):
    # Drug A sits in stock unsold after mid-November, so H2 rows would pull its rate down.
    full = sealed[0].with_name("wh.duckdb")
    with duckdb.connect(str(full), read_only=True) as con:
        m, t = policies.mimic(con), policies.textbook(con)
    with holdout.agent_view(config.CUTOFF, sealed[0]) as con:
        pd.testing.assert_frame_equal(m, policies.mimic(con))
        pd.testing.assert_frame_equal(t, policies.textbook(con))


@pytest.fixture(scope="module")
def real(real_db, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sealed")
    agent, hold = holdout.seal(real_db, tmp / "agent.duckdb", tmp / "holdout.duckdb")
    with holdout.agent_view(config.CUTOFF, agent) as con:
        mim = policies.mimic(con)
    return replay.load(agent, hold), mim


@pytest.mark.data
def test_real_pharmacist_holdout(real):
    h, _ = real
    a = replay.summarise(replay.actual(h))
    assert 800 <= a["drugs"] <= 850  # PLAN: about 820
    assert 300e6 < a["avg_stock_value"] < 330e6  # PLAN: about Rp 318M
    assert h.store_opens == date(2025, 7, 29)


@pytest.mark.data
def test_real_mimic_at_matched_stock_is_near_the_pharmacist_on_short_spells(real):
    h, mim = real
    matched, _ = replay.match_stock_value(mim, h)
    got = replay.score({"mimic": matched}, h)
    short = got["stockout_days"] - got["stockout_days_long"]
    ratio = short["mimic"] / short["pharmacist (actual)"]
    assert 0.7 < ratio < 1.3
