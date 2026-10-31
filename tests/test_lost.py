import csv
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from irpd import holdout, lost, replay, warehouse
from irpd.tables import ingredients

# A generated ledger, 1 Jan - 31 Mar 2024:
# - A sells 2 lines of 2 units a day at Rp 1,000 a unit. It sells out on the
#   evening of 1 Mar and is restocked on 11 Mar, so 2-10 Mar are fully out.
# - B has A's ingredient and form. It sells 2 lines of 1 unit a day, and 3 on
#   A's out days: half of A's customers switch.
# - C (another ingredient) sells 20 lines a day; D sells 2 a day off the floor,
#   but the floor is empty 10-14 Feb with stock upstairs, and only 1 a day is
#   fetched from the store then. Traffic is 26 lines a normal day, 25 on A's
#   out days and on D's shelf-out days.
D0, D1 = date(2024, 1, 1), date(2024, 3, 31)
A_OUT = (date(2024, 3, 2), date(2024, 3, 10))
D_SHELF = (date(2024, 2, 10), date(2024, 2, 14))


def _days(a, b):
    return [a + timedelta(days=i) for i in range((b - a).days + 1)]


def write_ledger(root: Path) -> Path:
    moves, sales = [], []
    bal: dict[tuple[str, str], int] = {}

    def move(day, doc, kind, drug, loc, qin, qout):
        o = bal.get((drug, loc), 0)
        bal[(drug, loc)] = o + qin - qout
        moves.append((day, doc, kind, drug, loc, o, qin, qout, o + qin - qout))

    def sell(day, drug, n_lines, qty, price, loc="Apotek"):
        for k in range(n_lines):
            inv = f"PJ{drug}{day:%m%d}{k}"
            move(day, inv, "Penjualan", drug, loc, 0, qty)
            sales.append((day, inv, drug, qty, price))

    a_units_before = 4 * len(_days(D0, A_OUT[0] - timedelta(days=1)))
    d_floor_before = 2 * len(_days(D0, D_SHELF[0] - timedelta(days=1)))
    for day in _days(D0, D1):
        if day == D0:
            move(day, "PB1", "Pembelian (Sup X)", "A", "Apotek", a_units_before, 0)
            move(day, "PB2", "Pembelian (Sup X)", "B", "Apotek", 1000, 0)
            move(day, "PB3", "Pembelian (Sup X)", "C", "Apotek", 10_000, 0)
            move(day, "PB4", "Pembelian (Sup X)", "D", "Apotek", d_floor_before, 0)
            move(day, "PB5", "Pembelian (Sup X)", "D", "Gudang Obat 1", 100, 0)
        if day == A_OUT[1] + timedelta(days=1):
            move(day, "PB6", "Pembelian (Sup X)", "A", "Apotek", 400, 0)
        if day == D_SHELF[1] + timedelta(days=1):
            move(day, "MT1", "Mutasi", "D", "Gudang Obat 1", 0, 95)
            move(day, "MT1", "Mutasi", "D", "Apotek", 95, 0)
        a_out = A_OUT[0] <= day <= A_OUT[1]
        if not a_out:
            sell(day, "A", 2, 2, 1000)
        sell(day, "B", 3 if a_out else 2, 1, 500)
        sell(day, "C", 20, 1, 100)
        if D_SHELF[0] <= day <= D_SHELF[1]:
            sell(day, "D", 1, 1, 300, loc="Gudang Obat 1")
        else:
            sell(day, "D", 2, 1, 300)

    proc, raw = root / "processed_dataset", root / "raw_dataset"
    proc.mkdir(parents=True)
    raw.mkdir(parents=True)
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
        for n, (d, inv, drug, q, price) in enumerate(sales, 1):
            w.writerow([n, d, "09:00:00", inv, "C1", drug, f"Drug {drug}", "Generik",
                        q, "Tab", price, 0, q * price, "TUNAI"])
    return root


def write_map(path: Path) -> Path:
    rows = [("A", "Drug A", "x", "", "tablet"), ("B", "Drug B", "x", "", ""),
            ("C", "Drug C", "y", "", "tablet"), ("D", "Drug D", "z", "", "syrup")]
    pd.DataFrame(rows, columns=["drug_code", "drug_name", "ingredients", "strength", "form"]) \
        .assign(kind="drug", mapped=True, source_model="test").to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def ledger(tmp_path_factory) -> Path:
    return write_ledger(tmp_path_factory.mktemp("lost") / "data")


@pytest.fixture(scope="module")
def mapped(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setattr(ingredients, "MAP", write_map(tmp_path_factory.mktemp("map") / "map.csv"))
    yield
    mp.undo()


@pytest.fixture(scope="module")
def db(ledger, mapped, tmp_path_factory) -> Path:
    return warehouse.build(ledger, tmp_path_factory.mktemp("wh") / "lost.duckdb")


@pytest.fixture(scope="module")
def sp(db) -> pd.DataFrame:
    with duckdb.connect(str(db), read_only=True) as con:
        return lost.spells(con, D1).set_index(["drug_code", "kind"])


TRAFFIC, TRAFFIC_OUT = 26, 25  # lines a normal day, and on A's (or D's) out days
# A's baseline: its 56 clean days before 2 Mar (5 Jan - 29 Feb), 5 of them D's shelf-out.
A_BASE = 56 / (51 * TRAFFIC + 5 * TRAFFIC_OUT)  # days per line of traffic


def test_lost_demand_is_the_rate_before_the_spell_times_the_traffic(sp):
    a = sp.loc[("A", "stockout")]
    out_days = (A_OUT[1] - A_OUT[0]).days + 1
    assert (a.start, a.counted_days, a.days_beyond) == (A_OUT[0] - timedelta(days=1), 9, 0)
    assert a.expected_units == pytest.approx(4 * A_BASE * TRAFFIC_OUT * out_days)
    assert a.recorded_units == 0
    assert a.lost_units == a.expected_units
    assert a.price == 1000
    assert a.lost_revenue == pytest.approx(1000 * a.lost_units)
    assert a.lost_lo_revenue <= a.lost_revenue <= a.lost_hi_revenue


def test_a_shelf_out_counts_only_the_shortfall_below_normal(sp):
    d = sp.loc[("D", "shelfout")]
    assert d.counted_days == 5
    assert d.expected_units == pytest.approx(2 / TRAFFIC * TRAFFIC_OUT * 5)
    assert d.recorded_units == 5
    assert d.lost_units == pytest.approx(d.expected_units - 5)


def test_switching_splits_lost_into_switched_and_truly_lost(sp):
    a = sp.loc[("A", "stockout")]
    b_expected = 2 * A_BASE * TRAFFIC_OUT * 9
    assert a.has_alternative
    assert a.alt_expected_lines == pytest.approx(b_expected)
    assert a.alt_uplift_lines == pytest.approx(27 - b_expected)
    share = (27 - b_expected) / a.lost_lines
    assert a.switched_raw == pytest.approx(share)
    assert a.switch_rate == pytest.approx(share)  # one spell: its own rate is the overall
    assert a.switched_revenue == pytest.approx(share * a.lost_revenue)
    assert a.truly_lost_revenue == pytest.approx((1 - share) * a.lost_revenue)


def test_switch_rates_shrink_small_groups_toward_the_overall_rate():
    df = pd.DataFrame({
        "kind": "stockout", "has_alternative": True, "estimated": True,
        "ingredients": ["big"] * 10 + ["small"],
        "lost_lines": [100.0] * 10 + [5.0], "alt_uplift_lines": [80.0] * 10 + [0.0]})
    r = lost.switch_rates(df)
    assert r.attrs["overall"] == pytest.approx(800 / 1005)
    assert r.loc["big", "rate"] == pytest.approx((800 + 50 * 800 / 1005) / 1050)
    assert r.loc["small", "raw"] == 0
    assert r.loc["small", "rate"] == pytest.approx(50 * 800 / 1005 / 55)


def test_placebo_windows_show_no_uplift(db):
    with duckdb.connect(str(db), read_only=True) as con:
        pl = lost.placebo(con, D1)
    same = pl.loc["placebo: same drug, in stock"]
    assert same.windows > 0
    # Not exactly 0: B's sales don't move with traffic when D's shelf-out lowers it.
    assert same.uplift_over_drug_expected == pytest.approx(0, abs=0.01)
    assert pl.loc["stock-outs", "uplift_over_drug_expected"] > 0.4


def test_fake_spells_recover_steady_sales(db):
    with duckdb.connect(str(db), read_only=True) as con:
        fake = lost.fake_spells(con, D1, max_out_share=0.2)
    c = fake[fake.drug_code == "C"]
    assert len(c) > 0 and (c.base_days >= lost.MIN_BASE).all()
    # C sells 20 a day; traffic dips by a line on A's and D's out days, so the
    # estimate is within a few percent.
    assert (c.expected / c.actual - 1).abs().max() < 0.05
    assert lost.fake_score(fake).loc["fast", "spells"] == len(fake)


def _rows(df: pd.DataFrame) -> list:
    return sorted(df.round(9).itertuples(index=False, name=None), key=str)


def test_nothing_after_as_of_changes_the_estimates(db, ledger, mapped, tmp_path):
    from test_as_of import cut

    for day in [date(2024, 2, 12), date(2024, 3, 1), date(2024, 3, 5), date(2024, 3, 11)]:
        part = warehouse.build(cut(ledger, tmp_path / f"cut{day}", day), tmp_path / f"{day}.db")
        with duckdb.connect(str(part), read_only=True) as con:
            want = _rows(lost.spells(con, day))
        with duckdb.connect(str(db), read_only=True) as con:
            assert _rows(lost.spells(con, day)) == want, day
        with warehouse.as_of(day, db) as con:
            assert _rows(lost.spells(con, day)) == want, day


def test_a_running_spell_is_clipped_at_as_of(db):
    with warehouse.as_of(date(2024, 3, 5), db) as con:
        a = lost.spells(con, date(2024, 3, 5)).set_index(["drug_code", "kind"]) \
            .loc[("A", "stockout")]
    assert (a.stop, a.counted_days) == (date(2024, 3, 5), 4)
    assert a.expected_units == pytest.approx(4 * A_BASE * TRAFFIC_OUT * 4)


def test_uncensor_adds_lost_demand_and_takes_switched_off_the_alternative(db, sp):
    with duckdb.connect(str(db), read_only=True) as con:
        p = lost.panel(con, D1)
        out = lost.uncensor(p, lost.spells(con, D1, p=p), lost.groups(con))
    a, b, d = (p.index[x] for x in "ABD")
    days = slice(p.day(A_OUT[0]), p.day(A_OUT[1]) + 1)
    spell = sp.loc[("A", "stockout")]
    added = out - p.units
    assert added[a].sum() == pytest.approx(spell.lost_units)
    assert added[a, days].sum() == pytest.approx(spell.lost_units)
    # B gives up the switched lines, in its own units (1 per line).
    assert added[b].sum() == pytest.approx(-spell.switch_rate * spell.lost_lines)
    assert added[b, days].sum() == pytest.approx(added[b].sum())
    assert not added[d].any()  # shelf-outs are left as recorded
    assert (out >= 0).all()


def test_the_replay_hook_returns_drugs_by_days(db, tmp_path):
    _, sealed = holdout.seal(db, tmp_path / "agent.duckdb", tmp_path / "holdout.duckdb")
    drugs = ["A", "B", "nope"]
    days = _days(date(2024, 3, 1), date(2024, 3, 31))
    # The fixture is all before the cutoff, so the sealed file is empty and the
    # whole fixture plays the agent's past; these March days stand in for the holdout.
    with holdout.open_sealed(sealed) as hc:
        got = lost.uncensored_demand(db)(hc, drugs, days)
        assert replay.recorded_demand(hc, drugs, days).shape == got.shape == (3, len(days))
    assert not got[2].any() and np.isfinite(got).all()
    a_out = [days.index(d) for d in _days(*A_OUT)]
    assert (got[0, a_out] > 0).all()
    assert got[0].sum() == pytest.approx(4 * (31 - 9) + 4 * A_BASE * TRAFFIC_OUT * 9)
    # No stock-out started before these days, so no switch rate is known yet
    # and B keeps its recorded demand (2 a day, 3 on A's out days).
    assert got[1].sum() == 2 * 31 + 9


@pytest.mark.data
def test_real_lost_demand_passes_the_fake_test_and_the_placebo(real_db):
    from irpd import config

    with duckdb.connect(str(real_db), read_only=True) as con:
        p = lost.panel(con, config.CUTOFF)
        score = lost.fake_score(lost.fake_spells(con, config.CUTOFF, p=p))
        sp = lost.spells(con, config.CUTOFF, p=p)
        pl = lost.placebo(con, config.CUTOFF, p=p)
    for tier, limit in (("fast", 0.03), ("regular", 0.1), ("slow", 0.05)):
        assert abs(score.loc[tier, "bias"]) < limit, tier
    so = sp[(sp.kind == "stockout") & sp.estimated]
    assert so.recorded_units.sum() / so.expected_units.sum() < 0.05  # FINDINGS: ~3%
    real = pl.loc["stock-outs", "uplift_over_drug_expected"]
    assert real > 0.15
    assert abs(pl.loc["placebo: unrelated drugs", "uplift_over_drug_expected"]) < real / 2
