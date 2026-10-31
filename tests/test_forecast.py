from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd
import pytest

from irpd import config, forecast
from irpd.tables import sales

AS_OF = date(2024, 6, 30)  # a Sunday


def make_db(daily: dict[str, callable], out_days: set[tuple[str, date]] = frozenset(),
            last: date = date(2024, 12, 31)):
    """An in-memory warehouse with sales tables built from a daily units rule per drug.

    daily: drug -> f(day) giving units sold that day (0 = no sale line).
    out_days: (drug, day) pairs where total stock ends the day at 0.
    """
    days = pd.date_range(config.FIRST_DAY, last).date
    rows = [(d, f"{c}{d}", c, float(u), "Tab", 100)
            for d in days for c, f in daily.items() if (u := f(d)) > 0]
    lines = pd.DataFrame(rows, columns=["date", "invoice", "drug_code", "qty", "unit",
                                        "unit_price"])
    con = duckdb.connect()
    con.register("l", lines)
    con.execute("""CREATE TABLE sales_lines AS SELECT date, invoice, drug_code,
        false AS is_service, qty, unit, unit_price, (qty * unit_price)::BIGINT AS revenue
        FROM l""")
    con.execute("""CREATE TABLE stock_moves AS SELECT date, invoice AS document, drug_code,
        'sale' AS move_type, 0 AS qty_in, qty::BIGINT AS qty_out, unit FROM l""")
    con.execute("CREATE TABLE drugs AS SELECT DISTINCT drug_code, 'Tab' AS stock_unit FROM l")
    sales.build(con)
    stock = pd.DataFrame([(d, c, "total", 0 if (c, d) in out_days else 50)
                          for d in days for c in daily], columns=["date", "drug_code",
                                                                  "location", "closing"])
    con.register("s", stock)
    con.execute("CREATE TABLE daily_stock AS SELECT * FROM s")
    return con


def steady(units):
    return lambda d: units


def test_methods_on_a_flat_series_return_its_level():
    y = np.array([[5.0] * 20, [0, 0, 4, 0, 0, 4] * 3 + [0, 0]])
    y[0, 7] = np.nan  # a censored week is skipped, not read as zero
    assert forecast._ses(y)[0] == pytest.approx(5)
    assert forecast._mean_last(y)[0] == pytest.approx(5)
    assert forecast._tsb(y)[1] == pytest.approx(4 / 3, rel=0.3)
    assert forecast._croston_sba(y)[1] == pytest.approx((1 - 0.15 / 2) * 4 / 3, rel=0.05)


def test_demand_quantile_matches_poisson_and_negative_binomial():
    q = forecast.demand_quantile
    assert q(np.array([5.0]), np.array([5.0]), 0.5)[0] == 5  # Poisson(5)
    assert q(np.array([5.0]), np.array([5.0]), 0.9)[0] == 8
    assert q(np.array([0.0]), np.array([0.0]), 0.9)[0] == 0
    rng = np.random.default_rng(1)
    for mean, var in [(2.0, 10.0), (300.0, 3000.0), (3000.0, 9000.0)]:
        r = mean**2 / (var - mean)
        sim = rng.negative_binomial(r, r / (r + mean), 200_000)
        got = q(np.array([mean]), np.array([var]), 0.9)[0]
        assert got == pytest.approx(np.quantile(sim, 0.9), rel=0.01, abs=1)


def test_forecast_rows_and_columns():
    con = make_db({"A": steady(10), "B": lambda d: 3 if d.weekday() == 0 else 0})
    df = forecast.forecast(con, ["A", "B", "NEW"], AS_OF, horizon_weeks=4)
    assert list(df.columns) == ["drug_code", "week_start", "week_end", "mean", "var",
                                "p50", "p90", "tier", "method"]
    assert len(df) == 12
    a = df[df.drug_code == "A"]
    assert a.week_start.tolist() == [AS_OF + timedelta(days=1 + 7 * i) for i in range(4)]
    assert (a.week_end == a.week_start + timedelta(days=6)).all()
    assert a["mean"].to_numpy() == pytest.approx(70)
    assert (a.p90 >= a.p50).all() and (a.p50 > 0).all()
    assert set(a.tier) == {"fast"} and set(a.method) == {"ses+eid"}
    b = df[df.drug_code == "B"]
    assert b["mean"].to_numpy() == pytest.approx(3) and set(b.tier) == {"regular"}
    new = df[df.drug_code == "NEW"]
    assert (new["mean"] == 0).all() and set(new.method) == {"none"}


def test_forecast_uses_nothing_after_as_of():
    def after(units, later):
        return lambda d: units if d <= AS_OF else later

    con = make_db({"A": steady(10), "B": steady(2)})
    changed = make_db({"A": after(10, 500), "B": after(2, 0)},
                      out_days={("A", AS_OF + timedelta(days=i)) for i in range(1, 60)})
    pd.testing.assert_frame_equal(forecast.forecast(con, None, AS_OF, 8),
                                  forecast.forecast(changed, None, AS_OF, 8))


def test_stockout_weeks_are_censored_not_read_as_low_demand():
    # A sells 10 a day, but runs out (and sells nothing) for the last 3 weeks.
    out = {("A", AS_OF - timedelta(days=i)) for i in range(21)}
    con = make_db({"A": lambda d: 0 if ("A", d) in out else 10}, out_days=out)
    df = forecast.forecast(con, ["A"], AS_OF, 2)
    assert df["mean"].to_numpy() == pytest.approx(70)


def test_eid_uplift_is_learned_from_past_eids_and_applied_to_the_next():
    # In the week after each Eid, A sells twice as much and C and D (which
    # sell nothing otherwise) double the store's sale lines.
    def after_eid(d):
        return any(1 <= (d - e).days <= 7 for e in forecast.EID)

    con = make_db({"A": lambda d: 20 if after_eid(d) else 10, "B": steady(1),
                   "C": after_eid, "D": after_eid}, last=date(2025, 6, 30))
    lines = pd.Series(con.execute("""SELECT sum(lines) FROM daily_demand
        GROUP BY date ORDER BY date""").df().iloc[:, 0].to_numpy(),
        index=pd.date_range(config.FIRST_DAY, date(2025, 6, 30)).date)
    f = forecast.eid_factors(lines[lines.index <= date(2024, 12, 31)])
    assert f[3] == pytest.approx(1 + forecast.EID_SHRINK)  # doubled lines, shrunk
    assert f[-3] == pytest.approx(1) and f[20] == pytest.approx(1)

    df = forecast.forecast(con, ["A"], date(2025, 3, 23), 4).set_index("week_start")
    plain, eid_week = df.loc[date(2025, 3, 24), "mean"], df.loc[date(2025, 3, 31), "mean"]
    assert plain == pytest.approx(70, rel=0.05)  # history de-Eided before fitting
    assert eid_week > 1.4 * plain


def test_backtest_origins_stay_before_the_cutoff():
    origins = forecast.backtest_origins()
    assert origins[0] == date(2024, 1, 7) and all(o.weekday() == 6 for o in origins)
    assert origins[-1] + timedelta(weeks=8) <= config.CUTOFF


def test_backtest_and_score_on_a_tiny_db():
    con = make_db({"A": steady(10), "B": lambda d: 3 if d.weekday() == 0 else 0},
                  last=config.CUTOFF)
    bt = forecast.backtest(con, [date(2024, 3, 3), date(2025, 3, 2)], horizon=4)
    s = forecast.score(bt).set_index(["period", "tier", "method"])
    assert s.loc[("2024", "fast", "ses"), "wape"] == pytest.approx(0)
    assert s.loc[("2024", "fast", "zero"), "wape"] == pytest.approx(1)
    assert s.loc[("2025H1", "regular", "mean8"), "cover90"] == 1


@pytest.mark.data
def test_real_forecast_is_fast_and_covers_the_drugs(real_db):
    import time

    with duckdb.connect(str(real_db), read_only=True) as con:
        start = time.time()
        df = forecast.forecast(con, None, config.CUTOFF, 8)
        assert time.time() - start < 10
    assert df.drug_code.nunique() > 2000
    assert set(df.method) <= {"ses+eid", "snaive", "none"}
