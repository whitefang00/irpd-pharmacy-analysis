from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from irpd import contest, replay
from irpd.evalx import rolling

D0 = date(2024, 7, 1)


def holdout(n_days=40, demand=2.0, price=1000.0, start_floor=10.0, start_store=20.0):
    T = n_days
    z = np.zeros((2, T))
    return replay.Holdout(
        drugs=["A", "B"], days=[D0 + timedelta(days=i) for i in range(T)],
        price=np.array([price, price]), lead=pd.Series({"A": 2.0, "B": 3.0}),
        start_floor=np.array([start_floor, start_floor]),
        start_store=np.array([start_store, start_store]),
        demand=np.full((2, T), demand), adjust=z.copy(), store_opens=D0,
        rate=np.full((2, T), demand), actual_floor=z.copy(), actual_store=z.copy(),
        actual_sold=z.copy(), live=np.ones((2, T), bool))


def window(h, switch=(0.0, 0.5), cap=(np.nan, np.nan)):
    return rolling.Window("toy", h.days[0], h.days[-1], h, target=1.0,
                          switch=np.array(switch), switch_shelf=np.array(switch),
                          floor_cap=np.array(cap, float))


def pol(s, S, fmin=1.0, fmax=5.0):
    return pd.DataFrame({"drug_code": ["A", "B"], "s": [s, s], "S": [S, S],
                         "floor_min": [fmin, fmin], "floor_max": [fmax, fmax]})


def test_one_policy_matches_the_static_simulate():
    h = holdout()
    p = pol(6, 20)
    a = replay.simulate(replay.scaled(p, 1.3), h)
    b = replay.simulate_tv(replay.schedule([(0, p)], h), h, k=1.3)
    for x in ("floor", "store", "sold", "ordered"):
        assert np.allclose(getattr(a, x), getattr(b, x))


def test_schedule_switches_policy_on_its_day_and_keeps_missing_drugs():
    h = holdout()
    later = pol(1, 2).iloc[[0]]  # only drug A re-planned
    p = replay.schedule([(0, pol(6, 20)), (10, later)], h)
    assert p["S"][0, 9] == 20 and p["S"][0, 10] == 2
    assert p["S"][1, 10] == 20  # B keeps its last rule
    with pytest.raises(ValueError):
        replay.schedule([(5, pol(6, 20))], h)


def test_floor_cap_limits_top_ups():
    h = holdout(start_floor=0.0, start_store=100.0)
    p = replay.schedule([(0, pol(-1, 0, fmin=0, fmax=30))], h)
    free = replay.simulate_tv(p, h)
    capped = replay.simulate_tv(p, h, floor_cap=np.array([4.0, np.nan]))
    assert free.floor[0, 0] == 28  # moved 30, sold 2
    assert capped.floor[0, 0] == 2  # moved 4, sold 2
    assert capped.floor[1, 0] == 28  # NaN = no cap


def test_outcome_nets_switching_and_splits_losses():
    h = holdout(start_floor=0.0, start_store=0.0)  # nothing in stock, nothing ordered
    w = window(h, switch=(0.0, 0.5))
    p = replay.schedule([(0, pol(-1, 0))], h)
    o = rolling.run_at(p, w, 1.0)
    T = len(h.days)
    assert o.summary["gross_out"] == pytest.approx(2 * 2.0 * 1000 * T)
    assert o.summary["net_out"] == pytest.approx((1 + 0.5) * 2.0 * 1000 * T)
    assert o.summary["gross_shelf"] == 0 and o.summary["fill_rate"] == 0
    assert set(o.periods["period"]) == {0, 1, 2}
    assert o.periods["gross_out"].sum() == pytest.approx(o.summary["gross_out"])


def test_cycle_service_counts_orders_with_a_stockout_before_arrival():
    stockout = np.array([[0, 0, 1, 0, 0, 0, 0, 0]], bool)
    ordered = np.zeros((1, 8))
    ordered[0, [0, 4]] = 5  # order on day 0 (arrives day 2), day 4 (arrives day 6)
    run = replay.Run(*(np.zeros((1, 8)),) * 3, np.zeros((1, 8)), ordered)
    cs, n = rolling.cycle_service(run, stockout, np.array([2]))
    assert n == 2 and cs == pytest.approx(0.5)


def test_match_hits_the_tolerance():
    h = holdout(n_days=60)
    w = window(h)
    p = replay.schedule([(0, pol(6, 20, fmin=1, fmax=50))], h)
    target = rolling.stock_value(replay.simulate_tv(p, h, 1.7), w)
    o = rolling.match(p, w, target=target, tol=0.0005)
    assert abs(o.summary["stock_value"] / target - 1) < 0.0005


def test_window_var_factor():
    f = contest.window_var_factor(np.array([3, 14, 28]), np.array(["fast"] * 3), {"fast": 0.2})
    assert f.tolist() == pytest.approx([1.0, 1.2, 1.6])
    assert contest.window_var_factor(np.array([28]), np.array(["rare"]), {"fast": 0.2})[0] == 1
    assert (contest.window_var_factor(np.array([28]), np.array(["fast"]), None) == 1).all()


def test_grid_and_first_grid_monday():
    g = rolling.grid()
    assert g[0] == date(2024, 1, 7) and g[-1] == date(2025, 12, 7) and len(g) == 51
    assert len(rolling.grid(last=date(2024, 10, 13))) == 21  # selection origins
    assert rolling.first_grid_monday(date(2024, 7, 1)) == date(2024, 7, 8)
    assert rolling.first_grid_monday(date(2025, 1, 1)) == date(2025, 1, 6)
    assert rolling.first_grid_monday(date(2025, 7, 1)) == date(2025, 7, 7)


def test_arm_source_uses_latest_origin_and_skips_past_weeks(monkeypatch):
    f = pd.DataFrame({"lead_days": [2, 3], "tier": ["fast", "fast"]},
                     index=pd.Index(["A", "B"], name="drug_code"))
    monkeypatch.setattr(rolling, "facts", lambda day: f)
    o = date(2024, 7, 7)
    rows = [{"origin": o, "drug_code": d, "h": h, "week_start": o + timedelta(days=1 + 7 * (h - 1)),
             "mean": 7.0 * h, "var": 14.0 * h, "tier": "fast"} for d in "AB" for h in range(1, 9)]
    src = rolling.ArmSource(pd.DataFrame(rows), name="toy")
    b = src.basis(o)
    assert b.mean[0, 0] == pytest.approx(1.0) and b.mean[0, 7] == pytest.approx(2.0)
    late = src.basis(o + timedelta(days=7))  # a week later: starts at h = 2
    assert late.mean[0, 0] == pytest.approx(2.0)
    assert late.mean.shape[1] == contest.LIMITS["cycle_days"][1] + 3
    assert late.mean[0, -1] == pytest.approx(8.0)  # past h = 8 repeats h = 8
    with pytest.raises(ValueError):
        src.basis(o + timedelta(days=20))
