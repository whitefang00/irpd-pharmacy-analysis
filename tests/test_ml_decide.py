"""The decision replay: bases from any forecaster, the matched comparison, the rule."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from test_contest import WINDOW, files, no_sealed  # noqa: F401  (fixtures)

from irpd import config, contest, holdout, replay
from irpd import forecast as fc
from irpd.ml import decide, schema


class HistoryForecaster:
    """The baseline behind the Forecaster protocol, for tests: forecast.predict on a
    history, the method from each row's tier."""

    name = "baseline-test"

    def __init__(self, hist: dict):
        self.hist = hist

    def fit(self, rows, origin):
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        tier = rows.groupby("drug_code")["tier"].first()
        method = np.array([fc.METHOD_BY_TIER.get(tier.get(c, "none"), "snaive")
                           for c in self.hist["drugs"]], dtype=object)
        p = fc.predict(self.hist, method, 8)
        at = {c: i for i, c in enumerate(self.hist["drugs"])}
        return p["mean"][rows["drug_code"].map(at).to_numpy(), rows["h"].to_numpy() - 1]


class ByWeek:
    """Predicts h units for week h, whatever the drug."""

    name = "by-week"

    def fit(self, rows, origin):
        return self

    def predict(self, rows):
        return rows["h"].to_numpy(float)


def panel_rows(codes, origin: date, tiers: dict | None = None) -> pd.DataFrame:
    rows = pd.DataFrame([(c, h) for c in codes for h in range(1, 9)], columns=["drug_code", "h"])
    rows["origin"] = pd.Timestamp(origin)
    rows["tier"] = [(tiers or {}).get(c, "none") for c in rows["drug_code"]]
    # Shuffled, so the forecaster's output has to be put back in order.
    return rows.sample(frac=1, random_state=0).reset_index(drop=True)


def assert_same_basis(a: contest.Basis, b: contest.Basis, days: int | None = None):
    assert a.day == b.day
    pd.testing.assert_frame_equal(a.facts, b.facts)
    assert a.mean.shape == b.mean.shape
    np.testing.assert_array_equal(a.mean[:, :days], b.mean[:, :days])
    np.testing.assert_array_equal(a.var[:, :days], b.var[:, :days])


# ---- bases on the fixture warehouse ---------------------------------------


def test_baseline_basis_reproduces_contest_basis_exactly(files):  # noqa: F811
    _, agent, _ = files
    day = config.CUTOFF  # a Monday
    with holdout.agent_view(day, agent) as con:
        want = contest.basis(con, day)
        full = decide.baseline_basis(con, day, horizon=None)
        eight = decide.baseline_basis(con, day)
    assert len(want.facts) and want.mean.sum() > 0
    assert_same_basis(full, want)
    # Capped at 8 weeks, it matches on those weeks and gives the same plain policy.
    skip = (day + timedelta(days=1) - decide.week_starts(day, 1)[0]).days
    assert_same_basis(eight, want, days=8 * 7 - skip)
    pd.testing.assert_frame_equal(contest.to_policy({}, eight), contest.to_policy({}, want))


def test_basis_from_a_forecaster_on_panel_rows_matches_the_baseline(files):  # noqa: F811
    _, agent, _ = files
    day = config.CUTOFF
    with holdout.agent_view(day, agent) as con:
        codes = list(contest.facts(con, day).index)
        hist = fc.load_history(con, day, codes)
        rows = panel_rows(codes, schema.policy_origin(day), fc.tiers(con, day))
        got = decide.basis(con, day, HistoryForecaster(hist), rows,
                           decide.baseline_dispersion(con, day, codes))
        want = decide.baseline_basis(con, day)
    assert_same_basis(got, want)


def test_a_tuesday_policy_day_stands_on_the_sunday_and_extends_past_week_8(files):  # noqa: F811
    _, agent, _ = files
    day = date(2025, 6, 24)  # Tuesday
    origin = schema.policy_origin(day)
    assert origin == date(2025, 6, 22)
    with holdout.agent_view(day, agent) as con:
        codes = list(contest.facts(con, day).index)
        disp = pd.Series(2.0, index=codes)
        with pytest.raises(ValueError, match="policy origin"):
            decide.basis(con, day, ByWeek(), panel_rows(codes, day + timedelta(days=5)), disp)
        with pytest.raises(ValueError, match="no forecast for"):
            decide.basis(con, day, ByWeek(), panel_rows(codes[1:], origin), disp)
        b = decide.basis(con, day, ByWeek(), panel_rows(codes, origin), disp)
    # Days from Wed 25 Jun: Wed-Sun are the rest of week 1, then Monday starts week 2.
    np.testing.assert_allclose(b.mean[:, :5], 1 / 7)
    np.testing.assert_allclose(b.mean[:, 5:12], 2 / 7)
    # Week 8 runs to day 5 + 6*7 + 7 = 54; after it, week 8's rate repeats.
    np.testing.assert_allclose(b.mean[:, 47:54], 8 / 7)
    np.testing.assert_allclose(b.mean[:, 54:], 8 / 7)
    assert b.mean.shape[1] == decide.horizon_days(b.facts) > 54
    np.testing.assert_allclose(b.var, 2 * b.mean)


def test_compare_on_a_fixture_window_matches_the_contest_basis(files, no_sealed):  # noqa: F811
    _, agent, _ = files
    (w,) = contest.load_windows(agent, WINDOW)
    with holdout.agent_view(w.day, agent) as con:  # policy day is a Saturday
        mine = decide.baseline_basis(con, w.day)
    res = decide.compare([w], {"contest": w.basis, "decide": mine})
    s = res.summary.set_index("policy")
    assert list(s.index) == ["pharmacist (ledger)", "contest", "decide"]
    cols = ["stockout_days", "shelfout_days", "lost_revenue", "avg_stock_value"]
    pd.testing.assert_series_equal(s.loc["contest", cols], s.loc["decide", cols],
                                   check_names=False)
    assert set(res.per_drug[w.day]) == {"pharmacist (ledger)", "contest", "decide"}


# ---- compare on a hand-made window ----------------------------------------

D0 = date(2025, 1, 1)


def holdout_of(demand: np.ndarray, price=1000.0) -> replay.Holdout:
    n, T = demand.shape
    zeros = np.zeros((n, T))
    drugs = [f"D{i}" for i in range(n)]
    floor = np.full((n, T), 20.0)
    return replay.Holdout(
        drugs=drugs, days=[D0 + timedelta(days=i) for i in range(T)],
        price=np.full(n, price), lead=pd.Series(2.0, index=drugs),
        start_floor=np.full(n, 10.0), start_store=np.zeros(n), demand=demand,
        adjust=zeros, store_opens=D0 + timedelta(days=T), rate=demand.mean(1)[:, None] + zeros,
        actual_floor=floor, actual_store=zeros, actual_sold=demand, live=np.ones((n, T), bool))


def flat_basis(day: date, rates: np.ndarray, disp: float = 1.5) -> contest.Basis:
    n = len(rates)
    f = pd.DataFrame({"lead_days": 2, "floor_rate": np.nan, "tier": "fast", "cadence": "short",
                      "trend": "flat", "long_stockout": False, "switching": "none",
                      "idle_days": 0}, index=pd.Index([f"D{i}" for i in range(n)],
                                                      name="drug_code"))
    mean = np.tile(rates[:, None], (1, 80)).astype(float)
    return contest.Basis(day, f, mean, mean * disp)


def test_compare_replays_each_basis_at_matched_stock():
    rng = np.random.default_rng(0)
    rates = np.array([1.0, 3.0, 6.0, 10.0])
    demand = rng.poisson(rates[:, None], (4, 60)).astype(float)
    day = D0 - timedelta(days=1)
    w = contest.Window(day, D0 + timedelta(days=59), flat_basis(day, rates), holdout_of(demand))
    wrong = flat_basis(day, rates[::-1].copy())  # stock in the wrong drugs
    res = decide.compare([w], {"right": w.basis, "wrong": wrong})
    s = res.summary.set_index("policy")
    # Both hold the pharmacist's stock value; the right forecast loses less.
    target = s.loc["pharmacist (ledger)", "avg_stock_value"]
    assert s.loc["right", "avg_stock_value"] == pytest.approx(target, rel=0.01)
    assert s.loc["wrong", "avg_stock_value"] == pytest.approx(target, rel=0.01)
    assert s.loc["right", "lost_revenue"] < s.loc["wrong", "lost_revenue"]
    assert set(res.factors[day]) == {"right", "wrong"}
    with pytest.raises(ValueError, match="not the policy day"):
        decide.compare([w], {"late": flat_basis(D0, rates)})
    with pytest.raises(ValueError, match="bases for"):
        decide.compare([w, w], {"one": [w.basis]})


# ---- bootstrap and the rule ------------------------------------------------


def per(lost, out, shelf=None) -> pd.DataFrame:
    n = len(lost)
    return pd.DataFrame({"drug_code": [f"D{i}" for i in range(n)],
                         "lost_revenue_stockout": np.asarray(lost, float),
                         "lost_revenue_shelfout": np.zeros(n),
                         "stockout_days": np.asarray(out, float),
                         "shelfout_days": np.zeros(n) if shelf is None else shelf})


def test_bootstrap_is_paired_over_drugs():
    rng = np.random.default_rng(1)
    base = rng.gamma(2, 1000, 200)
    a, b = per(base * 0.8, np.full(200, 10)), per(base, np.full(200, 10))
    boot = decide.paired_bootstrap_replay(a, b, n=500)
    lost = boot.loc["lost_revenue"]
    assert lost["diff"] == pytest.approx(-0.2 * base.sum())
    assert lost["diff_lo"] < lost["diff"] < lost["diff_hi"] < 0
    assert lost["ratio_lo"] == pytest.approx(0.8) and lost["ratio_hi"] == pytest.approx(0.8)
    # The same run against itself: no spread at all.
    same = decide.paired_bootstrap_replay(b, b, n=200).loc["lost_revenue"]
    assert same["diff_lo"] == same["diff_hi"] == 0
    # Drug order doesn't matter; different drug sets do.
    shuffled = a.sample(frac=1, random_state=0)
    pd.testing.assert_frame_equal(decide.paired_bootstrap_replay(shuffled, b, n=500)
                                  .drop(columns=["diff_lo", "diff_hi", "ratio_lo", "ratio_hi"]),
                                  boot.drop(columns=["diff_lo", "diff_hi", "ratio_lo",
                                                     "ratio_hi"]))
    with pytest.raises(ValueError, match="same drugs"):
        decide.paired_bootstrap_replay(a.iloc[1:], b)


def test_decision_rule():
    rng = np.random.default_rng(2)
    base = rng.gamma(2, 1000, 200)
    out = np.full(200, 10.0)

    def verdict(lost_a, out_a, lost_b=base, out_b=out):
        return decide.decide_winner(decide.paired_bootstrap_replay(
            per(lost_a, out_a), per(lost_b, out_b), n=500))

    assert verdict(base * 0.8, out)["winner"] == "challenger"
    assert verdict(base * 0.8, out * 1.04)["winner"] == "challenger"  # within 5%
    v = verdict(base * 0.8, out * 1.10)
    assert v["winner"] is None and "more than 5%" in v["reason"]
    assert verdict(base * 1.25, out)["winner"] == "baseline"
    noisy = base * rng.choice([0.5, 1.5], 200)  # no consistent direction
    v = verdict(noisy, out)
    assert v["winner"] is None and "includes zero" in v["reason"]
    # Shelf-outs are reported, never decisive.
    boot = decide.paired_bootstrap_replay(per(base * 0.8, out, np.full(200, 99.0)),
                                          per(base, out), n=500)
    v = decide.decide_winner(boot)
    assert v["winner"] == "challenger" and v["shelfout_diff"] == 99 * 200
