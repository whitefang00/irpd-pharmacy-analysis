from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from irpd import forecast as fc
from irpd.evalx import scorer

O = date(2024, 1, 7)  # a Sunday


def test_pinball_known_values():
    assert scorer.pinball(np.array([10.0]), np.array([8.0]), 0.95)[0] == pytest.approx(1.9)
    assert scorer.pinball(np.array([8.0]), np.array([10.0]), 0.95)[0] == pytest.approx(0.1)


def _crps_by_expectation(mean, var, y):
    """E|X - y| - 0.5 E|X - X'| for an NB, by enumeration."""
    d = var / mean
    r, p = mean / (d - 1), 1 / d
    z = np.arange(0, int(mean + 40 * np.sqrt(var)) + 50)
    pmf = stats.nbinom.pmf(z, r, p)
    e1 = (pmf * np.abs(z - y)).sum()
    e2 = (pmf[:, None] * pmf[None, :] * np.abs(z[:, None] - z[None, :])).sum()
    return e1 - 0.5 * e2


@pytest.mark.parametrize("mean,var,y", [(3.0, 6.0, 0.0), (3.0, 6.0, 4.0), (3.0, 6.0, 4.6),
                                        (20.0, 60.0, 31.2), (0.4, 0.5, 2.0)])
def test_crps_nb_matches_the_expectation_form(mean, var, y):
    got = scorer.crps_nb(np.array([mean]), np.array([var]), np.array([y]))[0]
    assert got == pytest.approx(_crps_by_expectation(mean, var, y), rel=1e-6, abs=1e-9)


def test_crps_of_a_point_mass_at_zero_is_y():
    got = scorer.crps_nb(np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 3.5]))
    assert got.tolist() == [0.0, 3.5]


def test_window_weights_spread_weeks_over_days():
    w = scorer.window_weights(np.array([3, 17]), weeks=4)
    assert w[0].tolist() == pytest.approx([3 / 7, 0, 0, 0])
    assert w[1].tolist() == pytest.approx([1, 1, 3 / 7, 0])


def _toy(mean_by_drug, lead=3):
    rows = []
    for d, m in mean_by_drug.items():
        for h in range(1, 9):
            rows.append({"origin": O, "drug_code": d, "h": h,
                         "week_start": O + timedelta(days=1 + 7 * (h - 1)),
                         "mean": m, "var": 2 * m, "tier": "fast"})
    preds = pd.DataFrame(rows)
    days = [O + timedelta(days=i) for i in range(1, 60)]
    daily = pd.DataFrame([{"drug_code": d, "date": day, "units": 1.0, "uncensored_units": 1.0,
                           "stockout": (d == "B" and day == O + timedelta(days=2))}
                          for d in mean_by_drug for day in days])
    labels = pd.DataFrame({"origin": O, "drug_code": list(mean_by_drug), "lead_days": lead,
                           "price": 100.0})
    return preds, daily, labels


def test_score_arm_builds_windows_like_to_policy():
    preds, daily, labels = _toy({"A": 7.0, "B": 14.0})
    s = scorer.score_arm(preds, labels=labels, daily=daily)
    w = s.windows.set_index("drug_code")
    # Lead 3 days: 3/7 of week 1; lead + cycle = 17 days.
    assert w.loc["A", "m_l"] == pytest.approx(3.0)
    assert w.loc["A", "m_lc"] == pytest.approx(17.0)
    assert w.loc["A", "y_lc"] == pytest.approx(17.0)
    q = fc.demand_quantile(np.array([17.0]), np.array([34.0]), 0.95)[0]
    assert w.loc["A", "q_lc"] == q
    assert w.loc["A", "pin_lc"] == pytest.approx(0.05 * (q - 17))
    assert w.loc["A", "f4"] == pytest.approx(28.0) and w.loc["A", "y4"] == pytest.approx(28.0)
    # B has a stock-out day inside its window: censored, scored, reported apart.
    assert not w.loc["A", "censored"] and w.loc["B", "censored"]
    summ = s.summary
    assert summ.loc[("all", "fast"), "windows"] == 2
    assert summ.loc[("censored", "fast"), "windows"] == 1
    assert summ.loc[("all", "fast"), "bias4"] == pytest.approx((28 + 56) / 56)
    assert summ.loc[("all", "fast"), "cov_lc"] == pytest.approx(1.0)


def test_variance_factor_widens_only_multiweek_windows():
    preds, daily, labels = _toy({"A": 7.0})
    plain = scorer.score_arm(preds, labels=labels, daily=daily).windows.iloc[0]
    wide = scorer.score_arm(preds, labels=labels, daily=daily, rho={"fast": 0.5}).windows.iloc[0]
    assert wide["v_l"] == pytest.approx(plain["v_l"])          # 3 days: factor 1
    assert wide["v_lc"] == pytest.approx(plain["v_lc"] * (1 + 0.5 * (17 / 7 - 1)))


def test_weekly_truth_is_prorated_when_no_daily():
    preds, daily, labels = _toy({"A": 7.0})
    weekly = scorer.weekly_from_daily(daily)
    s = scorer.score_arm(preds, actuals=weekly, labels=labels)
    assert s.windows.iloc[0]["y_lc"] == pytest.approx(17.0)


def test_compare_arms_identical_is_zero_and_shift_is_detected():
    preds, daily, labels = _toy({f"D{i}": 5.0 + i for i in range(30)})
    a = scorer.score_arm(preds, labels=labels, daily=daily)
    same = scorer.compare_arms(a, a, metrics=["pinball_lc"], tiers=["pooled"], subsets=["all"])
    assert same.iloc[0]["diff"] == 0 and same.iloc[0]["lo"] == 0 and same.iloc[0]["hi"] == 0
    worse = scorer.score_arm(preds.assign(mean=preds["mean"] * 3, var=preds["var"] * 3),
                             labels=labels, daily=daily)
    c = scorer.compare_arms(worse, a, metrics=["pinball_lc"], tiers=["pooled"], subsets=["all"])
    assert c.iloc[0]["worse"] and not c.iloc[0]["non_inferior"]
