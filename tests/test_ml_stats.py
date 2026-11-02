from datetime import date

import numpy as np
import pandas as pd
import pytest

from irpd.ml import stats

ORIGINS = [pd.Timestamp(date(2024, 1, 7)), pd.Timestamp(date(2024, 2, 4)),
           pd.Timestamp(date(2024, 3, 3))]


def _pred(actual, mean, drugs, origins, tiers):
    return pd.DataFrame({"origin": origins, "drug_code": drugs, "h": 1, "tier": tiers,
                         "actual": actual, "mean": mean})


def test_constant_ratio_gives_exact_interval():
    rng = np.random.default_rng(1)
    n = 60
    actual = rng.uniform(1, 50, n)
    drugs, origins = [f"D{i}" for i in range(n)], [ORIGINS[0]] * n
    a = _pred(actual, actual * 1.3, drugs, origins, "fast")     # WAPE 0.3 on any resample
    b = _pred(actual, actual * 0.9, drugs, origins, "fast")     # WAPE 0.1
    r = stats.paired_bootstrap(a, b, n=500).iloc[0]
    assert r["diff"] == pytest.approx(0.2)
    assert r["lo"] == pytest.approx(0.2) and r["hi"] == pytest.approx(0.2)


def test_interval_contains_population_difference():
    # A population of drugs where the true pooled WAPE difference is known.
    rng = np.random.default_rng(7)
    n = 400
    actual = np.full(n, 10.0)
    err_b = rng.uniform(0, 4, n)                 # E = 2
    err_a = err_b + rng.normal(0.5, 1.0, n)      # E[a - b] = 0.5, so true diff = 0.05
    err_a = np.abs(err_a)
    truth = 0.05
    drugs = [f"D{i}" for i in range(n)]
    o = [ORIGINS[0]] * n
    a = _pred(actual, actual + err_a, drugs, o, "slow")
    b = _pred(actual, actual + err_b, drugs, o, "slow")
    r = stats.paired_bootstrap(a, b, n=2000, seed=3).iloc[0]
    assert r["lo"] < truth < r["hi"]
    assert r["lo"] < r["diff"] < r["hi"]
    assert r["drugs"] == n


def test_resamples_drugs_not_rows_and_groups_by_tier():
    # Two drugs with 3 rows each; per tier the drugs are the resampling unit.
    drugs = ["A"] * 3 + ["B"] * 3 + ["C"] * 3
    origins = ORIGINS * 3
    tiers = ["fast"] * 6 + ["rare"] * 3
    actual = np.full(9, 10.0)
    a = _pred(actual, actual + 1, drugs, origins, tiers)
    b = _pred(actual, actual, drugs, origins, tiers)
    r = stats.paired_bootstrap(a, b, n=200).set_index("tier")
    assert r.loc["fast", "drugs"] == 2 and r.loc["fast", "rows"] == 6
    assert r.loc["fast", "diff"] == pytest.approx(0.1)
    assert r.loc["rare", "drugs"] == 1
    overall = stats.paired_bootstrap(a, b, by=None, n=200)
    assert overall["group"].tolist() == ["all"] and overall["drugs"].iloc[0] == 3


def test_only_shared_known_rows_are_compared():
    a = _pred([10.0, np.nan, 10.0], [12.0, 5.0, 10.0], ["A", "A", "B"], ORIGINS, "fast")
    b = _pred([10.0, np.nan], [10.0, 5.0], ["A", "A"], ORIGINS[:2], "fast")
    r = stats.paired_bootstrap(a, b, n=50).iloc[0]
    assert r["rows"] == 1 and r["diff"] == pytest.approx(0.2)


def test_duplicates_refused():
    a = _pred([10.0, 10.0], [1.0, 2.0], ["A", "A"], [ORIGINS[0]] * 2, "fast")
    with pytest.raises(ValueError, match="one method"):
        stats.paired_bootstrap(a, a)


def test_origin_wins():
    drugs = ["A", "B"] * 3
    origins = [o for o in ORIGINS for _ in range(2)]
    actual = np.full(6, 10.0)
    # A is better at the first two origins, worse at the third.
    mean_a = np.array([11, 11, 11, 11, 15, 15], float)
    mean_b = np.array([12, 12, 12, 12, 10, 10], float)
    per, summary = stats.origin_wins(_pred(actual, mean_a, drugs, origins, "fast"),
                                     _pred(actual, mean_b, drugs, origins, "fast"))
    assert per["diff"].round(6).tolist() == [-0.1, -0.1, 0.5]
    s = summary.iloc[0]
    assert (s["origins"], s["a_wins"], s["b_wins"], s["ties"]) == (3, 2, 1, 0)
