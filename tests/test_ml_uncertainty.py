"""U1 dispersion and the coverage / pinball tables, on synthetic predictions."""

from datetime import date, timedelta

import ml_fixture
import numpy as np
import pandas as pd
import pytest

from irpd import forecast as fc
from irpd.ml import schema, uncertainty

O0 = date(2024, 1, 7)  # a Sunday


def nb(rng, mu, d):
    """Negative binomial draws with mean mu and variance mu x d (d > 1)."""
    r = mu / (d - 1)
    return rng.negative_binomial(r, 1 / d).astype(float)


def predictions(n_drugs=200, n_origins=60, step=1, d=3.0, tiers=None, seed=0,
                horizon=8, mu_range=(5, 40)) -> pd.DataFrame:
    """Weekly origins; each target week's actual drawn once, shared by every origin."""
    rng = np.random.default_rng(seed)
    drugs = [f"X{i:03d}" for i in range(n_drugs)]
    tiers = tiers or {c: "fast" for c in drugs}
    disp = d if isinstance(d, dict) else {c: d for c in drugs}
    level = dict(zip(drugs, rng.uniform(*mu_range, n_drugs), strict=True))
    weeks = n_origins * step + horizon + 1
    true = {c: level[c] * (1 + 0.3 * np.sin(np.arange(weeks) / 5)) for c in drugs}
    actual = {c: nb(rng, true[c], disp[c]) for c in drugs}
    rows = []
    for k in range(n_origins):
        origin = O0 + timedelta(weeks=k * step)
        for c in drugs:
            for h in range(1, horizon + 1):
                t = k * step + h
                rows.append((origin, c, tiers[c], h, actual[c][t], true[c][t]))
    p = pd.DataFrame(rows, columns=["origin", "drug_code", "tier", "h", "actual", "mean"])
    p["origin"] = pd.to_datetime(p["origin"])
    p["method"], p["eid"], p["scale"] = "m", False, 1.0
    p["var"], p["config_hash"], p["seed"] = p["mean"], "abc", schema.ENSEMBLE_SEED
    return p[schema.PREDICTION_COLUMNS]


def at(k: int) -> date:
    return O0 + timedelta(weeks=k)


# ---- the estimator ----------------------------------------------------------


def test_u1_recovers_a_known_dispersion_within_ten_percent():
    p = predictions(d=3.0)
    tiers = pd.Series("fast", index=p.drug_code.unique())
    d = uncertainty.u1_dispersion(p, at(60), tiers)
    assert d.index.name == "drug_code" and len(d) == 200
    assert d.mean() == pytest.approx(3.0, rel=0.10)
    assert d.min() >= 1


def test_u1_recovers_each_tiers_dispersion():
    drugs = [f"X{i:03d}" for i in range(200)]
    tiers = {c: "fast" if i < 100 else "slow" for i, c in enumerate(drugs)}
    disp = {c: 2.0 if tiers[c] == "fast" else 6.0 for c in drugs}
    p = predictions(d=disp, tiers=tiers, seed=3)
    d = uncertainty.u1_dispersion(p, at(60), pd.Series(tiers))
    by = d.groupby(pd.Series(tiers)).mean()
    assert by["fast"] == pytest.approx(2.0, rel=0.10)
    assert by["slow"] == pytest.approx(6.0, rel=0.10)


def test_poisson_errors_floor_at_one():
    p = predictions(d=3.0)
    p["actual"] = p["mean"]  # perfect forecasts: no error at all
    d = uncertainty.u1_dispersion(p, at(30), pd.Series("fast", index=p.drug_code.unique()))
    assert (d == 1.0).all()


# ---- the residual rule ------------------------------------------------------


def toy(rows) -> pd.DataFrame:
    p = pd.DataFrame(rows, columns=["origin", "h", "actual", "mean"])
    p["origin"] = pd.to_datetime(p["origin"])
    p["drug_code"], p["tier"], p["method"], p["eid"] = "A", "fast", "m", False
    p["var"], p["scale"], p["config_hash"], p["seed"] = p["mean"], 1.0, "c", schema.ENSEMBLE_SEED
    return p[schema.PREDICTION_COLUMNS]


def test_residuals_use_earlier_origins_h_up_to_4_smallest_h_and_skip_censored():
    o1, o2, o3 = at(0), at(1), at(3)
    p = toy([
        (at(-6), 5, 1.0, 1.0),     # week of 25 Dec: only an h = 5 forecast, never counts
        (at(-4), 4, 2.0, 1.0),     # week of 1 Jan, h = 4: counts
        (o1, 1, 5.0, 1.0),         # week of 8 Jan
        (o1, 2, 6.0, 2.0),         # week of 15 Jan, superseded by (o2, 1)
        (o2, 1, 6.0, 9.0),
        (o1, 3, np.nan, 3.0),      # week of 22 Jan: censored at every origin
        (o2, 2, np.nan, 4.0),
        (o2, 3, 8.0, 4.0),         # ends after o3
        (o3, 1, 1.0, 1.0),         # the scoring origin's own forecast
    ])
    r = uncertainty.residuals(p, o3)
    got = {(row.target_week.date(), row.h): (row.y, row.mu) for row in r.itertuples()}
    assert got == {(at(-1) + timedelta(days=1), 4): (2.0, 1.0),
                   (o1 + timedelta(days=1), 1): (5.0, 1.0),
                   (o2 + timedelta(days=1), 1): (6.0, 9.0)}
    # A day before o2's week ends, the week of 15 Jan isn't usable yet.
    assert len(uncertainty.residuals(p, o2 + timedelta(days=6))) == 2


def test_seed_rows_and_configs():
    p = predictions(n_drugs=5, n_origins=5)
    per_seed = p.assign(seed=0, mean=p["mean"] * 2)
    both = pd.concat([p, per_seed])
    r = uncertainty.residuals(both, at(5))
    pd.testing.assert_frame_equal(r, uncertainty.residuals(p, at(5)))  # ensemble rows only
    with pytest.raises(ValueError, match="one config_hash"):
        uncertainty.residuals(pd.concat([p, p.assign(config_hash="other")]), at(5))


# ---- shrinkage and fallbacks ------------------------------------------------


def test_shrinkage_toward_the_tier_pool():
    # Two drugs in one tier: A with 30 residual weeks, B with 5.
    rng = np.random.default_rng(0)
    rows = []
    for c, weeks in (("A", 30), ("B", 5)):
        for k in range(weeks):
            rows.append((at(k), c, 1, float(rng.poisson(10) + (5 if c == "A" else 0)), 10.0))
    p = pd.DataFrame(rows, columns=["origin", "drug_code", "h", "actual", "mean"])
    p["tier"], p["seed"], p["config_hash"] = "fast", schema.ENSEMBLE_SEED, "c"
    d = uncertainty.u1_dispersion(p, at(40), pd.Series({"A": "fast", "B": "fast", "C": "fast"}))
    se = (p["actual"] - p["mean"]) ** 2
    own_a = max(se[p.drug_code == "A"].sum() / 300, 1)
    pooled = max(se.sum() / p["mean"].sum(), 1)
    w = 30 / 50
    assert d["A"] == pytest.approx(w * own_a + (1 - w) * pooled)
    assert d["B"] == pytest.approx(pooled)  # fewer than 8 weeks
    assert d["C"] == pytest.approx(pooled)  # no residuals at all
    # Only the last 52 weeks count.
    long = predictions(n_drugs=1, n_origins=100, d=4.0)
    late = long[long.origin >= pd.Timestamp(at(48))]
    tiers = pd.Series("fast", index=long.drug_code.unique())
    pd.testing.assert_series_equal(uncertainty.u1_dispersion(long, at(100), tiers),
                                   uncertainty.u1_dispersion(late, at(100), tiers))


def test_fallbacks_at_early_origins_and_for_empty_tiers():
    p = predictions(n_drugs=4, n_origins=20)
    tiers = pd.Series({"X000": "fast", "X001": "fast", "X002": "rare", "X003": "rare"})
    base = pd.Series({"X000": 1.5, "X001": 2.5, "X002": 3.5, "X003": 4.5})
    with pytest.raises(ValueError, match="baseline fallback"):
        uncertainty.u1_dispersion(p, at(0), tiers)
    # No earlier predictions: the baseline recipe, whole.
    pd.testing.assert_series_equal(uncertainty.u1_dispersion(p, at(0), tiers, base),
                                   base.rename("dispersion").rename_axis("drug_code"))
    called = []
    uncertainty.u1_dispersion(p, at(0), tiers, lambda o: called.append(o) or base)
    assert called == [at(0)]
    # A tier with no residual weeks takes its drugs' fallback values.
    only_fast = p[p.drug_code.isin(["X000", "X001"])]
    d = uncertainty.u1_dispersion(only_fast, at(20), tiers, base)
    assert d["X002"] == 3.5 and d["X003"] == 4.5 and d["X000"] != 1.5


def test_baseline_dispersion_matches_forecast_predict():
    h, _ = ml_fixture.synthetic_history()
    ctx = schema.PanelContext(None, date(2025, 6, 29), [date(2024, 6, 30)], h)
    origin = date(2024, 6, 30)
    d = uncertainty.baseline_dispersion(ctx, origin)
    cut = ctx.history_at(origin)
    p = fc.predict(cut, np.full(len(cut["drugs"]), "ses+eid", dtype=object), 8)
    np.testing.assert_allclose(d.to_numpy(), p["var"][:, 0] / p["mean"][:, 0])
    assert list(d.index) == list(cut["drugs"])


def test_apply_and_with_u1():
    p = predictions(n_drugs=30, n_origins=12, step=4, d=2.5)
    disp = pd.Series(2.0, index=p.drug_code.unique())
    out = uncertainty.apply(p, disp)
    np.testing.assert_allclose(out["var"], 2 * out["mean"])
    with pytest.raises(ValueError, match="no dispersion"):
        uncertainty.apply(p, disp.iloc[1:])
    base = pd.Series(7.0, index=p.drug_code.unique())
    u = uncertainty.with_u1(p, fallback=lambda o: base)
    schema.validate_predictions(u)
    first = u[u.origin == u.origin.min()]
    np.testing.assert_allclose(first["var"], 7 * first["mean"])  # baseline at the first origin
    second = u[u.origin == pd.Timestamp(at(4))]
    ratio = (second["var"] / second["mean"]).round(9)
    assert ratio.nunique() == 1  # 4 residual weeks each: the tier pool
    last = u[u.origin == u.origin.max()]
    assert (last["var"] / last["mean"]).mean() == pytest.approx(2.5, rel=0.15)


# ---- coverage and pinball ---------------------------------------------------


def test_window_totals():
    p = toy([(at(0), h, float(h), 1.0) for h in range(1, 9)])
    p.loc[p.h == 6, "actual"] = np.nan
    t = uncertainty.window_totals(p, 4)
    assert t["mean"].tolist() == [4.0, 4.0] and t["var"].tolist() == [4.0, 4.0]
    assert t["actual"].iloc[0] == 10 and np.isnan(t["actual"].iloc[1])
    t2 = uncertainty.window_totals(p.iloc[:7], 2)  # h = 7 alone is an incomplete block
    assert len(t2) == 3 and t2["actual"].tolist()[:2] == [3, 7] and np.isnan(t2.actual.iloc[2])


def test_coverage_of_a_correctly_specified_nb_is_near_nominal():
    # Weekly actuals independent given the mean, so window totals are NB-ish too.
    p = predictions(n_drugs=150, n_origins=10, step=8, d=3.0, seed=5)
    p = uncertainty.apply(p, pd.Series(3.0, index=p.drug_code.unique()))
    table = uncertainty.coverage_pinball(p).set_index(["tier", "window", "q"])
    for w in (1, 2, 4):
        for q in (0.9, 0.95):
            row = table.loc[("fast", w, q)]
            assert row["coverage"] == pytest.approx(q, abs=0.03), (w, q)
            assert row["n"] == 150 * 10 * (8 // w)
    # Too little variance under-covers and loses more.
    thin = uncertainty.coverage_pinball(uncertainty.apply(p, pd.Series(
        1.0, index=p.drug_code.unique()))).set_index(["tier", "window", "q"])
    assert thin.loc[("fast", 4, 0.95), "coverage"] < 0.9
    assert thin.loc[("fast", 4, 0.95), "pinball"] > table.loc[("fast", 4, 0.95), "pinball"]


def test_pinball_loss():
    y, q_hat = np.array([10.0, 10.0]), np.array([8.0, 12.0])
    np.testing.assert_allclose(uncertainty.pinball(y, q_hat, 0.9), [1.8, 0.2])
