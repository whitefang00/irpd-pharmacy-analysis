"""Public fixture contracts and as-of safety, without private data."""

from datetime import date, timedelta
from time import perf_counter

import numpy as np
import pandas as pd
import pytest
from ml_fixture import (
    PATTERNS,
    synthetic_drugs,
    synthetic_history,
    synthetic_panel,
    synthetic_prices,
    synthetic_store,
)

from irpd import config, forecast
from irpd.ml import schema, split


@pytest.fixture(scope="module")
def generated():
    return synthetic_history()


def test_determinism(generated):
    history, truth = generated
    again, again_truth = synthetic_history()
    for key in ("drugs", "weeks", "y", "out"):
        np.testing.assert_array_equal(history[key], again[key])
    pd.testing.assert_series_equal(history["lines"], again["lines"])
    pd.testing.assert_frame_equal(truth, again_truth)
    for build in (synthetic_drugs, synthetic_prices, synthetic_store):
        pd.testing.assert_frame_equal(build(history), build(again))
    pd.testing.assert_frame_equal(synthetic_panel(history, split.selection_origins()),
                                  synthetic_panel(again, split.selection_origins()))
    other, _ = synthetic_history(seed=1)
    assert not np.array_equal(history["y"], other["y"], equal_nan=True)


def test_history_format(generated):
    history, truth = generated
    assert set(history) == {"drugs", "weeks", "y", "out", "lines", "stock_known", "revenue",
                            "shelfout_days", "first_sale"}
    assert history["drugs"].dtype == object
    assert list(history["drugs"]) == sorted(history["drugs"])
    assert history["weeks"].dtype == object
    assert history["y"].dtype == np.float64
    assert history["out"].dtype == bool
    assert history["y"].shape == history["out"].shape == (28, 130)
    assert all(type(w) is date and w.weekday() == 0 for w in history["weeks"])
    assert history["weeks"][0] == date(2023, 1, 2)
    assert history["weeks"][-1] == date(2025, 6, 23)
    assert set(truth.columns) == {"drug_code", "week_start", "true_mean", "pattern"}
    assert truth.groupby("pattern").drug_code.nunique().to_dict() == dict.fromkeys(PATTERNS, 4)
    assert not truth.duplicated(["drug_code", "week_start"]).any()
    lines = history["lines"]
    assert lines.dtype == np.float64
    assert list(lines.index) == [date(2023, 1, 2) + timedelta(days=i) for i in range(910)]
    store = synthetic_store(history)
    pd.testing.assert_series_equal(store.set_index("date").lines, lines.rename("lines"))


def test_censoring_and_late_starts(generated):
    history, truth = generated
    assert not history["out"][:, history["weeks"] < config.STOCK_START].any()
    mean = truth.pivot(index="drug_code", columns="week_start", values="true_mean")
    mean = mean.reindex(index=history["drugs"], columns=history["weeks"]).to_numpy()
    assert history["out"].any()
    assert not history["out"].all(axis=0).any()
    assert np.all(history["y"][history["out"]] <= 0.3 * mean[history["out"]])
    lengths = set()
    for i, code in enumerate(history["drugs"]):
        mask = np.r_[False, history["out"][i], False].astype(int)
        starts, ends = np.flatnonzero(np.diff(mask) == 1), np.flatnonzero(np.diff(mask) == -1)
        lengths.update(ends - starts)
        if "late_start" in code:
            y = history["y"][i]
            first = np.flatnonzero(y > 0)[0]
            assert date(2024, 6, 1) <= history["weeks"][first] <= date(2024, 8, 1)
            assert np.isnan(y[:first]).all()
            assert np.isfinite(y[first:]).all()
    assert lengths and min(lengths) >= 1 and max(lengths) <= 6


def test_eid_and_baseline_compatibility(generated):
    history, _ = generated
    factors = forecast.eid_factors(history["lines"])
    assert set(factors) == set(forecast.EID_DAYS)
    assert min(factors.values()) > 1.2
    # Full weeks inside the window should retain a substantial uplift.
    indices = forecast.eid_index(history["weeks"], factors)
    assert indices.max() > 1.2
    for method in forecast.CANDIDATES:
        prediction = forecast.predict(history, np.full(len(history["drugs"]), method), 8)
        assert prediction["mean"].shape == prediction["var"].shape == (28, 8)
        assert np.isfinite(prediction["mean"]).all()
        assert (prediction["mean"] >= 0).all()
        assert (prediction["var"] >= prediction["mean"]).all()


def test_panel_contract_and_bookkeeping(generated):
    history, _ = generated
    origins = split.selection_origins()
    panel = synthetic_panel(history, origins)
    schema.validate_panel(panel)
    assert set(panel.origin.dt.date) == set(origins)
    assert panel.groupby(schema.KEYS[:2], observed=True).size().eq(8).all()
    assert panel.censored.any()
    assert panel.loc[panel.censored, "target"].isna().all()
    assert panel.groupby("tier", observed=True).size().gt(0).all()
    for origin in origins:
        n = sum(w + timedelta(days=6) <= origin for w in history["weeks"])
        expected = history["drugs"][np.any(history["y"][:, max(0, n - 52):n] > 0, axis=1)]
        subset = panel[panel.origin.dt.date == origin]
        assert set(subset.drug_code) == set(expected)
        for row in subset.itertuples(index=False):
            i = list(history["drugs"]).index(row.drug_code)
            target_i = n + row.h - 1
            censored = history["out"][i, target_i]
            expected_target = np.nan if censored else history["y"][i, target_i]
            np.testing.assert_equal(row.target, np.float32(expected_target))
    # Verify rolling statistics against the masked grid, including adjacent MASE gaps.
    origin = origins[-1]
    n = sum(w + timedelta(days=6) <= origin for w in history["weeks"])
    for row in panel[(panel.origin.dt.date == origin) & (panel.h == 1)].itertuples():
        i = list(history["drugs"]).index(row.drug_code)
        masked = np.where(history["out"][i, :n], np.nan, history["y"][i, :n])
        np.testing.assert_allclose(row.mase_scale, np.nanmean(np.abs(np.diff(masked))), rtol=1e-6)
        np.testing.assert_allclose([getattr(row, f"d_lag_{k}") for k in range(1, 5)],
                                   masked[-1:-5:-1], equal_nan=True)
        for window in (4, 13):
            np.testing.assert_allclose(getattr(row, f"d_mean_{window}"),
                                       np.nanmean(masked[-window:]), rtol=1e-6)


def test_features_do_not_read_future(generated):
    history, _ = generated
    origin = split.selection_origins()[5]
    n = sum(w + timedelta(days=6) <= origin for w in history["weeks"])
    rng = np.random.default_rng(99)
    changed = {**history, "y": history["y"].copy(), "out": history["out"].copy(),
               "lines": history["lines"].copy()}
    changed["y"][:, n:] = rng.uniform(0, 10000, changed["y"][:, n:].shape)
    changed["out"][:, n:] = rng.random(changed["out"][:, n:].shape) < 0.5
    changed["lines"].loc[changed["lines"].index > origin] = 100000
    before, after = synthetic_panel(history, [origin]), synthetic_panel(changed, [origin])
    columns = [*schema.KEYS, "tier", "target_scale", "mase_scale",
               *schema.feature_columns(before, ("d_", "c_"))]
    pd.testing.assert_frame_equal(before[columns], after[columns])
    truncated = {**history, "weeks": history["weeks"][:n], "y": history["y"][:, :n],
                 "out": history["out"][:, :n],
                 "lines": history["lines"].loc[history["lines"].index <= origin]}
    pd.testing.assert_frame_equal(before[columns], synthetic_panel(truncated, [origin])[columns])


def test_ingredients_and_prices(generated):
    history, _ = generated
    drugs = synthetic_drugs(history)
    assert len(drugs) == len(history["drugs"])
    assert drugs.drug_code.is_unique
    assert set(drugs.columns) == {"drug_code", "drug_name", "ingredients", "strength", "form",
                                  "kind", "mapped", "source_model"}
    mapped = drugs[drugs.ingredients != ""]
    sizes = mapped.groupby("ingredients").size()
    assert sizes.eq(1).any()
    assert sizes.between(2, 4).any()
    assert mapped.groupby("ingredients").form.nunique().gt(1).any()
    assert drugs.ingredients.eq("").sum() >= 2
    assert not drugs.loc[drugs.ingredients == "", "mapped"].any()
    assert mapped.ingredients.str.contains(" + ", regex=False).any()
    prices = synthetic_prices(history)
    assert set(prices.columns) == {"drug_code", "valid_from", "price"}
    assert set(prices.drug_code) == set(history["drugs"])
    assert prices.price.gt(0).all()
    assert prices.groupby("drug_code").price.nunique().eq(3).all()


def test_boundaries_and_scale_fallbacks(generated):
    history, _ = generated
    early, _ = synthetic_history(n_per_pattern=1, until=date(2023, 10, 4))
    assert early["weeks"][-1] == date(2023, 9, 25)
    panel = synthetic_panel(early, [date(2023, 9, 24)])
    schema.validate_panel(panel)
    assert not panel.loc[panel.target_week.dt.date < config.STOCK_START, "censor_known"].any()
    assert panel.loc[panel.target_week.dt.date > date(2023, 9, 25), "target"].isna().all()
    assert synthetic_panel(early, []).empty
    with pytest.raises(ValueError, match="Sundays"):
        synthetic_panel(history, [date(2024, 1, 8)])
    with pytest.raises(ValueError, match="positive"):
        synthetic_history(n_per_pattern=0)
    with pytest.raises(ValueError, match="complete week"):
        synthetic_history(until=date(2023, 1, 7))
    # An observed sale followed by zero/censored windows exercises both fallbacks.
    weeks = history["weeks"][:60]
    y = np.zeros((1, 60))
    y[0, 10] = 52
    small = {**history, "drugs": np.array(["SYN-test"], dtype=object), "weeks": weeks,
             "y": y, "out": np.zeros_like(y, dtype=bool)}
    origin = weeks[-1] + timedelta(days=6)
    assert synthetic_panel(small, [origin]).target_scale.eq(np.float32(1)).all()
    small["out"][0, 10] = True
    assert synthetic_panel(small, [origin]).target_scale.eq(np.float32(schema.SCALE_FLOOR)).all()


def test_build_time():
    start = perf_counter()
    history, _ = synthetic_history()
    synthetic_panel(history, split.selection_origins())
    assert perf_counter() - start < 3


def test_fixture_matches_the_shared_history_helpers(generated):
    """The fixture's panel scales agree with irpd.ml.history, which the real panel uses."""
    from irpd.ml import history as hist

    history, _ = generated
    origin = split.selection_origins()[5]
    panel = synthetic_panel(history, [origin])
    rows = panel[panel.h == 1].set_index("drug_code")
    idx = {c: i for i, c in enumerate(history["drugs"])}
    ts, ms = hist.target_scale(history, origin), hist.mase_scale(history, origin)
    for code, row in rows.iterrows():
        np.testing.assert_allclose(row.target_scale, ts[idx[code]], rtol=1e-5)
        np.testing.assert_allclose(row.mase_scale, ms[idx[code]], rtol=1e-5, equal_nan=True)
