"""Per-drug Prophet on the synthetic fixture: subset, holidays, forecasts, fallback."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("prophet")

import ml_fixture  # noqa: E402

from irpd import forecast  # noqa: E402
from irpd.ml import schema, split  # noqa: E402
from irpd.ml.config import MLConfig  # noqa: E402
from irpd.ml.models.base import Forecaster  # noqa: E402
from irpd.ml.models.prophet import ProphetForecaster, holidays  # noqa: E402

ORIGIN = date(2024, 3, 3)  # h = 5..8 fall in the Eid 2024 window
END = date(2025, 6, 29)


def _ctx(history) -> schema.PanelContext:
    return schema.PanelContext(con=None, data_end=END, origins=[ORIGIN], history=history)


@pytest.fixture(scope="module")
def setup():
    history, _ = ml_fixture.synthetic_history()
    panel = ml_fixture.synthetic_panel(history, [ORIGIN])
    model = ProphetForecaster(_ctx(history), trend="flat", yearly=False, n_jobs=2, seed=0)
    model.fit(schema.trainable(panel, ORIGIN), ORIGIN)
    mean = model.predict(panel)
    return history, panel, model, mean


def _tiers(panel) -> dict:
    first = panel.drop_duplicates("drug_code")
    return dict(zip(first.drug_code, first.tier.astype(str), strict=True))


def test_is_a_forecaster(setup):
    assert isinstance(setup[2], Forecaster)


def test_subset_rule(setup):
    history, panel, model, _ = setup
    tiers = _tiers(panel)
    got = model.subset(ORIGIN, tiers)
    h = model.ctx.history_at(ORIGIN)
    n = len(h["weeks"])
    expect = [c for i, c in enumerate(h["drugs"])
              if tiers.get(c) in ("fast", "regular") and h["first_sale"][i] >= 0
              and n - h["first_sale"][i] >= 52 and h["out"][i, -52:].sum() < 0.2 * 52]
    assert got == expect
    assert got and all(tiers[c] in ("fast", "regular") for c in got)
    assert not any("intermittent" in c or "rare" in c for c in got)


def test_subset_history_and_censoring_limits():
    history, _ = ml_fixture.synthetic_history()
    model = ProphetForecaster(_ctx(history))
    origin = date(2024, 9, 15)  # late starts have well under 52 weeks here
    n = model.ctx.weeks_to(origin)
    codes = list(history["drugs"])
    smooth = codes.index("SYN-smooth-1")  # odd index: no fixture stock-outs
    tiers = dict.fromkeys(codes, "fast")
    out = history["out"]
    out[smooth, :] = False
    out[smooth, n - 10:n] = True  # 10 of 52 censored: below 20%, kept
    got = model.subset(origin, tiers)
    assert "SYN-smooth-1" in got
    assert not any("late_start" in c for c in got)
    out[smooth, n - 11:n] = True  # 11 of 52: excluded
    assert "SYN-smooth-1" not in model.subset(origin, tiers)
    assert "SYN-smooth-1" not in model.subset(origin, {**tiers, "SYN-smooth-1": "slow"})


def test_holidays_on_mondays():
    hol = holidays()
    assert (hol.ds.dt.day_of_week == 0).all()
    eid = hol[hol.holiday == "eid"]
    assert len(eid) == len(forecast.EID)
    assert (eid.lower_window == -7).all() and (eid.upper_window == 28).all()
    for e, (_, row) in zip(forecast.EID, eid.iterrows(), strict=True):
        assert row.ds.date() <= e < row.ds.date() + timedelta(days=7)
    ram = hol[hol.holiday == "ramadan"]
    for e, (_, row) in zip(forecast.EID, ram.iterrows(), strict=True):
        start, stop = e - timedelta(days=30), e - timedelta(days=1)
        first = row.ds.date()
        last = first + timedelta(days=int(row.upper_window))
        assert row.lower_window == 0 and first <= start < first + timedelta(days=7)
        assert last <= stop < last + timedelta(days=7) and last.weekday() == 0


def test_predictions_finite_and_non_negative(setup):
    _, panel, model, mean = setup
    assert mean.shape == (len(panel),)
    assert np.isfinite(mean).all() and (mean >= 0).all()
    assert model.last_used_prophet.any() and not model.last_used_prophet.all()
    assert not model.failed
    assert set(model.fit_seconds) == set(model._means)


def test_eid_drug_rises_in_eid_weeks(setup):
    _, panel, _, mean = setup
    p = panel.assign(mean=mean)
    assert p.loc[p.h >= 5, "eid"].all() and not p.loc[p.h <= 3, "eid"].any()

    def ratio(pattern):
        d = p[p.drug_code.str.contains(pattern)]
        return (d[d.eid].groupby("drug_code")["mean"].mean()
                / d[~d.eid].groupby("drug_code")["mean"].mean())

    eid, flat = ratio("-eid-"), ratio("smooth")
    assert (eid > 1.4).all() and eid.min() > flat.max() + 0.2


def test_fallback_is_ses_eid(setup):
    _, panel, model, mean = setup
    used = model.last_used_prophet
    assert not used[panel.drug_code.str.contains("intermittent|lumpy|rare").to_numpy()].any()
    h = model.ctx.history_at(ORIGIN)
    base = forecast.predict(h, np.full(len(h["drugs"]), "ses+eid", dtype=object), 8)["mean"]
    idx = {c: i for i, c in enumerate(h["drugs"])}
    expect = np.array([base[idx[c], k - 1] for c, k in zip(panel.drug_code, panel.h,
                                                             strict=True)])
    np.testing.assert_allclose(mean[~used], expect[~used])


def test_deterministic_with_seed(setup):
    history, panel, _, mean = setup
    rows = panel[panel.drug_code.str.contains("-eid-|smooth")]
    model = ProphetForecaster(_ctx(history), n_jobs=1, seed=0).fit(rows.iloc[:0], ORIGIN)
    again = model.predict(rows)
    np.testing.assert_allclose(again, mean[rows.index.to_numpy()], rtol=1e-6)


def test_options_and_config(setup):
    history, panel, _, _ = setup
    rows = panel[panel.drug_code == "SYN-trend-0"]
    cfg = MLConfig(model="prophet", params={"trend": "linear", "yearly": True}, seeds=(3,))
    model = ProphetForecaster.from_config(_ctx(history), cfg).fit(rows.iloc[:0], ORIGIN)
    assert (model.trend, model.yearly, model.seed) == ("linear", True, 3)
    mean = model.predict(rows)
    assert model.last_used_prophet.all() and np.isfinite(mean).all() and (mean >= 0).all()
    with pytest.raises(ValueError):
        ProphetForecaster(_ctx(history), trend="logistic")


def test_predict_checks_origin(setup):
    history, panel, _, _ = setup
    model = ProphetForecaster(_ctx(history))
    with pytest.raises(RuntimeError):
        model.predict(panel)
    model.fit(panel.iloc[:0], ORIGIN - timedelta(weeks=1))
    with pytest.raises(ValueError):
        model.predict(panel)
    assert pd.Timestamp(split.FIRST_ORIGIN) <= panel.origin.min()
