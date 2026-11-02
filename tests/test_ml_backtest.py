from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from ml_fixture import synthetic_history, synthetic_panel

from irpd import forecast as fc
from irpd.ml import backtest, schema, split
from irpd.ml import history as hist
from irpd.ml.models.base import Forecaster
from irpd.ml.models.baseline import BaselineForecaster, baseline_dispersion, config_hash

DATA_END = date(2025, 6, 29)
SCORED = split.selection_origins()[:4]


@pytest.fixture(scope="module")
def setup():
    history, _ = synthetic_history(n_per_pattern=2, until=DATA_END)
    origins = split.training_origins(SCORED[-1] + timedelta(weeks=1))
    panel = synthetic_panel(history, origins)
    ctx = schema.PanelContext(con=None, data_end=DATA_END, origins=origins, history=history)
    return history, panel, ctx


class DrugMean:
    """A toy learner: each drug's mean training target. Records what it was fitted on."""

    name = "drugmean"
    fits: list = []

    def fit(self, rows, origin):
        end = rows["target_week"] + pd.Timedelta(days=6)
        assert (end <= pd.Timestamp(origin)).all() and rows["target"].notna().all()
        DrugMean.fits.append(origin)
        self.level = rows.groupby("drug_code")["target"].mean()
        return self

    def predict(self, rows):
        return self.level.reindex(rows["drug_code"]).fillna(1.0).to_numpy(float)


class Seeded(DrugMean):
    name = "seeded"

    def predict_seeds(self, rows):
        base = self.predict(rows)
        return {s: base * (1 + 0.1 * s) for s in (0, 1, 2)}


def test_baseline_is_forecast_predict(setup):
    history, panel, ctx = setup
    origin = SCORED[1]
    rows = panel[(panel["origin"] == pd.Timestamp(origin)) & panel["scored"]]
    model = BaselineForecaster(ctx).fit(schema.trainable(panel, origin), origin)
    assert isinstance(model, Forecaster)
    got, var = model.predict(rows), model.predict_var(rows)

    h = hist.cut(history, origin)
    tier = rows.drop_duplicates("drug_code").set_index("drug_code")["tier"].astype(str)
    method = np.array([fc.METHOD_BY_TIER[tier.get(d, "rare")] for d in h["drugs"]], dtype=object)
    p = fc.predict(h, method, 8)
    i = pd.Series(range(len(h["drugs"])), index=h["drugs"])[rows["drug_code"]].to_numpy()
    j = rows["h"].to_numpy(int) - 1
    np.testing.assert_allclose(got, p["mean"][i, j])
    np.testing.assert_allclose(var, p["var"][i, j])


def test_baseline_rejects_other_origin(setup):
    _, panel, ctx = setup
    rows = panel[panel["origin"] == pd.Timestamp(SCORED[1])]
    model = BaselineForecaster(ctx).fit(rows.iloc[:0], SCORED[0])
    with pytest.raises(ValueError, match="fitted on"):
        model.predict(rows)


def test_run_baseline_predictions(setup):
    _, panel, ctx = setup
    model = BaselineForecaster(ctx)
    pred = backtest.run(lambda: BaselineForecaster(ctx), config_hash(fc.METHOD_BY_TIER),
                        SCORED, panel, ctx)
    schema.validate_predictions(pred)
    assert list(pred.columns) == schema.PREDICTION_COLUMNS
    assert (pred["seed"] == schema.ENSEMBLE_SEED).all()
    assert pred["method"].unique().tolist() == ["baseline"]
    scored = panel[panel["scored"] & panel["origin"].isin([pd.Timestamp(o) for o in SCORED])]
    assert len(pred) == len(scored)
    m = pred.merge(scored, on=["drug_code", "origin", "h"], suffixes=("", "_p"))
    np.testing.assert_array_equal(m["actual"].to_numpy(), m["target"].to_numpy(float))
    np.testing.assert_allclose(m["scale"], m["mase_scale"].astype(float), equal_nan=True)
    # var is the baseline's own
    o = SCORED[2]
    rows = scored[scored["origin"] == pd.Timestamp(o)]
    model.fit(rows.iloc[:0], o)
    np.testing.assert_allclose(pred[pred["origin"] == pd.Timestamp(o)]["var"],
                               np.maximum(model.predict_var(rows), model.predict(rows)))


def test_var_recipe_without_predict_var(setup):
    history, panel, ctx = setup
    pred = backtest.run(DrugMean, "toy", SCORED[:1], panel, ctx)
    h = hist.cut(history, SCORED[0])
    disp = pd.Series(baseline_dispersion(h), index=h["drugs"])[pred["drug_code"]].to_numpy()
    np.testing.assert_allclose(pred["var"], np.maximum(pred["mean"] * disp, pred["mean"]))
    assert (disp >= 1).all()


def test_seeds_and_ensemble(setup):
    _, panel, ctx = setup
    pred = backtest.run(Seeded, "seeded", SCORED[:2], panel, ctx)
    assert sorted(pred["seed"].unique()) == [schema.ENSEMBLE_SEED, 0, 1, 2]
    wide = pred.pivot_table(index=["origin", "drug_code", "h"], columns="seed", values="mean")
    np.testing.assert_allclose(wide[schema.ENSEMBLE_SEED], wide[[0, 1, 2]].mean(axis=1))
    assert pred.groupby("seed").size().nunique() == 1


def test_retrain_every(setup):
    _, panel, ctx = setup
    DrugMean.fits = []
    backtest.run(DrugMean, "toy", SCORED, panel, ctx, retrain_every=2)
    assert DrugMean.fits == [SCORED[0], SCORED[2]]
    DrugMean.fits = []
    backtest.run(DrugMean, "toy", SCORED, panel, ctx)
    assert DrugMean.fits == SCORED


def test_cache_reuse(setup, tmp_path):
    _, panel, ctx = setup
    DrugMean.fits = []
    first = backtest.run(DrugMean, "toy", SCORED[:2], panel, ctx, cache_dir=tmp_path)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == [f"toy_{o:%Y%m%d}_v{schema.PANEL_VERSION}.parquet" for o in SCORED[:2]]

    def refuse():
        raise AssertionError("should have read the cache")

    again = backtest.run(refuse, "toy", SCORED[:2], panel, ctx, cache_dir=tmp_path)
    pd.testing.assert_frame_equal(first, again, check_dtype=False)
    # A third origin is computed; the cached two are not refitted.
    DrugMean.fits = []
    more = backtest.run(DrugMean, "toy", SCORED[:3], panel, ctx, cache_dir=tmp_path)
    assert DrugMean.fits == [SCORED[2]] and len(more) > len(first)
    # Another config hash doesn't see the cache.
    with pytest.raises(AssertionError):
        backtest.run(refuse, "other", SCORED[:1], panel, ctx, cache_dir=tmp_path)


def test_bad_forecaster_output(setup):
    _, panel, ctx = setup

    class Negative(DrugMean):
        def predict(self, rows):
            return -np.ones(len(rows))

    with pytest.raises(ValueError, match="negative"):
        backtest.run(Negative, "neg", SCORED[:1], panel, ctx)


def test_period_of():
    s = pd.Series(pd.to_datetime([date(2024, 1, 7), date(2024, 10, 13), date(2024, 11, 10),
                                  date(2024, 12, 8), date(2025, 1, 5)]))
    assert backtest.period_of(s).tolist() == ["selection", "selection", "purged", "purged",
                                              "eval"]
    assert all(backtest.period_of(pd.Series(pd.to_datetime(split.eval_origins(2)))) == "eval")


def _toy_pred(origins, config="c1", seed=schema.ENSEMBLE_SEED):
    rows = []
    for o in origins:
        for d in ("A", "B"):
            for h in (1, 2):
                rows.append({"origin": pd.Timestamp(o), "method": "m", "drug_code": d,
                             "tier": "fast", "h": h, "eid": False, "actual": 10.0,
                             "mean": 12.0 if o.year == 2024 else 8.0, "var": 20.0,
                             "scale": 1.0, "config_hash": config, "seed": seed})
    return pd.DataFrame(rows)


def test_score_period_and_filters():
    sel, purged = date(2024, 10, 13), date(2024, 11, 10)
    pred = pd.concat([_toy_pred([sel, purged, date(2025, 1, 5)]),
                      _toy_pred([sel], seed=0).assign(mean=100.0),
                      _toy_pred([sel], config="c2").assign(mean=100.0)])
    with pytest.raises(ValueError, match="config_hash"):
        backtest.score(pred, "selection")
    s = backtest.score(pred, "selection", config_hash="c1")
    assert s["period"].tolist() == ["selection"] and s["n"].tolist() == [4]
    assert s["bias"].iloc[0] == pytest.approx(1.2) and s["wape"].iloc[0] == pytest.approx(0.2)
    p = backtest.score(pred, "purged", config_hash="c1")
    assert p["period"].tolist() == ["purged"] and p["n"].tolist() == [4]
    e = backtest.score(pred, "eval", config_hash="c1")
    assert e["bias"].iloc[0] == pytest.approx(0.8)
    with pytest.raises(ValueError, match="period"):
        backtest.score(pred, "2024", config_hash="c1")

