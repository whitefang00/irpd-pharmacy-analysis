"""LightGBM and XGBoost forecasters on the synthetic panel."""

import importlib

import numpy as np
import pandas as pd
import pytest

from irpd.ml import schema, split
from irpd.ml.config import MLConfig
from irpd.ml.models.base import Forecaster

pytest.importorskip("lightgbm")
pytest.importorskip("xgboost")
ml_fixture = importlib.import_module("ml_fixture")
gbm = importlib.import_module("irpd.ml.models.lgbm")
xgbm = importlib.import_module("irpd.ml.models.xgb")

ORIGIN = split.selection_origins()[5]
# Small, fast settings; the wrapper keys are documented in models/lgbm.py.
FAST = {"num_boost_round": 60, "early_stopping_rounds": 10, "fixed_rounds": 30}
LIB_FAST = {"lgbm": {"num_leaves": 15, "min_data_in_leaf": 20, "num_threads": 2},
            "xgb": {"max_leaves": 15, "nthread": 2}}


def cfg(model="lgbm", **kw):
    params = {**FAST, **LIB_FAST[model], **kw.pop("params", {})}
    kw.setdefault("seeds", (0,))
    return MLConfig(model=model, params=params, **kw)


@pytest.fixture(scope="module")
def panel():
    history, _ = ml_fixture.synthetic_history()
    return ml_fixture.synthetic_panel(history, [*split.training_origins(ORIGIN), ORIGIN])


@pytest.fixture(scope="module")
def rows(panel):
    return schema.trainable(panel, ORIGIN)


@pytest.fixture(scope="module")
def test_rows(panel):
    return panel[panel["origin"].dt.date == ORIGIN]


def check(pred, n):
    assert pred.shape == (n,)
    assert np.isfinite(pred).all() and (pred >= 0).all()


def test_purge(rows, test_rows):
    f = gbm.make_gbm(cfg()).fit(rows, ORIGIN)
    info = f.fit_info
    block = pd.Timestamp(ORIGIN) - pd.Timedelta(weeks=schema.VALID_WEEKS)
    assert info["block_start"] == block
    assert info["train_max_week_end"] <= block
    assert info["train_max_origin"] < block
    used = rows.loc[info["train_index"]]
    assert (used["target_week"] + pd.Timedelta(days=6) <= block).all()
    valid = rows.loc[info["valid_index"]]
    assert len(valid) >= 200 and (valid["target_week"] + pd.Timedelta(days=6) > block).all()
    assert info["n_train"] + info["n_valid"] == len(rows)
    # Rows that aren't trainable at the origin are refused.
    with pytest.raises(ValueError):
        gbm.make_gbm(cfg()).fit(test_rows[test_rows["target"].notna()], ORIGIN)


@pytest.mark.parametrize("model", ["lgbm", "xgb"])
@pytest.mark.parametrize("target,loss,power", [
    ("raw", "l2", None), ("raw", "poisson", None), ("raw", "tweedie", 1.5),
    ("raw", "log1p_l2", None), ("raw", "huber", None),
    ("scaled", "l2", None), ("scaled", "poisson", None)])
def test_every_loss_trains(rows, test_rows, model, target, loss, power):
    f = gbm.make_gbm(cfg(model, target=target, loss=loss, tweedie_power=power))
    f.fit(rows, ORIGIN)
    assert all(1 <= it <= FAST["num_boost_round"] for it in f.best_iteration.values())
    check(f.predict(test_rows), len(test_rows))
    assert f.predict(test_rows).mean() > 0


@pytest.mark.parametrize("loss,power", [("tweedie", 1.5), ("log1p_l2", None), ("huber", None)])
def test_scaled_refuses_other_losses(loss, power):
    with pytest.raises(ValueError):
        gbm.make_gbm(cfg(target="scaled", loss=loss, tweedie_power=power))


def test_objective_names():
    assert gbm.LGBMForecaster(cfg(loss="tweedie", tweedie_power=1.3)).lib_params[
        "tweedie_variance_power"] == 1.3
    x = xgbm.XGBForecaster(cfg("xgb", loss="huber"))
    assert x.lib_params["objective"] == "reg:pseudohubererror"
    assert gbm.LGBMForecaster(cfg(loss="huber")).lib_params["objective"] == "huber"
    # cfg.params override the defaults.
    assert gbm.LGBMForecaster(MLConfig(model="lgbm", params={"num_leaves": 7})).lib_params[
        "num_leaves"] == 7
    assert gbm.LGBMForecaster(MLConfig(model="lgbm")).wrapper["num_boost_round"] == 2000


def test_scaled_round_trip(rows, test_rows):
    """A target that is exactly 3x target_scale is recovered row by row."""
    ratio = rows.assign(target=(3 * rows["target_scale"]).astype("float32"))
    f = gbm.make_gbm(cfg(target="scaled", loss="l2")).fit(ratio, ORIGIN)
    pred = f.predict(test_rows)
    np.testing.assert_allclose(pred, 3 * test_rows["target_scale"].to_numpy(), rtol=1e-3)


@pytest.mark.parametrize("model", ["lgbm", "xgb"])
def test_seeds_differ_and_average(rows, test_rows, model):
    f = gbm.make_gbm(cfg(model, seeds=(0, 1, 2))).fit(rows, ORIGIN)
    per = f.predict_seeds(test_rows)
    assert sorted(per) == [0, 1, 2]
    for p in per.values():
        check(p, len(test_rows))
    assert not np.allclose(per[0], per[1])
    np.testing.assert_allclose(f.predict(test_rows), np.mean(list(per.values()), axis=0))


def test_per_tier(rows, test_rows):
    base = gbm.make_gbm(cfg()).fit(rows, ORIGIN)
    tiered = gbm.make_gbm(cfg(per_tier=True, params={"min_tier_rows": 2500})).fit(rows, ORIGIN)
    counts = schema.train_valid(rows, ORIGIN)[0]["tier"].astype(object).value_counts()
    own = {t for t in counts.index if counts[t] >= 2500}
    assert own and set(tiered.fit_info["models"]) - {None} == own
    assert None in tiered.fit_info["models"]  # some tier falls back to the global model
    pred = tiered.predict(test_rows)
    check(pred, len(test_rows))
    assert not np.allclose(pred, base.predict(test_rows))
    # A tier without its own model gets the global model's prediction.
    rest = ~test_rows["tier"].astype(object).isin(own).to_numpy()
    assert rest.any()
    glob = tiered.boosters[(None, 0)]
    raw = tiered._predict_one(glob, tiered._design(test_rows[rest]),
                              tiered.best_iteration[(None, 0)])
    np.testing.assert_allclose(pred[rest], np.clip(raw, 0, None))


def test_recency_weights(rows, test_rows):
    f = gbm.make_gbm(cfg(recency_halflife_weeks=10))
    w = f._weights(pd.DataFrame({"target_week": [pd.Timestamp(ORIGIN) - pd.Timedelta(weeks=10),
                                                 pd.Timestamp(ORIGIN) - pd.Timedelta(weeks=30)]}),
                   ORIGIN)
    np.testing.assert_allclose(w, [0.5, 0.125])
    f.fit(rows, ORIGIN)
    plain = gbm.make_gbm(cfg()).fit(rows, ORIGIN)
    assert not np.allclose(f.predict(test_rows), plain.predict(test_rows))


@pytest.mark.parametrize("model", ["lgbm", "xgb"])
def test_categorical_unseen_level(rows, test_rows, model):
    def add(df, values):
        return df.assign(x_kind=pd.Categorical(values))

    codes = rows["drug_code"].str.len() % 3
    train = add(rows, np.array(["a", "b", "c"])[codes.to_numpy()])
    f = gbm.make_gbm(cfg(model, groups=("demand", "calendar", "drug"))).fit(train, ORIGIN)
    assert f.features[-2:] == ["h", "tier"] and "x_kind" in f.features
    assert set(f.categories) == {"tier", "x_kind"}
    unseen = f.predict(add(test_rows, ["z"] * len(test_rows)))
    check(unseen, len(test_rows))
    missing = f.predict(add(test_rows, pd.Categorical([None] * len(test_rows),
                                                      categories=["a", "b", "c"])))
    np.testing.assert_allclose(unseen, missing)
    design = f._design(add(test_rows, ["z"] * len(test_rows)))
    assert design["x_kind"].isna().all()
    assert list(design["x_kind"].cat.categories) == ["a", "b", "c"]


def test_drop_features(rows):
    f = gbm.make_gbm(cfg(drop_features=("y_week_of_year",))).fit(rows, ORIGIN)
    assert "y_week_of_year" not in f.features and "d_lag_1" in f.features
    imp = f.feature_importance()
    assert set(imp["feature"]) == set(f.features) and (imp["gain"] >= 0).all()


def test_fixed_rounds_without_validation(rows, test_rows):
    f = gbm.make_gbm(cfg(params={"min_valid_rows": 10**9})).fit(rows, ORIGIN)
    assert f.best_iteration == {(None, 0): FAST["fixed_rounds"]}
    check(f.predict(test_rows), len(test_rows))


def test_xgboost_parity(rows, test_rows):
    scored = test_rows[test_rows["target"].notna()]
    y = scored["target"].to_numpy()
    fits = {m: gbm.make_gbm(cfg(m, loss="poisson")).fit(rows, ORIGIN) for m in ("lgbm", "xgb")}
    assert fits["lgbm"].features == fits["xgb"].features
    wape = {m: np.abs(f.predict(scored) - y).sum() / y.sum() for m, f in fits.items()}
    assert wape["xgb"] < 1.3 * wape["lgbm"] and wape["lgbm"] < 1.3 * wape["xgb"]


def test_make_gbm_dispatch():
    assert isinstance(gbm.make_gbm(cfg()), gbm.LGBMForecaster)
    assert isinstance(gbm.make_gbm(cfg()), Forecaster)
    assert isinstance(gbm.make_gbm(cfg("xgb")), xgbm.XGBForecaster)
    with pytest.raises(ValueError):
        gbm.make_gbm(MLConfig(model="prophet"))
    with pytest.raises(ValueError):
        gbm.LGBMForecaster(cfg("xgb"))


def test_predict_before_fit():
    with pytest.raises(RuntimeError):
        gbm.make_gbm(cfg()).predict(pd.DataFrame())

