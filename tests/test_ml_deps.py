"""Small, synthetic smoke checks for the optional forecasting dependencies."""

import importlib
import logging
from time import perf_counter

import pytest

lgb = pytest.importorskip("lightgbm")
np = importlib.import_module("numpy")


def training_data():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(200, 4))
    y = rng.poisson(np.exp(0.3 * x[:, 0]) + 1).astype(float)
    return x, y


def lightgbm_model(objective):
    x, y = training_data()
    model = lgb.train(
        {
            "objective": objective,
            "alpha": 0.9,
            "num_threads": 1,
            "num_leaves": 7,
            "verbosity": -1,
            "seed": 42,
            "force_col_wise": True,
        },
        lgb.Dataset(x, label=y),
        num_boost_round=10,
    )
    return model, x


@pytest.mark.parametrize("library", ["xgboost", "optuna", "shap", "pyarrow", "joblib"])
def test_ml_imports(library):
    assert importlib.import_module(library) is not None


@pytest.mark.parametrize("objective", ["tweedie", "quantile"])
def test_lightgbm(objective):
    model, x = lightgbm_model(objective)
    predictions = model.predict(x[:5])
    print(f"LightGBM {objective}: {predictions}")
    assert predictions.shape == (5,)
    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()


@pytest.mark.parametrize("objective", ["reg:tweedie", "count:poisson"])
def test_xgboost(objective):
    xgb = importlib.import_module("xgboost")
    x, y = training_data()
    model = xgb.train(
        {"objective": objective, "nthread": 1, "max_depth": 2, "seed": 42},
        xgb.DMatrix(x, label=y),
        num_boost_round=10,
    )
    predictions = model.predict(xgb.DMatrix(x[:5]))
    print(f"XGBoost {objective}: {predictions}")
    assert predictions.shape == (5,)
    assert np.isfinite(predictions).all()
    assert (predictions > 0).all()


def test_shap_tree_explainer():
    shap = importlib.import_module("shap")
    model, x = lightgbm_model("tweedie")
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x[:5])
    assert values.shape == (5, 4)
    assert np.isfinite(values).all()
    np.testing.assert_allclose(
        values.sum(axis=1) + explainer.expected_value,
        model.predict(x[:5], raw_score=True),
        rtol=1e-6,
        atol=1e-6,
    )


def test_prophet():
    prophet = pytest.importorskip("prophet")
    importlib.import_module("cmdstanpy")
    pd = importlib.import_module("pandas")
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=104, freq="W-MON")
    frame = pd.DataFrame({"ds": dates, "y": 20 + rng.normal(size=104)})
    frame.loc[[3, 17, 52, 81], "y"] = np.nan
    holidays = pd.DataFrame(
        {
            "holiday": ["custom_event"] * 3,
            "ds": pd.to_datetime(["2024-12-23", "2025-12-22", "2026-12-21"]),
            "lower_window": -7,
            "upper_window": 28,
        }
    )
    model = prophet.Prophet(holidays=holidays)
    # Initialize CmdStanPy's handler before setting the level; its first use resets it.
    logger = importlib.import_module("cmdstanpy.utils").get_logger()
    previous_level = logger.level
    try:
        logger.setLevel(logging.WARNING)
        started = perf_counter()
        model.fit(frame, seed=42)
        elapsed = perf_counter() - started
    finally:
        logger.setLevel(previous_level)
    print(f"Prophet fit: {elapsed:.3f} seconds")
    assert model.stan_fit is not None
    assert len(model.history) == 100
    future = model.make_future_dataframe(periods=8, freq="W-MON", include_history=False)
    predictions = model.predict(future)
    print(f"Prophet predictions: {predictions.yhat.to_numpy()}")
    assert len(predictions) == 8
    assert (predictions.ds.dt.day_of_week == 0).all()
    assert predictions.ds.min() == dates[-1] + pd.Timedelta(weeks=1)
    assert predictions.ds.max() == dates[-1] + pd.Timedelta(weeks=8)
    assert np.isfinite(predictions[["yhat", "yhat_lower", "yhat_upper"]]).all().all()
