"""The tuning stages on the synthetic panel: grid, objective wiring, caching, the pick rule."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from ml_fixture import synthetic_history, synthetic_panel

from irpd import config
from irpd.ml import schema, select, split
from irpd.ml.config import MLConfig

pytest.importorskip("lightgbm")
optuna = pytest.importorskip("optuna")
tune = pytest.importorskip("irpd.ml.tune")

ORIGINS = split.selection_origins()[4:6]
FAST = {"num_boost_round": 30, "early_stopping_rounds": 5, "fixed_rounds": 20,
        "num_leaves": 7, "min_data_in_leaf": 20, "num_threads": 2}


@pytest.fixture(scope="module")
def data():
    history, _ = synthetic_history(n_per_pattern=2, until=split.SELECTION_END)
    origins = split.training_origins(ORIGINS[-1] + timedelta(weeks=1))
    panel = synthetic_panel(history, origins)
    ctx = schema.PanelContext(con=None, data_end=split.SELECTION_END, origins=origins,
                              history=history)
    return panel, ctx


@pytest.fixture
def ml_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ML_DIR", tmp_path)
    return tmp_path


def fast_configs():
    return [tune.make_config(tune.Setting("raw", "tweedie", 1.5), "mask", FAST),
            tune.make_config(tune.Setting("scaled", "l2"), "mask", FAST)]


def test_screen_grid():
    cfgs = tune.screen_configs()
    names = [s.name for s in tune.SCREEN_SETTINGS]
    assert len(names) == len(set(names)) == 12  # 11 settings, huber at two deltas
    assert len(cfgs) == 12 * 3
    assert len({c.hash() for c in cfgs}) == len(cfgs)
    assert len({tune.setup_name(c) for c in cfgs}) == len(cfgs)
    assert {c.censor_mode for c in cfgs} == {"mask", "raw", "impute"}
    assert sorted(c.tweedie_power for c in cfgs if c.loss == "tweedie") == \
        sorted([1.1, 1.3, 1.5, 1.7, 1.9] * 3)
    assert {c.loss for c in cfgs if c.target == "scaled"} == {"l2", "poisson"}
    assert sorted(c.params["alpha"] for c in cfgs if c.loss == "huber") == [5.0] * 3 + [20.0] * 3
    for c in cfgs:
        assert c.model == "lgbm" and c.seeds == (0,)
        assert c.groups == ("demand", "calendar", "store", "price", "drug")
        if c.loss != "huber":
            assert dict(c.params) == {}


def _table(rows):
    return pd.DataFrame(rows, columns=["config_hash", "family", "objective"])


def test_pick_two_different_families():
    t = _table([("baseline", "baseline", 0.1), ("a", "raw-tweedie", 0.50),
                ("b", "raw-tweedie", 0.51), ("c", "raw-tweedie", 0.52),
                ("d", "scaled-l2", 0.53), ("e", "raw-l2", 0.60)])
    assert tune.pick_two(t) == ["a", "d"]
    t = _table([("a", "raw-poisson", 0.50), ("b", "raw-l2", 0.51), ("c", "raw-poisson", 0.4)])
    assert tune.pick_two(t) == ["c", "b"]
    with pytest.raises(ValueError):
        tune.pick_two(_table([("a", "raw-l2", 0.5), ("b", "raw-l2", 0.6)]))


def test_pick_two_on_real_configs():
    cfgs = tune.screen_configs()
    rows = [(c.hash(), "-".join(tune.family(c)), 1.0 + i / 100) for i, c in enumerate(cfgs)]
    # Make two Tweedie powers under the same censoring the best two.
    tw = [c for c in cfgs if c.loss == "tweedie" and c.censor_mode == "mask"]
    best = {tw[1].hash(): 0.1, tw[2].hash(): 0.11}
    rows = [(h, f, best.get(h, v)) for h, f, v in rows]
    first, second = tune.chosen_setups(_table(rows))
    assert first.hash() == tw[1].hash()
    assert second.loss != "tweedie"


def test_objective_wiring(data):
    panel, ctx = data
    cfg = MLConfig.from_dict({**fast_configs()[0].to_dict(), "seeds": [0, 1]})
    pred, _ = tune.run_config(cfg, ORIGINS, panel, ctx, cache_dir=None)
    ens = pred[pred["seed"] == schema.ENSEMBLE_SEED]
    assert tune.objective(pred) == pytest.approx(select.objective(ens))
    one = pred[pred["seed"] == 1]
    assert tune.objective(pred, seed=1) == pytest.approx(select.objective(one))
    scores = tune.wide(tune.tier_scores(pred))
    assert set(scores) == {f"{t}_{m}" for t in ("fast", "regular", "slow", "rare")
                           for m in tune.METRICS}


def test_screen_writes_table_and_resumes(data, ml_dir, monkeypatch):
    panel, ctx = data
    table = tune.screen(fast_configs(), ORIGINS, panels=lambda mode: (panel, ctx))
    assert list(table["setup"]).count("baseline") == 1
    assert len(table) == 3 and table["objective"].is_monotonic_increasing
    assert (ml_dir / "tune/screen.csv").exists()
    assert {"fast_wape", "rare_cover90", "fit_s", "objective"} <= set(table.columns)

    def refuse(cfg):
        raise AssertionError("refitted instead of reading the cache")

    monkeypatch.setattr(tune, "make_gbm", refuse)
    again = tune.screen(fast_configs(), ORIGINS, panels=lambda mode: (panel, ctx))
    pd.testing.assert_frame_equal(table, again)


def test_search_resumes(data, ml_dir):
    panel, ctx = data
    base = fast_configs()[0]
    storage = f"sqlite:///{ml_dir / 'optuna.db'}"
    study = tune.search(base, ORIGINS, panel, ctx, n_trials=2, storage=storage)
    assert len(study.trials) == 2
    # The objective is reported after each origin, for the pruner.
    assert len(study.trials[0].intermediate_values) == len(ORIGINS)
    assert study.trials[0].intermediate_values[len(ORIGINS) - 1] == pytest.approx(
        study.trials[0].value)
    out = pd.read_csv(ml_dir / f"tune/search_{tune.setup_name(base)}.csv")
    assert len(out) == 2 and {"learning_rate", "tweedie_variance_power", "fast_wape",
                              "config_hash"} <= set(out.columns)
    # A trial left running by a killed process is failed and re-queued.
    orphan = study.ask()
    tune.suggest(orphan, base)
    study = tune.search(base, ORIGINS, panel, ctx, n_trials=3, storage=storage)
    states = [t.state for t in study.trials]
    assert states.count(optuna.trial.TrialState.COMPLETE) == 3
    assert states[2] == optuna.trial.TrialState.FAIL
    assert study.trials[3].params == study.trials[2].params
    # Nothing left to do: no new trials.
    study = tune.search(base, ORIGINS, panel, ctx, n_trials=3, storage=storage)
    assert len(study.trials) == 4
    best = tune.best_config(study, base)
    assert best.seeds == (0, 1, 2) and best.loss == "tweedie"
    assert best.tweedie_power == pytest.approx(study.best_params["tweedie_variance_power"])
    assert np.isclose(best.params["learning_rate"], study.best_params["learning_rate"])


def test_focused_search_starts_from_defaults(data, ml_dir):
    panel, ctx = data
    base = fast_configs()[0]
    storage = f"sqlite:///{ml_dir / 'optuna.db'}"
    study = tune.focused_search(base, ORIGINS, panel, ctx, n_trials=2, storage=storage)
    assert study.trials[0].params == tune.focused_defaults(base)
    assert study.trials[0].params["tweedie_variance_power"] == 1.5
    for t in study.trials:
        assert 31 <= t.params["num_leaves"] <= 255
        assert 1.05 <= t.params["tweedie_variance_power"] <= 1.4 or t.number == 0
    cfg = tune.focused_config(base, study.best_params)
    assert cfg.params["learning_rate"] == 0.05 and cfg.seeds == (0, 1, 2)
    assert cfg.params["num_boost_round"] == FAST["num_boost_round"]  # base params kept


def test_summary_routes_and_writes(data, ml_dir):
    panel, ctx = data
    setups = [("tw", MLConfig.from_dict({**fast_configs()[0].to_dict(), "seeds": [0, 1]})),
              ("l2", tune.make_config(tune.Setting("raw", "l2"), "mask", FAST, seeds=(0, 1)))]
    text = tune.summary(setups, ["a note"], origins=ORIGINS,
                        panels=lambda mode: (panel, ctx))
    assert "a note" in text and "Routing" in text
    routing = pd.read_csv(ml_dir / "tune/routing.csv")
    assert set(routing["tier"]) <= {"fast", "regular", "slow", "rare"}
    assert "zero" not in set(routing["candidate"])
    best = __import__("json").loads((ml_dir / "tune/best_configs.json").read_text())
    assert set(best) == {"tw", "l2"}
    assert MLConfig.from_dict(best["tw"]["config"]).hash() == best["tw"]["config_hash"]
