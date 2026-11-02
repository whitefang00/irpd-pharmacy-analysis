from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from irpd.ml import features, schema
from irpd.ml import history as hist
from irpd.ml.config import FrozenSystem, MLConfig
from irpd.ml.models.base import Forecaster

ORIGINS = (date(2024, 3, 3), date(2024, 3, 31))


def _panel(n_drugs=2, origins=ORIGINS):
    rows = []
    for d in range(n_drugs):
        for o in origins:
            for h in range(1, 9):
                rows.append({"drug_code": f"D{d}", "origin": pd.Timestamp(o), "h": h,
                             "target_week": pd.Timestamp(schema.target_week(o, h)),
                             "target": float(h), "censored": False, "censor_known": True,
                             "available": True, "scored": True, "tier": "fast", "eid": False,
                             "target_scale": 1.0, "mase_scale": 1.0, "d_lag_1": 1.0})
    p = pd.DataFrame(rows)
    return p.astype({"h": "int8", "target": "float32", "tier": "category",
                     "target_scale": "float32", "mase_scale": "float32", "d_lag_1": "float32",
                     "origin": "datetime64[s]", "target_week": "datetime64[s]"})


def test_valid_panel_passes():
    schema.validate_panel(_panel())


@pytest.mark.parametrize("breakage, match", [
    (lambda p: p.drop(columns="target_scale"), "missing"),
    (lambda p: p.assign(extra=1.0), "prefix"),
    (lambda p: p.assign(d_x=p.d_lag_1.astype("float64")), "float32"),
    (lambda p: p.assign(d_x=np.float32(np.inf)), "infinite"),
    (lambda p: pd.concat([p, p.head(1)]), "duplicate"),
    (lambda p: p.assign(target_week=p.target_week + pd.Timedelta(days=1)), "Monday"),
    (lambda p: p.assign(censored=True), "NaN target"),
    (lambda p: p.assign(available=False), "NaN target"),
    (lambda p: p.assign(censored=True, target=np.float32(np.nan), censor_known=False),
     "censoring isn't known"),
    (lambda p: p.assign(target=np.float32(np.inf)), "finite"),
    (lambda p: p.assign(target_scale=np.float32(0)), "positive"),
    (lambda p: p.assign(target_scale=np.float32(np.inf)), "positive"),
])
def test_broken_panels_fail(breakage, match):
    with pytest.raises(ValueError, match=match):
        schema.validate_panel(breakage(_panel()))


def test_target_week_and_policy_origin():
    assert schema.target_week(date(2024, 3, 3), 1) == date(2024, 3, 4)
    assert schema.target_week(date(2024, 3, 3), 8) == date(2024, 4, 22)
    assert schema.policy_origin(date(2024, 12, 31)) == date(2024, 12, 29)   # Tue
    assert schema.policy_origin(date(2025, 6, 30)) == date(2025, 6, 29)     # Mon
    assert schema.policy_origin(date(2025, 6, 29)) == date(2025, 6, 29)     # Sun
    m = np.arange(16, dtype=float).reshape(2, 8)
    assert schema.extend_horizon(m, 10)[:, 8:].tolist() == [[7, 7], [15, 15]]


def test_trainable_and_train_valid():
    p = _panel(origins=(date(2024, 1, 7), date(2024, 2, 4), date(2024, 3, 3)))
    o = date(2024, 3, 31)
    rows = schema.trainable(p, o)
    assert (rows.target_week + pd.Timedelta(days=6) <= pd.Timestamp(o)).all()
    # From the 3 Mar origin only h 1..4 have ended by 31 Mar.
    assert set(rows[rows.origin == pd.Timestamp(date(2024, 3, 3))].h) == {1, 2, 3, 4}
    train, valid = schema.train_valid(rows, o)
    block = pd.Timestamp(o - timedelta(weeks=8))
    assert (train.target_week + pd.Timedelta(days=6) <= block).all()
    assert (valid.target_week + pd.Timedelta(days=6) > block).all()
    assert (train.origin < block).all()  # no training row's features reach the block
    assert len(train) + len(valid) == len(rows)
    with pytest.raises(ValueError):
        schema.train_valid(p, o)


def _hist():
    weeks = np.array([date(2023, 9, 18) + timedelta(weeks=i) for i in range(30)], dtype=object)
    y = np.tile(np.arange(30, dtype=float), (2, 1))
    y[1, :5] = np.nan
    return {"drugs": np.array(["A", "B"], dtype=object), "weeks": weeks, "y": y,
            "out": np.zeros_like(y, dtype=bool),
            "stock_known": np.array([w >= hist.first_stock_week() for w in weeks]),
            "revenue": y * 100, "shelfout_days": np.zeros_like(y, dtype=np.int16),
            "first_sale": np.array([1, 5]),
            "lines": pd.Series(1.0, index=[weeks[0] + timedelta(days=i) for i in range(210)])}


def test_history_cut_cuts_lines_too():
    h = _hist()
    origin = date(2023, 10, 29)
    c = hist.cut(h, origin)
    assert c["y"].shape[1] == hist.weeks_to(h, origin) == 6
    assert c["lines"].index.max() == origin
    assert c["weeks"][-1] + timedelta(days=6) == origin


def test_first_stock_week():
    assert hist.first_stock_week() == date(2023, 10, 2)
    assert hist.first_stock_week().weekday() == 0


def test_target_scale_fallbacks():
    h = _hist()
    h["y"][0, :] = 0.0
    s = hist.target_scale(h, date(2024, 3, 31))
    assert s[0] == pytest.approx(schema.SCALE_FLOOR)
    assert s[1] > 0


def test_feature_registry():
    assert features.groups_for_step(1) == ("demand", "calendar")
    assert features.groups_for_step(3) == ("demand", "calendar", "store", "price", "drug",
                                           "groups")
    assert schema.feature_columns(_panel(), ("d_",)) == ["d_lag_1"]


def test_config_hash_is_stable_and_normalised():
    a = MLConfig(model="lgbm", groups=("calendar", "demand"), params={"b": 1, "a": 2})
    b = MLConfig(model="lgbm", groups=("demand", "calendar"), params={"a": 2, "b": 1})
    assert a.hash() == b.hash() and len(a.hash()) == 12
    assert MLConfig.from_dict(a.to_dict()) == a
    assert MLConfig.from_dict({"model": "lgbm"}) == MLConfig(model="lgbm")
    assert MLConfig(model="lgbm", loss="poisson").hash() != a.hash()
    assert MLConfig(model="lgbm", per_tier=True).hash() != MLConfig(model="lgbm").hash()
    assert a.prefixes() == ("d_", "c_")


def test_config_settings_are_copied_and_read_only():
    params = {"num_leaves": 31, "nested": {"x": [1, 2]}}
    cfg = MLConfig(model="lgbm", params=params)
    before = cfg.hash()
    params["num_leaves"] = 63
    params["nested"]["x"].append(3)
    assert cfg.hash() == before
    with pytest.raises(TypeError):
        cfg.params["num_leaves"] = 7


@pytest.mark.parametrize("kw", [
    {"model": "nope"}, {"model": "lgbm", "loss": "tweedie"},
    {"model": "lgbm", "tweedie_power": 1.5}, {"model": "lgbm", "groups": ("nope",)},
    {"model": "lgbm", "censor_mode": "nope"}, {"model": "lgbm", "retrain_every": 0},
    {"model": "lgbm", "params": {"x": float("nan")}},
])
def test_config_rejects_bad_settings(kw):
    with pytest.raises(ValueError):
        MLConfig(**kw)


def test_config_rejects_non_json_params():
    with pytest.raises(TypeError):
        MLConfig(model="lgbm", params={"x": object()})
    with pytest.raises(ValueError, match="unknown config keys"):
        MLConfig.from_dict({"model": "lgbm", "nope": 1})


def test_frozen_system_hash_covers_routing_and_uncertainty():
    cfg = MLConfig(model="lgbm")
    r = {"fast": "ml", "regular": "ml", "slow": "ml", "rare": "snaive"}
    a = FrozenSystem(cfg, r)
    assert a.hash() != FrozenSystem(cfg, {**r, "slow": "ses+eid"}).hash()
    assert a.hash() != FrozenSystem(cfg, r, uncertainty="u1_nb").hash()
    assert a.hash() == FrozenSystem(cfg, r, note="different note").hash()
    assert FrozenSystem.from_json(a.to_json()) == a
    with pytest.raises(ValueError):
        FrozenSystem(cfg, {"fast": "ml"})


def test_forecaster_protocol():
    class Mean:
        name = "mean"

        def fit(self, rows, origin):
            self.m = float(rows.target.mean())
            return self

        def predict(self, rows):
            return np.full(len(rows), self.m)

    assert isinstance(Mean(), Forecaster)


def _pred(**kw):
    base = {"origin": [date(2024, 3, 3)], "method": ["m"], "drug_code": ["D"],
            "tier": ["fast"], "h": [1], "eid": [False], "actual": [2.0], "mean": [1.5],
            "var": [3.0], "scale": [1.0], "config_hash": ["x"], "seed": [schema.ENSEMBLE_SEED]}
    return pd.DataFrame({**base, **{k: [v] for k, v in kw.items()}})


def test_predictions_contract():
    schema.validate_predictions(_pred())
    schema.validate_predictions(_pred(actual=np.nan, scale=np.nan))
    for kw, match in [({"var": 1.0}, "var"), ({"var": np.nan}, "var"),
                      ({"mean": np.inf}, "mean"), ({"actual": -1.0}, "actual"),
                      ({"h": 9}, "h must"), ({"seed": None}, "null")]:
        with pytest.raises(ValueError, match=match):
            schema.validate_predictions(_pred(**kw))
