"""Global gradient-boosted forecasters: one model across all drugs (plan 6.2, 6.3).

`LGBMForecaster` lives here; `XGBForecaster` (models/xgb.py) shares the
`GBMForecaster` base below, so both read the same rows, build the same
features and differ only in the library calls. `make_gbm(cfg)` returns the
right one for `cfg.model`.

**Features.** `schema.feature_columns(rows, cfg.prefixes())` minus
`cfg.drop_features`, plus `schema.MODEL_COLUMNS` (`h` numeric, `tier`
categorical). Every category-dtype column is passed as a native categorical
(LightGBM's own handling; XGBoost with `enable_categorical` and `hist`).
The levels seen at fit are kept and re-applied at predict, so a level the
model never saw becomes missing. Bool columns become 0/1 floats.

**Targets.** `raw` fits units. `scaled` fits target / `target_scale` and
multiplies predictions back by each row's `target_scale`.

**Losses** (library names in `OBJECTIVES` and xgb.OBJECTIVES):

    loss      LightGBM                   XGBoost
    l2        regression                 reg:squarederror
    poisson   poisson                    count:poisson
    tweedie   tweedie (variance power)   reg:tweedie (variance power)
    log1p_l2  regression on log1p(y)     reg:squarederror on log1p(y)
    huber     huber (alpha = delta)      reg:pseudohubererror (huber_slope)

log1p_l2 predicts expm1 of the fit, with no bias correction: the bias rule
(plan 7.3) judges it. Huber's delta defaults to 1.0 in both libraries, in
the target's units. With `scaled`, only l2 and poisson are allowed (plan
6.2); the others raise ValueError.

**Training.** `fit(rows, origin)` takes `schema.trainable(panel, origin)`
and splits it with `schema.train_valid` only. Training rows' target weeks
end before the 8-week validation block (checked in `fit`; what was used is
kept in `fit_info`). The model is fitted on the training part and early
stops on the validation part (unweighted, the library's default metric for
the objective). The best iteration is stored per booster. If validation has
fewer than `min_valid_rows` rows (default 200), it trains `fixed_rounds`
rounds (default 300) with no early stopping. The validation rows are never
refitted on: the purge costs those 8 weeks, as the plan says.

**Hyperparameters.** `cfg.params` holds library-native names that override
`DEFAULTS`, plus these wrapper keys (removed before the library sees them):
`num_boost_round` (max rounds, 2000), `early_stopping_rounds` (100),
`fixed_rounds` (300), `min_valid_rows` (200), `min_tier_rows` (1000).

**Seeds.** One booster per seed in `cfg.seeds`. `predict` is the seed
average; `predict_seeds` gives each seed's predictions.

**Per tier.** With `cfg.per_tier`, each tier with at least `min_tier_rows`
training rows gets its own models. If any tier in `config.TIERS` falls short
(or has no rows), a global model is also trained on all rows, and that
tier's rows, at predict, go to it.

**Recency.** With `cfg.recency_halflife_weeks`, training rows are weighted
0.5 ** (age / halflife), where age is the weeks from the row's target week
(its Monday) to the fit origin.

Predictions are clipped at zero; a non-finite prediction raises.
`boosters` maps (tier or None, seed) to the trained library model, and
`feature_importance()` gives gain per feature for later SHAP work.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from irpd.ml import schema
from irpd.ml.config import TIERS, MLConfig

GLOBAL = None  # the key of the all-tier model in `boosters`

WRAPPER_DEFAULTS = {"num_boost_round": 2000, "early_stopping_rounds": 100,
                    "fixed_rounds": 300, "min_valid_rows": 200, "min_tier_rows": 1000}
SCALED_LOSSES = ("l2", "poisson")
HUBER_DELTA = 1.0


class GBMForecaster:
    """Shared fit/predict logic; subclasses supply the library calls."""

    model: str = ""

    def __init__(self, cfg: MLConfig):
        if cfg.model != self.model:
            raise ValueError(f"{type(self).__name__} needs model={self.model!r}, "
                             f"not {cfg.model!r}")
        if cfg.target == "scaled" and cfg.loss not in SCALED_LOSSES:
            raise ValueError(f"the scaled target supports losses {SCALED_LOSSES}, "
                             f"not {cfg.loss!r}")
        self.cfg = cfg
        power = f"{cfg.tweedie_power:g}" if cfg.tweedie_power else ""
        self.name = f"{cfg.model}-{cfg.target}-{cfg.loss}{power}"
        params = dict(cfg.params)
        self.wrapper = {k: params.pop(k, v) for k, v in WRAPPER_DEFAULTS.items()}
        self.lib_params = {**self.defaults(), **self.objective(), **params}
        self.boosters: dict[tuple[str | None, int], object] = {}
        self.best_iteration: dict[tuple[str | None, int], int] = {}
        self.features: list[str] = []
        self.categories: dict[str, pd.Index] = {}
        self.fit_info: dict = {}

    # library hooks -----------------------------------------------------
    def defaults(self) -> dict:
        raise NotImplementedError

    def objective(self) -> dict:
        raise NotImplementedError

    def _train(self, x, y, w, xv, yv, seed: int, rounds: int,
               patience: int | None) -> tuple[object, int]:
        raise NotImplementedError

    def _predict_one(self, booster, x: pd.DataFrame, rounds: int) -> np.ndarray:
        raise NotImplementedError

    def _gain(self, booster) -> dict[str, float]:
        raise NotImplementedError

    # shared ------------------------------------------------------------
    def _columns(self, rows: pd.DataFrame) -> list[str]:
        cols = schema.feature_columns(rows, self.cfg.prefixes())
        drop = set(self.cfg.drop_features)
        return [c for c in cols if c not in drop and c not in schema.MODEL_COLUMNS] \
            + list(schema.MODEL_COLUMNS)

    def _design(self, rows: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in self.features:
            s = rows[c]
            if c in self.categories:
                levels = self.categories[c]
                values = s.astype(object)
                values = values.where(values.isin(levels), None)  # unseen -> missing
                out[c] = pd.Categorical(values, categories=levels)
            elif pd.api.types.is_bool_dtype(s):
                out[c] = s.to_numpy(dtype=np.float32)
            else:
                out[c] = s.to_numpy(dtype=np.float32, na_value=np.nan)
        return pd.DataFrame(out, index=rows.index)

    def _label(self, rows: pd.DataFrame) -> np.ndarray:
        y = rows["target"].to_numpy(dtype=np.float64)
        if self.cfg.target == "scaled":
            y = y / rows["target_scale"].to_numpy(dtype=np.float64)
        if self.cfg.loss == "log1p_l2":
            y = np.log1p(y)
        return y

    def _back(self, pred: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
        if self.cfg.loss == "log1p_l2":
            pred = np.expm1(pred)
        if self.cfg.target == "scaled":
            pred = pred * rows["target_scale"].to_numpy(dtype=np.float64)
        if not np.isfinite(pred).all():
            raise FloatingPointError(f"{self.name} produced non-finite predictions")
        return np.clip(pred, 0.0, None)

    def _weights(self, rows: pd.DataFrame, origin: date) -> np.ndarray | None:
        half = self.cfg.recency_halflife_weeks
        if half is None:
            return None
        age = (pd.Timestamp(origin) - rows["target_week"]).dt.days.to_numpy() / 7
        return 0.5 ** (age / half)

    def fit(self, rows: pd.DataFrame, origin: date) -> GBMForecaster:
        train, valid = schema.train_valid(rows, origin)
        block = pd.Timestamp(origin) - pd.Timedelta(weeks=schema.VALID_WEEKS)
        week_end = train["target_week"] + pd.Timedelta(days=6)
        if len(train) == 0:
            raise ValueError(f"no training rows before the validation block at {origin}")
        # The purge: training targets end by the block start, and so do their features.
        assert (week_end <= block).all() and (train["origin"] < block).all()
        self.fit_info = {"origin": origin, "block_start": block,
                         "train_max_week_end": week_end.max(),
                         "train_max_origin": train["origin"].max(),
                         "n_train": len(train), "n_valid": len(valid),
                         "train_index": train.index, "valid_index": valid.index}

        self.features = self._columns(rows)
        self.categories = {c: rows[c].cat.categories for c in self.features
                           if isinstance(rows[c].dtype, pd.CategoricalDtype)}
        self.boosters, self.best_iteration = {}, {}

        groups: dict[str | None, tuple[pd.DataFrame, pd.DataFrame]] = {}
        if self.cfg.per_tier:
            counts = train["tier"].astype(object).value_counts()
            for tier in TIERS:
                if counts.get(tier, 0) >= self.wrapper["min_tier_rows"]:
                    groups[tier] = (train[train["tier"] == tier], valid[valid["tier"] == tier])
        if not self.cfg.per_tier or len(groups) < len(TIERS):
            groups[GLOBAL] = (train, valid)
        self.fit_info["models"] = list(groups)

        for key, (tr, va) in groups.items():
            x, y, w = self._design(tr), self._label(tr), self._weights(tr, origin)
            if len(va) >= self.wrapper["min_valid_rows"]:
                xv, yv = self._design(va), self._label(va)
                rounds, patience = self.wrapper["num_boost_round"], \
                    self.wrapper["early_stopping_rounds"]
            else:
                xv = yv = None
                rounds, patience = self.wrapper["fixed_rounds"], None
            for seed in self.cfg.seeds:
                booster, best = self._train(x, y, w, xv, yv, seed, rounds, patience)
                self.boosters[(key, seed)] = booster
                self.best_iteration[(key, seed)] = best
        return self

    def _route(self, rows: pd.DataFrame) -> dict[str | None, np.ndarray]:
        """Row positions per model key."""
        if not self.cfg.per_tier:
            return {GLOBAL: np.arange(len(rows))}
        tiers = rows["tier"].astype(object).to_numpy()
        own = {k for k, _ in self.boosters if k is not GLOBAL}
        routes: dict[str | None, np.ndarray] = {}
        for key in own:
            routes[key] = np.flatnonzero(tiers == key)
        rest = np.flatnonzero(~np.isin(tiers, list(own)))
        if len(rest):
            routes[GLOBAL] = rest
        return routes

    def predict_seeds(self, rows: pd.DataFrame) -> dict[int, np.ndarray]:
        """Each seed's predictions, in row order: finite and non-negative."""
        if not self.boosters:
            raise RuntimeError("fit first")
        x = self._design(rows)
        out = {s: np.zeros(len(rows)) for s in self.cfg.seeds}
        for key, pos in self._route(rows).items():
            if not len(pos):
                continue
            for s in self.cfg.seeds:
                raw = self._predict_one(self.boosters[(key, s)], x.iloc[pos],
                                        self.best_iteration[(key, s)])
                out[s][pos] = self._back(np.asarray(raw, dtype=np.float64), rows.iloc[pos])
        return out

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        """The seed-average weekly mean, in row order."""
        return np.mean(list(self.predict_seeds(rows).values()), axis=0)

    def feature_importance(self) -> pd.DataFrame:
        """Gain per feature: columns tier (None = global), seed, feature, gain."""
        rows = []
        for (key, seed), booster in self.boosters.items():
            gain = self._gain(booster)
            rows += [(key, seed, f, float(gain.get(f, 0.0))) for f in self.features]
        return pd.DataFrame(rows, columns=["tier", "seed", "feature", "gain"])


OBJECTIVES = {"l2": "regression", "poisson": "poisson", "tweedie": "tweedie",
              "log1p_l2": "regression", "huber": "huber"}


class LGBMForecaster(GBMForecaster):
    model = "lgbm"

    def defaults(self) -> dict:
        return {"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
                "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
                "lambda_l2": 1.0, "verbosity": -1}

    def objective(self) -> dict:
        p = {"objective": OBJECTIVES[self.cfg.loss]}
        if self.cfg.loss == "tweedie":
            p["tweedie_variance_power"] = self.cfg.tweedie_power
        if self.cfg.loss == "huber":
            p["alpha"] = HUBER_DELTA
        return p

    def _train(self, x, y, w, xv, yv, seed, rounds, patience):
        import lightgbm as lgb

        cats = [c for c in x.columns if c in self.categories]
        params = {**self.lib_params, "seed": seed}
        dtrain = lgb.Dataset(x, y, weight=w, categorical_feature=cats, free_raw_data=False)
        if xv is None:
            booster = lgb.train(params, dtrain, num_boost_round=rounds)
            return booster, rounds
        dvalid = lgb.Dataset(xv, yv, reference=dtrain, categorical_feature=cats)
        booster = lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=[dvalid],
                            callbacks=[lgb.early_stopping(patience, verbose=False)])
        return booster, booster.best_iteration or rounds

    def _predict_one(self, booster, x, rounds):
        return booster.predict(x, num_iteration=rounds)

    def _gain(self, booster):
        return dict(zip(booster.feature_name(),
                        booster.feature_importance("gain"), strict=True))


def make_gbm(cfg: MLConfig) -> GBMForecaster:
    """The gradient-boosted forecaster for `cfg.model` ('lgbm' or 'xgb')."""
    if cfg.model == "lgbm":
        return LGBMForecaster(cfg)
    if cfg.model == "xgb":
        from irpd.ml.models.xgb import XGBForecaster

        return XGBForecaster(cfg)
    raise ValueError(f"not a gradient-boosted model: {cfg.model!r}")
