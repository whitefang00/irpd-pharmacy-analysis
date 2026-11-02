"""XGBoost behind the same interface as LightGBM (plan 6.3).

Everything but the library calls is shared with `lgbm.GBMForecaster`; see
its docstring for features, targets, early stopping, seeds and tiers.

Defaults mirror LightGBM's: `hist` trees grown leaf-wise (`lossguide`,
63 leaves, no depth limit), learning rate 0.05, 0.8 row and column
subsampling, L2 1.0. XGBoost has no minimum rows per leaf, only a minimum
hessian sum (`min_child_weight`). For l2 and log1p_l2 the hessian is 1 per
row, so 100 matches LightGBM's 100 rows exactly. For poisson, tweedie and
pseudo-Huber the hessian depends on the prediction and can be far below 1
per row, so 100 would block splits; there the XGBoost default of 1 is used.
Override with `params`.
"""

from __future__ import annotations

import pandas as pd

from irpd.ml.models.lgbm import HUBER_DELTA, GBMForecaster

OBJECTIVES = {"l2": "reg:squarederror", "poisson": "count:poisson", "tweedie": "reg:tweedie",
              "log1p_l2": "reg:squarederror", "huber": "reg:pseudohubererror"}
UNIT_HESSIAN = ("l2", "log1p_l2")


class XGBForecaster(GBMForecaster):
    model = "xgb"

    def defaults(self) -> dict:
        return {"eta": 0.05, "tree_method": "hist", "grow_policy": "lossguide",
                "max_leaves": 63, "max_depth": 0,
                "min_child_weight": 100.0 if self.cfg.loss in UNIT_HESSIAN else 1.0,
                "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 1.0,
                "verbosity": 0}

    def objective(self) -> dict:
        p = {"objective": OBJECTIVES[self.cfg.loss]}
        if self.cfg.loss == "tweedie":
            p["tweedie_variance_power"] = self.cfg.tweedie_power
        if self.cfg.loss == "huber":
            p["huber_slope"] = HUBER_DELTA
        return p

    def _matrix(self, x: pd.DataFrame, y=None, w=None):
        import xgboost as xgb

        return xgb.DMatrix(x, label=y, weight=w, enable_categorical=True)

    def _train(self, x, y, w, xv, yv, seed, rounds, patience):
        import xgboost as xgb

        params = {**self.lib_params, "seed": seed}
        dtrain = self._matrix(x, y, w)
        if xv is None:
            return xgb.train(params, dtrain, num_boost_round=rounds), rounds
        booster = xgb.train(params, dtrain, num_boost_round=rounds,
                            evals=[(self._matrix(xv, yv), "valid")],
                            early_stopping_rounds=patience, verbose_eval=False)
        return booster, booster.best_iteration + 1

    def _predict_one(self, booster, x, rounds):
        return booster.predict(self._matrix(x), iteration_range=(0, rounds))

    def _gain(self, booster):
        return {k: float(v) for k, v in booster.get_score(importance_type="total_gain").items()}

