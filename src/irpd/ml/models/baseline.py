"""The existing statistical methods behind the `Forecaster` interface.

`BaselineForecaster` doesn't learn from panel rows. `fit` records the origin;
`predict` runs `forecast.predict` on `ctx.history_at(origin)` with each
drug's method picked by its tier, exactly as `forecast.forecast` does. Tiers
come from the rows (the tier as of the origin). A tier missing from
`method_by_tier` gets 'snaive', as in `forecast.forecast`.

`predict_var` gives `forecast.predict`'s variance (mean times the drug's
dispersion on Eid-adjusted history), which `backtest.run` uses to fill `var`.
`baseline_dispersion` is the same dispersion on its own, the recipe the
backtest applies to forecasters that offer no variance.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date

import numpy as np
import pandas as pd

from irpd import forecast as fc
from irpd.ml import schema, split


def baseline_dispersion(h: dict) -> np.ndarray:
    """Per drug of history `h`: variance / mean of the Eid-adjusted weekly units, at least 1.

    The same calculation `forecast.predict` multiplies its mean by. `h` must
    already be cut to the origin.
    """
    y = np.where(h["out"], np.nan, h["y"])
    past = fc.eid_index(h["weeks"], fc.eid_factors(h["lines"]))
    return fc.dispersion(y / past)


def config_hash(method_by_tier: dict[str, str]) -> str:
    """A stable name for a baseline routing, for `backtest.run`'s cache key."""
    body = json.dumps({"model": "baseline", "method_by_tier": dict(sorted(method_by_tier.items()))},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()[:12]


class BaselineForecaster:
    """`forecast.predict` per tier, standing on the fitted origin's history."""

    def __init__(self, ctx: schema.PanelContext, method_by_tier: dict[str, str] | None = None,
                 name: str | None = None) -> None:
        self.ctx = ctx
        self.method_by_tier = dict(method_by_tier or fc.METHOD_BY_TIER)
        unknown = {m.removesuffix("+eid") for m in self.method_by_tier.values()} - {
            "zero", "snaive", *fc.LEVEL_METHODS}
        if unknown:
            raise ValueError(f"unknown methods {sorted(unknown)}")
        methods = set(self.method_by_tier.values())
        self.name = name or (methods.pop() if len(methods) == 1 else "baseline")
        self.origin: date | None = None

    def fit(self, rows: pd.DataFrame, origin: date) -> BaselineForecaster:
        self.origin = origin
        return self

    def _run(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.origin is None:
            raise RuntimeError("fit before predict")
        origins = pd.to_datetime(rows["origin"]).dt.date.unique()
        if len(origins) and set(origins) != {self.origin}:
            raise ValueError(f"rows are for {sorted(origins)}, fitted on {self.origin}")
        h = self.ctx.history_at(self.origin)
        idx = {c: i for i, c in enumerate(h["drugs"])}
        drugs = pd.unique(rows["drug_code"])
        missing = [d for d in drugs if d not in idx]
        if missing:
            raise ValueError(f"drugs not in the history: {missing[:5]}")
        pick = np.array([idx[d] for d in drugs], dtype=int)
        tier = rows.drop_duplicates("drug_code").set_index("drug_code")["tier"].astype(str)
        method = np.array([self.method_by_tier.get(tier[d], "snaive") for d in drugs],
                          dtype=object)
        sub = {**h, "drugs": h["drugs"][pick], "y": h["y"][pick], "out": h["out"][pick]}
        p = fc.predict(sub, method, split.HORIZON)
        row = pd.Series(np.arange(len(drugs)), index=drugs)[rows["drug_code"]].to_numpy()
        col = rows["h"].to_numpy(int) - 1
        return p["mean"][row, col], p["var"][row, col]

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return self._run(rows)[0]

    def predict_var(self, rows: pd.DataFrame) -> np.ndarray:
        """`forecast.predict`'s variance for the same rows, aligned like `predict`."""
        return self._run(rows)[1]
