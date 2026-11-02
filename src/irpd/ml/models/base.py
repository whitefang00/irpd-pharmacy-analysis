"""The interface every forecaster implements, ML or statistical.

`fit(rows, origin)` gets `schema.trainable(panel, origin)` for the scored
origin and may read nothing else. A model that early-stops splits `rows`
with `schema.train_valid(rows, origin)`, never its own way. `predict(rows)`
gets the origin's panel rows and returns the weekly mean in the same order:
finite and never negative.

Statistical baselines and Prophet don't use panel features; they take the
`PanelContext` at construction and read `ctx.history_at(origin)`, so the
backtest loop treats every model the same.

Variance is not the forecaster's job: `irpd.ml.uncertainty` adds it.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Forecaster(Protocol):
    name: str

    def fit(self, rows: pd.DataFrame, origin: date) -> Forecaster: ...

    def predict(self, rows: pd.DataFrame) -> np.ndarray: ...
