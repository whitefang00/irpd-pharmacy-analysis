"""The shared shapes: the panel, the context feature builders read, the predictions.

**The panel** has one row per (drug_code, origin, h). `origin` is a Sunday;
the row forecasts units in the Monday-start week `target_week`, h weeks
after it. Rows exist for drugs with a sale in the ACTIVE_WEEKS weeks before
the origin. `scored` marks the narrower set the backtest scores (a sale in
the SCORED_WEEKS weeks before), the same set `forecast.backtest` uses.

**What a row may know.** Only what was known on its origin. Calendar facts
about the target week (week of year, distance to Eid) are known in advance.
Everything else (sales, prices, traffic, stock) is as of the origin,
whatever h is. Builders get history through `PanelContext.history_at`,
which cuts weekly arrays and daily store lines alike, and tables rebuilt in
hindsight (prices, spells) through `PanelContext.connection_at`.

**Targets.** `target` is NaN when:
- the target week had a total stock-out day (`censored`);
- the week is before the drug's first sale;
- the week ends after the data the panel was built from (`available` False:
  rows of late training origins whose later weeks aren't known yet).

Stock is only trusted from the first full week after `config.STOCK_START`
(see `history.first_stock_week`). Before it `censor_known` is False, a
stock-out can't be seen, and the target is kept. Shelf-out days don't censor;
they are a demand feature, except that a week with 4+ shelf-out days counts as
censored (`history.SHELF_CENSOR_DAYS`).

**Model inputs.** Feature columns carry a group prefix (`irpd.ml.features`),
so an ablation is a column filter. Besides features, every model also gets
the bookkeeping columns in MODEL_COLUMNS (`h` and `tier`, as categoricals
where the library supports them).

**Training and early stopping.** For a scored origin O, a model may train on
`trainable(panel, O)`: rows with a known target whose week ended by O. For
early stopping, `train_valid(rows, O)` splits those by target week: the last
VALID_WEEKS weeks before O are validation, and training rows' targets end
before that block starts. A training row's features end at its own origin,
which is before its target week, so no training row sees the validation
block. Validation rows may come from earlier origins (h > 1), which is what
forecasting the block from the past looks like.

**Predictions** keep the columns of `forecast.backtest`, so `forecast.score`
reads them unchanged. Their `scale` column is the **MASE denominator**
(the panel's `mase_scale`), not the target scale. Score selection, purged
and evaluation origins in separate calls, and filter to one `config_hash`
and `seed` first (`seed` is ENSEMBLE_SEED for a seed average).

**Policy days.** The replay plans on days that aren't Sundays (Tue 31 Dec
2024, Mon 30 Jun 2025). A forecaster serves them from `policy_origin(day)`,
the last Sunday on or before it: the 1-2 days in between are not used, the
same as `forecast.forecast`, whose first week is the one after the last
complete week. Weeks past h = 8 repeat the h = 8 forecast
(`extend_horizon`); the plain policy needs at most 4 weeks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd

from irpd.ml import history as hist
from irpd.ml import split

KEYS = ["drug_code", "origin", "h"]
ACTIVE_WEEKS = 52
SCORED_WEEKS = 13
VALID_WEEKS = 8
ENSEMBLE_SEED = -1
# Bump when the panel's construction changes, so cached panels and predictions
# keyed by it are rebuilt.
PANEL_VERSION = 2  # 2: shelf-out censoring, season group, d_ewm_15, 26-week siblings

# Columns every panel has, besides features: name -> dtype.
BOOKKEEPING: dict[str, str] = {
    "drug_code": "string",            # any string dtype (str in pandas 3, object before)
    "origin": "datetime64",
    "h": "int8",
    "target_week": "datetime64",      # Monday of the target week
    "target": "float32",              # units; NaN if censored, before first sale or unavailable
    "censored": "bool",               # the target week had a total stock-out day
    "censor_known": "bool",           # the target week is covered by the stock ledger
    "available": "bool",              # the target week ended by the panel's data_end
    "scored": "bool",                 # sold in the SCORED_WEEKS weeks before the origin
    "tier": "category",               # tier as of the origin
    "eid": "bool",                    # the target week overlaps an Eid window
    "target_scale": "float32",        # denominator for the scaled target (SCALE_*)
    "mase_scale": "float32",          # in-sample naive MAE; NaN when there's no history
}
MODEL_COLUMNS = ("h", "tier")

CENSOR_MODES = ("mask", "raw", "impute")
GROUP_DEFS = ("ingredient", "ingredient_route", "ingredient_form")

# Scaled-target denominator: the drug's mean weekly units over the 13 weeks
# before the origin (uncensored weeks only); if that is 0 or missing, over 52
# weeks; if that too, SCALE_FLOOR (one unit in 13 weeks). See history.target_scale.
SCALE_WEEKS = (13, 52)
SCALE_FLOOR = 1 / 13

PREDICTION_COLUMNS = ["origin", "method", "drug_code", "tier", "h", "eid",
                      "actual", "mean", "var", "scale", "config_hash", "seed"]


@dataclass
class PanelContext:
    """What feature builders and history-based models may read. Nothing else.

    `history` is `ml.history.load(con, data_end)`. `con_at(day)` returns a
    connection as of `day` (for tables rebuilt in hindsight, such as prices
    and spells); without it, `con` itself is used, which suits tests. The real
    pipeline passes `split.selection_con` or an equivalent.
    """

    con: duckdb.DuckDBPyConnection | None
    data_end: date
    origins: list[date]
    history: dict
    censor_mode: str = "mask"
    group_def: str = "ingredient"
    con_at: Callable[[date], duckdb.DuckDBPyConnection] | None = None
    forecast_only: bool = False  # allow origins with no target week yet (a live forecast)
    # Experiment switches read by feature builders (e.g. calendar.OPTIONS). Empty
    # means the frozen behaviour; they are not part of MLConfig, so its hash and
    # the frozen system don't change.
    options: dict = field(default_factory=dict)
    drugs: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.censor_mode not in CENSOR_MODES:
            raise ValueError(f"censor_mode must be one of {CENSOR_MODES}")
        if self.group_def not in GROUP_DEFS:
            raise ValueError(f"group_def must be one of {GROUP_DEFS}")
        last = self.data_end if self.forecast_only else self.data_end - timedelta(weeks=1)
        late = [o for o in self.origins if o > last]
        if late:
            raise ValueError(f"origins need at least one target week by {self.data_end} "
                             "(or forecast_only, and by data_end); "
                             f"first bad one: {late[0]}")
        weeks = self.history["weeks"]
        if len(weeks) and weeks[-1] + timedelta(days=6) > self.data_end:
            raise ValueError("history runs past data_end")
        self.drugs = self.history["drugs"]

    def history_at(self, origin: date) -> dict:
        """The history as it stood on `origin` (weekly arrays and daily lines)."""
        return hist.cut(self.history, origin)

    def weeks_to(self, origin: date) -> int:
        """How many history weeks ended by `origin`."""
        return hist.weeks_to(self.history, origin)

    def connection_at(self, day: date) -> duckdb.DuckDBPyConnection:
        """A connection as of `day`, for tables rebuilt in hindsight."""
        if day > self.data_end:
            raise ValueError(f"{day} is after data_end {self.data_end}")
        return self.con_at(day) if self.con_at else self.con


def target_week(origin: date, h: int) -> date:
    """Monday of the week h weeks after the Sunday `origin`."""
    return origin + timedelta(days=1 + 7 * (h - 1))


def policy_origin(day: date) -> date:
    """The origin a forecast for policy day `day` stands on: the last Sunday by it."""
    return day - timedelta(days=(day.weekday() + 1) % 7)


def extend_horizon(mean: np.ndarray, weeks: int) -> np.ndarray:
    """Drugs x 8 weekly values extended to `weeks` by repeating the last week."""
    if weeks <= mean.shape[1]:
        return mean[:, :weeks]
    return np.concatenate([mean, np.repeat(mean[:, -1:], weeks - mean.shape[1], axis=1)], 1)


def _week_end(panel: pd.DataFrame) -> pd.Series:
    return panel["target_week"] + pd.Timedelta(days=6)


def trainable(panel: pd.DataFrame, origin: date) -> pd.DataFrame:
    """Rows a model fitted for `origin` may use: target known and its week ended by it."""
    return panel[panel["target"].notna() & (_week_end(panel) <= pd.Timestamp(origin))]


def train_valid(rows: pd.DataFrame, origin: date,
                valid_weeks: int = VALID_WEEKS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split `trainable` rows for early stopping: the last `valid_weeks` weeks validate."""
    end = _week_end(rows)
    if (end > pd.Timestamp(origin)).any():
        raise ValueError("pass trainable(panel, origin) rows only")
    block = pd.Timestamp(origin - timedelta(weeks=valid_weeks))
    return rows[end <= block], rows[end > block]


def feature_columns(panel: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    """The feature columns whose prefix is in `prefixes`, in panel order."""
    return [c for c in panel.columns if c.startswith(prefixes)]


def _check_dtype(s: pd.Series, dtype: str) -> bool:
    if dtype == "string":
        return pd.api.types.is_string_dtype(s)
    if dtype == "datetime64":
        return pd.api.types.is_datetime64_dtype(s)
    return str(s.dtype) == dtype


def validate_panel(panel: pd.DataFrame) -> None:
    """Raise if the panel breaks the contract."""
    from irpd.ml.features import PREFIXES

    missing = [c for c in BOOKKEEPING if c not in panel.columns]
    if missing:
        raise ValueError(f"panel is missing {missing}")
    for c, dtype in BOOKKEEPING.items():
        if not _check_dtype(panel[c], dtype):
            raise ValueError(f"column {c} is {panel[c].dtype}, expected {dtype}")
    extra = [c for c in panel.columns if c not in BOOKKEEPING]
    stray = [c for c in extra if not c.startswith(tuple(PREFIXES))]
    if stray:
        raise ValueError(f"columns without a feature-group prefix: {stray}")
    bad = [c for c in extra
           if str(panel[c].dtype) not in ("float32", "category", "bool", "int8")]
    if bad:
        raise ValueError(f"feature columns must be float32, int8, bool or category: {bad}")
    num = [c for c in extra if str(panel[c].dtype) == "float32"]
    if num and np.isinf(panel[num].to_numpy()).any():
        raise ValueError("infinite feature values")
    if panel["drug_code"].isna().any() or panel["origin"].isna().any():
        raise ValueError("null keys")
    if panel.duplicated(KEYS).any():
        raise ValueError("duplicate (drug_code, origin, h) rows")
    if not panel["h"].between(1, split.HORIZON).all():
        raise ValueError(f"h must be 1..{split.HORIZON}")
    if (panel["origin"].dt.day_of_week != 6).any():
        raise ValueError("origins must be Sundays")
    expect = panel["origin"] + pd.to_timedelta(1 + 7 * (panel["h"].astype(int) - 1), unit="D")
    if (panel["target_week"] != expect).any():
        raise ValueError("target_week must be the Monday h weeks after the origin")
    t = panel["target"]
    if ((panel["censored"] | ~panel["available"]) & t.notna()).any():
        raise ValueError("censored or unavailable rows must have a NaN target")
    if (panel["censored"] & ~panel["censor_known"]).any():
        raise ValueError("a week can't be censored where censoring isn't known")
    if np.isinf(t).any() or (t < 0).any():
        raise ValueError("targets must be finite and non-negative")
    s = panel["target_scale"]
    if s.isna().any() or np.isinf(s).any() or (s <= 0).any():
        raise ValueError("target_scale must be finite and positive (see SCALE_FLOOR)")
    if np.isinf(panel["mase_scale"]).any() or (panel["mase_scale"] < 0).any():
        raise ValueError("mase_scale must be finite and non-negative, or NaN")


def validate_predictions(pred: pd.DataFrame) -> None:
    """Raise if a predictions table breaks the contract."""
    missing = [c for c in PREDICTION_COLUMNS if c not in pred.columns]
    if missing:
        raise ValueError(f"predictions are missing {missing}")
    for c in ("origin", "method", "drug_code", "h", "config_hash", "seed"):
        if pred[c].isna().any():
            raise ValueError(f"null {c}")
    if not pred["h"].between(1, split.HORIZON).all():
        raise ValueError(f"h must be 1..{split.HORIZON}")
    m, v = pred["mean"].astype(float), pred["var"].astype(float)
    if m.isna().any() or np.isinf(m).any() or (m < 0).any():
        raise ValueError("mean must be finite and non-negative")
    if v.isna().any() or np.isinf(v).any() or (v < m - 1e-9).any():
        raise ValueError("var must be finite and at least the mean (dispersion >= 1)")
    a = pred["actual"].astype(float)
    if np.isinf(a).any() or (a < 0).any():
        raise ValueError("actual must be non-negative, or NaN when censored")
    if pred.duplicated(["origin", "method", "drug_code", "h", "config_hash", "seed"]).any():
        raise ValueError("duplicate prediction rows")
