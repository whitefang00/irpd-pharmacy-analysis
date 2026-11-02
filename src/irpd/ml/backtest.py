"""Run any forecaster over scored origins into one predictions table, and score it.

`run` fits a fresh forecaster on `schema.trainable(panel, origin)` and
predicts that origin's `scored` rows. Every model, ML or statistical, goes
through it, so they are scored on the same rows.

**Variance.** If the forecaster has `predict_var(rows)`, that fills `var`.
Otherwise the baseline recipe does: the mean times the drug's dispersion
(`models.baseline.baseline_dispersion`, the one `forecast.predict` uses) on
the history cut at the origin. `var` is never below the mean.

**Seeds.** A forecaster may offer `predict_seeds(rows) -> {seed: mean}`.
Then each seed's rows are kept and their average is written with
`seed = schema.ENSEMBLE_SEED`. A forecaster without seeds writes its one
prediction with `seed = ENSEMBLE_SEED` too, so `ENSEMBLE_SEED` always means
"the model's final forecast" and scoring defaults to it. With seeds, `var`
always comes from the baseline recipe on each seed's (or the average's) mean.

**Cache.** With `cache_dir`, each origin's predictions are written as parquet
keyed by (config hash, origin, `schema.PANEL_VERSION`) and read back on a
rerun instead of refitting. The config hash is the caller's promise that the
forecaster is the same; `MLConfig.hash()` or `baseline.config_hash`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from irpd import config
from irpd import forecast as fc
from irpd.ml import schema, split
from irpd.ml.models.base import Forecaster
from irpd.ml.models.baseline import baseline_dispersion

CACHE_DIR = config.ML_DIR / "predictions"
PERIODS = ("selection", "purged", "eval")


def cache_path(cache_dir: Path, cfg_hash: str, origin: date) -> Path:
    return Path(cache_dir) / f"{cfg_hash}_{origin:%Y%m%d}_v{schema.PANEL_VERSION}.parquet"


def _as_date(o) -> date:
    return pd.Timestamp(o).date()


def _frame(rows: pd.DataFrame, name: str, cfg_hash: str, seed: int, mean: np.ndarray,
           var: np.ndarray) -> pd.DataFrame:
    mean = np.asarray(mean, float)
    if mean.shape != (len(rows),):
        raise ValueError(f"{name} returned {mean.shape} predictions for {len(rows)} rows")
    if not np.isfinite(mean).all() or (mean < 0).any():
        raise ValueError(f"{name} returned a negative or non-finite mean")
    return pd.DataFrame({
        "origin": rows["origin"].to_numpy(), "method": name,
        "drug_code": rows["drug_code"].astype(str).to_numpy(),
        "tier": rows["tier"].astype(str).to_numpy(),
        "h": rows["h"].to_numpy(), "eid": rows["eid"].to_numpy(bool),
        "actual": rows["target"].to_numpy(float), "mean": mean,
        "var": np.maximum(np.asarray(var, float), mean),
        "scale": rows["mase_scale"].to_numpy(float),
        "config_hash": cfg_hash, "seed": np.int64(seed)})


def predict_origin(model: Forecaster, rows: pd.DataFrame, ctx: schema.PanelContext,
                   origin: date, cfg_hash: str) -> pd.DataFrame:
    """One origin's predictions table from a fitted forecaster."""
    h = ctx.history_at(origin)
    disp = pd.Series(baseline_dispersion(h), index=h["drugs"])

    def recipe(mean):
        return np.asarray(mean, float) * disp[rows["drug_code"]].to_numpy()

    name = getattr(model, "name", type(model).__name__)
    if hasattr(model, "predict_seeds"):
        seeds = {int(s): np.asarray(m, float) for s, m in model.predict_seeds(rows).items()}
        if not seeds or schema.ENSEMBLE_SEED in seeds:
            raise ValueError(f"predict_seeds must return seeds other than {schema.ENSEMBLE_SEED}")
        ens = np.mean(list(seeds.values()), axis=0)
        parts = [_frame(rows, name, cfg_hash, s, m, recipe(m)) for s, m in sorted(seeds.items())]
        parts.append(_frame(rows, name, cfg_hash, schema.ENSEMBLE_SEED, ens, recipe(ens)))
        return pd.concat(parts, ignore_index=True)
    mean = model.predict(rows)
    var = model.predict_var(rows) if hasattr(model, "predict_var") else recipe(mean)
    return _frame(rows, name, cfg_hash, schema.ENSEMBLE_SEED, mean, var)


def run(make_forecaster: Callable[[], Forecaster], cfg_hash: str, origins: Iterable[date],
        panel: pd.DataFrame, ctx: schema.PanelContext, cache_dir: Path | None = None,
        retrain_every: int = 1) -> pd.DataFrame:
    """Predictions for every origin's scored rows (schema.PREDICTION_COLUMNS).

    `make_forecaster()` returns a new, unfitted forecaster. It is fitted at
    the first origin and then every `retrain_every` origins (in date order);
    the origins in between are predicted by the last fitted model. Each fit
    sees only `schema.trainable(panel, fit_origin)`.
    """
    if retrain_every < 1:
        raise ValueError("retrain_every must be at least 1")
    origins = sorted(_as_date(o) for o in origins)
    if len(set(origins)) != len(origins):
        raise ValueError("duplicate origins")
    split.check_scored(origins, ctx.data_end)
    by_origin = dict(tuple(panel.groupby(panel["origin"].dt.date, sort=False)))
    model, fitted_at, parts = None, None, []
    for i, origin in enumerate(origins):
        path = cache_path(cache_dir, cfg_hash, origin) if cache_dir else None
        if path is not None and path.exists():
            parts.append(pd.read_parquet(path))
            continue
        fit_origin = origins[i - i % retrain_every]
        if fitted_at != fit_origin:
            model = make_forecaster().fit(schema.trainable(panel, fit_origin), fit_origin)
            fitted_at = fit_origin
        rows = by_origin.get(origin)
        if rows is None:
            raise ValueError(f"no panel rows at {origin}")
        rows = rows[rows["scored"]]
        part = predict_origin(model, rows, ctx, origin, cfg_hash)
        schema.validate_predictions(part)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            part.to_parquet(path, index=False)
        parts.append(part)
    pred = pd.concat(parts, ignore_index=True)
    schema.validate_predictions(pred)
    return pred


def period_of(origins: pd.Series) -> pd.Series:
    """'selection', 'purged' or 'eval' per origin, from where its targets end (split)."""
    o = pd.to_datetime(origins)
    end = o + pd.Timedelta(weeks=split.HORIZON)
    return pd.Series(np.select(
        [end <= pd.Timestamp(split.SELECTION_END), o < pd.Timestamp(split.EVAL_START)],
        ["selection", "purged"], "eval"), index=origins.index)


def score(pred: pd.DataFrame, period: str, config_hash: str | None = None,
          seed: int = schema.ENSEMBLE_SEED, by: tuple[str, ...] = ("tier", "method")
          ) -> pd.DataFrame:
    """`forecast.score` on one period, one config and one seed.

    `forecast.score` labels its own 'period' by origin year, which would put
    the purged 2024 origins in with selection. So this groups without it and
    writes `period` as the split's name instead. With several configs in
    `pred`, `config_hash` is required.
    """
    if period not in PERIODS:
        raise ValueError(f"period must be one of {PERIODS}")
    if "period" in by:
        raise ValueError("'period' is set by this function; leave it out of `by`")
    if config_hash is None:
        hashes = pred["config_hash"].unique()
        if len(hashes) != 1:
            raise ValueError(f"several configs in pred; pass config_hash (one of {list(hashes)})")
        config_hash = hashes[0]
    sub = pred[(pred["config_hash"] == config_hash) & (pred["seed"] == seed)]
    sub = sub[period_of(sub["origin"]) == period]
    if sub.empty:
        raise ValueError(f"no predictions for period={period}, config={config_hash}, seed={seed}")
    out = fc.score(sub, by=by)
    out.insert(0, "period", period)
    return out
