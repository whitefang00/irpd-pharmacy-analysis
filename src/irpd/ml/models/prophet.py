"""Per-drug Prophet with explicit Eid and Ramadan holidays (plan 6.4).

A side experiment: can a per-drug model that is told when Eid falls beat
the global model on the drugs where it has the best chance? So Prophet only
runs on a fixed subset at each origin; every other drug gets the existing
SES+Eid forecast and is flagged as such.

**Subset** at origin O (`ProphetForecaster.subset`): drugs whose tier as of
O is fast or regular, with at least SUBSET_WEEKS weeks from their first sale
to O, and fewer than MAX_CENSORED of the last SUBSET_WEEKS weeks censored.

**Series.** One row per history week ended by O, dated by the week's
Monday, from the drug's first sale on. Censored weeks (`out`) keep their
row with y = NaN; Prophet leaves them out of the fit.

**Holidays.** Prophet counts holiday windows in days, but with weekly rows
it only ever sees the dated Mondays. So each holiday is placed on a Monday
and its day window chosen to land on whole weeks:

- `eid`: each date in `forecast.EID`, moved to the Monday of its week M,
  with `lower_window=-7` and `upper_window=28`. The Mondays it reaches are
  M-7 .. M+28: the week before Eid's week, Eid's week and the four after
  (the same span as `forecast.EID_DAYS`). Each of those six weeks gets its
  own effect; the other 30 offsets never fall on a row and stay at zero.
- `ramadan`: the 30 days before Eid (Eid-30 .. Eid-1), placed on the Monday
  R of the week Ramadan starts in, `lower_window=0`, `upper_window` reaching
  the Monday of the week holding Eid-1. Its last week or so overlaps Eid's
  week-before offset; the two effects add, and the priors keep them apart.

**Forecast.** The eight Mondays after O (`schema.target_week(O, 1..8)`),
clipped at zero. Uncertainty sampling is off: variance is not this model's
job (`irpd.ml.uncertainty` adds it).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from datetime import date, timedelta
from time import perf_counter

import numpy as np
import pandas as pd

from irpd import forecast
from irpd.ml import schema, split
from irpd.ml.config import MLConfig

SUBSET_TIERS = ("fast", "regular")
SUBSET_WEEKS = 52
MAX_CENSORED = 0.2   # share of the last SUBSET_WEEKS weeks; the subset needs fewer
RAMADAN_DAYS = 30
FALLBACK = "ses+eid"
TRENDS = ("flat", "linear")


def _monday(day: date) -> date:
    return day - timedelta(days=day.weekday())


def holidays() -> pd.DataFrame:
    """Eid and Ramadan placed on Mondays, with day windows covering whole weeks."""
    rows = []
    for eid in forecast.EID:
        rows.append(("eid", _monday(eid), -7, 28))
        start = _monday(eid - timedelta(days=RAMADAN_DAYS))
        last = _monday(eid - timedelta(days=1))
        rows.append(("ramadan", start, 0, (last - start).days))
    out = pd.DataFrame(rows, columns=["holiday", "ds", "lower_window", "upper_window"])
    out["ds"] = pd.to_datetime(out["ds"])
    return out


def _quiet() -> None:
    """Silence cmdstanpy and Prophet's info logging in this process."""
    import cmdstanpy.utils

    logging.getLogger("prophet.plot").setLevel(logging.CRITICAL)  # "Importing plotly failed"
    cmdstanpy.utils.get_logger().setLevel(logging.ERROR)  # first use resets the level
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)


def _fit_one(ds: np.ndarray, y: np.ndarray, future: np.ndarray, trend: str, yearly: bool,
             hol: pd.DataFrame, seed: int) -> tuple[np.ndarray | None, float]:
    """Fit one drug and forecast `future`; None if the fit fails."""
    _quiet()
    from prophet import Prophet

    started = perf_counter()
    try:
        model = Prophet(growth=trend, yearly_seasonality=yearly, weekly_seasonality=False,
                        daily_seasonality=False, holidays=hol, uncertainty_samples=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(pd.DataFrame({"ds": pd.to_datetime(ds), "y": y}), seed=seed)
            yhat = model.predict(pd.DataFrame({"ds": pd.to_datetime(future)}))["yhat"]
        mean = np.maximum(yhat.to_numpy(float), 0)
        if not np.isfinite(mean).all():
            return None, perf_counter() - started
        return mean, perf_counter() - started
    except Exception:  # a failed Stan fit falls back to SES+Eid
        return None, perf_counter() - started


class ProphetForecaster:
    """Prophet per drug on the plan 6.4 subset; SES+Eid for everyone else.

    `fit(rows, origin)` only fixes the origin: Prophet reads history, not
    panel rows, and the subset needs each drug's tier as of the origin,
    which comes from the `tier` column of the rows `predict` gets. So the
    per-drug fits run on the first `predict` for an origin and are kept.

    After `predict`, `last_used_prophet` marks the rows Prophet forecast,
    and `fit_seconds` holds the time per fitted drug.
    """

    name = "prophet"

    def __init__(self, ctx: schema.PanelContext, trend: str = "flat", yearly: bool = False,
                 n_jobs: int = -1, seed: int = 0) -> None:
        if trend not in TRENDS:
            raise ValueError(f"trend must be one of {TRENDS}")
        self.ctx, self.trend, self.yearly = ctx, trend, bool(yearly)
        self.n_jobs, self.seed = n_jobs, seed
        self.origin: date | None = None
        self.last_used_prophet = np.zeros(0, dtype=bool)
        self.fit_seconds: dict[str, float] = {}
        self.failed: list[str] = []
        self._means: dict[str, np.ndarray] = {}

    @classmethod
    def from_config(cls, ctx: schema.PanelContext, cfg: MLConfig,
                    n_jobs: int = -1) -> ProphetForecaster:
        """Read `trend` and `yearly` from `cfg.params`; the first seed seeds Stan."""
        if cfg.model != "prophet":
            raise ValueError("not a prophet config")
        p = cfg.params
        return cls(ctx, trend=p.get("trend", "flat"), yearly=p.get("yearly", False),
                   n_jobs=n_jobs, seed=cfg.seeds[0])

    def subset(self, origin: date, tiers: Mapping[str, str]) -> list[str]:
        """Drugs Prophet forecasts at `origin`, given each drug's tier as of it."""
        h = self.ctx.history_at(origin)
        n = len(h["weeks"])
        first = h["first_sale"]
        recent = h["out"][:, -SUBSET_WEEKS:]
        censored = recent.sum(axis=1) < MAX_CENSORED * SUBSET_WEEKS
        long = (first >= 0) & (n - first >= SUBSET_WEEKS)
        keep = long & censored
        return [str(c) for c, k in zip(h["drugs"], keep, strict=True)
                if k and tiers.get(c) in SUBSET_TIERS]

    def fit(self, rows: pd.DataFrame, origin: date) -> ProphetForecaster:
        if origin != self.origin:
            self.origin = origin
            self._means, self.fit_seconds, self.failed = {}, {}, []
        return self

    def _fit_drugs(self, codes: list[str]) -> None:
        from joblib import Parallel, delayed

        todo = [c for c in codes if c not in self._means and c not in self.failed]
        if not todo:
            return
        h = self.ctx.history_at(self.origin)
        idx = {c: i for i, c in enumerate(h["drugs"])}
        weeks = np.array(h["weeks"], dtype="datetime64[D]")
        future = np.array([schema.target_week(self.origin, k)
                           for k in range(1, split.HORIZON + 1)], dtype="datetime64[D]")
        hol = holidays()
        jobs = []
        for c in todo:
            i = idx[c]
            first = h["first_sale"][i]
            y = np.where(h["out"][i], np.nan, h["y"][i])[first:]
            jobs.append(delayed(_fit_one)(weeks[first:], y, future, self.trend, self.yearly,
                                          hol, self.seed))
        n_jobs = 1 if len(jobs) == 1 else self.n_jobs
        results = Parallel(n_jobs=n_jobs)(jobs)
        for c, (mean, seconds) in zip(todo, results, strict=True):
            self.fit_seconds[c] = seconds
            if mean is None:
                self.failed.append(c)
            else:
                self._means[c] = mean

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        if self.origin is None:
            raise RuntimeError("call fit first")
        origins = pd.to_datetime(rows["origin"]).dt.date.unique()
        if len(origins) and set(origins) != {self.origin}:
            raise ValueError(f"rows must all stand on the fitted origin {self.origin}")
        first = rows.drop_duplicates("drug_code")
        tiers = dict(zip(first["drug_code"].astype(str), first["tier"].astype(str),
                         strict=True))
        self._fit_drugs(self.subset(self.origin, tiers))

        h = self.ctx.history_at(self.origin)
        idx = {c: i for i, c in enumerate(h["drugs"])}
        base = forecast.predict(h, np.full(len(h["drugs"]), FALLBACK, dtype=object),
                                split.HORIZON)["mean"]
        codes = rows["drug_code"].astype(str).to_numpy()
        steps = rows["h"].astype(int).to_numpy() - 1
        out = np.zeros(len(rows))
        used = np.zeros(len(rows), dtype=bool)
        for r, (c, k) in enumerate(zip(codes, steps, strict=True)):
            if c in self._means:
                out[r], used[r] = self._means[c][k], True
            elif c in idx:
                out[r] = base[idx[c], k]
        self.last_used_prophet = used
        return np.maximum(np.nan_to_num(out), 0)
