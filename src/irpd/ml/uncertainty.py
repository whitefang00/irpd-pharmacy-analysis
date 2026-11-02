"""U1: turn a weekly mean forecast into the variance a reorder point needs.

Every week is negative binomial with variance = mean x dispersion, the same
form the baseline uses, so weekly means and variances still sum over any
lead time and plug into `contest.to_policy` (plan 10.1, 10.2).

**The dispersion for each drug at origin O** (`u1_dispersion`):

- Residuals come only from out-of-sample predictions: rows from origins
  before O whose target week ended by O, at h = 1..4. Where several origins
  forecast the same (drug, target week), the one from the latest origin
  (smallest h) is kept. Censored weeks (NaN `actual`) are skipped.
- Over the drug's last 52 such weeks, d = sum((y - mu)^2) / sum(mu), floored
  at 1 (Poisson).
- d is shrunk toward the tier's pooled value (the same estimator over every
  residual week of the tier's drugs, also floored at 1) with weight
  n / (n + 20), n the drug's residual weeks. Below 8 weeks the drug takes the
  pooled value. 20 and 8 are fixed by the plan, not tuned.
- Where no earlier predictions exist at all, the baseline recipe is used
  (`baseline_dispersion`: the drug's own variance / mean over its last year,
  on Eid-adjusted history, exactly as `forecast.predict` does it).

Tiers are as of O. A tier with no residual weeks at all takes the fallback
where one is given, otherwise the pool over every tier.

**Judging it** (`coverage_pinball`, plan 10.4): coverage and pinball loss at
0.9 and 0.95 for weekly demand and for 2- and 4-week totals. Totals sum the
means and variances over consecutive h for one drug and origin, which is
right only if weeks are independent given the mean; the window coverage is
what checks that assumption.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import numpy as np
import pandas as pd

from irpd import forecast as fc
from irpd.ml.schema import ENSEMBLE_SEED, PanelContext

MAX_H = 4            # residuals use forecasts at h = 1..MAX_H only
RESIDUAL_WEEKS = 52  # a drug's last this-many residual weeks
SHRINK_K = 20        # weight n / (n + SHRINK_K) on the drug's own estimate
MIN_WEEKS = 8        # below this, the drug takes the tier's pooled value

Fallback = pd.Series | Callable[[date], pd.Series] | None


# ---- the baseline recipe ------------------------------------------------


def dispersion_from_history(h: dict) -> np.ndarray:
    """The baseline's dispersion per drug in `h` (a `load_history`-shaped dict).

    The same calculation as inside `forecast.predict`: censored weeks dropped,
    history divided by the Eid factors learned from `h["lines"]`, then
    `forecast.dispersion`.
    """
    y = np.where(h["out"], np.nan, h["y"])
    factors = fc.eid_factors(h["lines"])
    return fc.dispersion(y / fc.eid_index(h["weeks"], factors))


def baseline_dispersion(ctx: PanelContext, origin: date) -> pd.Series:
    """The baseline recipe on the history as it stood on `origin`, by drug_code."""
    h = ctx.history_at(origin)
    return pd.Series(dispersion_from_history(h), index=pd.Index(h["drugs"], name="drug_code"),
                     name="dispersion")


# ---- U1 -----------------------------------------------------------------


def _one_run(pred: pd.DataFrame) -> pd.DataFrame:
    """The seed-average rows of a single config (all rows if there's no seed average)."""
    if "seed" in pred and (pred["seed"] == ENSEMBLE_SEED).any():
        pred = pred[pred["seed"] == ENSEMBLE_SEED]
    if "config_hash" in pred and pred["config_hash"].nunique() > 1:
        raise ValueError("pass predictions from one config_hash")
    if pred.duplicated(["origin", "drug_code", "h"]).any():
        raise ValueError("duplicate (origin, drug_code, h) rows: pass one seed")
    return pred


def residuals(past_pred: pd.DataFrame, origin: date) -> pd.DataFrame:
    """The residual weeks usable at `origin`: drug_code, target_week, h, y, mu.

    Only origins before `origin`, target weeks ended by it, h <= MAX_H, the
    smallest h per (drug, target week), censored weeks dropped.
    """
    p = _one_run(past_pred)
    o = pd.to_datetime(p["origin"])
    h = p["h"].astype(int)
    week_end = o + pd.to_timedelta(7 * h, unit="D")  # the Sunday the target week ends
    at = pd.Timestamp(origin)
    keep = (o < at) & (h <= MAX_H) & (week_end <= at)
    r = pd.DataFrame({"drug_code": p["drug_code"][keep].to_numpy(),
                      "target_week": (week_end[keep] - pd.Timedelta(days=6)).to_numpy(),
                      "h": h[keep].to_numpy(),
                      "y": p["actual"][keep].astype(float).to_numpy(),
                      "mu": p["mean"][keep].astype(float).to_numpy()})
    r = (r.sort_values(["drug_code", "target_week", "h"])
          .drop_duplicates(["drug_code", "target_week"], keep="first"))
    return r[r["y"].notna()].reset_index(drop=True)


def _estimate(sse: pd.Series, smu: pd.Series) -> pd.Series:
    d = sse / smu.where(smu > 0)
    return d.clip(lower=1.0)


def u1_dispersion(past_pred: pd.DataFrame, origin: date, tiers: pd.Series,
                  fallback: Fallback = None) -> pd.Series:
    """U1 dispersion per drug in `tiers` (drug_code -> tier as of `origin`).

    `past_pred` is a predictions table (`schema.PREDICTION_COLUMNS`) of one
    config; seed-average rows are used when present. Only rows the residual
    rule allows are read, so passing the whole backtest is safe. `fallback`
    (a Series by drug_code, or a function of the origin returning one) is the
    baseline recipe, used where there are no earlier predictions at all, and
    for a tier with no residual weeks. Without it, those cases raise or fall
    back to the all-tier pool.
    """
    tiers = pd.Series(tiers).astype(object)
    r = residuals(past_pred, origin)
    r = r[r["drug_code"].isin(tiers.index)]

    def fb() -> pd.Series | None:
        if fallback is None:
            return None
        f = fallback(origin) if callable(fallback) else fallback
        return f.reindex(tiers.index)

    if r.empty:
        f = fb()
        if f is None:
            raise ValueError(f"no predictions usable at {origin}; pass the baseline fallback")
        return f.fillna(1.0).rename("dispersion").rename_axis("drug_code")

    r = (r.sort_values("target_week")
          .groupby("drug_code", sort=False).tail(RESIDUAL_WEEKS))
    r = r.assign(se=(r["y"] - r["mu"]) ** 2, tier=r["drug_code"].map(tiers))
    by_drug = r.groupby("drug_code").agg(sse=("se", "sum"), smu=("mu", "sum"), n=("se", "size"))
    by_tier = r.groupby("tier").agg(sse=("se", "sum"), smu=("mu", "sum"))
    pooled_tier = _estimate(by_tier["sse"], by_tier["smu"])
    pooled_all = _estimate(pd.Series([r["se"].sum()]), pd.Series([r["mu"].sum()])).iloc[0]

    pooled = tiers.map(pooled_tier).astype(float)
    if pooled.isna().any():
        f = fb()
        pooled = pooled.fillna(f) if f is not None else pooled
        pooled = pooled.fillna(pooled_all if np.isfinite(pooled_all) else 1.0)

    own = _estimate(by_drug["sse"], by_drug["smu"]).reindex(tiers.index)
    n = by_drug["n"].reindex(tiers.index).fillna(0)
    w = n / (n + SHRINK_K)
    use = (n >= MIN_WEEKS) & own.notna()
    out = pooled.where(~use, w * own + (1 - w) * pooled)
    return out.astype(float).rename("dispersion").rename_axis("drug_code")


def apply(pred: pd.DataFrame, dispersion: pd.Series) -> pd.DataFrame:
    """`pred` with `var` = mean x dispersion.

    `dispersion` is indexed by drug_code (one origin) or by (origin,
    drug_code). Every row needs a value.
    """
    out = pred.copy()
    if isinstance(dispersion.index, pd.MultiIndex):
        key = pd.MultiIndex.from_arrays([pd.to_datetime(out["origin"]), out["drug_code"]])
        d = dispersion.copy()
        d.index = pd.MultiIndex.from_arrays([pd.to_datetime(d.index.get_level_values(0)),
                                             d.index.get_level_values(1)])
        d = d.reindex(key).to_numpy(float)
    else:
        d = dispersion.reindex(out["drug_code"]).to_numpy(float)
    if np.isnan(d).any():
        raise ValueError(f"no dispersion for {int(np.isnan(d).sum())} prediction rows")
    out["var"] = out["mean"].astype(float).to_numpy() * np.maximum(d, 1.0)
    return out


def with_u1(pred: pd.DataFrame, fallback: Callable[[date], pd.Series] | None = None
            ) -> pd.DataFrame:
    """Fill `var` with U1 at every origin of a backtest, each from earlier origins only.

    Tiers come from each origin's own rows. `fallback(origin)` is the baseline
    recipe (e.g. `lambda o: baseline_dispersion(ctx, o)`), needed at origins
    with no earlier predictions.
    """
    run = _one_run(pred)
    parts = {}
    for origin in sorted(pd.to_datetime(run["origin"]).unique()):
        o = pd.Timestamp(origin).date()
        rows = run[pd.to_datetime(run["origin"]) == origin]
        tiers = rows.groupby("drug_code")["tier"].first().astype(object)
        parts[pd.Timestamp(origin)] = u1_dispersion(run, o, tiers, fallback)
    disp = pd.concat(parts, names=["origin", "drug_code"])
    return apply(run, disp)


# ---- judging it ---------------------------------------------------------


def window_totals(pred: pd.DataFrame, window: int) -> pd.DataFrame:
    """Sums of mean, var and actual over consecutive h blocks of length `window`.

    Blocks are h = 1..w, w+1..2w, ... for each drug and origin; incomplete
    blocks are dropped. The actual total is NaN if any week in it is
    censored.
    """
    p = pred.copy()
    if p.duplicated(["origin", "drug_code", "h"]).any():
        raise ValueError("duplicate (origin, drug_code, h) rows: pass one config and seed")
    if window == 1:
        return p.assign(block=p["h"].astype(int) - 1)
    p["block"] = (p["h"].astype(int) - 1) // window
    p["missing"] = p["actual"].isna()
    g = p.groupby(["origin", "drug_code", "block"], sort=False)
    out = g.agg(tier=("tier", "first"), mean=("mean", "sum"), var=("var", "sum"),
                actual=("actual", "sum"), missing=("missing", "any"), n=("h", "size"))
    out = out[out["n"] == window].reset_index()
    out.loc[out["missing"], "actual"] = np.nan
    return out.drop(columns=["missing", "n"])


def pinball(y: np.ndarray, q_hat: np.ndarray, q: float) -> np.ndarray:
    """Pinball (quantile) loss per row."""
    diff = y - q_hat
    return np.where(diff >= 0, q * diff, (q - 1) * diff)


def coverage_pinball(pred: pd.DataFrame, qs: tuple[float, ...] = (0.9, 0.95),
                     windows: tuple[int, ...] = (1, 2, 4)) -> pd.DataFrame:
    """Coverage and pinball loss per tier, quantile level and window length.

    `pred` is one run (one config and seed) with `var` filled. Rows with a
    censored actual (NaN) and tier 'none' are left out. `pinball` is the
    summed loss over the summed actual, so tiers of different volume compare.
    """
    rows = []
    for w in windows:
        t = window_totals(pred, w)
        t = t[t["actual"].notna() & (t["tier"].astype(object) != "none")]
        mean = t["mean"].astype(float).to_numpy()
        var = np.maximum(t["var"].astype(float).to_numpy(), mean)
        y = t["actual"].astype(float).to_numpy()
        for q in qs:
            q_hat = fc.demand_quantile(mean, var, q)
            df = pd.DataFrame({"tier": t["tier"].astype(object).to_numpy(),
                               "covered": y <= q_hat, "loss": pinball(y, q_hat, q), "y": y})
            for tier, g in df.groupby("tier"):
                rows.append({"tier": tier, "window": w, "q": q,
                             "coverage": g["covered"].mean(),
                             "pinball": g["loss"].sum() / g["y"].sum() if g["y"].sum() > 0
                             else np.nan,
                             "n": len(g)})
    return pd.DataFrame(rows, columns=["tier", "window", "q", "coverage", "pinball", "n"])
