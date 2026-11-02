"""Weekly demand forecasts per drug, and the rolling backtest that picked the methods.

`forecast(con, drug_codes, as_of, horizon_weeks)` predicts each drug's units
(stock base units) for the next weeks, from data up to `as_of` only. Every
query filters on `as_of` itself, so it is safe on a full warehouse connection
as well as on one from `warehouse.as_of`.

How it works:

- History is `weekly_demand` for weeks that ended by `as_of`. Weeks with any
  stock-out day (total stock <= 0, from `config.STOCK_START`) are censored:
  sales there are a floor, not demand, so they are left out of fitting and
  scoring. Weeks before a drug's first sale are missing too.
- Eid al-Fitr: weeks near Eid get a store-wide uplift factor, learned from the
  Eids seen before `as_of`: each day from -7 to +28 days around Eid gets the
  store's sale lines that day over its normal level for that weekday; a week's
  factor is the mean of its days. History is divided by it before fitting and
  forecasts multiplied by it (methods marked '+eid').
- The method depends on the drug's tier, recomputed as of `as_of` with
  `sales.tiers_sql`. `METHOD_BY_TIER` was chosen on the 2024 backtest origins
  and checked on H1 2025 (`python -m irpd.forecast` prints the table).
- Uncertainty: each week is negative binomial with the forecast mean and
  variance = mean x dispersion, where dispersion is the drug's own variance /
  mean over its last year of observed weeks (at least 1, i.e. Poisson). On
  the backtest the p90 already covers 90-92% per tier, so it isn't widened.
  `var` is returned so a planner can sum weeks over a lead time and call
  `demand_quantile`.

The forecast weeks are the Monday-start weeks after the last week that ended
by `as_of`; if `as_of` is mid-week, the first one is the current week.
"""

import time
import warnings
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd

from irpd import config
from irpd.tables.sales import tiers_sql

# Eid al-Fitr, as set by Kemenag (see eda/demand.py). Add later years here.
# 2026 and 2027 are the expected dates (1 Syawal 1447 and 1448 H); Kemenag fixes
# each at its isbat a day or two before, so either may move by a day.
EID = [date(2023, 4, 22), date(2024, 4, 10), date(2025, 3, 31),
       date(2026, 3, 20), date(2027, 3, 10)]
# The list the ML system was frozen with (lab/ml-frozen.json). Only the ML's
# legacy `c_eid_days` reads it, so the frozen control stays reproducible; the
# later dates fall outside the data, so nothing else changes before 2026.
EID_LEGACY = EID[:3]
EID_DAYS = range(-7, 29)  # days around Eid that get their own uplift factor (FINDINGS.md)
EID_BASE = range(-63, -14)  # the normal days each Eid is compared with

SMOOTHING = 0.15  # alpha for SES, Croston-SBA and TSB (picked on the 2024 origins)
DISPERSION_WEEKS = 52

# Store-wide sale lines overstate the Eid uplift on drug units (forecasts ran
# about 10% high in Eid weeks), so only this share of it is applied. Set on the
# 2024 origins, where it brought Eid-week bias to about 1.0.
EID_SHRINK = 0.6
# The Eid profile: 'lines' (the default above) or 'units', the store's daily
# units over the same weekday's normal, unshrunk (experiment E3). Lines put
# more of the uplift after Eid than units do, which put the 2025 peak a week
# late: units peak in days -7..-1, lines in days +1..+6.
EID_PROFILES = ("lines", "units")
EID_UNIT_SHRINK = 1.0

# Chosen on the 2024 backtest origins and checked on H1 2025. Among methods
# with little bias, the lowest WAPE; Croston-SBA is a hair lower on fast drugs
# but runs 4-6% low, which a reorder point can't afford. On rare drugs the
# seasonal-naive baseline beats every model, so it is used there.
METHOD_BY_TIER = {"fast": "ses+eid", "regular": "ses+eid", "slow": "ses+eid",
                  "rare": "snaive"}


def _d(day: date) -> str:
    return f"DATE '{day.isoformat()}'"


def _monday(day: date) -> date:
    return day - timedelta(days=day.weekday())


def load_history(con: duckdb.DuckDBPyConnection, until: date,
                 drug_codes: list[str] | None = None) -> dict:
    """Weekly units per drug for weeks ended by `until`, with censored weeks marked.

    Returns {'drugs', 'weeks' (week starts), 'y' (units, NaN before the first
    sale), 'out' (week had a stock-out day), 'lines' (store-wide sale lines
    per day), 'store_units' (store-wide units per day)}.
    """
    only = ""
    if drug_codes is not None:
        con.execute("CREATE OR REPLACE TEMP TABLE _fc_drugs (drug_code VARCHAR)")
        con.executemany("INSERT INTO _fc_drugs VALUES (?)", [[c] for c in drug_codes])
        only = "AND drug_code IN (SELECT drug_code FROM _fc_drugs)"
    wk = con.execute(f"""
        SELECT drug_code, week_start, units FROM weekly_demand
        WHERE week_end <= {_d(until)} {only}""").df()
    out = con.execute(f"""
        SELECT drug_code, date_trunc('week', date)::DATE AS week_start
        FROM daily_stock
        WHERE location = 'total' AND closing <= 0
          AND date BETWEEN {_d(config.STOCK_START)} AND {_d(until)} {only}
        GROUP BY ALL""").df()
    weeks = con.execute(f"""
        SELECT DISTINCT week_start FROM weekly_demand
        WHERE week_end <= {_d(until)} ORDER BY 1""").df()
    weeks = pd.to_datetime(weeks.week_start).dt.date.to_numpy()
    lines = con.execute(f"""
        SELECT date, sum(lines) AS lines, sum(units) AS units FROM daily_demand
        WHERE date <= {_d(weeks[-1] + timedelta(days=6) if len(weeks) else until)}
        GROUP BY 1 ORDER BY 1""").df()
    drugs = np.array(sorted(set(wk.drug_code) | set(drug_codes or [])), dtype=object)
    y = (wk.assign(week_start=pd.to_datetime(wk.week_start).dt.date)
         .pivot(index="drug_code", columns="week_start", values="units")
         .reindex(index=drugs, columns=weeks).to_numpy(float))
    o = (out.assign(week_start=pd.to_datetime(out.week_start).dt.date, v=True)
         .pivot(index="drug_code", columns="week_start", values="v")
         .reindex(index=drugs, columns=weeks).notna().to_numpy())
    day = pd.to_datetime(lines.date).dt.date
    return {"drugs": drugs, "weeks": weeks, "y": y, "out": o,
            "lines": pd.Series(lines.lines.to_numpy(float), index=day),
            "store_units": pd.Series(lines.units.to_numpy(float), index=day)}


def _slice(h: dict, n_weeks: int) -> dict:
    """The history cut to its first `n_weeks` weeks (for backtest origins)."""
    end = h["weeks"][n_weeks - 1] + timedelta(days=6)
    out = {"drugs": h["drugs"], "weeks": h["weeks"][:n_weeks], "y": h["y"][:, :n_weeks],
           "out": h["out"][:, :n_weeks], "lines": h["lines"][h["lines"].index <= end]}
    if "store_units" in h:
        out["store_units"] = h["store_units"][h["store_units"].index <= end]
    return out


def eid_factors(lines: pd.Series, shrink: float = EID_SHRINK) -> dict[int, float]:
    """Uplift per day relative to Eid, from the Eids whose base days are in `lines`.

    `lines` is store-wide sale lines (or units) per day. Each day in the window
    is divided by the mean of the same weekday over EID_BASE days before Eid,
    then averaged over the Eids seen. Days not yet seen for any Eid get no
    factor. Only `shrink` of the uplift is kept.
    """
    ratios: dict[int, list[float]] = {}
    for eid in EID:
        base = lines[[eid + timedelta(days=k) for k in EID_BASE
                      if eid + timedelta(days=k) in lines.index]]
        if len(base) < len(EID_BASE) - 7:  # allow for a few closed days
            continue
        by_weekday = base.groupby([d.weekday() for d in base.index]).mean()
        for k in EID_DAYS:
            d = eid + timedelta(days=k)
            if d in lines.index or d <= lines.index.max():
                ratios.setdefault(k, []).append(lines.get(d, 0.0) / by_weekday[d.weekday()])
    return {k: 1 + shrink * (float(np.mean(r)) - 1) for k, r in ratios.items()}


def eid_profile(hist: dict, profile: str = "lines") -> dict[int, float]:
    """The day factors `predict` uses: from store lines (shrunk) or store units (unshrunk).

    Either way they come from the history as cut, so only Eids seen by its
    last day count.
    """
    if profile == "lines":
        return eid_factors(hist["lines"])
    if profile == "units":
        return eid_factors(hist["store_units"], EID_UNIT_SHRINK)
    raise ValueError(f"profile must be one of {EID_PROFILES}")


def eid_index(weeks, factors: dict[int, float]) -> np.ndarray:
    """The Eid factor for each Monday-start week: the mean of its days' factors."""
    out = np.ones(len(weeks))
    for i, w in enumerate(weeks):
        days = [w + timedelta(days=j) for j in range(7)]
        f = [next((factors[(d - e).days] for e in EID if (d - e).days in factors), 1.0)
             for d in days]
        out[i] = np.mean(f)
    return out


def _ses(y: np.ndarray, a: float = SMOOTHING) -> np.ndarray:
    level = np.full(len(y), np.nan)
    for t in range(y.shape[1]):
        v = y[:, t]
        seen = ~np.isnan(v)
        first = seen & np.isnan(level)
        level[first] = v[first]
        upd = seen & ~first
        level[upd] = a * v[upd] + (1 - a) * level[upd]
    return level


def _croston_sba(y: np.ndarray, a: float = SMOOTHING) -> np.ndarray:
    """Croston with the Syntetos-Boylan bias correction. Censored weeks don't count."""
    size = np.full(len(y), np.nan)
    gap = np.full(len(y), np.nan)
    since = np.zeros(len(y))
    for t in range(y.shape[1]):
        v = y[:, t]
        seen = ~np.isnan(v)
        since[seen] += 1
        hit = seen & (v > 0)
        first = hit & np.isnan(size)
        size[first], gap[first] = v[first], since[first]
        upd = hit & ~first
        size[upd] = a * v[upd] + (1 - a) * size[upd]
        gap[upd] = a * since[upd] + (1 - a) * gap[upd]
        since[hit] = 0
    out = (1 - a / 2) * size / gap
    return np.where(np.isnan(out) & ~np.all(np.isnan(y), axis=1), 0.0, out)


def _tsb(y: np.ndarray, a: float = SMOOTHING, b: float = SMOOTHING) -> np.ndarray:
    """Teunter-Syntetos-Babai: smooths the chance of a sale and its size separately."""
    prob = np.full(len(y), np.nan)
    size = np.full(len(y), np.nan)
    for t in range(y.shape[1]):
        v = y[:, t]
        seen = ~np.isnan(v)
        hit = (seen & (v > 0)).astype(float)
        prob = np.where(seen, np.where(np.isnan(prob), hit, b * hit + (1 - b) * prob), prob)
        pos = seen & (v > 0)
        size = np.where(pos, np.where(np.isnan(size), v, a * v + (1 - a) * size), size)
    return np.where(np.isnan(size), np.where(np.isnan(prob), np.nan, 0.0), prob * size)


def _mean_last(y: np.ndarray, n: int = 8) -> np.ndarray:
    """Mean of the last `n` observed weeks."""
    seen = ~np.isnan(y)
    rank = np.cumsum(seen[:, ::-1], axis=1)[:, ::-1]  # 1 = the latest observed week
    keep = seen & (rank <= n)
    with np.errstate(invalid="ignore"):
        return np.where(keep, y, 0).sum(1) / keep.sum(1)


# TSB for fading drugs (experiment E5): the chance of a sale decays with every
# observed zero week. Separate smoothing for that chance and for the sale size.
TSB_ALPHA_P = 0.15
TSB_ALPHA_D = 0.10


def tsb(hist: dict, horizon: int, alpha_p: float = TSB_ALPHA_P,
        alpha_d: float = TSB_ALPHA_D) -> dict:
    """TSB weekly mean and variance per drug (flat over `horizon`), on uncensored weeks.

    The variance is the usual recipe: mean x the drug's dispersion over its
    last year of observed weeks. `rate` is the one weekly level per drug.
    """
    y = np.where(hist["out"], np.nan, hist["y"])
    rate = np.nan_to_num(_tsb(y, a=alpha_d, b=alpha_p))
    mean = np.repeat(rate[:, None], horizon, axis=1)
    disp = dispersion(y)
    return {"mean": mean, "var": mean * disp[:, None], "rate": rate, "disp": disp}


# Each maps weekly units (NaN = unobserved) to one level per drug.
LEVEL_METHODS = {"ses": _ses, "croston_sba": _croston_sba, "tsb": _tsb,
                 "mean8": _mean_last}


def _seasonal_naive(y: np.ndarray, horizon: int) -> np.ndarray:
    """Same week a year earlier (52 weeks back); falls back to the 8-week mean."""
    T = y.shape[1]
    fallback = _mean_last(y)
    out = np.empty((len(y), horizon))
    for h in range(horizon):
        t = T + h - 52
        v = y[:, t] if 0 <= t < T else np.full(len(y), np.nan)
        out[:, h] = np.where(np.isnan(v), fallback, v)
    return out


def dispersion(y: np.ndarray, n: int = DISPERSION_WEEKS) -> np.ndarray:
    """Variance / mean of each drug's last `n` observed weeks, at least 1."""
    seen = ~np.isnan(y)
    rank = np.cumsum(seen[:, ::-1], axis=1)[:, ::-1]
    v = np.where(seen & (rank <= n), y, np.nan)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)  # drugs with < 2 observed weeks
        d = np.nanvar(v, axis=1, ddof=1) / np.nanmean(v, axis=1)
    return np.where(np.isfinite(d), np.maximum(d, 1.0), 1.0)


def demand_quantile(mean, var, q: float) -> np.ndarray:
    """The q-quantile of a negative binomial with this mean and variance (var >= mean).

    Exact for small means; a gamma (Wilson-Hilferty) approximation for large
    ones. Also works on demand summed over weeks (sum the means and the vars).
    """
    mean = np.asarray(mean, float)
    d = np.maximum(np.asarray(var, float) / np.where(mean > 0, mean, 1), 1 + 1e-6)
    out = np.zeros(mean.shape)
    sd = np.sqrt(mean * d)
    small = (mean > 0) & (mean + 8 * sd <= 2000)
    idx = np.flatnonzero(small)
    grid = np.ceil(mean + 8 * sd).astype(int) + 2  # pmf terms needed per row
    idx = idx[np.argsort(grid[idx])]
    while len(idx):  # chunks of rows with similar grid sizes, to bound memory
        cells = grid[idx] * np.arange(1, len(idx) + 1)
        n = max(1, int(np.searchsorted(cells, 2_000_000, side="right")))
        rows, idx = idx[:n], idx[n:]
        top = grid[rows].max()
        m, dd = mean[rows][:, None], d[rows][:, None]
        k = np.arange(top - 1)
        r = m / (dd - 1)
        step = np.log(k + r) - np.log(k + 1) + np.log1p(-1 / dd)
        logp = -r * np.log1p(dd - 1) + np.concatenate([np.zeros((n, 1)), np.cumsum(step, 1)], 1)
        out[rows] = (np.cumsum(np.exp(logp), axis=1) < q).sum(1)
    big = (mean > 0) & ~small
    if big.any():
        shape, z = mean[big] / d[big], _z(q)
        c = 1 / (9 * shape)
        out[big] = np.ceil(mean[big] * np.maximum(1 - c + z * np.sqrt(c), 0) ** 3)
    return out


def _z(q: float) -> float:
    """Standard normal quantile (Acklam's rational approximation, |error| < 1e-8)."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    s = np.sqrt(-2 * np.log(min(q, 1 - q)))
    if abs(q - 0.5) <= 0.475:
        r = (q - 0.5) ** 2
        num = ((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]
        den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
        return (q - 0.5) * num / den
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    e = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    num = ((((c[0] * s + c[1]) * s + c[2]) * s + c[3]) * s + c[4]) * s + c[5]
    x = num / ((((e[0] * s + e[1]) * s + e[2]) * s + e[3]) * s + 1)
    return -x if q < 0.5 else x


def predict(hist: dict, method: np.ndarray, horizon: int, eid: str = "lines") -> dict:
    """Mean and variance per drug and future week, one method per drug.

    `method` names a LEVEL_METHODS entry, 'snaive' (same week a year back)
    or 'zero'. A '+eid' suffix fits on Eid-adjusted history and scales the
    forecast by the target week's Eid factor. `eid` picks the day profile
    behind those factors (`eid_profile`); 'lines' is the shipped forecaster.
    """
    y = np.where(hist["out"], np.nan, hist["y"])
    weeks = hist["weeks"]
    last = weeks[-1] if len(weeks) else _monday(config.FIRST_DAY) - timedelta(weeks=1)
    future = [last + timedelta(weeks=h + 1) for h in range(horizon)]
    factors = eid_profile(hist, eid)
    past_f, future_f = eid_index(weeks, factors), eid_index(future, factors)
    adj = y / past_f

    mean = np.zeros((len(y), horizon))
    for m in np.unique(method):
        rows = method == m
        name, eid = m.removesuffix("+eid"), m.endswith("+eid")
        if name == "snaive":
            mean[rows] = _seasonal_naive(y[rows], horizon)
        elif name != "zero":
            level = LEVEL_METHODS[name]((adj if eid else y)[rows])
            mean[rows] = level[:, None] * (future_f if eid else np.ones(horizon))[None, :]
    mean = np.nan_to_num(np.maximum(mean, 0))
    disp = dispersion(adj)
    return {"future": future, "mean": mean, "var": mean * disp[:, None], "disp": disp}


def tiers(con: duckdb.DuckDBPyConnection, as_of: date) -> dict[str, str]:
    """Demand tier per drug, from daily demand up to `as_of` only."""
    return dict(con.execute(f"SELECT drug_code, tier FROM ({tiers_sql(as_of)})").fetchall())


FORECASTERS = ("current", "ml")


def _ml_means(as_of: date, drugs: np.ndarray, future: list[date], tier: np.ndarray,
              mean: np.ndarray, method: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Swap in the saved ML forecasts (`irpd.ml.serve`) where there are any.

    Weeks past the saved ones repeat the last saved week. Drugs with no saved
    forecast (no sale in the 52 weeks before `as_of`) keep the current
    method's forecast and name.
    """
    from irpd.ml import serve

    saved = serve.load(as_of)
    if saved is None:
        raise FileNotFoundError(f"no ML forecasts saved for {as_of}; run `irpd ml serve`")
    weeks = [w for w in future if w in saved.columns]
    if not weeks or weeks[0] != future[0]:
        raise ValueError(f"saved ML forecasts start {min(saved.columns)}, "
                         f"not {future[0]}; rerun `irpd ml serve`")
    m = saved.reindex(index=list(drugs), columns=weeks).to_numpy(float)
    m = np.concatenate([m, np.repeat(m[:, -1:], len(future) - len(weeks), axis=1)], axis=1)
    use = ~np.isnan(m).any(axis=1) & (tier != "none")
    mean, method = mean.copy(), method.copy()
    mean[use] = m[use]
    method[use] = "ml"
    return mean, method


def forecast(con: duckdb.DuckDBPyConnection, drug_codes: list[str] | None, as_of: date,
             horizon_weeks: int = 8, forecaster: str = "current") -> pd.DataFrame:
    """Weekly demand forecasts from data up to `as_of`.

    `drug_codes=None` forecasts every drug with a sale by `as_of`. Returns one
    row per drug and week: drug_code, week_start, week_end, mean, var, p50,
    p90, tier, method. Drugs with no sale by `as_of` get zeros, tier 'none'.

    `forecaster="ml"` takes the means from the saved ML forecasts (method
    'ml'); the spread is the same recipe either way: variance = mean x the
    drug's own dispersion.
    """
    if forecaster not in FORECASTERS:
        raise ValueError(f"forecaster must be one of {FORECASTERS}")
    hist = load_history(con, as_of, drug_codes)
    tier_of = tiers(con, as_of)
    tier = np.array([tier_of.get(c, "none") for c in hist["drugs"]], dtype=object)
    method = np.array([METHOD_BY_TIER.get(t, "snaive") for t in tier], dtype=object)
    p = predict(hist, method, horizon_weeks)
    mean = p["mean"]
    if forecaster == "ml":
        mean, method = _ml_means(as_of, hist["drugs"], p["future"], tier, mean, method)

    n, h = mean.shape
    week_start = np.tile(np.array(p["future"], dtype=object), n)
    var = (mean * p["disp"][:, None]).ravel()
    mean = mean.ravel()
    return pd.DataFrame({
        "drug_code": np.repeat(hist["drugs"], h),
        "week_start": week_start,
        "week_end": [w + timedelta(days=6) for w in week_start],
        "mean": mean, "var": var,
        "p50": demand_quantile(mean, var, 0.5),
        "p90": demand_quantile(mean, var, 0.9),
        "tier": np.repeat(tier, h),
        "method": np.repeat(np.where(tier == "none", "none", method), h),
    })


CANDIDATES = ["zero", "mean8", "snaive", "ses", "ses+eid", "croston_sba+eid", "tsb+eid"]


def backtest_origins(first: date = date(2024, 1, 7), horizon: int = 8) -> list[date]:
    """Sundays every 4 weeks, from `first`, whose whole horizon ends by the cutoff."""
    out, d = [], first
    while d + timedelta(weeks=horizon) <= config.CUTOFF:
        out.append(d)
        d += timedelta(weeks=4)
    return out


def backtest(con: duckdb.DuckDBPyConnection, origins: list[date] | None = None,
             horizon: int = 8, methods: list[str] = CANDIDATES) -> pd.DataFrame:
    """Every candidate method at every origin, scored on later weeks up to the cutoff.

    Drugs are scored if they sold in the 13 weeks before the origin. A target
    week with a stock-out day is censored and gets NaN `actual`. `eid` marks
    target weeks that overlap an Eid window.
    """
    origins = origins or backtest_origins(horizon=horizon)
    full = load_history(con, config.CUTOFF)
    in_eid = eid_index(full["weeks"], dict.fromkeys(EID_DAYS, 2.0)) > 1
    parts = []
    for origin in origins:
        n_weeks = sum(w + timedelta(days=6) <= origin for w in full["weeks"])
        hist = _slice(full, n_weeks)
        recent = np.nansum(hist["y"][:, -13:], axis=1) > 0
        hist = {**hist, **{k: hist[k][recent] for k in ("drugs", "y", "out")}}
        tgt = slice(n_weeks, n_weeks + horizon)
        actual = np.where(full["out"][recent, tgt], np.nan, full["y"][recent, tgt])
        tier_of = tiers(con, origin)
        tier = np.array([tier_of.get(c, "none") for c in hist["drugs"]], dtype=object)
        y = np.where(hist["out"], np.nan, hist["y"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            scale = np.nanmean(np.abs(np.diff(y, axis=1)), axis=1)  # MASE: in-sample naive
        for m in methods:
            p = predict(hist, np.full(len(tier), m, dtype=object), horizon)
            n, h = p["mean"].shape
            parts.append(pd.DataFrame({
                "origin": origin, "method": m,
                "drug_code": np.repeat(hist["drugs"], h), "tier": np.repeat(tier, h),
                "h": np.tile(np.arange(1, h + 1), n), "eid": np.tile(in_eid[tgt], n),
                "actual": actual.ravel(), "mean": p["mean"].ravel(),
                "var": p["var"].ravel(), "scale": np.repeat(scale, h)}))
    return pd.concat(parts, ignore_index=True)


def score(bt: pd.DataFrame, by: tuple[str, ...] = ("period", "tier", "method")
          ) -> pd.DataFrame:
    """WAPE, MASE, bias (forecast / actual) and p50 / p90 coverage per group.

    Censored target weeks are left out. Period: 2024 origins (used to choose
    methods) or 2025H1 (a check on choices already made).
    """
    bt = bt.dropna(subset=["actual"])
    bt = bt[bt.tier != "none"].copy()
    bt["p50"] = demand_quantile(bt["mean"], bt["var"], 0.5)
    bt["p90"] = demand_quantile(bt["mean"], bt["var"], 0.9)
    bt["in50"], bt["in90"] = bt.actual <= bt.p50, bt.actual <= bt.p90
    bt["err"] = (bt["mean"] - bt.actual).abs()
    bt["se"] = bt.err / bt.scale.where(bt.scale > 0)
    bt["period"] = np.where(pd.to_datetime(bt.origin).dt.year == 2024, "2024", "2025H1")
    g = bt.groupby(list(by))
    return pd.DataFrame({
        "wape": g.err.sum() / g.actual.sum(),
        "mase": g.se.mean(),
        "bias": g["mean"].sum() / g.actual.sum(),
        "cover50": g.in50.mean(),
        "cover90": g.in90.mean(),
        "n": g.size(),
    }).reset_index()


TIER_ORDER = ["fast", "regular", "slow", "rare"]


def _show(table: pd.DataFrame) -> str:
    order = {t: i for i, t in enumerate(TIER_ORDER)}
    table = table.sort_values([c for c in ("period", "tier", "wape") if c in table],
                              key=lambda s: s.map(order) if s.name == "tier" else s)
    if "tier" in table:
        chosen = table.method == table.tier.map(METHOD_BY_TIER)
        table.insert(table.columns.get_loc("method") + 1, "chosen",
                     np.where(chosen, "*", ""))
    return table.to_string(index=False, float_format=lambda v: f"{v:.3f}")


def main() -> None:
    """Print the backtest tables (`uv run python -m irpd.forecast`)."""
    start = time.time()
    with duckdb.connect(str(config.WAREHOUSE), read_only=True) as con:
        bt = backtest(con)
    print("Rolling-origin backtest, by tier (* = METHOD_BY_TIER)\n")
    print(_show(score(bt)))
    print("\nEid-window target weeks only, fast and regular drugs\n")
    eid = bt[bt.eid & bt.tier.isin(["fast", "regular"])]
    print(_show(score(eid, by=("period", "method"))))
    print(f"\n{len(backtest_origins())} origins every 4 weeks from 2024-01-07, horizons 1-8, "
          f"{time.time() - start:.0f}s.")


if __name__ == "__main__":
    main()
