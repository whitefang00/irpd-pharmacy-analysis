"""Rolling replay: (s, S) and floor levels re-planned during the window (E0a).

The project's replay (`ml.decide.compare`) sets one policy on the day before a
half-year and holds it. Here the plain policy (`contest.to_policy`, service
0.95, 14-day cycle, floor 2-7 days) is re-planned every `cadence` days from a
forecast source standing on the day before each re-plan day, and the window is
run with `replay.simulate_tv`. `cadence=None` is the static policy.

**Forecast sources** give a `contest.Basis` for any policy day:

- `CurrentSource`: the project's current forecaster, computed on that day from
  data up to it (`decide.baseline_basis`), so any day works;
- `ArmSource`: an arm parquet (IMPL-SPEC format). It uses the latest origin on
  or before the policy day; weeks already past are skipped and weeks past h = 8
  repeat h = 8. Its own `var` is used. Drugs it lacks fall back to the current
  forecaster (counted in `.fallbacks`);
- `Perturbed`: any source with each drug's mean (and variance) times a fixed
  factor, for power checks.

**Scoring.** As `replay.metrics`, plus: lost revenue net of switching (each
drug's ingredient switch rate from `lost.switch_rates` on spells before the
window: stock-out rates for the stock-out part, shelf-out rates for the
shelf-out part), achieved lead-time cycle service (share of orders with no
stock-out day between the order and its arrival, inclusive), fill rate (units
sold / demand on live days) and average stock value. Per drug and 14-day
period, for the two-way bootstrap.

**Matching.** `s` and `S` are scaled by one store-wide k until stock value is
within `tol` (0.05%) of a target: the pharmacist's average stock value, or a
reference run's. The frontier scans k over `KS`.

**Floor cap** (optional): the floor is never topped up past the pharmacist's
largest end-of-day floor stock in the 182 days before the window (at least 1
unit). The plain sim has unlimited shelf space, which overstates what floor
refreshes can do.
"""

from __future__ import annotations

import math
import os
import pickle
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from irpd import config, contest, holdout, lost, replay
from irpd.agent.tools import plan as plan_tool
from irpd.evalx import bootstrap
from irpd.ml import decide

KS = (0.7, 0.85, 1.0, 1.2, 1.45, 1.75, 2.1)
TOL = 0.0005
PERIOD = 14
CAP_DAYS = 182
G_FIRST, G_LAST = date(2024, 1, 7), date(2025, 12, 14)
WINDOWS = {"H2-2024": (date(2024, 7, 1), date(2024, 12, 31)),
           "H1-2025": (date(2025, 1, 1), date(2025, 6, 30)),
           "H2-2025": (date(2025, 7, 1), date(2025, 12, 31))}


def cache_dir() -> Path | None:
    p = os.environ.get("IRPD_EVALX_CACHE")
    return Path(p) if p else None


def _cached(key: str, make: Callable[[], object]) -> object:
    d = cache_dir()
    if d is None:
        return make()
    f = d / f"{key}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    obj = make()
    d.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps(obj))
    return obj


def grid(first: date = G_FIRST, last: date = G_LAST) -> list[date]:
    """The shared origin grid G: Sundays 7 Jan 2024 + 14k days, up to `last`."""
    out, d = [], G_FIRST
    while d <= last:
        if d >= first:
            out.append(d)
        d += timedelta(days=14)
    return out


def first_grid_monday(day: date) -> date:
    """The first Monday after a G origin, on or after `day`."""
    return next(o + timedelta(days=1) for o in grid() if o + timedelta(days=1) >= day)


def view(day: date):
    """A connection as of `day`: the agent file up to the cutoff, the full warehouse after."""
    return holdout.agent_view(day) if day <= config.CUTOFF else \
        holdout.agent_view(day, config.WAREHOUSE)


_FACTS: dict[date, pd.DataFrame] = {}


def facts(day: date) -> pd.DataFrame:
    if day not in _FACTS:
        def make():
            with view(day) as con:
                return contest.facts(con, day)
        _FACTS[day] = _cached(f"facts_{day}", make)
    return _FACTS[day]


# ---- forecast sources ---------------------------------------------------------


class CurrentSource:
    """The current forecaster (SES + Eid, seasonal naive for rare) on any day."""

    name = "current"

    def __init__(self) -> None:
        self._b: dict[date, contest.Basis] = {}

    def basis(self, day: date) -> contest.Basis:
        if day not in self._b:
            def make():
                with view(day) as con:
                    return decide.baseline_basis(con, day)
            self._b[day] = _cached(f"basis_current_{day}", make)
        return self._b[day]


class ArmSource:
    """An arm parquet: origin, drug_code, h, week_start, mean, var, tier."""

    def __init__(self, preds: pd.DataFrame | str | Path, name: str | None = None,
                 max_stale_days: int = 13, fallback: CurrentSource | None = None) -> None:
        if not isinstance(preds, pd.DataFrame):
            name = name or Path(preds).stem
            preds = pd.read_parquet(preds)
        self.name = name or "arm"
        p = preds.copy()
        p["origin"] = pd.to_datetime(p["origin"]).dt.date
        p["drug_code"] = p["drug_code"].astype(str)
        p = p[p["h"].between(1, 8)]
        self._by = {o: (g.pivot(index="drug_code", columns="h", values="mean"),
                        g.pivot(index="drug_code", columns="h", values="var"))
                    for o, g in p.groupby("origin")}
        self.origins = sorted(self._by)
        self.max_stale = max_stale_days
        self.fallback = fallback or CurrentSource()
        self.fallbacks: dict[date, int] = {}
        self._b: dict[date, contest.Basis] = {}

    def basis(self, day: date) -> contest.Basis:
        if day in self._b:
            return self._b[day]
        past = [o for o in self.origins if o <= day]
        if not past or (day - past[-1]).days > self.max_stale:
            raise ValueError(f"{self.name}: no origin within {self.max_stale} days of {day}")
        o = past[-1]
        f = facts(day)
        codes = list(f.index)
        mean, var = (x.reindex(codes) for x in self._by[o])
        miss = mean.isna().any(axis=1) | var.isna().any(axis=1)
        n_days = decide.horizon_days(f)
        weeks = math.ceil(((day - o).days + n_days) / 7) + 1
        m = np.nan_to_num(mean.to_numpy(float))
        v = np.nan_to_num(var.to_numpy(float))
        m = np.concatenate([m, np.repeat(m[:, -1:], max(weeks - 8, 0), axis=1)], 1)[:, :weeks]
        v = np.concatenate([v, np.repeat(v[:, -1:], max(weeks - 8, 0), axis=1)], 1)[:, :weeks]
        starts = [o + timedelta(days=1 + 7 * k) for k in range(weeks)]
        wk = pd.DataFrame({"drug_code": np.repeat(codes, weeks),
                           "week_start": np.tile(np.array(starts, dtype=object), len(codes)),
                           "mean": m.ravel(), "var": v.ravel()})
        dm, dv = plan_tool._daily(wk, codes, day, n_days)
        if miss.any():
            fb = self.fallback.basis(day)
            pos = {c: i for i, c in enumerate(fb.facts.index)}
            for i in np.flatnonzero(miss.to_numpy()):
                j = pos[codes[i]]
                dm[i], dv[i] = fb.mean[j, :n_days], fb.var[j, :n_days]
        self.fallbacks[day] = int(miss.sum())
        self._b[day] = contest.Basis(day, f, dm, dv)
        return self._b[day]


class Perturbed:
    """A source with each drug's mean and variance times a fixed factor."""

    def __init__(self, source, factor: pd.Series | float, name: str) -> None:
        self.source, self.factor, self.name = source, factor, name

    def basis(self, day: date) -> contest.Basis:
        b = self.source.basis(day)
        e = (np.full(len(b.facts), float(self.factor)) if np.isscalar(self.factor)
             else self.factor.reindex(b.facts.index).fillna(1.0).to_numpy(float))
        nb = copy(b)
        nb.mean, nb.var = b.mean * e[:, None], b.var * e[:, None]
        return nb


def noise_factors(codes: list[str], sigma: float, seed: int) -> pd.Series:
    """Persistent per-drug lognormal(-sigma^2/2, sigma) factors (mean 1)."""
    rng = np.random.default_rng(seed)
    return pd.Series(np.exp(rng.normal(-sigma * sigma / 2, sigma, len(codes))),
                     index=pd.Index(codes, name="drug_code"))


# ---- windows -----------------------------------------------------------------


@dataclass
class Window:
    """A replay window and the fixed facts its scoring needs."""

    label: str
    first: date
    last: date
    h: replay.Holdout
    target: float               # the pharmacist's average stock value
    switch: np.ndarray          # per drug: share of stock-out loss that switches
    switch_shelf: np.ndarray    # per drug: share of shelf-out loss that switches
    floor_cap: np.ndarray       # per drug: the pharmacist's recent max floor stock


def _load_holdout(first: date, last: date) -> replay.Holdout:
    if last <= config.CUTOFF:
        return contest.load_window(first, last)
    return contest.load_window(first, last, agent=config.WAREHOUSE, after_cutoff=True)


def load_window(label: str = "H2-2024", start: date | None = None,
                end: date | None = None) -> Window:
    """A named window (`WINDOWS`), optionally starting later (`start`)."""
    first, last = WINDOWS.get(label, (start, end))
    first, last = start or first, end or last
    h = _cached(f"holdout_{first}_{last}", lambda: _load_holdout(first, last))
    day = first - timedelta(days=1)
    sw, sw_shelf = _cached(f"switch_{day}", lambda: switch_rates(day, h.drugs))

    def cap():
        with view(day) as con:
            return con.execute(f"""
                SELECT drug_code, max(greatest(closing, 0)) AS cap FROM daily_stock
                WHERE location = 'floor' AND date > DATE '{day - timedelta(days=CAP_DAYS)}'
                  AND date <= DATE '{day}' GROUP BY 1""").df()
    c = _cached(f"floorcap_{day}", cap).set_index("drug_code")["cap"]
    floor_cap = np.maximum(c.reindex(h.drugs).fillna(0).to_numpy(float), 1.0)
    target = float(replay.actual(h)["avg_stock_value"].sum())
    return Window(label, first, last, h, target, sw, sw_shelf, floor_cap)


def switch_rates(day: date, drugs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per drug, the stock-out and shelf-out switch rates from spells before `day`.

    As `contest.facts` and `lost.apply_switching`: the ingredient's shrunk rate
    (pooled rate where the ingredient has none), 0 for drugs without an
    alternative in their ingredient group.
    """
    with view(day) as con:
        sp = lost.spells(con, day)
        g = lost.groups(con)
    sp = sp[sp.start <= day]
    sizes = g.group.value_counts()
    idx = pd.Index(drugs)
    has_alt = idx.map(g.group).map(sizes).fillna(0).to_numpy() > 1
    out = []
    for kind in ("stockout", "shelfout"):
        r = lost.switch_rates(sp, kind=kind)
        rate = idx.map(g.ingredients).map(r["rate"]).to_series(index=idx)
        out.append(np.where(has_alt, rate.fillna(r.attrs["overall"]).to_numpy(float), 0.0))
    return out[0], out[1]


# ---- planning, running, scoring ---------------------------------------------------


def replan_days(w: Window, cadence: int | None) -> list[int]:
    T = len(w.h.days)
    return [0] if cadence is None else list(range(0, T, cadence))


def make_plan(source, w: Window, cadence: int | None = 14, rho: dict[str, float] | None = None,
              knobs: dict | None = None, offset: int = 0) -> dict[str, np.ndarray]:
    """Policy arrays for the window: re-planned every `cadence` days (None: once).

    `offset` shifts the re-plan days after the first (day 0 always plans), for
    A/A checks on the re-plan phase.
    """
    idx = replan_days(w, cadence)
    if offset:
        idx = [0] + [t + offset for t in idx[1:] if t + offset < len(w.h.days)]
    pols = [(t, contest.to_policy(knobs or {}, source.basis(w.h.days[t] - timedelta(days=1)),
                                  rho)) for t in idx]
    return replay.schedule(pols, w.h)


@dataclass
class Outcome:
    k: float
    summary: dict[str, float]
    periods: pd.DataFrame       # drug_code x period: lost revenue parts, stock value


def stock_value(run: replay.Run, w: Window) -> float:
    return float((np.maximum(run.floor + run.store, 0) * w.h.price[:, None]).mean(1).sum())


def cycle_service(run: replay.Run, stockout: np.ndarray, lead: np.ndarray) -> tuple[float, int]:
    """Share of orders with no stock-out day from the day after the order to arrival."""
    T = stockout.shape[1]
    cs = np.concatenate([np.zeros((len(lead), 1)), np.cumsum(stockout, 1)], 1)
    i, t = np.nonzero(run.ordered > 0)
    end = t + lead[i]
    ok = end < T
    i, t, end = i[ok], t[ok], end[ok]
    hit = (cs[i, end + 1] - cs[i, t + 1]) > 0
    return (float(1 - hit.mean()) if len(hit) else np.nan), int(len(hit))


def outcome(run: replay.Run, w: Window, k: float, lead: np.ndarray,
            period: int = PERIOD) -> Outcome:
    h = w.h
    x = replay.losses(run.floor, run.store, run.sold, h)
    price = h.price[:, None]
    parts = {"gross_out": x.lost_out * price, "gross_shelf": x.lost_shelf * price}
    parts["net_out"] = parts["gross_out"] * (1 - w.switch)[:, None]
    parts["net_shelf"] = parts["gross_shelf"] * (1 - w.switch_shelf)[:, None]
    T = len(h.days)
    total = np.maximum(run.floor + run.store, 0) * price
    parts["stock_value"] = total / T
    pid = np.arange(T) // period
    rows = []
    for p in np.unique(pid):
        sel = pid == p
        rows.append(pd.DataFrame({"drug_code": h.drugs, "period": int(p),
                                  **{c: v[:, sel].sum(1) for c, v in parts.items()},
                                  "stockout_days": x.stockout[:, sel].sum(1),
                                  "shelfout_days": x.shelfout[:, sel].sum(1)}))
    per = pd.concat(rows, ignore_index=True)
    per["lost_gross"] = per["gross_out"] + per["gross_shelf"]
    per["lost_net"] = per["net_out"] + per["net_shelf"]
    live_dem = (h.demand * h.live).sum()
    cs, n_cycles = cycle_service(run, x.stockout, lead)
    s = {c: float(v.sum()) for c, v in parts.items()}
    s.update({"k": k, "lost_gross": s["gross_out"] + s["gross_shelf"],
              "lost_net": s["net_out"] + s["net_shelf"],
              "stockout_days": float(x.stockout.sum()), "shelfout_days": float(x.shelfout.sum()),
              "fill_rate": float((run.sold * h.live).sum() / live_dem) if live_dem else np.nan,
              "cycle_service": cs, "cycles": n_cycles})
    return Outcome(k, s, per)


def run_at(p: dict[str, np.ndarray], w: Window, k: float, floor_cap: bool = False) -> Outcome:
    run = replay.simulate_tv(p, w.h, k, w.floor_cap if floor_cap else None)
    return outcome(run, w, k, p["lead"])


def match(p: dict[str, np.ndarray], w: Window, target: float | None = None, tol: float = TOL,
          floor_cap: bool = False, hi0: float = 1.0) -> Outcome:
    """Bisect k until stock value is within `tol` of `target` (default: the pharmacist).

    `hi0` is the bisection's first upper bracket (the A/A check varies it).
    """
    target = w.target if target is None else target
    cap = w.floor_cap if floor_cap else None

    def val(k):
        return stock_value(replay.simulate_tv(p, w.h, k, cap), w)

    lo, hi = 0.0, hi0
    while val(hi) < target:
        lo, hi = hi, hi * 2
        if hi > 64:
            raise ValueError("can't reach the target stock value")
    k = hi
    for _ in range(60):
        k = (lo + hi) / 2
        v = val(k)
        if abs(v / target - 1) < tol:
            break
        lo, hi = (k, hi) if v < target else (lo, k)
    return run_at(p, w, k, floor_cap)


@dataclass
class Result:
    """`rolling_frontier` output."""

    name: str
    window: str
    cadence: int | None
    frontier: pd.DataFrame        # one row per k
    matched: Outcome | None
    plan: dict[str, np.ndarray] = field(repr=False, default_factory=dict)


def rolling_frontier(source, window: str | Window = "H2-2024", cadence: int | None = 14,
                     ks: tuple[float, ...] = KS, match_to: float | str | None = "pharmacist",
                     floor_cap: bool = False, rho: dict[str, float] | None = None,
                     start: date | None = None, tol: float = TOL, offset: int = 0) -> Result:
    """Run a forecast source through the rolling replay.

    source: `CurrentSource()`, `ArmSource("preds/<arm>.parquet")`, or `Perturbed`.
    window: a `WINDOWS` label or a loaded `Window`; `start` moves its first day
        (use `first_grid_monday` for arm parquets, whose origins are on G only).
    cadence: re-plan every this many days; None = the static policy.
    ks: the scales of s and S for the frontier (k = 1 is the app's policy).
    match_to: "pharmacist" (its average stock value), a number (e.g. a reference
        run's matched stock value), or None (frontier only).
    floor_cap: cap floor top-ups at the pharmacist's recent max floor stock.
    rho: tier -> rho variance factor in the policy (`contest.to_policy`).

    Returns the frontier (stock value, gross/net lost revenue split into
    stock-out and shelf-out, stock-out days, fill rate, cycle service) and the
    matched run with its per-drug, per-14-day-period results.
    """
    w = window if isinstance(window, Window) else load_window(window, start)
    p = make_plan(source, w, cadence, rho, offset=offset)
    rows = [run_at(p, w, k, floor_cap).summary for k in ks]
    m = None
    if match_to is not None:
        m = match(p, w, None if match_to == "pharmacist" else float(match_to), tol, floor_cap)
    return Result(getattr(source, "name", "source"), w.label, cadence, pd.DataFrame(rows), m, p)


def compare_replay(a: Outcome, b: Outcome, metric: str = "net_out", n: int = bootstrap.N_DRAWS,
                   seed: int = 0, drugs_only: bool = False) -> dict[str, float]:
    """Ratio a / b of a lost-revenue part at matched stock, two-way bootstrap 90%.

    Origins here are the 14-day periods (blocks of 2 = 28 days) and drugs.
    """
    x = a.periods[["drug_code", "period", metric]].merge(
        b.periods[["drug_code", "period", metric]], on=["drug_code", "period"],
        suffixes=("_a", "_b"))
    iv = bootstrap.two_way(x, [f"{metric}_a", f"{metric}_b"],
                           lambda s: s[f"{metric}_a"] / s[f"{metric}_b"], origin="period",
                           n=n, seed=seed, drugs_only=drugs_only)
    d = bootstrap.two_way(x, [f"{metric}_a", f"{metric}_b"],
                          lambda s: s[f"{metric}_a"] - s[f"{metric}_b"], origin="period",
                          n=n, seed=seed, drugs_only=drugs_only)
    return {"metric": metric, "a": a.summary[metric], "b": b.summary[metric],
            "ratio": iv.point, "ratio_lo": iv.lo, "ratio_hi": iv.hi,
            "diff": d.point, "diff_lo": d.lo, "diff_hi": d.hi,
            "stock_a": a.summary["stock_value"], "stock_b": b.summary["stock_value"]}
