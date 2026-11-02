"""The decision-level test: the same reorder policy, only the forecaster swapped.

Plan 9.1. The policy is the plain one (`contest.to_policy({}, basis)`:
service level 0.95, 14-day cycle, floor 2 to 7 days). Each forecaster feeds
it through a `contest.Basis`: the same facts, and a daily mean and variance
spread from weekly forecasts exactly as `contest.basis` spreads them
(`plan._daily`). Each policy is replayed at the pharmacist's average stock
value (`contest.matched`) and scored per drug.

**A basis from any forecaster** (`basis`). On a policy day (often not a
Sunday) the forecaster stands on `schema.policy_origin(day)`, the last Sunday
by it, and predicts h = 1..8 for the candidate drugs. Weeks past 8 repeat
week 8 (`schema.extend_horizon`); the plain policy needs at most 4 weeks
(lead up to 14 days plus the 14-day cycle). Variance is mean x dispersion,
with the dispersion passed in: the baseline recipe or U1.

`baseline_basis` is the baseline forecaster through this same path. With
`horizon=None` (every week the basis covers) it reproduces `contest.basis`
exactly; with the default 8 weeks it matches it on the first 8 weeks, and so
gives the same plain policy.

**The two comparisons** are two calls to `compare` with different bases:

- mean model only: every forecaster with the baseline dispersion
  (`baseline_dispersion(con, day, codes)`), so only the mean differs;
- full system: each forecaster with its own uncertainty (U1 for ML).

**Intervals** (`paired_bootstrap_replay`) resample drugs over the per-drug
results of the matched runs. Matching is a store-wide scale fixed per run,
so it isn't redone per sample. This is variation across drugs only.

**The rule** (`decide_winner`), fixed in plan 9.1: lost revenue decides. A
forecaster is better if its lost revenue is lower with the interval
excluding zero and its stock-out days are no more than 5% higher.
Shelf-out days are reported, never decisive.

Nothing here opens the sealed holdout. Windows come from the caller
(`contest.load_windows`, for the lead in wave 4).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from irpd import contest
from irpd import forecast as fc
from irpd.agent.tools import plan
from irpd.ml import schema, split
from irpd.ml.models.base import Forecaster
from irpd.ml.uncertainty import dispersion_from_history

STOCKOUT_SLACK = 1.05  # a winner's stock-out days may be at most 5% higher
LEVEL = 0.9            # bootstrap interval, as in plan 3.3
BOOT_METRICS = ("lost_revenue", "stockout_days", "shelfout_days")


# ---- bases --------------------------------------------------------------


def horizon_days(f: pd.DataFrame) -> int:
    """Days of daily forecast a basis holds, as `contest.basis` sizes it."""
    return int(f.lead_days.max()) + contest.LIMITS["cycle_days"][1]


def horizon_weeks(f: pd.DataFrame) -> int:
    return math.ceil((horizon_days(f) + 7) / 7)


def week_starts(day: date, weeks: int) -> list[date]:
    """Mondays of the forecast weeks for policy day `day`: the weeks after its origin."""
    first = schema.policy_origin(day) + timedelta(days=1)
    return [first + timedelta(weeks=k) for k in range(weeks)]


def basis_from_weekly(con: duckdb.DuckDBPyConnection, day: date, mean: pd.DataFrame,
                      dispersion: pd.Series, f: pd.DataFrame | None = None
                      ) -> contest.Basis:
    """A Basis from weekly means (drug_code x h = 1.., in order) and a dispersion per drug.

    Weeks past the last column repeat it (`schema.extend_horizon`). `f` is
    `contest.facts(con, day)`, computed if not given; `con` stands on `day`.
    """
    f = contest.facts(con, day) if f is None else f
    codes = list(f.index)
    missing = [c for c in codes if c not in mean.index]
    if missing:
        raise ValueError(f"no forecast for {len(missing)} candidate drugs, e.g. {missing[:5]}")
    d = dispersion.reindex(codes)
    if d.isna().any():
        raise ValueError(f"no dispersion for {int(d.isna().sum())} candidate drugs")
    weeks = horizon_weeks(f)
    m = schema.extend_horizon(mean.reindex(codes).to_numpy(float), weeks)
    v = m * d.to_numpy(float)[:, None]
    starts = week_starts(day, weeks)
    weekly = pd.DataFrame({"drug_code": np.repeat(codes, weeks),
                           "week_start": np.tile(np.array(starts, dtype=object), len(codes)),
                           "mean": m.ravel(), "var": v.ravel()})
    dm, dv = plan._daily(weekly, codes, day, horizon_days(f))
    return contest.Basis(day, f, dm, dv)


def predict_weekly(forecaster: Forecaster, rows: pd.DataFrame, day: date,
                   codes: Sequence[str]) -> pd.DataFrame:
    """Weekly means (drug_code x h) from a fitted forecaster on the policy origin's rows."""
    origin = pd.Timestamp(schema.policy_origin(day))
    if (pd.to_datetime(rows["origin"]) != origin).any():
        raise ValueError(f"panel rows must stand on the policy origin {origin.date()}")
    rows = rows[rows["drug_code"].isin(set(codes))]
    mu = np.asarray(forecaster.predict(rows), float)
    if len(mu) != len(rows) or not np.isfinite(mu).all() or (mu < 0).any():
        raise ValueError("predict must return one finite, non-negative mean per row")
    w = (pd.DataFrame({"drug_code": rows["drug_code"].to_numpy(),
                       "h": rows["h"].astype(int).to_numpy(), "mean": mu})
         .pivot(index="drug_code", columns="h", values="mean"))
    want = list(range(1, split.HORIZON + 1))
    if list(w.columns) != want or w.isna().any().any():
        raise ValueError(f"each drug needs h = 1..{split.HORIZON}")
    return w


def basis(con: duckdb.DuckDBPyConnection, day: date, forecaster: Forecaster,
          panel_rows: pd.DataFrame, dispersion: pd.Series) -> contest.Basis:
    """The basis for policy day `day` from any fitted forecaster.

    `con` stands on `day`. `panel_rows` are the panel's rows at
    `schema.policy_origin(day)` (h = 1..8) covering every candidate drug;
    `forecaster` has been fitted for that origin. `dispersion` (by drug_code)
    is the baseline recipe or U1 as of that origin.
    """
    f = contest.facts(con, day)
    mean = predict_weekly(forecaster, panel_rows, day, list(f.index))
    return basis_from_weekly(con, day, mean, dispersion, f)


def _baseline_history(con: duckdb.DuckDBPyConnection, day: date, codes: list[str]) -> dict:
    return fc.load_history(con, day, codes)


def baseline_dispersion(con: duckdb.DuckDBPyConnection, day: date,
                        codes: list[str]) -> pd.Series:
    """The baseline recipe for `codes` on `day` (for the mean-only comparison)."""
    h = _baseline_history(con, day, codes)
    return pd.Series(dispersion_from_history(h), index=pd.Index(h["drugs"], name="drug_code"))


def baseline_basis(con: duckdb.DuckDBPyConnection, day: date,
                   horizon: int | None = split.HORIZON) -> contest.Basis:
    """The baseline forecaster and its dispersion through `basis_from_weekly`.

    Tiers and methods as `forecast.forecast` sets them on `day`. With
    `horizon=None` every needed week is forecast and the result equals
    `contest.basis(con, day)`; with 8 it is what any forecaster limited to
    h <= 8 would give.
    """
    f = contest.facts(con, day)
    codes = list(f.index)
    h = _baseline_history(con, day, codes)
    tier_of = fc.tiers(con, day)
    method = np.array([fc.METHOD_BY_TIER.get(tier_of.get(c, "none"), "snaive")
                       for c in h["drugs"]], dtype=object)
    p = fc.predict(h, method, horizon or horizon_weeks(f))
    mean = pd.DataFrame(p["mean"], index=pd.Index(h["drugs"], name="drug_code"),
                        columns=range(1, p["mean"].shape[1] + 1))
    disp = pd.Series(dispersion_from_history(h), index=mean.index)
    return basis_from_weekly(con, day, mean, disp, f)


# ---- replay -------------------------------------------------------------


@dataclass
class Results:
    """`compare` output: one summary row per (window, policy), per-drug scores, scales."""

    summary: pd.DataFrame
    per_drug: dict[date, dict[str, pd.DataFrame]] = field(default_factory=dict)
    factors: dict[date, dict[str, float]] = field(default_factory=dict)


def _bases_for(bases: Mapping[str, Any], i: int, n: int) -> dict[str, contest.Basis]:
    out = {}
    for name, b in bases.items():
        if isinstance(b, contest.Basis):
            if n != 1:
                raise ValueError(f"{name}: pass one Basis per window")
            out[name] = b
        else:
            if len(b) != n:
                raise ValueError(f"{name}: {len(b)} bases for {n} windows")
            out[name] = b[i]
    return out


def compare(windows: Sequence[contest.Window],
            bases: Mapping[str, contest.Basis | Sequence[contest.Basis]]) -> Results:
    """The plain policy from each forecaster's basis, replayed at matched stock.

    `windows` are `contest.Window`s (anything with `day`, `end` and the replay
    inputs `h`); `bases` maps a forecaster name to its Basis for each window,
    in order (a single Basis if there's one window). Each basis must stand on
    its window's policy day.

    Run it twice for plan 9.1: once with every forecaster on the baseline
    dispersion (mean model only) and once with each on its own uncertainty
    (full system).
    """
    rows, per_all, fac_all = [], {}, {}
    for i, w in enumerate(windows):
        bs = _bases_for(bases, i, len(windows))
        for name, b in bs.items():
            if b.day != w.day:
                raise ValueError(f"{name}: basis day {b.day} is not the policy day {w.day}")
        policies = {name: contest.to_policy({}, b) for name, b in bs.items()}
        table, per, factors = contest.matched(policies, w.h)
        for name, r in table.iterrows():
            rows.append({"window": f"{w.day + timedelta(days=1)} to {w.end}",
                         "policy_day": w.day, "policy": name,
                         "scale": factors.get(name, 1.0),
                         **{k: r[k] for k in (*contest.COLUMNS, "stockout_days_long",
                                              "lost_units_stockout", "lost_units_shelfout",
                                              "drugs")}})
        per_all[w.day], fac_all[w.day] = per, factors
    return Results(pd.DataFrame(rows), per_all, fac_all)


def _per_drug(per: pd.DataFrame) -> pd.DataFrame:
    p = per.set_index("drug_code")
    return pd.DataFrame({"lost_revenue": p["lost_revenue_stockout"] + p["lost_revenue_shelfout"],
                         "stockout_days": p["stockout_days"].astype(float),
                         "shelfout_days": p["shelfout_days"].astype(float)})


def paired_bootstrap_replay(per_a: pd.DataFrame, per_b: pd.DataFrame, n: int = 2000,
                            level: float = LEVEL, seed: int = 0) -> pd.DataFrame:
    """Totals of A and B and an interval on A - B, resampling drugs (paired).

    `per_a` and `per_b` are per-drug results (`replay.metrics` rows) of two
    matched runs on the same window. One row per metric: a, b, diff (a - b),
    ratio (a / b), and the `level` percentile interval of the diff and ratio.
    """
    a, b = _per_drug(per_a), _per_drug(per_b)
    if set(a.index) != set(b.index):
        raise ValueError("the two runs must score the same drugs")
    b = b.reindex(a.index)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    rows = []
    for k in BOOT_METRICS:
        x, y = a[k].to_numpy(float), b[k].to_numpy(float)
        sa, sb = x[idx].sum(1), y[idx].sum(1)
        diff = sa - sb
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = sa / sb
        rows.append({"metric": k, "a": x.sum(), "b": y.sum(), "diff": x.sum() - y.sum(),
                     "ratio": x.sum() / y.sum() if y.sum() else np.nan,
                     "diff_lo": np.quantile(diff, lo_q), "diff_hi": np.quantile(diff, hi_q),
                     "ratio_lo": np.nanquantile(ratio, lo_q) if np.isfinite(ratio).any()
                     else np.nan,
                     "ratio_hi": np.nanquantile(ratio, hi_q) if np.isfinite(ratio).any()
                     else np.nan,
                     "drugs": len(a), "n": n})
    return pd.DataFrame(rows).set_index("metric")


def _better(lost_hi: float, out_a: float, out_b: float) -> bool:
    return lost_hi < 0 and out_a <= STOCKOUT_SLACK * out_b


def decide_winner(boot: pd.DataFrame, a: str = "challenger", b: str = "baseline"
                  ) -> dict[str, Any]:
    """Plan 9.1's rule on `paired_bootstrap_replay(per_a, per_b)` output.

    A wins if its lost revenue is lower with the interval excluding zero and
    its stock-out days are at most 5% above B's; the same test the other way
    round for B. Otherwise there's no difference. Shelf-out days are carried
    along for the report only.
    """
    lost, out, shelf = boot.loc["lost_revenue"], boot.loc["stockout_days"], \
        boot.loc["shelfout_days"]
    a_wins = _better(lost["diff_hi"], out["a"], out["b"])
    b_wins = _better(-lost["diff_lo"], out["b"], out["a"])
    winner = a if a_wins else b if b_wins else None
    if winner:
        why = (f"{winner} loses less revenue (interval {lost['diff_lo']:.0f} to "
               f"{lost['diff_hi']:.0f}, {a} - {b}) with stock-out days within 5%")
    elif lost["diff_hi"] < 0 or lost["diff_lo"] > 0:
        side = a if lost["diff_hi"] < 0 else b
        why = f"{side} loses less revenue, but its stock-out days are more than 5% higher"
    else:
        why = "the lost-revenue interval includes zero"
    return {"winner": winner, "reason": why,
            "lost_revenue_diff": float(lost["diff"]),
            "lost_revenue_interval": (float(lost["diff_lo"]), float(lost["diff_hi"])),
            "stockout_ratio": float(out["a"] / out["b"]) if out["b"]
            else (1.0 if not out["a"] else float("inf")),
            "shelfout_diff": float(shelf["diff"])}
