"""The choosing rules of ml-plan 7.2 and 7.3, and the baseline re-selection.

**Per tier (7.3).** On the selection origins, for each tier:

1. keep candidates whose bias is inside the tier's band: 0.97 to 1.03 for
   fast and regular, 0.93 to 1.07 for slow and rare;
2. among those, the lowest WAPE wins, except that any candidate within 0.5
   WAPE points of it that is simpler (lower `complexity`) beats it. Among
   the simplest of the near ones, the lowest WAPE;
3. if no candidate is in the band, take the one with bias closest to 1
   among those within 1 WAPE point of the best WAPE. This is the pick; the
   0.5-point tie-break applies only to step 2.

WAPE here is a fraction, as `forecast.score` gives it, so a point is 0.01.
Complexity is the caller's: lower is simpler (a baseline below any ML
model; fewer features below more).

**Pooled objective (7.2).** WAPE over fast, regular and slow rows together,
plus `max(0, |bias - 1| - 0.03)` for fast and for regular, each tier's own
bias. Rare rows are left out. Lower is better.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from irpd import config
from irpd import forecast as fc
from irpd.ml import backtest, schema, split
from irpd.ml import history as hist
from irpd.ml.models.baseline import BaselineForecaster, config_hash

BANDS = {"fast": (0.97, 1.03), "regular": (0.97, 1.03),
         "slow": (0.93, 1.07), "rare": (0.93, 1.07)}
POINT = 0.01
NEAR = 1 * POINT       # fallback: within 1 WAPE point of the best
TIE = 0.5 * POINT      # tie-break: within 0.5 WAPE points, the simpler wins
OBJECTIVE_TIERS = ("fast", "regular", "slow")
PENALTY_TIERS = ("fast", "regular")
BIAS_SLACK = 0.03

# Not a candidate: a forecast of zero never reorders, so it can't stock a
# shelf. It stays in the scores as a reference (on slow and rare drugs its
# WAPE of 1.0 beats every method, which is why those tiers are judged on
# coverage too). Decided at gate 1, before any ML result.
NOT_CANDIDATES = ("zero",)

# Simpler first. Used to break near-ties between the statistical methods
# (gate 1: 8-week mean, then SES, then SES + Eid, then Croston/TSB).
BASELINE_COMPLEXITY = {"zero": 0, "mean8": 1, "snaive": 1, "ses": 2, "ses+eid": 3,
                       "croston_sba+eid": 4, "tsb+eid": 4}


def choose(table: pd.DataFrame, candidate: str = "method", tier: str = "tier",
           bands: dict[str, tuple[float, float]] = BANDS, near: float = NEAR,
           tie: float = TIE) -> pd.DataFrame:
    """One row per tier: the chosen candidate and why.

    `table` has one row per (tier, candidate) with columns `wape`, `bias` and
    `complexity`. Returns tier, candidate, wape, bias, complexity, `rule`
    ('band' or 'fallback'), `in_band` (how many passed the band) and `best`
    (the lowest-WAPE candidate that passed, before the tie-break).
    """
    need = {tier, candidate, "wape", "bias", "complexity"} - set(table.columns)
    if need:
        raise ValueError(f"table is missing {sorted(need)}")
    if table.duplicated([tier, candidate]).any():
        raise ValueError("one row per (tier, candidate)")
    out = []
    for t, g in table.groupby(tier, sort=False, observed=True):
        if t not in bands:
            raise ValueError(f"no bias band for tier {t!r}")
        g = g.dropna(subset=["wape", "bias"])
        if g.empty:
            continue
        lo, hi = bands[t]
        ok = g[g["bias"].between(lo, hi)]
        if len(ok):
            best = ok.loc[ok["wape"].idxmin()]
            close = ok[ok["wape"] <= best["wape"] + tie + 1e-12]
            simplest = close[close["complexity"] == close["complexity"].min()]
            pick, rule = simplest.loc[simplest["wape"].idxmin()], "band"
            best_name = best[candidate]
        else:
            close = g[g["wape"] <= g["wape"].min() + near + 1e-12]
            gap = (close["bias"] - 1).abs()
            pick, rule, best_name = close.loc[gap.idxmin()], "fallback", None
        out.append({"tier": t, "candidate": pick[candidate], "wape": pick["wape"],
                    "bias": pick["bias"], "complexity": pick["complexity"], "rule": rule,
                    "in_band": len(ok), "best": best_name})
    return pd.DataFrame(out)


def objective(pred: pd.DataFrame) -> float:
    """The pooled tuning objective (plan 7.2) on one config's predictions; lower is better.

    `pred` is a predictions table already filtered to one config, one seed
    and the selection origins. Censored rows (NaN actual) are left out.
    """
    if pred["config_hash"].nunique() > 1 or pred["seed"].nunique() > 1:
        raise ValueError("filter pred to one config_hash and one seed first")
    p = pred.dropna(subset=["actual"])
    core = p[p["tier"].isin(OBJECTIVE_TIERS)]
    if core["actual"].sum() <= 0:
        raise ValueError("no demand in fast, regular or slow rows")
    wape = (core["mean"] - core["actual"]).abs().sum() / core["actual"].sum()
    penalty = 0.0
    for t in PENALTY_TIERS:
        g = p[p["tier"] == t]
        if g["actual"].sum() > 0:
            bias = g["mean"].sum() / g["actual"].sum()
            penalty += max(0.0, abs(bias - 1) - BIAS_SLACK)
    return float(wape + penalty)


def candidate_panel(con, h: dict, origins: list) -> pd.DataFrame:
    """The bookkeeping rows `forecast.backtest` scores, built from `ml.history` data `h`.

    One row per (scored drug, origin, h): drugs with a sale in the 13 weeks
    before the origin, tier from `forecast.tiers(con, origin)`, NaN target in
    censored weeks, `mase_scale` from `ml.history`. Enough for `backtest.run`
    with baselines; ML models need the real panel.
    """
    in_eid = fc.eid_index(h["weeks"], dict.fromkeys(fc.EID_DAYS, 2.0)) > 1
    y = np.where(h["out"], np.nan, h["y"])
    parts = []
    for o in origins:
        n = hist.weeks_to(h, o)
        keep = hist.active(h, o, schema.SCORED_WEEKS)
        if n + split.HORIZON > y.shape[1]:
            raise ValueError(f"targets of {o} run past the history")
        tier_of = fc.tiers(con, o)
        drugs = h["drugs"][keep]
        k, hs = len(drugs), np.arange(1, split.HORIZON + 1)
        parts.append(pd.DataFrame({
            "drug_code": np.repeat(drugs, split.HORIZON).astype(str),
            "origin": pd.Timestamp(o), "h": np.tile(hs, k).astype("int8"),
            "target_week": pd.to_datetime(np.tile([schema.target_week(o, int(i)) for i in hs], k)),
            "target": y[keep, n:n + split.HORIZON].ravel().astype("float32"),
            "scored": True,
            "tier": np.repeat([tier_of.get(c, "none") for c in drugs], split.HORIZON),
            "eid": np.tile(in_eid[n:n + split.HORIZON], k),
            "mase_scale": np.repeat(hist.mase_scale(h, o)[keep], split.HORIZON)}))
    return pd.concat(parts, ignore_index=True).astype({"tier": "category"})


def candidate_predictions(con, origins: list, methods: list[str] = fc.CANDIDATES,
                          until=split.SELECTION_END) -> pd.DataFrame:
    """Every candidate method through `backtest.run` on `ml.history` data to `until`."""
    h = hist.load(con, until)
    ctx = schema.PanelContext(con=None, data_end=until, origins=list(origins), history=h)
    panel = candidate_panel(con, h, origins)
    tiers = (*fc.TIER_ORDER, "none")
    preds = []
    for m in methods:
        routing = dict.fromkeys(tiers, m)
        preds.append(backtest.run(lambda r=routing: BaselineForecaster(ctx, r),
                                  config_hash(routing), origins, panel, ctx))
    return pd.concat(preds, ignore_index=True)


def reselect_baselines(con=None, out: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-choose `METHOD_BY_TIER` on the selection origins (plan 3.1, 6.1).

    Runs every `forecast.CANDIDATES` method through the harness on
    `split.selection_origins()`, reading only `split.selection_con()` (data
    to 8 Dec 2024) via `ml.history`, scores by tier and applies `choose`.
    Returns (scores, choice) and writes the scores, with the chosen rows
    marked, to `config.ML_DIR / baseline_selection.csv`.

    `ml.history` differs from `forecast.load_history` in one place: the week
    of 25 Sep 2023, before the stock ledger's first full week, is not marked
    censored. That moves some inputs a little; actuals are the same.
    """
    con = con or split.selection_con()
    pred = candidate_predictions(con, split.selection_origins())
    scores = pd.concat([backtest.score(pred, "selection", config_hash=c)
                        for c in pred["config_hash"].unique()], ignore_index=True)
    scores = scores[scores["tier"] != "none"]
    scores["complexity"] = scores["method"].map(BASELINE_COMPLEXITY)
    choice = choose(scores[~scores["method"].isin(NOT_CANDIDATES)])
    picked = dict(zip(choice["tier"], choice["candidate"], strict=True))
    scores["current"] = scores["method"] == scores["tier"].map(fc.METHOD_BY_TIER)
    scores["chosen"] = scores["method"] == scores["tier"].map(picked)
    order = {t: i for i, t in enumerate(fc.TIER_ORDER)}
    scores = scores.sort_values(["tier", "wape"], key=lambda s: s.map(order)
                                if s.name == "tier" else s).reset_index(drop=True)
    choice = choice.merge(pd.DataFrame({"tier": list(fc.METHOD_BY_TIER),
                                        "current": list(fc.METHOD_BY_TIER.values())}),
                          on="tier", how="left")
    choice["changed"] = choice["candidate"] != choice["current"]
    choice = choice.sort_values("tier", key=lambda s: s.map(order)).reset_index(drop=True)
    out = out or config.ML_DIR / "baseline_selection.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out, index=False)
    return scores, choice


def main() -> None:
    """Print the baseline re-selection (`uv run python -m irpd.ml.select`)."""
    scores, choice = reselect_baselines()
    fmt = {"float_format": lambda v: f"{v:.3f}", "index": False}
    print("Statistical methods on the 11 selection origins (data to 8 Dec 2024)\n")
    print(scores.to_string(**fmt))
    print("\nChoice by the plan 7.3 rule\n")
    print(choice.to_string(**fmt))
    changed = choice[choice["changed"]]
    print("\nMETHOD_BY_TIER " + ("changes: " + ", ".join(
        f"{t} {c} -> {n}" for t, c, n in zip(changed.tier, changed.current, changed.candidate,
                                            strict=True)) if len(changed) else "is unchanged."))


if __name__ == "__main__":
    main()
