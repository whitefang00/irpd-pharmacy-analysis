"""Forecast arms on the shared 14-day origin grid (see eda/followup/experiment-plan.md).

Run from the repo root (each command writes data/followup/preds/<arm>.parquet + .json;
the ML arms retrain LightGBM at every origin and take hours):

    uv run python eda/followup/scripts/arms/run_arms.py panel            # base panel + Eid calendar, once
    uv run python eda/followup/scripts/arms/run_arms.py ml ml_control    # one ML arm (resumable)
    uv run python eda/followup/scripts/arms/run_arms.py cur              # cur_control, cur_eid, tsb
    uv run python eda/followup/scripts/arms/run_arms.py labels
    uv run python eda/followup/scripts/arms/run_arms.py actuals
    uv run python eda/followup/scripts/arms/run_arms.py validate

Reads the full warehouse (irpd.duckdb, data to 31 Dec 2025) read-only. Every
forecast at origin o uses data up to o only: the ML panel rows are built from
history cut at their origin and connections as of it, and each fit sees
`schema.trainable(panel, o)`; the current forecaster and TSB run on the
history sliced at o.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402

# Keep any model-fit side files out of the project's data/ml.
os.environ.setdefault("IRPD_ML_DIR", str(fp.out("arms") / "ml-dir"))

from irpd import config, warehouse  # noqa: E402
from irpd import forecast as fc
from irpd.ml import backtest, censor, panel, schema, segments, split
from irpd.ml import history as hist
from irpd.ml.features import calendar as cal
from irpd.ml.models.baseline import baseline_dispersion
from irpd.ml.models.lgbm import make_gbm
from irpd.ml.serve import CONFIG_HASH, frozen_config


WH = fp.WAREHOUSE / "irpd.duckdb"
PREDS = fp.PREDS
NOTES = fp.out("arms")
WORK = NOTES / "work"
DATA_END = date(2025, 12, 28)  # the last Sunday of a whole week (22-28 Dec 2025)
GRID = [date(2024, 1, 7) + timedelta(days=14 * k) for k in range(52)]
assert GRID[-1] == date(2025, 12, 21)  # 14 Dec 2025 is not on the 14-day grid; see experiment-notes.md

EID_OPTIONS = {"eid_list": "full", "eid_days_cap": 45, "eid_day_counts": True,
               "eid_uplift": "units"}
ML_ARMS = {
    "ml_control": {"calendar": {}, "target": "mask", "shelf": "mask"},
    "ml_eid": {"calendar": EID_OPTIONS, "target": "mask", "shelf": "mask"},
    "ml_censor": {"calendar": {}, "target": "impute", "shelf": "flag"},
    "ml_eid_censor": {"calendar": EID_OPTIONS, "target": "impute", "shelf": "flag"},
    # Post hoc (added after the 2025 placement check, see experiment-notes.md): the store-level
    # Eid factor taken out of the tree, as the current forecaster does.
    "ml_eid_offset": {"calendar": EID_OPTIONS, "target": "mask", "shelf": "mask",
                      "offset": True},
}
DESCRIPTIONS = {
    "ml_control": "Frozen ML system 36ad7c615b96 (LightGBM Tweedie 1.3, A2, mask, 3 seeds), "
                  "retrained at every origin; var = baseline NB recipe.",
    "ml_eid": "ml_control + E3: Eid list to 2027, c_eid_days missing beyond 45 days, "
              "target-week Eid day counts, c_eid_uplift from store units (unshrunk).",
    "ml_censor": "ml_control + E4: stock-out target weeks put back with the leak-free "
                 "per-day lost.py estimate (gross); shelf-out weeks kept with a target-week "
                 "shelf-out-days feature, set to 0 at prediction.",
    "ml_eid_censor": "ml_eid + ml_censor changes together.",
    "ml_eid_offset": "POST HOC. ml_eid, plus: training targets divided by the store-units Eid "
                     "factor of their target week (factors learned from Eids before the fit "
                     "origin), predictions multiplied by the target week's factor.",
    "cur_control": "Current forecaster (forecast.METHOD_BY_TIER, SES+Eid with the store-lines "
                   "Eid profile, shrink 0.6; snaive for rare), as forecast.forecast runs it.",
    "cur_eid": "Current forecaster with the E3 Eid profile: store units per day, unshrunk, "
               "summed (averaged) over each target week's days.",
    "tsb": "TSB (alpha_p 0.15 on the chance of a sale, alpha_d 0.10 on its size) on "
           "uncensored weeks; var = mean x the drug's dispersion. stop_flag = rate x 26 < 1.",
}


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


@lru_cache(maxsize=4)  # each connection holds threads; ~150 open at once ran out
def con_at(day: date) -> duckdb.DuckDBPyConnection:
    con = warehouse.as_of(day, WH)
    con.execute("SET threads = 2")
    return con


def full_con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(WH), read_only=True)


def ml_context(options: dict | None = None) -> schema.PanelContext:
    origins = set(split.training_origins(DATA_END)) | set(GRID)
    return panel.context(full_con(), DATA_END, origins, censor_mode="mask", con_at=con_at,
                         options=options)


def _restore(p: pd.DataFrame) -> pd.DataFrame:
    p = p.astype({"drug_code": "str", "origin": panel.DATETIME,
                  "target_week": panel.DATETIME})
    p["tier"] = p["tier"].astype(pd.CategoricalDtype(panel.TIERS))
    return p


def build_panels() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    cfg = frozen_config()
    t = time.time()
    ctx = ml_context()
    p = panel.build_panel(ctx, cfg.groups)
    log(f"control panel {p.shape} in {time.time() - t:.0f}s")
    p.to_parquet(WORK / "panel_control.parquet", index=False)
    t = time.time()
    c1 = cal.build(ml_context(EID_OPTIONS))
    c1.reset_index().to_parquet(WORK / "calendar_eid.parquet", index=False)
    log(f"eid calendar {c1.shape} in {time.time() - t:.0f}s")


def with_calendar(p: pd.DataFrame, c1: pd.DataFrame) -> pd.DataFrame:
    """The control panel with its c_ block swapped for `c1`, in the same column position."""
    cols = list(p.columns)
    i = next(k for k, c in enumerate(cols) if c.startswith("c_"))
    base = p.drop(columns=[c for c in cols if c.startswith("c_")])
    out = panel.join(base, "calendar", c1.set_index(["drug_code", "origin", "h"]))
    order = ([c for c in cols[:i]] + list(c1.columns.drop(["drug_code", "origin", "h"]))
             + [c for c in cols[i:] if not c.startswith("c_")])
    out = out[order]
    schema.validate_panel(out)
    return out


def arm_panel(arm: str, ctx: schema.PanelContext) -> pd.DataFrame:
    spec = ML_ARMS[arm]
    p = _restore(pd.read_parquet(WORK / "panel_control.parquet"))
    if spec["calendar"]:
        c1 = pd.read_parquet(WORK / "calendar_eid.parquet")
        c1["origin"] = pd.to_datetime(c1["origin"]).astype(panel.DATETIME)
        c1["h"] = c1["h"].astype("int8")
        p = with_calendar(p, c1)
    else:
        schema.validate_panel(p)
    if spec["target"] != "mask" or spec["shelf"] != "mask":
        p = censor.censor_aware(p, ctx, spec["target"], spec["shelf"])
    return p


def run_ml(arm: str, only: list[date] | None = None) -> None:
    cfg = frozen_config()
    spec = ML_ARMS[arm]
    parts_dir = WORK / "parts" / arm
    parts_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ctx = ml_context(spec["calendar"])
    p = arm_panel(arm, ctx)
    feats = [c for c in schema.feature_columns(p, cfg.prefixes())]
    log(f"{arm}: panel {p.shape}, {len(feats)} features, ready in {time.time() - t0:.0f}s")
    by_origin = dict(tuple(p.groupby(p["origin"].dt.date, sort=False)))
    fits = []
    for k, o in enumerate(GRID, 1):
        path = parts_dir / f"{o:%Y%m%d}.parquet"
        if path.exists() or (only and o not in only):
            continue
        t = time.time()
        train = schema.trainable(p, o)
        rows = censor.for_prediction(by_origin[o])
        factor = None
        if spec.get("offset"):
            f = fc.eid_profile(ctx.history_at(o), "units")
            weeks = sorted({x.date() for x in train["target_week"]} |
                           {x.date() for x in rows["target_week"]})
            by_week = dict(zip(weeks, fc.eid_index(weeks, f), strict=True))
            div = np.array([by_week[x.date()] for x in train["target_week"]])
            train = train.assign(target=(train["target"] / div).astype(np.float32))
            factor = np.array([by_week[x.date()] for x in rows["target_week"]])
        model = make_gbm(cfg).fit(train, o)
        seeds = {int(s): np.asarray(m, float) for s, m in model.predict_seeds(rows).items()}
        if factor is not None:
            seeds = {s_: m * factor for s_, m in seeds.items()}
        mean = np.mean(list(seeds.values()), axis=0)
        h = ctx.history_at(o)
        disp = pd.Series(baseline_dispersion(h), index=h["drugs"])
        d = disp[rows["drug_code"]].to_numpy()
        out = pd.DataFrame({
            "origin": o, "drug_code": rows["drug_code"].astype(str).to_numpy(),
            "h": rows["h"].astype(int).to_numpy(),
            "week_start": [x.date() for x in rows["target_week"]],
            "mean": mean, "var": np.maximum(mean * d, mean),
            "tier": rows["tier"].astype(str).to_numpy(),
            "scored": rows["scored"].to_numpy(bool)})
        for s, m in sorted(seeds.items()):
            out[f"mean_seed{s}"] = m
        out.to_parquet(path, index=False)
        info = {"origin": o.isoformat(), "seconds": round(time.time() - t, 1),
                "n_train": int(model.fit_info["n_train"]),
                "n_valid": int(model.fit_info["n_valid"]),
                "best_iteration": {str(k_[1]): int(v) for k_, v in model.best_iteration.items()}}
        fits.append(info)
        with (parts_dir / "fits.jsonl").open("a") as f:
            f.write(json.dumps(info) + "\n")
        log(f"{arm} {k}/{len(GRID)} {o}: {len(out)} rows, {info['seconds']}s, "
            f"iters {info['best_iteration']}")
    if only:
        return
    combine(arm, parts_dir, {"code_flags": spec, "features": feats,
                             "config": cfg.to_dict(),
                             "config_hash": CONFIG_HASH, "seeds": list(cfg.seeds)},
            time.time() - t0)


def combine(arm: str, parts_dir: Path, meta: dict, seconds: float) -> None:
    files = sorted(parts_dir.glob("*.parquet"))
    pred = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    pred["origin"] = pd.to_datetime(pred["origin"]).dt.date
    pred["week_start"] = pd.to_datetime(pred["week_start"]).dt.date
    check_pred(pred)
    PREDS.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(PREDS / f"{arm}.parquet", index=False)
    fits = []
    if (parts_dir / "fits.jsonl").exists():
        fits = [json.loads(x) for x in (parts_dir / "fits.jsonl").read_text().splitlines()]
    note = {"arm": arm, "description": DESCRIPTIONS[arm], **meta,
            "origins": [o.isoformat() for o in sorted(pred.origin.unique())],
            "n_origins": int(pred.origin.nunique()), "rows": len(pred),
            "drug_set": "ML arms: drugs with a sale in the 52 weeks before the origin "
                        "(the panel's active rows); `scored` = a sale in the last 13 weeks.",
            "data": str(WH), "data_end": DATA_END.isoformat(),
            "runtime_seconds_this_run": round(seconds),
            "fit_seconds_total": round(sum(f["seconds"] for f in fits), 1)}
    (PREDS / f"{arm}.json").write_text(json.dumps(note, indent=1, default=str) + "\n")
    log(f"wrote {arm}: {len(pred)} rows, {pred.origin.nunique()} origins")


def check_pred(pred: pd.DataFrame) -> None:
    assert not pred.duplicated(["origin", "drug_code", "h"]).any()
    assert pred.h.between(1, 8).all()
    expect = [o + timedelta(days=1 + 7 * (h - 1)) for o, h in zip(pred.origin, pred.h)]
    assert (pred.week_start.to_numpy() == np.array(expect, dtype=object)).all()
    assert np.isfinite(pred["mean"]).all() and (pred["mean"] >= 0).all()
    assert np.isfinite(pred["var"]).all() and (pred["var"] >= pred["mean"] - 1e-9).all()


# ---- current forecaster and TSB ------------------------------------------------


def run_cur() -> None:
    t0 = time.time()
    con = full_con()
    full = fc.load_history(con, DATA_END)
    parts = {"cur_control": [], "cur_eid": [], "tsb": []}
    for o in GRID:
        n = sum(w + timedelta(days=6) <= o for w in full["weeks"])
        h = fc._slice(full, n)
        tier_of = fc.tiers(con, o)
        tier = np.array([tier_of.get(c, "none") for c in h["drugs"]], dtype=object)
        keep = tier != "none"
        method = np.array([fc.METHOD_BY_TIER.get(t, "snaive") for t in tier], dtype=object)
        future = [h["weeks"][-1] + timedelta(weeks=k + 1) for k in range(8)]
        assert future[0] == o + timedelta(days=1)
        runs = {"cur_control": fc.predict(h, method, 8, eid="lines"),
                "cur_eid": fc.predict(h, method, 8, eid="units"),
                "tsb": fc.tsb(h, 8)}
        for arm, r in runs.items():
            m, v = r["mean"][keep], r["var"][keep]
            k_ = keep.sum()
            df = pd.DataFrame({
                "origin": o, "drug_code": np.repeat(h["drugs"][keep].astype(str), 8),
                "h": np.tile(np.arange(1, 9), k_),
                "week_start": np.tile(np.array(future, dtype=object), k_),
                "mean": m.ravel(), "var": np.maximum(v.ravel(), m.ravel()),
                "tier": np.repeat(tier[keep].astype(str), 8)})
            if arm != "tsb":
                df["method"] = np.repeat(method[keep], 8)
            else:
                df["stop_flag"] = np.repeat(r["rate"][keep] * 26 < 1, 8)
            parts[arm].append(df)
        log(f"cur/tsb {o}: {keep.sum()} drugs")
    for arm, ps in parts.items():
        pred = pd.concat(ps, ignore_index=True)
        check_pred(pred)
        pred.to_parquet(PREDS / f"{arm}.parquet", index=False)
        flags = {"cur_control": {"eid": "lines", "shrink": fc.EID_SHRINK},
                 "cur_eid": {"eid": "units", "shrink": fc.EID_UNIT_SHRINK},
                 "tsb": {"alpha_p": fc.TSB_ALPHA_P, "alpha_d": fc.TSB_ALPHA_D}}[arm]
        note = {"arm": arm, "description": DESCRIPTIONS[arm], "code_flags": flags,
                "config": {"method_by_tier": fc.METHOD_BY_TIER} if arm != "tsb" else flags,
                "origins": [o.isoformat() for o in GRID], "n_origins": len(GRID),
                "rows": len(pred), "seeds": None,
                "drug_set": "every drug with a sale by the origin (tier != 'none'), as "
                            "forecast.forecast does; censoring = total stock-out weeks only "
                            "(forecast.load_history), not the ML's shelf-out rule.",
                "data": str(WH), "runtime_seconds_all_three_arms": round(time.time() - t0)}
        (PREDS / f"{arm}.json").write_text(json.dumps(note, indent=1, default=str) + "\n")
        log(f"wrote {arm}: {len(pred)} rows")


# ---- labels and actuals ----------------------------------------------------------


def run_labels() -> None:
    from irpd import contest, replay

    t0 = time.time()
    con = full_con()
    h_all = hist.load(con, DATA_END)
    parts = []
    for o in GRID:
        h = hist.cut(h_all, o)
        tier_of = fc.tiers(con, o)
        tier = np.array([tier_of.get(c, "none") for c in h["drugs"]], dtype=object)
        lab = segments.labels(h, tier)
        n = h["y"].shape[1]
        lab["active52"] = np.nansum(h["y"][:, max(0, n - 52):], axis=1) > 0
        lab["scored13"] = np.nansum(h["y"][:, max(0, n - 13):], axis=1) > 0
        c = con_at(o)
        try:
            contest.ensure_baseline(c, o)
            lead = replay.default_lead(c)
        except Exception as e:  # no supplier history yet
            log(f"lead days at {o}: {e}")
            lead = pd.Series(dtype=float)
        lab["lead_days"] = (lab.drug_code.map(lead).fillna(replay.LEAD_FALLBACK)
                            .clip(*replay.LEAD_CLIP).astype(int))
        price = con.execute(f"""
            SELECT drug_code,
                   sum(revenue) FILTER (WHERE date > DATE '{o - timedelta(days=182)}')
                     / nullif(sum(units) FILTER (WHERE date > DATE '{o - timedelta(days=182)}'), 0)
                     AS p182,
                   sum(revenue) / nullif(sum(units), 0) AS p_all
            FROM daily_demand WHERE date <= DATE '{o}' AND units > 0 GROUP BY 1""").df()
        price = price.set_index("drug_code")
        lab["price_avg182"] = lab.drug_code.map(price.p182.fillna(price.p_all))
        lab.insert(0, "origin", o)
        parts.append(lab[lab.tier != "none"])
        log(f"labels {o}: {(lab.tier != 'none').sum()} drugs")
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(PREDS / "labels.parquet", index=False)
    (PREDS / "labels.json").write_text(json.dumps({
        "file": "labels.parquet", "origins": len(GRID), "rows": len(out),
        "definitions": segments.__doc__,
        "extra": {"active52": "sale in the 52 weeks before the origin (ML rows)",
                  "scored13": "sale in the 13 weeks before (the project's scored set)",
                  "lead_days": "replay.default_lead on a connection as of the origin "
                               "(supplier cadence, clipped 1-14, fallback 7)",
                  "price_avg182": "revenue / units over the 182 days to the origin "
                                  "(all history if no sale in them)"},
        "runtime_seconds": round(time.time() - t0)}, indent=1) + "\n")
    log(f"labels: {len(out)} rows in {time.time() - t0:.0f}s")


def run_actuals() -> None:
    from irpd import lost

    t0 = time.time()
    con = full_con()
    h = hist.load(con, DATA_END)
    ctx = panel.context(con, DATA_END, [GRID[0]], con_at=con_at)
    daily_est, ok = censor.stockout_estimate(ctx)
    p = lost.panel(con, config.LAST_DAY)
    sp = lost.spells(con, config.LAST_DAY, p=p)
    log(f"spells {len(sp)} in {time.time() - t0:.0f}s")
    units = p.units.copy()
    s = sp[(sp.kind == "stockout") & sp.estimated & (sp.lost_units > 0)]
    for r in s.itertuples():
        a = p.day(r.counted_from)
        days = slice(a, a + r.counted_days)
        if p.w[days].sum() > 0:
            units[p.index[r.drug_code], days] += r.lost_units * p.w[days] / p.w[days].sum()
    weeks = h["weeks"]
    unc = np.full(h["y"].shape, np.nan)
    rows = np.array([p.index.get(c, -1) for c in h["drugs"]])
    have = rows >= 0
    for jw, w in enumerate(weeks):
        a = (w - p.days[0]).days
        unc[have, jw] = units[rows[have], a:a + 7].sum(axis=1)
    y = h["y"]
    keep = ~np.isnan(y)
    jj, tt = np.nonzero(keep)
    out = pd.DataFrame({
        "drug_code": h["drugs"][jj].astype(str),
        "week_start": np.array(weeks, dtype=object)[tt],
        "units": y[jj, tt],
        "censored": h["stockout"][jj, tt],
        "uncensored_units": unc[jj, tt],
        "shelfout_days": h["shelfout_days"][jj, tt].astype(int),
        "shelf_censored": (h["out"] & ~h["stockout"])[jj, tt],
        "uncensored_units_daily": np.where(ok[jj, tt], daily_est[jj, tt], np.nan),
        "stock_known": h["stock_known"][tt],
    })
    out.to_parquet(PREDS / "actuals.parquet", index=False)
    (PREDS / "actuals.json").write_text(json.dumps({
        "file": "actuals.parquet", "rows": len(out),
        "weeks": [str(weeks[0]), str(weeks[-1])],
        "columns": {
            "units": "recorded units (weekly_demand); rows from each drug's first sale on",
            "censored": "the week had a total stock-out day (ledger from 2 Oct 2023)",
            "uncensored_units": "gross: units plus each stock-out spell's lost units "
                                "(lost.spells as of 31 Dec 2025, spells with positive "
                                "estimated loss, first MAX_DAYS days, spread by traffic as "
                                "lost.uncensor does); no switching taken off alternatives",
            "uncensored_units_daily": "the E4 training estimator (censor.stockout_estimate): "
                                      "out days replaced by rate x traffic; NaN where an out "
                                      "day can't be estimated",
            "shelfout_days": "days in a shelf-out spell (floor empty, store stocked)",
            "shelf_censored": "4+ shelf-out days and no total stock-out (the ML panel "
                              "also drops these)",
        },
        "runtime_seconds": round(time.time() - t0)}, indent=1) + "\n")
    log(f"actuals: {len(out)} rows in {time.time() - t0:.0f}s")


# ---- validation ------------------------------------------------------------------


def validate() -> None:
    ev = fp.ML / "eval/predictions"
    parts = WORK / "parts/ml_control"
    ml = pd.concat([pd.read_parquet(f) for f in sorted(parts.glob("*.parquet"))])
    ml["origin"] = pd.to_datetime(ml["origin"]).dt.date
    rows = []
    for f in sorted(ev.glob(f"{CONFIG_HASH}_*_v2.parquet")):
        e = pd.read_parquet(f)
        e = e[e.seed == schema.ENSEMBLE_SEED]
        o = pd.Timestamp(e.origin.iloc[0]).date()
        if o not in set(ml.origin):
            continue
        m = ml[ml.origin == o].merge(e[["drug_code", "h", "mean", "var"]],
                                     on=["drug_code", "h"], suffixes=("", "_eval"))
        d = (m["mean"] - m["mean_eval"]).abs()
        rows.append({"origin": o, "rows_eval": len(e), "matched": len(m),
                     "max_abs_diff": d.max(), "mean_abs_diff": d.mean(),
                     "max_rel_diff": (d / m["mean_eval"].clip(lower=1e-6)).max(),
                     "sum_ratio": m["mean"].sum() / m["mean_eval"].sum(),
                     "var_max_abs_diff": (m["var"] - m["var_eval"]).abs().max()})
    out = pd.DataFrame(rows)
    out.to_csv(NOTES / "validate_ml_control.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "panel":
        build_panels()
    elif cmd == "ml":
        only = ([date.fromisoformat(x) for x in sys.argv[3].split(",")]
                if len(sys.argv) > 3 else None)
        run_ml(sys.argv[2], only)
    elif cmd == "cur":
        run_cur()
    elif cmd == "labels":
        run_labels()
    elif cmd == "actuals":
        run_actuals()
    elif cmd == "validate":
        validate()
    else:
        raise SystemExit(f"unknown command {cmd}")
