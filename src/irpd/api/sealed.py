"""The holdout side of the app: H2 2025 actuals and the replay, for display only.

This is the only module in `irpd.api` that opens the sealed holdout file. Its
output goes to people looking at the app (the drug page with `holdout=true`,
the agent-vs-pharmacist page). Nothing here is ever handed to the agent: the
agent's path is `irpd.api.seen` and the tools, both on `holdout.agent_view()`.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from irpd import holdout, policies, replay

FAIR_CONTEST = (
    "The simulation delivers every order on time, so it can't reproduce the long "
    "stock-outs in the real ledger. The fair contest is a policy against the mimic "
    "(a copy of the pharmacist's own habits) in the same simulation at the same stock "
    "value; the real ledger is shown as a reference.")


def drug_actuals(code: str, sealed: Path) -> dict:
    """What really happened to one drug after the cutoff."""
    with holdout.open_sealed(sealed) as con:
        stock = con.execute("""
            SELECT date,
                   max(closing) FILTER (WHERE location = 'floor') AS floor,
                   max(closing) FILTER (WHERE location = 'store') AS store,
                   max(closing) FILTER (WHERE location = 'total') AS total
            FROM daily_stock WHERE drug_code = ? GROUP BY date ORDER BY date""", [code]).df()
        sales = con.execute("""
            SELECT date, units, revenue::BIGINT AS revenue FROM daily_demand
            WHERE drug_code = ? ORDER BY date""", [code]).df()
        purchases = con.execute("""
            SELECT date, qty, supplier, location FROM purchases
            WHERE drug_code = ? ORDER BY date, row_number""", [code]).df()
    stock = stock.fillna(0)
    out_days = stock["total"] <= 0
    shelf_days = (stock["floor"] <= 0) & (stock["store"] > 0) & ~out_days
    return {
        "stock": _records(stock),
        "sales": _records(sales),
        "purchases": _records(purchases),
        "stockouts": _spells(stock["date"], out_days),
        "shelfouts": _spells(stock["date"], shelf_days),
    }


def _records(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    for c in df.columns:
        if c == "date":
            df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d")
    return df.astype(object).where(df.notna(), None).to_dict("records")


def _spells(dates: pd.Series, flag: pd.Series) -> list[dict]:
    """Runs of consecutive flagged days as {start, stop, days}."""
    out, start, prev = [], None, None
    for d, f in zip(pd.to_datetime(dates).dt.date, flag, strict=True):
        if f and start is None:
            start = d
        if not f and start is not None:
            out.append((start, prev))
            start = None
        prev = d
    if start is not None:
        out.append((start, prev))
    return [{"start": a.isoformat(), "stop": b.isoformat(), "days": (b - a).days + 1}
            for a, b in out]


@lru_cache(maxsize=4)
def replay_report(agent: Path, sealed: Path) -> dict:
    """The replay table, per-drug scores and tradeoff curves. Cached per file pair."""
    h = replay.load(agent, sealed)
    with holdout.agent_view(agent=agent) as con:
        refs = {"mimic": policies.mimic(con), "textbook": policies.textbook(con)}
        names = dict(con.execute("SELECT drug_code, drug_name FROM drugs").fetchall())
    runs, factors = dict(refs), {}
    for name, policy in refs.items():
        try:
            matched, k = replay.match_stock_value(policy, h)
        except ValueError:  # the policy can't hold that much stock; leave it unmatched
            factors[name] = None
            continue
        runs[f"{name} at matched stock"] = matched
        factors[name] = k

    table = replay.score(runs, h)
    cols = ["drugs", "days", "stockout_days", "stockout_days_long", "shelfout_days",
            "stockout_share", "shelfout_share", "lost_units_stockout", "lost_units_shelfout",
            "lost_revenue_stockout", "lost_revenue_shelfout", "lost_revenue",
            "avg_stock_value", "units_sold"]
    summary = [{"policy": name, "kind": _kind(name), **{c: _num(row[c]) for c in cols}}
               for name, row in table.iterrows()]

    per = {"pharmacist (actual)": replay.actual(h)}
    for name in (n for n in runs if n.endswith("matched stock")):
        run = replay.simulate(runs[name], h)
        per[name] = replay.metrics(run.floor, run.store, run.sold, h)
    keys = ["stockout_days", "shelfout_days", "lost_revenue", "avg_stock_value"]
    per_drug = [{"code": code, "name": names.get(code, code)} for code in h.drugs]
    for name, df in per.items():
        df = df.assign(lost_revenue=df.lost_revenue_stockout + df.lost_revenue_shelfout)
        for row, r in zip(per_drug, df[keys].to_dict("records"), strict=True):
            row[name] = {k: _num(v) for k, v in r.items()}

    grid = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0)
    curves = {name: [{k: _num(v) for k, v in r.items()}
                     for r in replay.tradeoff(policy, h, grid).to_dict("records")]
              for name, policy in refs.items()}
    return {
        "period": {"first": h.days[0].isoformat(), "last": h.days[-1].isoformat()},
        "scope": "fast and regular drugs at the cutoff that sold again after it",
        "fair_contest": FAIR_CONTEST,
        "matched_factor": factors,
        "summary": summary,
        "tradeoff": curves,
        "per_drug": per_drug,
    }


def _kind(name: str) -> str:
    if name.startswith("pharmacist"):
        return "ledger"
    return "matched" if "matched" in name else "reference"


def _num(v) -> float | int | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    v = float(v)
    return int(v) if v.is_integer() else round(v, 4)


# The frozen contest runs. Their holdout scores were computed from the sealed file
# when each run was scored; here they are only read back.

REFERENCES = ("pharmacist (ledger)", "mimic", "textbook", "plain tools")
METRICS = ("stockout_days", "shelfout_days", "lost_revenue", "avg_stock_value",
           "stockout_days_long", "scale")
WHEN = {  # a segment filter, in plain words (thresholds from irpd.contest)
    "tier": lambda v: f"{v}-selling",
    "cadence": lambda v: f"{v} supplier cadence",
    "trend": lambda v: {"falling": "demand falling (last 8 weeks under 0.75x the half-year rate)",
                        "rising": "demand rising (last 8 weeks over 1.25x the half-year rate)",
                        "flat": "demand flat"}[v],
    "long_stockout": lambda v: ("a stock-out over 14 days in the last half-year" if v
                                else "no stock-out over 14 days in the last half-year"),
    "switching": lambda v: {"none": "no same-ingredient alternative",
                            "high": "customers often switch to an alternative (30%+)",
                            "low": "customers rarely switch to an alternative"}[v],
    "min_idle_days": lambda v: f"no sale for {v}+ days",
}


def _sets(rule: dict) -> str:
    """What a rule sets, in plain words; unset values are left out."""
    parts = []
    if rule.get("service_level") is not None:
        parts.append(f"{rule['service_level']:.0%} service level")
    if rule.get("cycle_days") is not None:
        parts.append(f"orders cover the lead time plus {rule['cycle_days']} days")
    lo, hi = rule.get("floor_min_days"), rule.get("floor_max_days")
    if lo is not None or hi is not None:
        parts.append(f"refill the shop floor at {lo if lo is not None else 'the default'} "
                     f"days of floor sales, up to {hi if hi is not None else 'the default'}")
    return "; ".join(parts) or "the plain defaults"


def _plain_rules(knobs: dict, counts: dict[str, int]) -> list[dict]:
    """The knobs as a list of rules in plain words, first match wins, with drug counts."""
    rules = []
    for seg in knobs.get("segments") or []:
        when = [WHEN[k](seg[k]) for k in WHEN if seg.get(k) is not None]
        rules.append({"name": seg["name"], "when": "Drugs with " + ", and ".join(when),
                      "sets": _sets(seg), "drugs": counts.get(seg["name"], 0)})
    if knobs.get("stop_after_idle_days") is not None:
        rules.append({"name": "stopped (idle)",
                      "when": f"Drugs with no sale for {knobs['stop_after_idle_days']}+ days",
                      "sets": "stop reordering", "drugs": counts.get("stopped (idle)", 0)})
    for o in knobs.get("overrides") or []:
        rules.append({"name": f"override {o['drug']}", "when": o.get("reason") or "",
                      "sets": "stop reordering" if o.get("stop") else _sets(o), "drugs": 1})
    rules.append({"name": "default", "when": "Every other drug",
                  "sets": _sets(knobs.get("default") or {}), "drugs": counts.get("default", 0)})
    return rules


def _rows(df: pd.DataFrame) -> list[dict]:
    return [{"policy": p, **{c: _num(r[c]) for c in METRICS}} for p, r in df.iterrows()]


def _curve(path: Path, who: str) -> list[dict]:
    df = pd.read_csv(path, index_col=[0, 1])
    df = df.loc[who] if who in df.index.get_level_values(0) else df.iloc[:0]
    return [{k: _num(v) for k, v in r.items()} for r in df.to_dict("records")]


def _run(d: Path) -> dict:
    meta = json.loads((d / "knobs.json").read_text())
    score = pd.read_csv(d / "holdout_score.csv", index_col=0)
    counts = pd.read_csv(d / "rules.csv").rule.value_counts().to_dict()
    in_sample = json.loads((d / "in_sample.json").read_text())["windows"]
    return {
        "run_id": d.name, "model": meta["model"], "as_of": meta["as_of"],
        "frozen_at": meta["frozen_at"], "usd": meta.get("usd"),
        "rationale": meta["rationale"], "rules": _plain_rules(meta["knobs"], counts),
        "score": {m: _rows(t.drop(columns="demand"))
                  for m, t in score.groupby("demand", sort=False)},
        "tradeoff": {m: _curve(d / f"tradeoff_{m}.csv", "agent") for m in score.demand.unique()},
        "in_sample": [{"window": w["window"], "rows": [
            {"policy": r["policy"], "stockout_days": r["stockout_days"],
             "shelfout_days": r["shelfout_days"], "lost_revenue": r["lost_revenue_rp_m"] * 1e6,
             "avg_stock_value": r["avg_stock_rp_m"] * 1e6} for r in w["table"]]}
            for w in in_sample],
    }


def contest_report(out: Path) -> dict | None:
    """The frozen contest runs under `out` (`irpd.contest`), or None if none was scored.

    A run counts once it is frozen (`final-*`) and scored (`holdout_score.csv`).
    Runs are listed best first by lost revenue under uncensored demand. Only
    files are read; the sealed holdout was opened when the runs were scored.
    """
    dirs = sorted(d for d in out.glob("final-*") if (d / "holdout_score.csv").exists())
    if not dirs:
        return None
    runs = sorted((_run(d) for d in dirs), key=lambda r: next(
        x["lost_revenue"] for x in r["score"]["uncensored"] if x["policy"] == "agent"))
    first = pd.read_csv(dirs[0] / "holdout_score.csv", index_col=0)
    refs = {m: _rows(t.loc[[p for p in REFERENCES if p in t.index]].drop(columns="demand"))
            for m, t in first.groupby("demand", sort=False)}
    table = {m: rows + [{**next(r for r in run["score"][m] if r["policy"] == "agent"),
                         "policy": f"agent ({run['model']})"} for run in runs]
             for m, rows in refs.items()}
    log = out / "holdout_runs.jsonl"
    return {
        "drugs": int(first.drugs.iloc[0]),
        "holdout_runs": len(log.read_text().splitlines()) if log.exists() else None,
        "table": table,
        "mimic_tradeoff": {m: _curve(dirs[0] / f"tradeoff_{m}.csv", "mimic") for m in refs},
        "runs": runs,
    }
