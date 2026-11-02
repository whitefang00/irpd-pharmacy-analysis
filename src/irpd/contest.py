"""The contest: an agent sets the H2 2025 reorder and floor policy, and the replay scores it.

**The contest.** Standing on the cutoff (30 Jun 2025), a planner agent picks a
few *knobs* (below) that turn into an (s, S) and floor policy for every
`replay.candidates` drug. The policy is frozen to `data/contest/<run_id>/`,
and only then scored on the sealed holdout, at the pharmacist's average stock
value, against:

- the mimic: a copy of the pharmacist's habits in the same sim (the headline);
- the textbook (s, S) rule;
- the plain tool policy: `reorder_rows` (service level 0.95) with the
  `transfer_rows` floor rule, no LLM. It is exactly `DEFAULT` knobs, so it
  shows what the LLM's choices add;
- the real ledger, as a reference.

**No peeking.** While it plans, the agent can only backtest: knobs turn into a
policy as of an earlier day and are replayed on the half-year after it, up to
the cutoff (`WINDOWS`). `load_window` builds the replay's arrays from the
agent file alone, so tuning never opens the sealed file.

**Knobs.** A default rule, then segment rules (the first that matches a drug
wins), then `stop_after_idle_days`, then drug overrides, each with a reason.
A rule sets:

- `service_level`: the chance the forecast demand over the lead time is
  covered; s is that quantile;
- `cycle_days`: S covers the lead time plus this many days, at the same level;
- `floor_min_days`, `floor_max_days`: move stock down to the shop floor when it
  holds this many days of floor sales or less, up to the max.

Segments match on facts known on the policy day (`facts`). At matched stock
the replay rescales every policy to the same stock value, so the knobs matter
by *where* they put stock, not how much.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from irpd import config, holdout, llm, lost, policies, replay
from irpd import forecast as fc
from irpd.agent import loop, registry
from irpd.agent.tools import common, plan, stock
from irpd.tables import supply

OUT = config.ROOT / "data/contest"
RUNS_LOG = OUT / "holdout_runs.jsonl"  # every holdout scoring, appended
# In-sample backtests: (policy day, last day replayed). Both end by the cutoff.
WINDOWS = [(date(2024, 12, 31), date(2025, 6, 30)), (date(2024, 6, 30), date(2024, 12, 31))]

DEFAULT = {"service_level": 0.95, "cycle_days": plan.CYCLE_DAYS,
           "floor_min_days": stock.FLOOR_LOW_DAYS, "floor_max_days": stock.FLOOR_TARGET_DAYS}
LIMITS = {"service_level": (0.5, 0.995), "cycle_days": (0, 60),
          "floor_min_days": (0, 14), "floor_max_days": (1, 30)}
TREND_DAYS = (56, 182)        # recent vs half-year in-stock rate
TREND_EDGES = (0.75, 1.25)    # below: falling, above: rising
CADENCE_EDGES = (3, 7)        # lead days: short <= 3 < medium <= 7 < long
HIGH_SWITCH = 0.3             # ingredient switch rate at or above this is "high"
LONG_OUT = 14                 # a stock-out spell longer than this is "long"
FACT_DAYS = 182               # stock-out history behind the facts


def ensure_baseline(con: duckdb.DuckDBPyConnection, day: date) -> None:
    """Give an as-of connection the pharmacist's baseline as of `day`.

    `pharmacist_baseline` and `supplier_cadence` are hidden before the cutoff
    (they are computed through it). Before it, compute them from the ledger up
    to `day`, which is what the pharmacist's habits looked like then.
    """
    have = {r[0] for r in con.execute("""
        SELECT table_name FROM duckdb_tables() WHERE database_name = current_database()
        UNION SELECT view_name FROM duckdb_views() WHERE database_name = current_database()
        """).fetchall()}
    if "pharmacist_baseline" not in have:
        supply.build_baseline(con, day)


def window_tables(start: date, end: date, source: Path) -> duckdb.DuckDBPyConnection:
    """The sealed tables' shape (`holdout.SEALED_TABLES`) for start..end, from `source`."""
    con = duckdb.connect()
    con.execute(f"ATTACH '{source}' AS ag (READ_ONLY)")
    for name, col in holdout.SEALED_TABLES.items():
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM ag.{name} "
                    f"WHERE {col} BETWEEN DATE '{start}' AND DATE '{end}'")
    return con


def load_window(start: date, end: date, agent: Path = holdout.AGENT,
                uncensored: bool = True, after_cutoff: bool = False) -> replay.Holdout:
    """`replay.load` for a window before the cutoff, from agent-visible data only.

    The policy stands on the day before `start`; the window's demand, stock
    and events come from the agent file. With `uncensored`, stock-out losses
    are added back as in `lost.uncensored_demand`, from history up to `end`.
    `after_cutoff` allows a later window when `agent` is the full warehouse
    (evaluation only: diagnostic windows in H2 2025).
    """
    if end > config.CUTOFF and not after_cutoff:
        raise ValueError(f"a backtest window must end by the cutoff, not {end}")
    day = start - timedelta(days=1)
    with holdout.agent_view(day, agent) as past, holdout.agent_view(end, agent) as now, \
            window_tables(start, end, agent) as win:
        ensure_baseline(past, day)

        def demand(hc, drugs, days):
            return lost.uncensored_units(now, drugs, days)

        return replay.load_from(past, win, start, end,
                                demand if uncensored else replay.recorded_demand)


@dataclass
class Basis:
    """What a policy is built from on one day: facts and the daily forecast."""

    day: date
    facts: pd.DataFrame       # one row per candidate drug, indexed by drug_code
    mean: np.ndarray          # forecast daily mean, drugs x days after `day`
    var: np.ndarray


def _bucket(x: pd.Series, edges: tuple[float, float], names: tuple[str, str, str]) -> pd.Series:
    return pd.Series(np.select([x <= edges[0], x <= edges[1]], names[:2], names[2]),
                     index=x.index).where(x.notna())


def facts(con: duckdb.DuckDBPyConnection, day: date) -> pd.DataFrame:
    """Per candidate drug, what's known on `day`.

    `con` should be as of `day`: the tiers and the pharmacist's baseline are
    read as the connection shows them.
    """
    ensure_baseline(con, day)
    codes = replay.candidates(con)
    d = common.d
    f = pd.DataFrame(index=pd.Index(codes, name="drug_code"))
    f["drug_name"] = pd.Series(dict(con.execute(
        "SELECT drug_code, drug_name FROM drugs").fetchall())).reindex(codes)
    f["tier"] = pd.Series(dict(con.execute(
        "SELECT drug_code, tier FROM demand_tiers").fetchall())).reindex(codes)
    f["lead_days"] = (replay.default_lead(con).reindex(codes)
                      .fillna(replay.LEAD_FALLBACK).astype(int))
    f["cadence"] = _bucket(f.lead_days, CADENCE_EDGES, ("short", "medium", "long"))

    rates = con.execute(f"""
        SELECT s.drug_code, s.date > {d(day - timedelta(days=TREND_DAYS[0]))} AS recent,
               sum(coalesce(dd.units, 0)) AS units, count(*) AS days
        FROM daily_stock s LEFT JOIN daily_demand dd USING (drug_code, date)
        WHERE s.location = 'total' AND s.closing > 0
          AND s.date > {d(day - timedelta(days=TREND_DAYS[1]))} AND s.date <= {d(day)}
        GROUP BY ALL""").df()
    r = rates.pivot_table(index="drug_code", columns="recent", values=["units", "days"],
                          aggfunc="sum").fillna(0)

    def get(k, rec):
        return r[(k, rec)] if (k, rec) in r else 0

    rate_recent = get("units", True) / get("days", True).where(lambda x: x > 0)
    rate_half = ((get("units", True) + get("units", False))
                 / (get("days", True) + get("days", False)).where(lambda x: x > 0))
    f["trend_ratio"] = (rate_recent / rate_half.where(rate_half > 0)).reindex(codes)
    f["trend"] = _bucket(f.trend_ratio, TREND_EDGES, ("falling", "flat", "rising"))

    last = dict(con.execute("SELECT drug_code, max(date) FROM daily_demand "
                            f"WHERE units > 0 AND date <= {d(day)} GROUP BY 1").fetchall())
    f["idle_days"] = [(day - last[c]).days if c in last else 9999 for c in codes]
    rev = con.execute(f"""
        SELECT drug_code, sum(revenue) AS revenue FROM daily_demand
        WHERE date > {d(day - timedelta(days=FACT_DAYS))} AND date <= {d(day)}
        GROUP BY 1""").df()
    f["revenue_182d"] = rev.set_index("drug_code")["revenue"].reindex(codes).fillna(0)
    f["price"] = replay.prices(con, day).reindex(codes).fillna(0)
    f["floor_rate"] = common.recent(con, day, codes).floor_rate.reindex(codes)

    sp = lost.spells(con, day)
    g = lost.groups(con)
    since = day - timedelta(days=FACT_DAYS)
    out = sp[(sp.kind == "stockout") & (sp.stop > since)]
    out = out.assign(in_window=[(b - max(a, since + timedelta(days=1))).days + 1
                                for a, b in zip(out.start, out.stop, strict=True)])
    by = out.groupby("drug_code")
    f["stockout_days_182d"] = by.in_window.sum().reindex(codes).fillna(0).astype(int)
    f["lost_revenue_182d"] = by.lost_revenue.sum().reindex(codes).fillna(0)
    f["long_stockout"] = (by.days.max().reindex(codes) > LONG_OUT).fillna(False)
    shelf = sp[(sp.kind == "shelfout") & (sp.stop > since)]
    f["shelfout_days_182d"] = (shelf.assign(n=[(b - max(a, since + timedelta(days=1))).days + 1
                                                for a, b in zip(shelf.start, shelf.stop,
                                                                strict=True)])
                               .groupby("drug_code").n.sum().reindex(codes).fillna(0).astype(int))
    sizes = g.group.value_counts()
    has_alt = f.index.map(g.group).map(sizes).fillna(0).to_numpy() > 1
    rates_by = lost.switch_rates(sp)
    rate = f.index.map(g.ingredients).map(rates_by["rate"]).to_series(index=f.index)
    f["switch_rate"] = np.where(has_alt, rate.fillna(rates_by.attrs["overall"]), np.nan)
    f["switching"] = np.where(~has_alt, "none",
                              np.where(f.switch_rate >= HIGH_SWITCH, "high", "low"))
    return f


def basis(con: duckdb.DuckDBPyConnection, day: date) -> Basis:
    f = facts(con, day)
    codes = list(f.index)
    n_days = int(f.lead_days.max()) + LIMITS["cycle_days"][1]
    weekly = fc.forecast(con, codes, day, math.ceil((n_days + 7) / 7))
    mean, var = plan._daily(weekly, codes, day, n_days)
    return Basis(day, f, mean, var)


FACTS = ("tier", "cadence", "trend", "long_stockout", "switching")
SEGMENT_FILTERS = ("tier", "cadence", "trend", "long_stockout", "switching", "min_idle_days")


def _check_rule(r: dict[str, Any], where: str) -> None:
    for k, (lo, hi) in LIMITS.items():
        if r.get(k) is not None and not lo <= r[k] <= hi:
            raise ValueError(f"{where}: {k} must be between {lo} and {hi}, not {r[k]}")


def assign(knobs: dict[str, Any], f: pd.DataFrame) -> pd.DataFrame:
    """Per drug: the rule it gets (name and the four values) and whether it is stopped."""
    base = {**DEFAULT, **{k: v for k, v in (knobs.get("default") or {}).items()
                          if v is not None}}
    _check_rule(base, "default")
    out = pd.DataFrame({k: base[k] for k in DEFAULT}, index=f.index)
    out["rule"], out["stop"], out["reason"] = "default", False, ""
    free = pd.Series(True, index=f.index)
    for i, seg in enumerate(knobs.get("segments") or []):
        name = seg.get("name") or f"segment {i + 1}"
        _check_rule(seg, name)
        hit = free.copy()
        for k in SEGMENT_FILTERS:
            v = seg.get(k)
            if v is None:
                continue
            hit &= f.idle_days >= v if k == "min_idle_days" else f[k] == v
        for k in DEFAULT:
            if seg.get(k) is not None:
                out.loc[hit, k] = seg[k]
        out.loc[hit, "rule"] = name
        free &= ~hit
    idle = knobs.get("stop_after_idle_days")
    if idle is not None:
        stale = f.idle_days >= idle
        out.loc[stale, ["stop", "rule", "reason"]] = [True, "stopped (idle)",
                                                        f"no sale in {idle}+ days"]
    for o in knobs.get("overrides") or []:
        code = o["drug"]
        if code not in out.index:
            raise ValueError(f"override: {code!r} is not a candidate drug (use its drug_code)")
        _check_rule(o, f"override {code}")
        for k in DEFAULT:
            if o.get(k) is not None:
                out.loc[code, k] = o[k]
        if o.get("stop") is not None:
            out.loc[code, "stop"] = bool(o["stop"])
        out.loc[code, ["rule", "reason"]] = ["override", o.get("reason") or ""]
    return out


def window_var_factor(days: np.ndarray, tier: pd.Series | np.ndarray,
                      rho: dict[str, float] | None) -> np.ndarray:
    """1 + rho_tier (w - 1) for a window of `days` (w = days / 7, at least 1 week).

    Weekly errors are positively correlated, so the variance of a w-week total
    grows faster than w. No rho (or a tier missing from it) gives 1.
    """
    if not rho:
        return np.ones(len(days))
    r = pd.Series(np.asarray(tier, dtype=object)).map(rho).fillna(0).to_numpy(float)
    return 1 + r * (np.maximum(np.asarray(days, float) / 7, 1) - 1)


def to_policy(knobs: dict[str, Any], b: Basis,
              rho: dict[str, float] | None = None) -> pd.DataFrame:
    """The replay policy (`plan.POLICY_COLUMNS`) the knobs give on the basis day.

    s and S are forecast quantiles as in `plan.reorder_rows`, at each drug's
    own service level and cycle; the floor levels are days of floor sales (the
    forecast daily mean where the floor sold nothing lately). A stopped drug
    never orders (s = -1). `rho` (tier -> rho) scales each window's summed
    variance by `window_var_factor`; None keeps the plain recipe.
    """
    f, a = b.facts, assign(knobs, b.facts)
    lead = f.lead_days.to_numpy(int)
    cycle = a.cycle_days.round().to_numpy(int)
    sl = a.service_level.to_numpy(float)
    m_l, v_l = plan._window_sum(b.mean, lead), plan._window_sum(b.var, lead)
    m_s, v_s = plan._window_sum(b.mean, lead + cycle), plan._window_sum(b.var, lead + cycle)
    if rho:
        v_l = v_l * window_var_factor(lead, f.tier, rho)
        v_s = v_s * window_var_factor(lead + cycle, f.tier, rho)
    s, S = np.zeros(len(f)), np.zeros(len(f))
    for q in np.unique(sl):
        i = sl == q
        s[i] = fc.demand_quantile(m_l[i], np.maximum(v_l[i], m_l[i]), q)
        S[i] = fc.demand_quantile(m_s[i], np.maximum(v_s[i], m_s[i]), q)
    S = np.maximum(S, np.where(m_s > 0, s + 1, s))
    rate = f.floor_rate.where(f.floor_rate > 0).fillna(
        pd.Series(b.mean[:, :7].mean(axis=1), index=f.index)).to_numpy()
    floor_min = a.floor_min_days.to_numpy() * rate
    floor_max = np.maximum(np.ceil(a.floor_max_days.to_numpy() * rate), np.ceil(floor_min) + 1)
    stop = a.stop.to_numpy(bool)
    return pd.DataFrame({"drug_code": f.index, "s": np.where(stop, -1, s),
                         "S": np.where(stop, 0, S), "lead_days": lead,
                         "floor_min": floor_min, "floor_max": floor_max})


def plain_policy(b: Basis) -> pd.DataFrame:
    """The tools' default output as a policy, no LLM: `reorder_rows` at 0.95 with the
    `transfer_rows` floor rule. It is `to_policy` with no knobs set."""
    return to_policy({}, b)


COLUMNS = ["stockout_days", "shelfout_days", "lost_revenue", "avg_stock_value"]


def matched(policies_: dict[str, pd.DataFrame], h: replay.Holdout,
            ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, float]]:
    """Summary (with the real ledger first), per-drug scores and scale factors, each
    policy scaled to the pharmacist's average stock value."""
    rows = {"pharmacist (ledger)": replay.summarise(replay.actual(h))}
    per, factors = {"pharmacist (ledger)": replay.actual(h)}, {}
    for name, p in policies_.items():
        m, k = replay.match_stock_value(p, h)
        run = replay.simulate(m, h)
        per[name] = replay.metrics(run.floor, run.store, run.sold, h)
        rows[name], factors[name] = replay.summarise(per[name]), k
    return pd.DataFrame(rows).T, per, factors


@dataclass
class Window:
    """One in-sample backtest: the basis on its policy day, its replay, the references."""

    day: date
    end: date
    basis: Basis
    h: replay.Holdout
    refs: dict[str, pd.DataFrame] = field(default_factory=dict)


def load_windows(agent: Path = holdout.AGENT,
                 windows: list[tuple[date, date]] = WINDOWS) -> list[Window]:
    out = []
    for day, end in windows:
        h = load_window(day + timedelta(days=1), end, agent)
        with holdout.agent_view(day, agent) as con:
            b = basis(con, day)
            refs = {"mimic": policies.mimic(con, day), "textbook": policies.textbook(con, day=day),
                    "plain tools": plain_policy(b)}
        out.append(Window(day, end, b, h, refs))
    return out


def _m(x: float) -> float:
    return round(float(x) / 1e6, 2)


def backtest(knobs: dict[str, Any], windows: list[Window], top: int = 8) -> dict[str, Any]:
    """The knobs on each window at matched stock, vs the references, with the biggest losers."""
    out = []
    for w in windows:
        p = to_policy(knobs, w.basis)
        table, per, factors = matched({"agent": p, **w.refs}, w.h)
        rows = [{"policy": n, "scale": round(factors.get(n, 1.0), 2),
                 "stockout_days": int(r.stockout_days), "shelfout_days": int(r.shelfout_days),
                 "lost_revenue_rp_m": _m(r.lost_revenue), "avg_stock_rp_m": _m(r.avg_stock_value)}
                for n, r in table.iterrows()]
        a = assign(knobs, w.basis.facts).reindex(w.h.drugs)
        ag, mi = per["agent"].set_index("drug_code"), per["mimic"].set_index("drug_code")
        lost_a = ag.lost_revenue_stockout + ag.lost_revenue_shelfout
        lost_m = mi.lost_revenue_stockout + mi.lost_revenue_shelfout
        by_rule = pd.DataFrame({"rule": a.rule, "stock": ag.avg_stock_value,
                                "agent_out": ag.stockout_days, "mimic_out": mi.stockout_days,
                                "agent_lost": lost_a, "mimic_lost": lost_m}).groupby("rule").agg(
            drugs=("stock", "size"), stock=("stock", "sum"), agent_out=("agent_out", "sum"),
            mimic_out=("mimic_out", "sum"), agent_lost=("agent_lost", "sum"),
            mimic_lost=("mimic_lost", "sum"))
        f = w.basis.facts.reindex(w.h.drugs)
        losers = pd.DataFrame({
            "drug_code": w.h.drugs, "drug_name": f.drug_name.to_numpy(), "tier": f.tier.to_numpy(),
            "rule": a.rule.to_numpy(), "agent_stockout_days": ag.stockout_days.to_numpy(),
            "mimic_stockout_days": mi.stockout_days.to_numpy(),
            "agent_lost_rp_k": (lost_a / 1e3).round().to_numpy(),
            "mimic_lost_rp_k": (lost_m / 1e3).round().to_numpy(),
            "gap": (lost_a - lost_m).to_numpy()}).sort_values("gap", ascending=False)
        out.append({
            "window": f"{w.day + timedelta(days=1)} to {w.end}", "policy_day": w.day.isoformat(),
            "drugs_scored": len(w.h.drugs),
            "agent_stock_before_scaling_rp_m": _m(replay.stock_value(p, w.h)),
            "table": rows,
            "by_rule": [{"rule": r, "drugs": int(x.drugs),
                         "stock_share_pct": round(100 * x.stock / by_rule.stock.sum(), 1),
                         "agent_stockout_days": int(x.agent_out),
                         "mimic_stockout_days": int(x.mimic_out),
                         "agent_lost_rp_m": _m(x.agent_lost), "mimic_lost_rp_m": _m(x.mimic_lost)}
                        for r, x in by_rule.iterrows()],
            "biggest_losers_vs_mimic": common.clean(
                losers[losers.gap > 0].head(top).drop(columns="gap").to_dict("records")),
        })
    return {"note": "Every policy is rescaled (scale) to the pharmacist's average stock value "
                    "in the window. Demand is uncensored (stock-out losses added back). Lost "
                    "revenue counts stock-outs and shelf-outs, in rupiah millions.",
            "windows": out}


PLANNER_SYSTEM = """\
You are the stock planner for one retail pharmacy in Indonesia. Today is {as_of}. \
You set the reorder and shop-floor policy for the next six months for its {n} fast and \
regular drugs. You can't see anything after today.

The goal: fewer stock-out days, fewer shelf-out days (shop floor empty while the \
upstairs store has stock) and less lost revenue than the pharmacist, at the same stock \
value. Every policy is rescaled to hold the pharmacist's average stock value before it \
is compared, so raising stock everywhere gains nothing. What counts is where the stock \
goes: which drugs get more cover and which get less.

You don't write rows per drug. You set knobs:
- default: service_level (chance lead-time demand is covered, 0.5-0.995), cycle_days \
(S covers the lead time plus this many days, 0-60), floor_min_days and floor_max_days \
(refill the shop floor from the store when it holds floor_min_days of floor sales or less, \
up to floor_max_days).
- segments: rules on facts per drug, the first match wins. Facts: tier (fast, regular); \
cadence (main supplier delivers every <=3 days short, <=7 medium, else long); trend \
(last 8 weeks vs half-year sales rate: falling <0.75x, rising >1.25x, else flat); \
long_stockout (a stock-out over 14 days in the last half-year); switching (when the drug \
ran out, did customers buy a same-ingredient alternative: high >=30%, low, or none = no \
alternative); min_idle_days (no sale for at least this many days). Null means any.
- stop_after_idle_days: stop reordering drugs with no sale for this long (null: never).
- overrides: a few drugs by drug_code, each with a reason.

The no-knobs default is the plain tool policy (service 0.95, cycle 14, floor 2 to 7 \
days). Steps: look at segment_facts, form a view, call backtest_policy to test knobs on \
past half-years, adjust, then call submit_policy once with your final knobs and a short \
rationale. Submit only knobs you have backtested. The backtest is in-sample: prefer \
simple rules you can explain over fitting it. Be done within {steps} tool calls."""

RULE_PROPS = {
    "service_level": {"type": ["number", "null"], "description": "0.5-0.995"},
    "cycle_days": {"type": ["integer", "null"], "description": "0-60"},
    "floor_min_days": {"type": ["number", "null"], "description": "0-14"},
    "floor_max_days": {"type": ["number", "null"], "description": "1-30"},
}


def _obj(props: dict[str, Any], **kw: Any) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "required": list(props),
            "properties": props, **kw}


def _nullable_enum(values: list[Any], desc: str) -> dict[str, Any]:
    return {"type": ["string", "null"], "enum": [*values, None], "description": desc}


KNOBS_SCHEMA = _obj({
    "default": _obj(RULE_PROPS, description="Null values keep the plain defaults."),
    "segments": {"type": "array", "items": _obj({
        "name": {"type": "string"},
        "tier": _nullable_enum(["fast", "regular"], "Match this tier, or null for any."),
        "cadence": _nullable_enum(["short", "medium", "long"], "Supplier cadence."),
        "trend": _nullable_enum(["falling", "flat", "rising"], "Demand trend."),
        "long_stockout": {"type": ["boolean", "null"]},
        "switching": _nullable_enum(["high", "low", "none"], "Switching to alternatives."),
        "min_idle_days": {"type": ["integer", "null"]},
        **RULE_PROPS})},
    "stop_after_idle_days": {"type": ["integer", "null"]},
    "overrides": {"type": "array", "items": _obj({
        "drug": {"type": "string", "description": "drug_code"},
        "stop": {"type": ["boolean", "null"]},
        "service_level": RULE_PROPS["service_level"], "cycle_days": RULE_PROPS["cycle_days"],
        "reason": {"type": "string"}})},
})


@dataclass
class Planner:
    """State the planner tools share: the backtest windows and the day's basis."""

    windows: list[Window]
    basis: Basis
    submitted: dict[str, Any] | None = None


def segment_summary(f: pd.DataFrame, by: list[str]) -> list[dict[str, Any]]:
    g = f.assign(drugs=1).groupby(by, dropna=False).agg(
        drugs=("drugs", "sum"), revenue=("revenue_182d", "sum"),
        stockout_days=("stockout_days_182d", "sum"), lost=("lost_revenue_182d", "sum"),
        shelfout_days_182d=("shelfout_days_182d", "sum"),
        switch_rate=("switch_rate", "mean"), idle_90=("idle_days", lambda x: int((x >= 90).sum())))
    g["revenue_share_pct"] = (100 * g.revenue / g.revenue.sum()).round(1)
    g["lost_revenue_rp_m"] = (g.lost / 1e6).round(2)
    return common.clean(g.drop(columns=["revenue", "lost"]).rename(columns={
        "stockout_days": "stockout_days_182d", "lost_revenue_rp_m": "lost_revenue_182d_rp_m",
        "idle_90": "no_sale_90d"}).reset_index().to_dict("records"))


def planner_tools(ctx: Planner) -> dict[str, registry.Tool]:
    """The planner's tools: three of its own plus a few read-only agent tools."""
    def segment_facts(con, by: list[str]) -> dict[str, Any]:
        bad = set(by) - set(FACTS)
        if bad:
            raise ValueError(f"group by {', '.join(FACTS)}, not {sorted(bad)}")
        rows = ({"by_" + "_and_".join(by): segment_summary(ctx.basis.facts, by)} if by else
                {f"by_{k}": segment_summary(ctx.basis.facts, [k]) for k in FACTS})
        return {"as_of": ctx.basis.day.isoformat(), "drugs": len(ctx.basis.facts), **rows,
                "note": "Over the last 182 days: revenue share, stock-out days and lost "
                        "revenue (rupiah millions). switch_rate: mean share of lost sales "
                        "that went to an alternative, drugs with one."}

    def backtest_policy(con, knobs: dict[str, Any]) -> dict[str, Any]:
        return backtest(knobs, ctx.windows)

    def submit_policy(con, knobs: dict[str, Any], rationale: str) -> dict[str, Any]:
        a = assign(knobs, ctx.basis.facts)
        ctx.submitted = {"knobs": knobs, "rationale": rationale}
        return {"ok": True, "drugs": len(a), "stopped": int(a.stop.sum()),
                "by_rule": a.rule.value_counts().to_dict()}

    own = {
        "segment_facts": registry.Tool(
            "segment_facts", "Drug counts and the last half-year's revenue share, stock-out "
            "and shelf-out days, lost revenue and switching, grouped by facts. An empty `by` "
            "gives every fact on its own; one or two facts give their cross.",
            _obj({"by": {"type": "array", "items": {"type": "string", "enum": list(FACTS)}}}),
            segment_facts),
        "backtest_policy": registry.Tool(
            "backtest_policy", "Replay the knobs on two past half-years (policy formed at "
            "their start, with what was known then) against the pharmacist's mimic, the "
            "textbook rule and the plain tool policy, all at the pharmacist's stock value. "
            "Also by rule, and the drugs that lost most vs the mimic.",
            _obj({"knobs": KNOBS_SCHEMA}), backtest_policy),
        "submit_policy": registry.Tool(
            "submit_policy", "Submit the final knobs and a short rationale (3-5 points). "
            "Ends the session.",
            _obj({"knobs": KNOBS_SCHEMA, "rationale": {"type": "string"}}), submit_policy),
    }
    shared = {n: registry.TOOLS[n] for n in ("find_drug", "stockout_history", "run_sql")}
    return {**own, **shared}


def run_planner(model: str = llm.BULK_MODEL, *, run_id: str | None = None,
                max_steps: int = 14, effort: str = "low", ledger: Path | None = None,
                client: Any = None, agent: Path = holdout.AGENT, out: Path = OUT,
                ctx: Planner | None = None, cap: float | None = None) -> Path:
    """Run the planner on the cutoff and freeze what it submits to out/<run_id>/."""
    run_id = run_id or f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{model}"
    with holdout.agent_view(config.CUTOFF, agent) as con:
        if ctx is None:
            ctx = Planner(load_windows(agent), basis(con, config.CUTOFF))
        system = PLANNER_SYSTEM.format(as_of=config.CUTOFF.isoformat(),
                                       n=len(ctx.basis.facts), steps=max_steps - 2)
        t = loop.run("Set the policy for the next six months.", as_of=config.CUTOFF,
                     model=model, effort=effort, max_steps=max_steps, con=con, client=client,
                     ledger=ledger, cap=cap, purpose=f"contest:{run_id}", system=system,
                     tools=planner_tools(ctx), final_tools=("submit_policy",),
                     max_output_tokens=2_000)
    d = out / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "transcript.json").write_text(json.dumps(t.to_dict(), indent=1, default=str))
    if ctx.submitted is None:
        raise RuntimeError(f"the planner stopped ({t.stopped}) without submitting; see {d}")
    knobs = ctx.submitted["knobs"]
    to_policy(knobs, ctx.basis).to_csv(d / "policy.csv", index=False)
    assign(knobs, ctx.basis.facts).to_csv(d / "rules.csv")
    (d / "knobs.json").write_text(json.dumps({
        "run_id": run_id, "model": model, "as_of": config.CUTOFF.isoformat(),
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"), "usd": round(t.usd, 6),
        **ctx.submitted}, indent=1))
    (d / "in_sample.json").write_text(json.dumps(backtest(knobs, ctx.windows), indent=1))
    return d


def score_frozen(run_id: str, out: Path = OUT, agent: Path = holdout.AGENT,
                 sealed: Path = holdout.SEALED) -> pd.DataFrame:
    """Score a frozen policy on the sealed holdout and print the contest table.

    Every policy is matched to the pharmacist's average stock value, under
    uncensored demand (the main score) and recorded demand (a check). Each
    call is logged to `RUNS_LOG`, so every holdout run is on record.
    """
    d = out / run_id
    agent_policy = pd.read_csv(d / "policy.csv", dtype={"drug_code": str})
    with holdout.agent_view(config.CUTOFF, agent) as con:
        refs = {"mimic": policies.mimic(con), "textbook": policies.textbook(con),
                "plain tools": plain_policy(basis(con, config.CUTOFF))}
    tables = []
    for name, demand in (("uncensored", lost.uncensored_demand(agent)),
                         ("recorded", replay.recorded_demand)):
        h = replay.load(agent, sealed, demand=demand)
        table, _, factors = matched({**refs, "agent": agent_policy}, h)
        table = table[COLUMNS + ["stockout_days_long", "drugs"]].assign(
            scale=pd.Series(factors), demand=name)
        tables.append(table)
        curve = pd.concat({n: replay.tradeoff(p, h) for n, p in
                           (("agent", agent_policy), ("mimic", refs["mimic"]))})
        curve.to_csv(d / f"tradeoff_{name}.csv")
    result = pd.concat(tables)
    result.to_csv(d / "holdout_score.csv")
    RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUNS_LOG.open("a") as fh:
        fh.write(json.dumps({"run_id": run_id,
                             "scored_at": datetime.now(UTC).isoformat(timespec="seconds"),
                             "rows": json.loads(result.reset_index(names="policy")
                                                .to_json(orient="records"))}) + "\n")
    print(show(result))
    return result


def show(result: pd.DataFrame) -> str:
    lines = []
    for name, t in result.groupby("demand", sort=False):
        lines.append(f"\nH2 2025 holdout, {name} demand, at matched stock "
                     f"({int(t.drugs.iloc[0])} drugs)")
        lines.append(f"{'':22}{'stock-out d':>12}{'shelf-out d':>12}{'lost Rp M':>11}"
                     f"{'stock Rp M':>11}{'scale':>7}")
        for n, r in t.iterrows():
            scale = "" if pd.isna(r.scale) else f"{r.scale:.2f}"
            lines.append(f"{n:22}{r.stockout_days:12,.0f}{r.shelfout_days:12,.0f}"
                         f"{r.lost_revenue / 1e6:11.1f}{r.avg_stock_value / 1e6:11.1f}{scale:>7}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="irpd contest")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="run the planner on the cutoff and freeze its policy")
    p.add_argument("--model", default=llm.BULK_MODEL)
    p.add_argument("--effort", default="low")
    p.add_argument("--max-steps", type=int, default=14)
    p.add_argument("--ledger", type=Path, default=None)
    p.add_argument("--cap", type=float, default=None, help="total spend cap in USD")
    s = sub.add_parser("score", help="score a frozen run on the sealed holdout")
    s.add_argument("run_id")
    args = ap.parse_args(argv)
    if args.command == "plan":
        d = run_planner(args.model, effort=args.effort, max_steps=args.max_steps,
                        ledger=args.ledger, cap=args.cap)
        print(f"frozen: {d}")
        print(json.dumps(json.loads((d / "in_sample.json").read_text())["windows"], indent=1))
    else:
        score_frozen(args.run_id)


if __name__ == "__main__":
    main()
