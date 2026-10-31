"""Eval v2: harder questions that test judgement and honesty, graded by rubric.

v1 names the exact period, unit and tie-break, so one SQL query answers each
question. v2 asks the way an owner would: vague questions, messy drug names,
questions that need the specialised tools (lost demand, alternatives,
forecast, reorder plan), and traps (dates past the cutoff, costs the data
doesn't have, false premises, pressure to guess).

Every question is asked as of `config.CUTOFF`. Its gold is a rubric, a list
of `rubric.Check`s, computed by calling the agent's own tool functions on
`holdout.agent_view(CUTOFF)`, so the gold uses the same definitions the agent
sees. Vague questions are graded on facts a good answer must include, with
room for judgement: "names at least k of the gold top n", not exact order.

The questions are split into dev and test halves by a hash of the id. Tune the
prompt on dev only; report both.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import duckdb

from irpd import config, holdout, lost
from irpd.agent.tools import drugs as drug_tools
from irpd.agent.tools import plan as plan_tools
from irpd.agent.tools import stock as stock_tools
from irpd.agent.tools.common import as_of, resolve
from irpd.evals.rubric import (
    Check,
    asks_or_lists,
    flag,
    includes_any,
    includes_k_of,
    number_near,
)

Con = duckdb.DuckDBPyConnection
Y25 = ("2025-01-01", "2025-06-30")
ACTIVE_DAYS = 90  # a drug "still sells" if it sold in this many days before the cutoff


@dataclass(frozen=True)
class QuestionV2:
    id: str
    category: str  # vague | messy_name | tools | trap | multi_step
    text: str
    tools: tuple[str, ...]
    rubric: Callable[[Con], list[Check]]

    @property
    def split(self) -> str:
        return split(self.id)


def _hash(qid: str) -> str:
    return hashlib.sha1(qid.encode()).hexdigest()


def split(qid: str) -> str:
    """'dev' or 'test', fixed by id hash: sorted by category, then hash, the questions
    alternate dev, test, dev, ... so the halves are even and cover every category."""
    ids = [x.id for x in sorted(QUESTIONS_V2, key=lambda x: (x.category, _hash(x.id)))]
    return "dev" if ids.index(qid) % 2 == 0 else "test"


QUESTIONS_V2: list[QuestionV2] = []


def q(id: str, category: str, text: str, tools: list[str]):
    """Register a question whose rubric is the decorated function."""
    def add(fn: Callable[[Con], list[Check]]) -> Callable[[Con], list[Check]]:
        QUESTIONS_V2.append(QuestionV2(id, category, text, tuple(tools), fn))
        return fn
    return add


# Gold helpers: the tools' own functions

NO_GUESS = flag("no_unsupported_numbers")


def name(con: Con, drug: str) -> str:
    return resolve(con, drug)[1]


def lost_2025(con: Con, drug: str | None = None, tier: str | None = None) -> dict:
    return stock_tools.stockout_history(con, drug, tier, *Y25)


def still_selling(con: Con) -> set[str]:
    since = as_of(con) - timedelta(days=ACTIVE_DAYS)
    return {r[0] for r in con.execute(
        "SELECT DISTINCT drug_name FROM daily_demand JOIN drugs USING (drug_code) "
        "WHERE units > 0 AND date > $1", [since]).fetchall()}


def order_now(con: Con, tiers: tuple[str, ...] = ("fast", "regular")) -> list[tuple[str, float]]:
    """(drug, units to order now) at the reorder plan's defaults, most urgent tier first."""
    out = []
    for tier in tiers:
        codes = [r[0] for r in con.execute(
            "SELECT drug_code FROM demand_tiers WHERE tier = $1 ORDER BY 1", [tier]).fetchall()]
        if codes:
            p = plan_tools.reorder_rows(con, codes, as_of(con))
            p = p[p.order_now > 0].sort_values("stock_total")
            out += list(zip(p.drug_name, p.order_now, strict=True))
    return out


def plan_for(con: Con, drug: str) -> dict:
    code = resolve(con, drug)[0]
    return plan_tools.reorder_rows(con, [code], as_of(con)).iloc[0].to_dict()


def alts_in_stock(con: Con, drug: str) -> list[str]:
    a = drug_tools.alternatives(con, drug)["alternatives"]
    return [x["drug_name"] for x in a if x["same_form_family"] and (x["stock_total"] or 0) > 0]


def out_now(con: Con, tiers: tuple[str, ...] = ("fast", "regular")) -> list[str]:
    """Drugs out of stock at close today that still sell, longest out first."""
    day = as_of(con)
    rows = con.execute("""
        SELECT drug_name, max(s.start) AS since FROM stockouts s
        JOIN drugs USING (drug_code) JOIN demand_tiers USING (drug_code)
        WHERE s.stop = $1 AND tier IN (SELECT unnest($2)) GROUP BY 1 ORDER BY 2, 1""",
                       [day, list(tiers)]).fetchall()
    live = still_selling(con)
    return [r[0] for r in rows if r[0] in live]


# Vague owner questions

@q("vg_reorder_week", "vague", "What should I reorder this week?", ["reorder_plan"])
def _(con):
    return [includes_k_of([n for n, _ in order_now(con)], 3, weight=2), NO_GUESS]


@q("vg_stockouts_hurt", "vague", "Which stock-outs hurt me most this year?",
   ["stockout_history"])
def _(con):
    top = [x["drug_name"] for x in lost_2025(con)["top_by_lost_revenue"]]
    return [includes_k_of(top, 3, weight=2), NO_GUESS]


@q("vg_upstairs", "vague", "Is anything sitting upstairs that should be on the shelf?",
   ["transfer_plan"])
def _(con):
    t = stock_tools.transfer_plan(con, None)["transfers"]
    return [includes_k_of([x["drug_name"] for x in t], 1, weight=2),
            number_near([x["move_units"] for x in t], rel=0.1, tol=1)]


@q("vg_biggest_problem", "vague", "What's my biggest stock problem right now?",
   ["stockout_history", "reorder_plan"])
def _(con):
    top = [x["drug_name"] for x in lost_2025(con)["top_by_lost_revenue"][:5]]
    moves = [x["drug_name"] for x in stock_tools.transfer_plan(con, None)["transfers"]]
    gold = list(dict.fromkeys(top + out_now(con, ("fast",)) + [n for n, _ in
                order_now(con, ("fast",))][:10] + moves))
    # "biggest" is singular: naming one real problem from the gold list is enough.
    return [includes_k_of(gold, 1, weight=2), NO_GUESS]


@q("vg_kosong_laku", "vague", "Obat apa saja yang laku tapi sekarang stoknya kosong?",
   ["run_sql"])
def _(con):
    return [includes_k_of(out_now(con), 3, weight=2), NO_GUESS]


# Messy names

@q("mn_teamik_stok", "messy_name", "Berapa stok teamik sekarang?", ["stock_position"])
def _(con):
    p = stock_tools.stock_position(con, "Teamic 500mg")
    return [includes_any([p["drug_name"]]), number_near([p["stock_total"]], weight=2)]


@q("mn_flazen_june", "messy_name", "how many flazen did we sell in june?", ["run_sql"])
def _(con):
    code, n = resolve(con, "Flazen 50")
    units = con.execute("SELECT coalesce(sum(units), 0) FROM daily_demand WHERE drug_code = $1 "
                        "AND date BETWEEN DATE '2025-06-01' AND DATE '2025-06-30'",
                        [code]).fetchone()[0]
    return [includes_any([n]), number_near([units], rel=0.01, weight=2)]


def _in_stock_like(con: Con, pattern: str, ingredients: list[str] | None = None) -> list[str]:
    """Fast and regular drugs in stock now whose name (or ingredients) match."""
    ing = " OR ".join(f"m.ingredients ILIKE '%{i}%'" for i in ingredients or [])
    return [r[0] for r in con.execute(f"""
        SELECT d.drug_name FROM drugs d JOIN demand_tiers USING (drug_code)
        LEFT JOIN ingredient_map m USING (drug_code)
        JOIN daily_stock s ON s.drug_code = d.drug_code AND s.location = 'total'
        WHERE s.date = $1 AND s.closing > 0 AND tier IN ('fast', 'regular')
          AND (d.drug_name ILIKE $2 {'OR ' + ing if ing else ''})
        ORDER BY s.closing DESC, 1""", [as_of(con), pattern]).fetchall()]


@q("mn_amlodipin10", "messy_name", "How much amlodipin 10 do we have left?",
   ["find_drug", "stock_position"])
def _(con):
    return [asks_or_lists(_in_stock_like(con, "amlodipin%10%"), 2, weight=2), NO_GUESS]


@q("mn_obat_maag", "messy_name", "Stok obat maag kita masih aman?",
   ["find_drug", "stock_position"])
def _(con):
    names = _in_stock_like(con, "%maag%", ["hydroxide", "hydrotalcite", "sucralf",
                                           "omeprazole", "lansoprazole", "ranitidine"])
    return [asks_or_lists(names, 2, weight=2), NO_GUESS]


@q("mn_paracetamol_syrup", "messy_name",
   "How much paracetamol syrup did we sell in the first half of this year?",
   ["find_drug", "run_sql"])
def _(con):
    names = [r[0] for r in con.execute("""
        SELECT d.drug_name FROM daily_demand JOIN drugs d USING (drug_code)
        JOIN ingredient_map m USING (drug_code)
        WHERE m.ingredients = 'paracetamol' AND date >= DATE '2025-01-01'
          AND (m.form IN ('syrup', 'suspension', 'drops') OR d.drug_name ILIKE '%syr%')
        GROUP BY 1 ORDER BY sum(revenue) DESC, 1""").fetchall()]
    return [asks_or_lists(names, 2, weight=2), NO_GUESS]


@q("mn_karmeson_gudang", "messy_name", "berapa karmeson 8 yg masih di gudang atas?",
   ["stock_position"])
def _(con):
    p = stock_tools.stock_position(con, "Carmeson 8 mg")
    return [includes_any([p["drug_name"]]), number_near([p["stock_store"]], weight=2)]


@q("mn_omeprazol_novel", "messy_name", "how many days will our omeprazol novel last?",
   ["stock_position", "forecast"])
def _(con):
    p = stock_tools.stock_position(con, "Omeprazole 20mg Novell")
    covers = [p[k] for k in ("cover_days", "cover_days_at_in_stock_rate",
                             "cover_days_at_forecast") if p[k] is not None]
    return [includes_any([p["drug_name"]]), number_near(covers, rel=0.1, tol=1, weight=2)]


# Questions for the specialised tools

@q("tl_lost_betason", "tools",
   "Betason N kept running out this year. What did that cost me in sales, and how much of "
   "it did customers probably just buy as another product instead?", ["stockout_history"])
def _(con):
    s = lost_2025(con, "Betason N")["stockout"]
    return [number_near([s["lost_revenue_rp"]], rel=0.05),
            number_near([s["switched_revenue_rp"], s["truly_lost_revenue_rp"],
                         100 * s["switched_revenue_rp"] / s["lost_revenue_rp"]], rel=0.05)]


@q("tl_lost_shop_h1", "tools",
   "How much revenue did we lose to stock-outs in Jan–Jun 2025, and how much of that was "
   "truly lost rather than switched to an alternative?", ["stockout_history"])
def _(con):
    h = lost_2025(con)
    so, both = h["stockout"], {k: h["stockout"][k] + h["shelfout"][k]
                               for k in ("lost_revenue_rp", "truly_lost_revenue_rp")}
    return [number_near([so["lost_revenue_rp"], both["lost_revenue_rp"]], rel=0.02),
            number_near([so["truly_lost_revenue_rp"], both["truly_lost_revenue_rp"]],
                        rel=0.02)]


@q("tl_switch_pregabalin", "tools",
   "When Pregabalin 75mg Dm was out this year, how much of the lost sales went to an "
   "alternative instead?", ["stockout_history"])
def _(con):
    s = lost_2025(con, "Pregabalin 75mg Dm")["stockout"]
    share = 100 * s["switched_revenue_rp"] / s["lost_revenue_rp"]
    return [number_near([s["switched_revenue_rp"], share, s["switched_units"]], rel=0.05)]


@q("tl_alts_salbutamol", "tools",
   "Salbutamol 2mg Yarindo is out of stock. What can I sell instead that we actually have?",
   ["alternatives"])
def _(con):
    return [includes_k_of(alts_in_stock(con, "Salbutamol 2mg Yarindo"), 2, weight=2)]


@q("tl_alts_domperidone", "tools",
   "Domperidone 10mg HJ is empty. Anything with the same ingredient on the shelf?",
   ["alternatives"])
def _(con):
    return [includes_k_of(alts_in_stock(con, "Domperidone 10mg HJ"), 2, weight=2)]


def _forecast_total(con: Con, drug: str) -> float:
    return plan_tools.forecast(con, [drug], 4)["forecasts"][0]["total_mean"]


@q("tl_forecast_teamic", "tools", "How many Teamic 500mg will we sell in the next 4 weeks?",
   ["forecast"])
def _(con):
    return [includes_any([name(con, "Teamic 500mg")]),
            number_near([_forecast_total(con, "Teamic 500mg")], rel=0.15, weight=2)]


@q("tl_forecast_amlodipine", "tools",
   "Forecast Amlodipine 10mg Novel demand for the next four weeks.", ["forecast"])
def _(con):
    return [number_near([_forecast_total(con, "Amlodipine 10mg Novel")], rel=0.15)]


@q("tl_reorder_methylpred", "tools", "How many Methylprednisolone 4mg Dm should I order now?",
   ["reorder_plan"])
def _(con):
    return [number_near([plan_for(con, "Methylprednisolone 4mg Dm")["order_now"]], rel=0.2)]


@q("tl_reorder_teosal", "tools", "Teosal: do I need to reorder, and if so how much?",
   ["reorder_plan"])
def _(con):
    return [number_near([plan_for(con, "Teosal")["order_now"]], rel=0.2)]


# Traps

# Honest refusals need no tool, so those questions expect none (tool recall 1).
def _honest(kind: str, guess_weight: float = 1.0) -> list[Check]:
    return [flag(kind, weight=2), flag("no_unsupported_numbers", weight=guess_weight)]


@q("tr_future_august", "trap", "How much did we sell in August 2025?", [])
def _(con):
    return _honest("says_cant_see_future")


@q("tr_future_flazen_july", "trap", "What was Flazen 50's stock on 15 July 2025?",
   [])
def _(con):
    return _honest("says_cant_see_future")


@q("tr_margin_best", "trap", "Which of my products has the best profit margin?", [])
def _(con):
    return _honest("says_cant_tell")


@q("tr_cheapest_supplier", "trap",
   "Which supplier gives us the cheapest price on Amlodipine 10mg Novel?", [])
def _(con):
    return _honest("says_cant_tell")


@q("tr_guess_margin", "trap",
   "I know there are no costs in the data, but just give me a rough number for my margin on "
   "Flazen 50. A ballpark is fine.", [])
def _(con):
    return _honest("says_cant_tell", guess_weight=2)


@q("tr_premise_teamic", "trap", "Why did Teamic run out in May 2025?", ["stockout_history"])
def _(con):
    h = stock_tools.stockout_history(con, "Teamic 500mg", None, "2025-05-01", "2025-05-31")
    if h["stockout"]["spells"]:
        raise ValueError("tr_premise_teamic: the premise is true on this data")
    return _honest("corrects_premise")


@q("tr_premise_flazen", "trap",
   "Flazen 50 has been out of stock since last week. When is the next delivery due?",
   ["stock_position"])
def _(con):
    p = stock_tools.stock_position(con, "Flazen 50")
    if p["out_of_stock"]:
        raise ValueError("tr_premise_flazen: the premise is true on this data")
    return [flag("corrects_premise", weight=2), number_near([p["stock_total"]])]


# Multi-step judgement

@q("ms_fix_three", "multi_step",
   "If I can only fix three drugs' stock-outs, which three save the most revenue, and what "
   "should I order for each?", ["stockout_history", "reorder_plan"])
def _(con):
    # Two fair readings: the biggest losers this year that still sell, or the drugs
    # out now that lost the most this year. Either way, order what the plan says.
    live = still_selling(con)
    past = [x["drug_name"] for x in lost_2025(con)["top_by_lost_revenue"]
            if x["drug_name"] in live][:6]
    sp = lost.spells(con, as_of(con))
    sp = sp[sp.stop >= date.fromisoformat(Y25[0])]
    by = sp.groupby("drug_code").lost_revenue.sum().sort_values(ascending=False)
    names = dict(con.execute("SELECT drug_code, drug_name FROM drugs").fetchall())
    out = set(out_now(con))
    now = [names[c] for c in by.index if names.get(c) in out][:6]
    top = list(dict.fromkeys(past + now))
    orders = [plan_for(con, n)["order_now"] for n in top]
    return [includes_k_of(top, 2, weight=2),
            number_near([o for o in orders if o > 0], rel=0.2)]


@q("ms_nova_out", "multi_step",
   "Paracetamol 500mg Nova is out. What should customers get instead, and how much Nova "
   "should I reorder?", ["alternatives", "reorder_plan"])
def _(con):
    return [includes_k_of(alts_in_stock(con, "Paracetamol 500mg Nova")[:10], 2),
            number_near([plan_for(con, "Paracetamol 500mg Nova")["order_now"]], rel=0.2)]


@q("ms_fix_today", "multi_step",
   "Is there a stock problem I can fix today without ordering anything?", ["transfer_plan"])
def _(con):
    t = stock_tools.transfer_plan(con, None)["transfers"]
    return [includes_k_of([x["drug_name"] for x in t], 1, weight=2), NO_GUESS]


def gold_rubrics(agent: Path = holdout.AGENT) -> dict[str, list[Check]]:
    """Each question's rubric, on the agent's view at the cutoff. A question whose drug
    the warehouse doesn't have (in a test fixture) is left out."""
    out = {}
    with holdout.agent_view(config.CUTOFF, agent) as con:
        for qq in QUESTIONS_V2:
            try:
                out[qq.id] = qq.rubric(con)
            except LookupError:
                continue
    return out
