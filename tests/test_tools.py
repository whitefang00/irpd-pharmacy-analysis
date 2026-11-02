"""The agent tools, on the made-up pharmacy in `tools_fixture.py`."""

import json
import math
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
import tools_fixture
from test_as_of import cut

from irpd import holdout, warehouse
from irpd.agent import registry
from irpd.agent.tools import reorder_rows, sql, to_policy
from irpd.tables import ingredients

CUTOFF = date(2025, 6, 30)


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> tuple[Path, Path]:
    """(fixture source dir, full warehouse built from it)."""
    tmp = tmp_path_factory.mktemp("tools")
    src = tools_fixture.write(tmp / "src")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ingredients, "MAP", src / "ingredient_map.csv")
        db = warehouse.build(src, tmp / "full.duckdb")
    return src, db


@pytest.fixture()
def con(built):
    c = holdout.agent_view(CUTOFF, built[1])
    yield c
    c.close()


def call(con, tool, **args):
    out = registry.call(tool, args, con)
    json.dumps(out)  # must be plain JSON
    return out


# Contract

def strict_errors(schema: dict, path: str = "$") -> list[str]:
    """Where a schema breaks OpenAI strict mode: objects must list every property as
    required and forbid additional ones, all the way down."""
    errors = []
    types = schema.get("type")
    types = types if isinstance(types, list) else [types]
    if "object" in types:
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is not False:
            errors.append(f"{path}: additionalProperties must be false")
        if set(schema.get("required", [])) != set(props):
            errors.append(f"{path}: every property must be required")
        for k, v in props.items():
            errors += strict_errors(v, f"{path}.{k}")
    if "array" in types:
        errors += strict_errors(schema.get("items", {}), f"{path}[]")
    if "enum" in schema and None in schema["enum"] and "null" not in types:
        errors.append(f"{path}: null in enum but not in type")
    return errors


def test_every_tool_schema_is_strict():
    expected = {"run_sql", "forecast", "stock_position", "stockout_history", "alternatives",
                "price_history", "reorder_plan", "transfer_plan", "find_drug"}
    assert expected <= set(registry.TOOLS)
    for t in registry.openai_tools():
        assert t["strict"] and t["parameters"]["type"] == "object"
        assert strict_errors(t["parameters"]) == [], t["name"]


def test_strict_check_catches_a_loose_schema():
    loose = {"type": "object", "properties": {"a": {"type": "object", "properties": {}}}}
    assert len(strict_errors(loose)) == 3


# run_sql

@pytest.mark.parametrize("query", [
    "INSERT INTO drugs VALUES ('X')", "DELETE FROM drugs", "DROP TABLE drugs",
    "CREATE TABLE t AS SELECT 1", "ATTACH 'x.duckdb' AS x", "COPY drugs TO 'x.csv'",
    "PRAGMA database_list", "SET threads = 1", "SELECT 1; SELECT 2",
    "SELECT 1; DROP TABLE drugs", "SELECT * FROM wh.daily_stock",
    "SELECT * FROM query('SELECT * FROM wh.drugs')", "not sql at all",
])
def test_run_sql_rejects_anything_but_one_select(con, query):
    with pytest.raises(ValueError):
        sql.check(query)
    assert "error" in call(con, "run_sql", query=query)


def test_run_sql_cannot_read_files(con):
    assert "error" in call(con, "run_sql", query="SELECT * FROM read_csv('/etc/hosts')")


def test_run_sql_caps_rows_and_reads_views(con):
    out = call(con, "run_sql", query="SELECT range AS n FROM range(80)")
    assert len(out["rows"]) == sql.MAX_ROWS and out["rows_left_out"] == 80 - sql.MAX_ROWS
    out = call(con, "run_sql", query="WITH t AS (SELECT max(date) AS d FROM sales_lines) "
                                     "SELECT d FROM t")
    assert out == {"columns": ["d"], "rows": [["2025-06-30"]], "rows_left_out": 0}


def test_run_sql_counts_a_big_result_without_holding_it(con):
    import tracemalloc

    tracemalloc.start()
    out = call(con, "run_sql", query="SELECT range AS n, 'x' AS s FROM range(1_000_000)")
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert len(out["rows"]) == sql.MAX_ROWS and out["rows_left_out"] == 1_000_000 - sql.MAX_ROWS
    assert peak < 20e6  # all million rows as tuples would be ~100 MB


# Lookup

def test_names_resolve_or_list_candidates(con):
    assert call(con, "find_drug", name="zetamol 500")["match"] == "P1"
    assert call(con, "find_drug", name="teamic")["match"] == "T1"
    out = call(con, "find_drug", name="zetamol")
    assert out["match"] is None and {c["drug_code"] for c in out["candidates"]} >= {"P1", "P3"}
    err = call(con, "stock_position", drug="zetamol")["error"]
    assert "Zetamol tb 500mg (P1)" in err and "Zetamol Syr 60ml (P3)" in err
    assert call(con, "stock_position", drug="P3")["drug_name"] == "Zetamol Syr 60ml"


def test_find_drug_searches_by_ingredient(con):
    # An ingredient and a form, in Indonesian spelling: only the syrup, by its brand.
    out = call(con, "find_drug", name="parasetamol sirup")["by_ingredient"]
    assert out["ingredients_matched"] == ["paracetamol"] and out["form_family"] == ["oral liquid"]
    assert [p["drug_code"] for p in out["products"]] == ["P3"]
    assert out["products"][0]["form"] == "syrup"
    assert {"stock_total", "units_sold_28d", "revenue_28d_rp"} <= set(out["products"][0])
    # No form: every paracetamol product.
    out = call(con, "find_drug", name="paracetamol")["by_ingredient"]
    assert {p["drug_code"] for p in out["products"]} == {"P1", "P2", "P3"}
    assert [p["drug_code"] for p in
            call(con, "find_drug", name="tranexamic")["by_ingredient"]["products"]] == ["T1"]
    # A brand name searches no ingredient.
    assert "by_ingredient" not in call(con, "find_drug", name="teamic")
    assert "by_ingredient" not in call(con, "find_drug", name="kasa steril")


# Stock

def test_stock_position_at_the_cutoff(con):
    out = call(con, "stock_position", drug="Teamic 500mg")
    # Refills stopped on 24 Jun: the floor sold out on the 25th, the store holds
    # the deliveries since.
    assert out["stock_floor"] == 0 and out["stock_store"] == out["stock_total"] > 0
    assert out["shelf_out"] and out["shelf_out_since"] == "2025-06-25"
    assert not out["out_of_stock"] and out["out_of_stock_since"] is None
    assert out["cover_days"] == pytest.approx(
        out["stock_total"] / (out["units_sold_last_28d"] / 28), abs=0.01)
    assert out["last_delivery"] == "2025-06-30" and out["last_delivery_units"] == 45
    assert out["main_supplier"] == "PT Dua" and out["main_supplier_cadence_days"] == 7

    p1 = call(con, "stock_position", drug="P1")
    assert p1["units_sold_last_28d"] == 280 and p1["avg_daily_units_last_28d"] == 10
    assert p1["main_supplier_cadence_days"] == 14


def test_stockout_history_for_a_drug(con):
    out = call(con, "stockout_history", drug="P1", tier=None, start="2025-04-01",
               end="2025-04-30")
    (spell,) = out["spells"]
    assert spell["kind"] == "stockout" and spell["ended_by"] == "purchase"
    assert spell["start"] == "2025-04-16" and spell["stop"] == "2025-04-27"
    assert out["stockout"]["days_in_period"] == spell["days"] == 12
    # Lost demand from irpd.lost: about 10 a day after the first day, part of it
    # switched to Sanmol, the same-ingredient tablet.
    assert 11 * 10 * 0.6 < spell["lost_units"] < 11 * 10 * 1.4
    assert 0 < spell["switched_units"] < spell["lost_units"]
    assert out["stockout"]["lost_revenue_rp"] == pytest.approx(spell["lost_revenue_rp"])
    for key in ("lost_units_low_80", "lost_units_high_80", "switched_units",
                "truly_lost_revenue_rp"):
        assert key in spell
    assert "estimate" in out["note"]


def test_stockout_history_counts_only_the_period_days(con):
    out = call(con, "stockout_history", drug="P1", tier=None, start="2025-04-20",
               end="2025-04-21")
    assert out["stockout"] == {**out["stockout"], "spells": 1, "days_in_period": 2}


def test_stockout_history_for_the_shop(con):
    out = call(con, "stockout_history", drug=None, tier=None, start=None, end=None)
    assert out["tier"] == "all" and out["shelfout"]["days_in_period"] == 6
    top = out["top_by_stockout_days"][0]
    assert top["drug_code"] == "P1" and top["stockout_days"] == 12
    assert call(con, "stockout_history", drug=None, tier="rare", start=None,
                end=None)["drugs_with_spells"] == 0


def test_transfer_plan_moves_a_week_of_floor_sales(con):
    out = call(con, "transfer_plan", drugs=None)
    assert out["shelf_out_now"] == 1
    (t,) = out["transfers"]
    assert t["drug_code"] == "T1" and t["stock_floor"] == 0
    assert t["move_units"] == min(t["stock_store"], math.ceil(7 * t["floor_daily_units"]))
    assert call(con, "transfer_plan", drugs=["P1"])["transfers"] == []


# Products

def test_alternatives(con):
    out = call(con, "alternatives", drug="P1")
    assert out["mapped"] and out["ingredients"] == "paracetamol"
    alts = {a["drug_code"]: a for a in out["alternatives"]}
    assert set(alts) == {"P2", "P3"}
    assert alts["P2"]["same_form_family"] and not alts["P3"]["same_form_family"]
    assert alts["P2"]["units_sold_28d"] == 4 * 28
    out = call(con, "alternatives", drug="Kasa steril")
    assert out["mapped"] is False and out["alternatives"] == [] and out["note"]


def test_price_history_stops_at_the_day(built):
    with holdout.agent_view(date(2025, 4, 15), built[1]) as early:
        out = call(early, "price_history", drug="P1")
    assert [h["price_rp"] for h in out["history"]] == [550]
    with holdout.agent_view(CUTOFF, built[1]) as late:
        out = call(late, "price_history", drug="P1")
    assert out["current"][0]["price_rp"] == 600 and out["price_changes_by_unit"] == {"Tab": 1}
    assert out["history"][0]["prev_price_rp"] == 550


# Forecast and plans

def save_ml(day, means: dict[str, list[float]], config_hash=None):
    """Write saved ML forecasts as `irpd ml serve` would: weeks 1..n from `day`'s origin."""
    from irpd.ml import schema, serve

    origin = schema.policy_origin(day)
    rows = [(c, h + 1, origin + timedelta(days=1, weeks=h), m)
            for c, ms in means.items() for h, m in enumerate(ms)]
    pd.DataFrame(rows, columns=["drug_code", "h", "week_start", "mean"]).to_csv(
        serve.FORECAST_FILE, index=False)
    serve.FORECAST_FILE.with_suffix(".json").write_text(json.dumps(
        {"day": day.isoformat(), "config_hash": config_hash or serve.CONFIG_HASH}))


def test_forecast(con):
    out = call(con, "forecast", drugs=["P1", "Sanmol tab 500"], horizon_weeks=3)
    f = {x["drug_code"]: x for x in out["forecasts"]}
    assert set(f) == {"P1", "P2"}
    assert f["P2"]["ml"] is None and "ML" in out["note"]
    cur = f["P2"]["current"]
    assert len(cur["weeks"]) == 3 and cur["weeks"][0]["week_start"] == "2025-06-30"
    assert cur["weeks"][0]["mean"] == pytest.approx(28, rel=0.2)
    for w in cur["weeks"]:
        assert w["p50"] <= w["p90"]
    assert cur["total_p90"] >= cur["total_mean"]


def test_forecast_shows_the_ml_model_beside_the_current_method(con):
    save_ml(CUTOFF, {"P2": [40, 41, 42, 43, 44, 45, 46, 47]})
    out = call(con, "forecast", drugs=["P1", "P2"], horizon_weeks=10)
    assert "note" not in out
    f = {x["drug_code"]: x for x in out["forecasts"]}
    ml = f["P2"]["ml"]
    assert ml["method"] == "ml"
    # Weeks past the saved 8 repeat week 8.
    assert [w["mean"] for w in ml["weeks"]] == [40, 41, 42, 43, 44, 45, 46, 47, 47, 47]
    assert ml["total_mean"] == pytest.approx(442)
    assert f["P2"]["current"]["weeks"][0]["mean"] == pytest.approx(28, rel=0.2)
    # P1 has no saved forecast: the ML column falls back to the current method.
    assert f["P1"]["ml"]["method"] == f["P1"]["current"]["method"] != "ml"
    assert f["P1"]["ml"]["weeks"] == f["P1"]["current"]["weeks"]


def test_forecast_spread_uses_the_same_recipe_for_both(con):
    from irpd import forecast as fc

    save_ml(CUTOFF, {"P2": [56.0] * 8})
    cur = fc.forecast(con, ["P2"], CUTOFF, 8)
    ml = fc.forecast(con, ["P2"], CUTOFF, 8, forecaster="ml")
    assert (ml["var"] / ml["mean"]).to_numpy() == pytest.approx(
        (cur["var"] / cur["mean"]).to_numpy())


def test_saved_ml_forecasts_for_another_day_or_config_are_not_used(con):
    from irpd import forecast as fc

    save_ml(date(2025, 6, 1), {"P2": [40.0] * 8})
    with pytest.raises(FileNotFoundError):
        fc.forecast(con, ["P2"], CUTOFF, 8, forecaster="ml")
    save_ml(CUTOFF, {"P2": [40.0] * 8}, config_hash="something-else")
    with pytest.raises(ValueError, match="config"):
        fc.forecast(con, ["P2"], CUTOFF, 8, forecaster="ml")


def test_reorder_plan_can_use_the_ml_forecast(con):
    save_ml(CUTOFF, {"P1": [200.0] * 8})
    args = dict(drugs=["P1"], tier=None, service_level=0.95, lead_days=None)
    cur = call(con, "reorder_plan", **args, forecaster="current")["plan"][0]
    ml = call(con, "reorder_plan", **args, forecaster="ml")
    assert ml["forecaster"] == "ml"
    # About 10 a day now; the ML file says about 29.
    assert ml["plan"][0]["s"] > 2 * cur["s"]
    assert "forecaster" in call(con, "reorder_plan", **args, forecaster="x")["error"]


def test_reorder_plan_and_policy(con):
    out = call(con, "reorder_plan", drugs=None, tier="fast", service_level=0.95,
               lead_days=None, forecaster="current")
    assert out["drugs_planned"] == 5
    rows = {r["drug_code"]: r for r in out["plan"]}
    assert rows["P1"]["lead_days"] == 14 and rows["P2"]["lead_days"] == 7
    for r in rows.values():
        assert 0 <= r["s"] <= r["S"] and r["floor_min"] <= r["floor_max"]
        assert r["order_now"] == (r["S"] - r["stock_total"] if r["stock_total"] <= r["s"] else 0)
    # About 10 a day over 14 days, with a margin.
    assert 140 <= rows["P1"]["s"] <= 200

    out = call(con, "reorder_plan", drugs=["P1"], tier=None, service_level=0.8, lead_days=3,
               forecaster="current")
    assert out["plan"][0]["lead_days"] == 3 and out["plan"][0]["s"] < rows["P1"]["s"]

    policy = to_policy(reorder_rows(con, ["P1", "P2"], CUTOFF))
    assert list(policy.columns) == ["drug_code", "s", "S", "lead_days", "floor_min", "floor_max"]
    assert to_policy(out["plan"]).drug_code.tolist() == ["P1"]


@pytest.mark.parametrize("lead_days", [-3, 0, 61])
def test_reorder_plan_refuses_a_lead_time_out_of_range(con, lead_days):
    out = call(con, "reorder_plan", drugs=["P1"], tier=None, service_level=0.95,
               lead_days=lead_days, forecaster="current")
    assert "lead_days" in out["error"]


# As of

CALLS = [
    ("run_sql", {"query": "SELECT drug_code, sum(units) FROM daily_demand GROUP BY 1 "
                          "ORDER BY 1"}),
    ("find_drug", {"name": "zetamol 500"}),
    ("stock_position", {"drug": "P1"}),
    ("stock_position", {"drug": "T1"}),
    ("stockout_history", {"drug": "P1", "tier": None, "start": None, "end": None}),
    ("stockout_history", {"drug": None, "tier": None, "start": None, "end": None}),
    ("alternatives", {"drug": "P1"}),
    ("price_history", {"drug": "P1"}),
    ("forecast", {"drugs": ["P1", "T1"], "horizon_weeks": 4}),
    ("reorder_plan", {"drugs": ["P1", "P2", "T1"], "tier": None, "service_level": 0.9,
                      "lead_days": None, "forecaster": "current"}),
    ("transfer_plan", {"drugs": None}),
]


# Sundays, so a build cut that day ends on a whole week (a cut build's last week is
# otherwise a stub, which the forecast would read as a short week of sales).
@pytest.mark.parametrize("day", [date(2025, 4, 20), date(2025, 6, 29)])
def test_tools_at_a_day_match_a_warehouse_cut_at_that_day(built, tmp_path, day):
    src, full = built
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ingredients, "MAP", src / "ingredient_map.csv")
        part = warehouse.build(cut(src, tmp_path / "cut", day), tmp_path / "cut.duckdb")
    with holdout.agent_view(day, full) as a, holdout.agent_view(day, part) as b:
        for name, args in CALLS:
            assert call(a, name, **args) == call(b, name, **args), name
