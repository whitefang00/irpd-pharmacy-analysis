"""The contest: window loader, knobs -> policy, planner tools, and no peeking."""

import json
from datetime import date

import duckdb
import numpy as np
import pandas as pd
import pytest
import tools_fixture
from test_agent_loop import FakeClient, call
from test_tools import strict_errors

from irpd import config, contest, holdout, lost, replay, warehouse
from irpd.agent.tools import plan
from irpd.tables import ingredients

WINDOW = [(date(2025, 5, 31), date(2025, 6, 30))]


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    """(full warehouse, agent file, sealed file) for the made-up pharmacy."""
    tmp = tmp_path_factory.mktemp("contest")
    src = tools_fixture.write(tmp / "src")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ingredients, "MAP", src / "ingredient_map.csv")
        wh = warehouse.build(src, tmp / "full.duckdb")
    return (wh, *holdout.seal(wh, tmp / "agent.duckdb", tmp / "holdout.duckdb"))


@pytest.fixture()
def no_sealed(monkeypatch, files):
    """Opening the sealed file, by any route, fails the test."""
    def refuse(*a, **k):
        raise AssertionError("the sealed holdout was opened")
    monkeypatch.setattr(holdout, "open_sealed", refuse)
    monkeypatch.setattr(lost, "combined", refuse)
    monkeypatch.setattr(holdout, "SEALED", files[2].with_name("nowhere.duckdb"))


def test_window_loader_reproduces_load_on_the_holdout_window(files):
    wh, agent, sealed = files
    want = replay.load(agent, sealed)
    # The full warehouse, cut to the holdout window, has the sealed file's shape.
    with holdout.agent_view(config.CUTOFF, agent) as past, \
            contest.window_tables(replay.FIRST, replay.LAST, wh) as win:
        got = replay.load_from(past, win, replay.FIRST, replay.LAST)
    assert got.drugs == want.drugs and got.days == want.days and want.drugs
    assert got.store_opens == want.store_opens
    for k in ("price", "start_floor", "start_store", "demand", "adjust", "rate",
              "actual_floor", "actual_store", "actual_sold", "live"):
        np.testing.assert_array_equal(getattr(got, k), getattr(want, k), err_msg=k)
    pd.testing.assert_series_equal(got.lead, want.lead)


def test_facts_ignore_rows_after_their_day(files):
    wh, agent, _ = files
    with duckdb.connect(str(wh), read_only=True) as con:
        full = contest.facts(con, config.CUTOFF)
    with holdout.agent_view(config.CUTOFF, agent) as con:
        pd.testing.assert_frame_equal(full, contest.facts(con, config.CUTOFF))


def test_a_backtest_window_must_end_by_the_cutoff(files):
    with pytest.raises(ValueError, match="by the cutoff"):
        contest.load_window(date(2025, 6, 1), date(2025, 7, 31), files[1])


def test_the_in_sample_backtest_never_opens_the_sealed_file(files, no_sealed):
    _, agent, _ = files
    ws = contest.load_windows(agent, WINDOW)
    h = ws[0].h
    assert h.days[0] == date(2025, 6, 1) and h.days[-1] == date(2025, 6, 30)
    assert set(h.drugs) <= set(ws[0].basis.facts.index)
    out = contest.backtest({}, ws)
    names = [r["policy"] for r in out["windows"][0]["table"]]
    assert names == ["pharmacist (ledger)", "agent", "mimic", "textbook", "plain tools"]
    json.dumps(out)


def basis() -> contest.Basis:
    f = pd.DataFrame({
        "drug_code": ["a", "b", "c", "d"], "drug_name": list("ABCD"),
        "tier": ["fast", "regular", "regular", "fast"], "cadence": ["short"] * 4,
        "trend": ["flat", "falling", "flat", "flat"], "long_stockout": [False] * 4,
        "switching": ["none", "high", "low", "none"], "idle_days": [0, 5, 120, 1],
        "lead_days": [2, 2, 2, 2], "floor_rate": [10.0, np.nan, 1.0, 10.0],
        "switch_rate": [np.nan, 0.5, 0.1, np.nan], "revenue_182d": [4.0, 3, 2, 1],
        "stockout_days_182d": [0, 3, 9, 0], "shelfout_days_182d": [0] * 4,
        "lost_revenue_182d": [0.0, 1, 2, 0],
    }).set_index("drug_code")
    mean = np.tile([[10.0], [2.0], [1.0], [10.0]], (1, 80))
    return contest.Basis(date(2025, 6, 30), f, mean, mean * 2)


KNOBS = {
    "default": {"service_level": 0.9, "cycle_days": None, "floor_min_days": None,
                "floor_max_days": None},
    "segments": [
        {"name": "fast", "tier": "fast", "service_level": 0.99, "cycle_days": 21,
         "floor_min_days": 3, "floor_max_days": 10},
        {"name": "switchers", "switching": "high", "service_level": 0.7},
        {"name": "fast none", "tier": "fast", "switching": "none", "service_level": 0.5},
    ],
    "stop_after_idle_days": 90,
    "overrides": [{"drug": "d", "stop": None, "service_level": 0.8, "cycle_days": None,
                   "reason": "test"}],
}


def test_rules_first_segment_wins_then_idle_stop_then_overrides():
    a = contest.assign(KNOBS, basis().facts)
    assert a.rule.tolist() == ["fast", "switchers", "stopped (idle)", "override"]
    assert a.service_level.tolist() == [0.99, 0.7, 0.9, 0.8]
    assert a.cycle_days.tolist() == [21, 14, 14, 21]  # d keeps its segment's cycle
    assert a.stop.tolist() == [False, False, True, False]
    with pytest.raises(ValueError, match="service_level must be between"):
        contest.assign({"default": {"service_level": 1.2}}, basis().facts)
    with pytest.raises(ValueError, match="not a candidate"):
        contest.assign({"overrides": [{"drug": "zz"}]}, basis().facts)


def test_knobs_to_policy():
    b = basis()
    p = contest.to_policy(KNOBS, b).set_index("drug_code")
    plain = contest.plain_policy(b).set_index("drug_code")
    assert (p.loc["c", "s"], p.loc["c", "S"]) == (-1, 0)  # stopped: never orders
    assert p.loc["a", "s"] > plain.loc["a", "s"]  # 0.99 vs 0.95
    assert p.loc["a", "S"] - p.loc["a", "s"] > plain.loc["a", "S"] - plain.loc["a", "s"]
    assert p.loc["b", "s"] < plain.loc["b", "s"]  # 0.7
    # Floor levels are days of floor sales (forecast mean where the floor sold nothing).
    assert (p.loc["a", "floor_min"], p.loc["a", "floor_max"]) == (30, 100)
    assert (plain.loc["b", "floor_min"], plain.loc["b", "floor_max"]) == (4, 14)
    assert list(p.reset_index().columns) == plan.POLICY_COLUMNS


def test_plain_policy_is_the_reorder_tool_at_its_defaults(files):
    _, agent, _ = files
    with holdout.agent_view(config.CUTOFF, agent) as con:
        b = contest.basis(con, config.CUTOFF)
        rows = plan.reorder_rows(con, list(b.facts.index), config.CUTOFF)
    p = contest.plain_policy(b)
    assert len(p) and p.drug_code.tolist() == rows.drug_code.tolist()
    np.testing.assert_array_equal(p.s, rows.s)
    np.testing.assert_array_equal(p.S, rows.S)


def test_planner_tool_schemas_are_strict(files):
    for t in contest.planner_tools(contest.Planner([], basis())).values():
        assert strict_errors(t.parameters) == [], t.name


def test_planner_backtests_submits_and_freezes(files, no_sealed, tmp_path):
    _, agent, _ = files
    with holdout.agent_view(config.CUTOFF, agent) as con:
        ctx = contest.Planner(contest.load_windows(agent, WINDOW),
                              contest.basis(con, config.CUTOFF))
    knobs = {**KNOBS, "overrides": []}
    client = FakeClient([
        [call("segment_facts", {"by": []})],
        [call("backtest_policy", {"knobs": knobs}, 1)],
        [call("submit_policy", {"knobs": {"default": {"service_level": 1.5}},
                                "rationale": "bad"}, 2)],  # rejected, the run goes on
        [call("submit_policy", {"knobs": knobs, "rationale": "Because."}, 3)],
        [call("run_sql", {"sql": "SELECT 1"}, 4)],  # never reached
    ])
    d = contest.run_planner("gpt-6-luna", run_id="t1", client=client, agent=agent,
                            out=tmp_path, ctx=ctx, ledger=tmp_path / "spend.jsonl")
    t = json.loads((d / "transcript.json").read_text())
    assert t["stopped"] == "submit_policy" and len(t["steps"]) == 4
    assert [c["name"] for c in t["tool_calls"]] == [
        "segment_facts", "backtest_policy", "submit_policy", "submit_policy"]
    assert "error" in t["tool_calls"][2]["output"]
    assert set(t["tool_calls"][0]["output"]) >= {"by_tier", "by_switching", "by_trend"}
    assert t["tool_calls"][1]["output"]["windows"][0]["table"][1]["policy"] == "agent"
    frozen = json.loads((d / "knobs.json").read_text())
    assert frozen["knobs"] == knobs and frozen["rationale"] == "Because."
    policy = pd.read_csv(d / "policy.csv")
    assert len(policy) == len(ctx.basis.facts)
    assert (d / "in_sample.json").exists() and (d / "rules.csv").exists()
    # Only the planner's own tools and a few read-only ones were offered.
    offered = {x["name"] for x in client.requests[0]["tools"]}
    assert offered == {"segment_facts", "backtest_policy", "submit_policy", "find_drug",
                       "stockout_history", "run_sql"}
    assert "stock planner" in client.requests[0]["instructions"]
