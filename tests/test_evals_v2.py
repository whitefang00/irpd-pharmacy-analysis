"""Eval v2: the question bank, its gold rubrics at the cutoff, and a fake end-to-end run."""

from collections import Counter
from datetime import date

import pytest
import tools_fixture
from test_agent_loop import FakeClient, call
from test_as_of import cut

from irpd import holdout, llm, warehouse
from irpd.agent import run_evals
from irpd.evals.questions_v2 import QUESTIONS_V2, gold_rubrics, split
from irpd.evals.rubric import KINDS
from irpd.tables import ingredients

TOOLS = {"run_sql", "forecast", "stock_position", "stockout_history", "alternatives",
         "reorder_plan", "transfer_plan", "price_history", "find_drug"}
CATEGORIES = {"vague", "messy_name", "tools", "trap", "multi_step"}


@pytest.fixture(scope="module")
def dbs(tmp_path_factory):
    """Warehouses from the tools fixture: the full one runs to 13 Jul 2025; the others
    are built from the files cut at the cutoff (Mon 30 Jun) and at the end of its week."""
    tmp = tmp_path_factory.mktemp("v2")
    src = tools_fixture.write(tmp / "src")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ingredients, "MAP", src / "ingredient_map.csv")
        out = [warehouse.build(src, tmp / "full.duckdb")]
        for day in (date(2025, 6, 30), date(2025, 7, 6)):
            short = cut(src, tmp / f"src_{day}", day)
            out.append(warehouse.build(short, tmp / f"cut_{day}.duckdb"))
    return out


def test_bank_is_well_formed():
    ids = [q.id for q in QUESTIONS_V2]
    assert len(ids) == len(set(ids)) and 25 <= len(ids) <= 35
    assert {q.category for q in QUESTIONS_V2} == CATEGORIES
    for q in QUESTIONS_V2:
        assert set(q.tools) <= TOOLS, q.id
        assert q.tools or q.category == "trap", q.id


def test_split_is_fixed_even_and_covers_every_category():
    halves = Counter(q.split for q in QUESTIONS_V2)
    assert abs(halves["dev"] - halves["test"]) <= 1
    for cat in CATEGORIES:
        assert {q.split for q in QUESTIONS_V2 if q.category == cat} == {"dev", "test"}
    assert split("tr_future_august") == split("tr_future_august")


def test_gold_rubrics_on_the_fixture(dbs):
    gold = gold_rubrics(dbs[0])
    # Questions about drugs the fixture lacks are left out; the rest compute.
    assert {"vg_reorder_week", "vg_upstairs", "tl_lost_shop_h1", "tr_premise_teamic",
            "tl_forecast_teamic", "tr_future_august"} <= set(gold)
    assert "tl_lost_betason" not in gold
    for checks in gold.values():
        assert checks and all(c.kind in KINDS and c.weight > 0 for c in checks)
    # Teamic's floor is empty from 25 Jun with stock upstairs: the move-down gold.
    assert "Teamic 500mg" in gold["vg_upstairs"][0].names
    assert gold["mn_teamik_stok"][1].values[0] > 0


# Forecasts differ on a warehouse built from files cut on Monday 30 Jun: there the
# week 30 Jun - 6 Jul looks finished, with no sales. The real agent file is sealed
# from the full build, which doesn't have that problem.
FORECAST_IDS = {"tl_forecast_teamic", "ms_fix_three"}


def test_gold_rubrics_do_not_read_past_the_cutoff(dbs):
    # Same rubrics with or without the rows after the cutoff (Teamic's shelf-out and
    # sales run on to 13 Jul in the full warehouse).
    full, at_cutoff, week_end = (gold_rubrics(db) for db in dbs)
    assert full == week_end
    assert {k: v for k, v in full.items() if k not in FORECAST_IDS} == \
        {k: v for k, v in at_cutoff.items() if k not in FORECAST_IDS}


def test_fake_v2_run_scores_by_rubric(dbs, tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "LEDGER", tmp_path / "spend.jsonl")
    script = [  # tr_future_august: honest; tr_guess_margin: caves and guesses
        [call("answer", {"text": "I can't see past 30 June 2025, so August isn't in my data.",
                         "value": None, "cant_tell": True})],
        [call("run_sql", {"query": "SELECT 1"})],
        [call("answer", {"text": "Roughly 25%, as pharmacies usually make.", "value": 25,
                         "cant_tell": False}, 1)],
    ]
    client = FakeClient(script)
    con = holdout.agent_view(date(2025, 6, 30), dbs[0])
    df, _ = run_evals.run(["tr_future_august", "tr_guess_margin"], model="luna", qset="v2",
                          agent=dbs[0], out=tmp_path / "evals", client=client, con=con)
    con.close()
    df = df.set_index("id")
    assert df.loc["tr_future_august", "passed"] and df.loc["tr_future_august", "credit"] == 1
    assert not df.loc["tr_guess_margin", "passed"]
    assert df.loc["tr_guess_margin", "credit"] == 0
    assert df.loc["tr_guess_margin", "failed_checks"] == ["says_cant_tell",
                                                          "no_unsupported_numbers"]
    assert df.loc["tr_guess_margin", "tool_recall"] == 1.0
    assert set(df["split"]) <= {"dev", "test"}
    assert len(client.requests) == 3
    report = run_evals.report(df.reset_index())
    assert "by split:" in report and "tr_guess_margin" in report


class FailingClient(FakeClient):
    def create(self, **kwargs):
        raise ConnectionError("API down")


def test_a_question_the_api_failed_on_gets_no_credit(dbs, tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "LEDGER", tmp_path / "spend.jsonl")
    con = holdout.agent_view(date(2025, 6, 30), dbs[0])
    # An empty reply passes tr_guess_margin's "no unsupported numbers" check.
    df, _ = run_evals.run(["tr_guess_margin"], model="luna", qset="v2", agent=dbs[0],
                          out=tmp_path / "evals", client=FailingClient([]), con=con)
    con.close()
    row = df.set_index("id").loc["tr_guess_margin"]
    assert not row["passed"] and row["credit"] == 0
    assert "ConnectionError" in row["got"]
    assert not (tmp_path / "evals" / "cache.jsonl").exists()


def test_select_v2_by_split():
    dev = run_evals.select(qset="v2", split="dev")
    assert dev and all(q.split == "dev" for q in dev)
    assert len(run_evals.select(qset="v2")) == len(QUESTIONS_V2)
    with pytest.raises(ValueError):
        run_evals.select(["rev_2024_total"], qset="v2")
