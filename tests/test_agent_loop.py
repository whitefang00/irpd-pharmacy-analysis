import json
from types import SimpleNamespace

import duckdb
import pytest

from irpd import llm
from irpd.agent import loop, registry, run_evals

SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["sql"], "properties": {"sql": {"type": "string"}}}


def call(name, args, i=0):
    return {"type": "function_call", "name": name, "arguments": json.dumps(args),
            "call_id": f"c{i}"}


def say(text):
    return {"type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": text}]}


class FakeClient:
    """Plays back scripted Responses API turns and keeps every request."""

    def __init__(self, turns, input_tokens=1_000, output_tokens=200):
        self.turns, self.requests = list(turns), []
        self.usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                                     input_tokens_details=SimpleNamespace(cached_tokens=0))
        self.responses = self

    def create(self, **kwargs):
        self.requests.append({**kwargs, "input": list(kwargs["input"])})
        output = self.turns.pop(0) if self.turns else [call("run_sql", {"sql": "x"})]
        text = "".join(c["text"] for it in output if it["type"] == "message"
                       for c in it["content"])
        return SimpleNamespace(output=output, output_text=text, usage=self.usage)


@pytest.fixture
def tools(monkeypatch, tmp_path):
    monkeypatch.setattr(registry, "TOOLS", {})
    monkeypatch.setattr(llm, "LEDGER", tmp_path / "spend.jsonl")

    @registry.tool("run_sql", "Run SQL.", SCHEMA)
    def run_sql(con, sql):
        return {"rows": [[con.execute(sql).fetchone()[0]]]}

    con = duckdb.connect()
    yield con
    con.close()


def test_loop_runs_tools_and_answers(tools):
    client = FakeClient([
        [call("run_sql", {"sql": "SELECT 3997275634"})],
        [call("answer", {"text": "Revenue was Rp 3997275634.", "value": 3997275634,
                         "cant_tell": False}, 1)],
    ])
    t = loop.run("Revenue in 2024?", con=tools, structured=True, client=client)
    assert t.stopped == "answer"
    assert t.answer == {"value": 3997275634, "cant_tell": False}
    assert t.tool_calls[0]["output"] == {"rows": [[3997275634]]}
    # the tool's output went back to the model with its call id
    sent = client.requests[1]["input"]
    assert {"type": "function_call_output", "call_id": "c0",
            "output": json.dumps({"rows": [[3997275634]]})} in sent
    assert "answer" in [x["name"] for x in client.requests[0]["tools"]]
    assert "Today is 2025-06-30" in client.requests[0]["instructions"]
    assert len(t.steps) == 2
    assert t.usd == pytest.approx(2 * llm.cost(llm.AGENT_MODEL, 1_000, 0, 200))
    assert llm.spent() == pytest.approx(t.usd)  # every request is in the ledger


def test_loop_stops_at_max_steps(tools):
    client = FakeClient([])  # calls run_sql forever
    t = loop.run("Loop?", con=tools, client=client, max_steps=3)
    assert t.stopped == "max_steps" and len(t.steps) == 3 and t.answer is None
    assert t.tools == ["run_sql"] * 3


def test_plain_text_gets_one_forced_nudge(tools):
    client = FakeClient([[say("The data can't tell.")],
                         [call("answer", {"text": "The data can't tell: no costs.",
                                          "value": None, "cant_tell": True})]])
    t = loop.run("Margin?", con=tools, structured=True, client=client)
    assert client.requests[1]["tool_choice"] == {"type": "function", "name": "answer"}
    assert t.answer == {"value": None, "cant_tell": True} and t.stopped == "answer"
    # without the answer tool, plain text ends the run
    t = loop.run("Hi", con=tools, client=FakeClient([[say("Hello.")]]))
    assert (t.stopped, t.text) == ("text", "Hello.")


def test_tool_errors_go_back_to_the_model(tools):
    client = FakeClient([[call("no_such_tool", {})], [say("Sorry.")]])
    t = loop.run("?", con=tools, client=client)
    assert "error" in t.tool_calls[0]["output"]


def test_cap_refuses_near_the_limit(tools, tmp_path):
    ledger = tmp_path / "near.jsonl"
    ledger.write_text(json.dumps({"usd": 0.995}) + "\n")
    client = FakeClient([[say("x")]])
    with pytest.raises(llm.BudgetExceeded):
        loop.run("?", con=tools, client=client, ledger=ledger, cap=1.0)
    assert client.requests == [] and llm.spent(ledger) == 0.995


def test_runner_scores_and_caches(tools, fixture_db, tmp_path, monkeypatch):
    monkeypatch.setattr("irpd.holdout.agent_view", lambda day: duckdb.connect())
    turns = {  # rev_2024_total and a can't-tell question
        "rev_2024_total": [[call("run_sql", {"sql": "SELECT 42"})],
                           [call("answer", {"text": "Rp 42, or 99 maybe.", "value": 42,
                                            "cant_tell": False}, 1)]],
        "ct_margin_flazen": [[call("answer", {"text": "The data can't tell.",
                                              "value": None, "cant_tell": True})]],
    }
    script = turns["rev_2024_total"] + turns["ct_margin_flazen"]
    client = FakeClient(script)
    ids = list(turns)
    kw = dict(out=tmp_path / "evals", map_path=tmp_path / "none.csv", db=fixture_db,
              client=client)
    df, _ = run_evals.run(ids, model="luna", **kw)
    df = df.set_index("id")
    assert df.loc["ct_margin_flazen", "passed"]
    assert df.loc["rev_2024_total", "tool_recall"] == 1.0
    assert df.loc["rev_2024_total", "unsupported"] == 1  # the 99
    assert len(client.requests) == 3
    assert len(list((tmp_path / "evals").glob("2*.jsonl"))) == 1

    df2, est = run_evals.run(ids, model="luna", **kw)  # all cached: no new requests
    assert len(client.requests) == 3 and est == 0
    assert df2["cached"].all()


def test_runner_refuses_when_estimate_passes_budget(tools, tmp_path):
    ledger = tmp_path / "spend.jsonl"
    ledger.write_text(json.dumps({"usd": 0.99}) + "\n")
    client = FakeClient([])
    with pytest.raises(llm.BudgetExceeded):
        run_evals.run(None, model="sol", out=tmp_path / "evals", ledger=ledger, client=client,
                      map_path=tmp_path / "none.csv")
    assert client.requests == []


def test_select_skips_map_questions_without_a_map():
    assert not any(q.needs for q in run_evals.select())
    assert any(q.needs for q in run_evals.select(with_map=True))
    assert [q.id for q in run_evals.select(["rev_2024_total"])] == ["rev_2024_total"]
    with pytest.raises(ValueError):
        run_evals.select(["nope"])
