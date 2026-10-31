"""The web API on a small sealed fixture (drug A sells all year, B only before the cutoff)."""

import json
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient
from test_replay import write_fixture

import irpd.agent.tools  # noqa: F401  (registers the tools)
from irpd import config, holdout, warehouse
from irpd.api import app as api_app
from irpd.api import seen

CUTOFF = config.CUTOFF.isoformat()


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> TestClient:
    tmp = tmp_path_factory.mktemp("api")
    wh = warehouse.build(write_fixture(tmp / "data"), tmp / "wh.duckdb")
    with duckdb.connect(str(wh)) as con:  # A and B are the same medicine
        con.execute("DELETE FROM ingredient_map")
        con.execute("""INSERT INTO ingredient_map VALUES
            ('A', 'Drug A', 'amlodipine', '10mg', 'tablet', 'drug', true, 'test'),
            ('B', 'Drug B', 'amlodipine', '10 mg', '', 'drug', true, 'test')""")
    agent, sealed = holdout.seal(wh, tmp / "agent.duckdb", tmp / "holdout.duckdb")
    return TestClient(api_app.create_app(agent, sealed))


def _dates(obj) -> list[str]:
    """Every ISO date string anywhere in a JSON value."""
    text = json.dumps(obj)
    return [t for t in text.replace('"', " ").split() if len(t) == 10 and t[4] == "-"
            and t[:4].isdigit()]


def test_search_returns_tier_revenue_and_ingredients(client):
    rows = client.get("/drugs", params={"q": "drug"}).json()
    assert [r["code"] for r in rows] == ["A", "B"]  # by revenue
    a = rows[0]
    assert a["tier"] in ("fast", "regular") and a["revenue"] == 91 * 2 * 500
    assert a["ingredients"] == "amlodipine" and a["mapped"] is True
    assert client.get("/drugs", params={"q": "zzz"}).json() == []


def test_drug_page_shows_only_what_the_agent_saw(client):
    page = client.get("/drugs/A").json()
    assert page["holdout"] is None
    assert max(_dates(page)) == CUTOFF
    assert page["stock"][-1]["date"] == CUTOFF and page["stock"][-1]["floor"] == 600 - 2 * 91
    assert page["purchases"][0] == {"date": "2025-04-01", "qty": 600, "supplier": "Sup X",
                                    "location": "floor"}
    assert page["prices"][0]["price"] == 500
    alt = page["alternatives"]
    assert [a["code"] for a in alt] == ["B"] and alt[0]["same_strength_form"] is True
    assert client.get("/drugs/nope").status_code == 404


def test_holdout_actuals_are_a_separate_section(client):
    page = client.get("/drugs/A", params={"holdout": "true"}).json()
    hold = page["holdout"]
    assert min(_dates(hold)) > CUTOFF
    assert hold["purchases"][0]["date"] == "2025-07-20"
    assert hold["purchases"][0]["location"] == "store"
    # The agent-side part is unchanged by asking for the holdout.
    assert {k: v for k, v in page.items() if k != "holdout"} == \
        {k: v for k, v in client.get("/drugs/A").json().items() if k != "holdout"}


def test_the_agent_side_module_never_opens_the_holdout():
    src = Path(seen.__file__).read_text()
    assert "open_sealed" not in src and "SEALED" not in src and "holdout.duckdb" not in src


def test_overview(client):
    o = client.get("/overview").json()
    assert o["as_of"] == CUTOFF and o["period"]["last"] <= CUTOFF
    assert o["revenue_total"] == sum(r["revenue"] for r in o["revenue_by_year"]) > 0
    assert o["revenue_share_mapped"] == 1.0
    assert {t["tier"] for t in o["tiers"]} <= {"fast", "regular", "slow", "rare"}


def test_replay_scores_the_pharmacist_and_matched_policies(client):
    r = client.get("/replay").json()
    names = [s["policy"] for s in r["summary"]]
    assert names[:3] == ["pharmacist (actual)", "mimic", "textbook"]
    # Matched runs appear for each policy that can reach the pharmacist's stock value.
    matched = [n for n, k in r["matched_factor"].items() if k is not None]
    assert names[3:] == [f"{n} at matched stock" for n in matched] and matched
    assert [p["code"] for p in r["per_drug"]] == ["A"]
    assert set(r["tradeoff"]) == {"mimic", "textbook"}
    assert "same simulation" in r["fair_contest"]


def _events(r) -> list[tuple[str, dict]]:
    out = []
    for block in r.text.strip().split("\n\n"):
        head, data = block.split("\n", 1)
        out.append((head.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


SSE = {"accept": "text/event-stream"}


def test_chat_demo_json_and_stream(client):
    t = client.post("/chat?demo=1").json()
    assert t["question"] and t["answer"] and t["as_of"] == CUTOFF
    calls = [s for s in t["steps"] if s["type"] == "tool_call"]
    assert calls and all({"id", "name", "args", "output"} <= set(s) for s in calls)

    r = client.post("/chat?demo=1", headers=SSE)
    assert r.headers["content-type"].startswith("text/event-stream")
    kinds = [k for k, _ in _events(r)]
    want = []
    for s in t["steps"]:
        want += ["message"] if s["type"] == "message" else ["tool_call", "tool_result"]
    assert kinds == want + ["answer", "done"]


@pytest.fixture
def live(tmp_path, monkeypatch):
    """An app whose chat runs the real loop on a fake client and one fake tool."""
    from test_agent_loop import SCHEMA, FakeClient, call, say

    from irpd.agent import registry

    monkeypatch.setattr(registry, "TOOLS", {})
    monkeypatch.setenv("IRPD_LEDGER", str(tmp_path / "spend.jsonl"))
    monkeypatch.setenv("IRPD_CHAT_MODEL", "gpt-6-luna")

    @registry.tool("run_sql", "Run SQL.", SCHEMA)
    def run_sql(con, sql):
        return {"rows": [[con.execute(sql).fetchone()[0]]], "pad": "x" * 9_000}

    def make(turns):
        fake = FakeClient(turns)
        wh = warehouse.build(write_fixture(tmp_path / "data"), tmp_path / "wh.duckdb")
        agent, sealed = holdout.seal(wh, tmp_path / "agent.duckdb", tmp_path / "holdout.duckdb")
        return TestClient(api_app.create_app(agent, sealed, chat_client=fake)), fake

    make.turns = lambda: [
        [say("Let me look."), call("run_sql", {"sql": "SELECT 42"})],
        [say("42 units.")],
    ]
    return make


def test_chat_live_streams_events_in_order(live, tmp_path):
    app, fake = live(live.turns())
    body = {"question": "How many?", "history": [{"question": "Hi?", "answer": "Hello."}]}
    ev = _events(app.post("/chat", json=body, headers=SSE))
    assert [k for k, _ in ev] == ["message", "tool_call", "tool_result", "answer", "done"]
    d = dict(ev)
    assert d["tool_call"] == {"id": "c0", "name": "run_sql", "args": {"sql": "SELECT 42"}}
    assert d["tool_result"]["truncated"] is True and "42" in d["tool_result"]["output"]
    assert d["answer"] == {"text": "42 units."}
    assert d["done"]["model"] == "gpt-6-luna" and d["done"]["usage"]["cost_usd"] > 0
    first = fake.requests[0]
    assert first["model"] == "gpt-6-luna"
    assert "Q: Hi?\nA: Hello." in first["input"][0]["content"]
    assert (tmp_path / "spend.jsonl").exists()  # the ledger from IRPD_LEDGER


def test_chat_live_json(live):
    app, _ = live(live.turns())
    t = app.post("/chat", json={"question": "How many?"}).json()
    assert t["answer"] == "42 units."
    assert [s["type"] for s in t["steps"]] == ["message", "tool_call"]


def test_chat_stops_at_the_cap(live, tmp_path):
    app, fake = live(live.turns())
    (tmp_path / "spend.jsonl").write_text(json.dumps({"usd": 1.0}) + "\n")
    ev = _events(app.post("/chat", json={"question": "How many?"}, headers=SSE))
    assert [k for k, _ in ev] == ["error"] and ev[0][1]["code"] == "budget"
    assert "cap" in ev[0][1]["message"] and not fake.requests
    assert app.post("/chat", json={"question": "x"}).status_code == 503


def test_chat_without_a_key_says_so(client, monkeypatch):
    from irpd import llm

    def no_key():
        raise RuntimeError("OPENAI_API_KEY not set")

    monkeypatch.setattr(llm, "load_key", no_key)
    ev = _events(client.post("/chat", json={"question": "x"}, headers=SSE))
    assert ev == [("error", {"code": "no_key", "message": ev[0][1]["message"]})]
    assert "key" in ev[0][1]["message"]
    assert client.post("/chat?demo=1").json()["answer"]  # the demo still works


def test_chat_failure_hides_the_exception(live, monkeypatch, caplog):
    from irpd.agent import loop

    def boom(*a, **k):
        raise RuntimeError("Incorrect API key provided: sk-proj-abc***xyz")

    monkeypatch.setattr(loop, "run", boom)
    app, _ = live(live.turns())
    with caplog.at_level("ERROR", logger="irpd.api.chat"):
        ev = _events(app.post("/chat", json={"question": "x"}, headers=SSE))
    assert [k for k, _ in ev] == ["error"] and ev[0][1]["code"] == "failed"
    assert "sk-" not in ev[0][1]["message"] and "RuntimeError" not in ev[0][1]["message"]
    assert "sk-proj-abc" in caplog.text  # the details stay in the server log
    r = app.post("/chat", json={"question": "x"})
    assert r.status_code == 503 and "sk-" not in r.text


def test_demo_transcript_matches_the_tools_and_the_page():
    from irpd.agent import registry
    from irpd.api import chat

    for step in chat.TRANSCRIPT["steps"]:
        if step["type"] != "tool_call":
            continue
        schema = registry.TOOLS[step["name"]].parameters
        assert set(step["args"]) == set(schema["required"]), step["name"]
    page = (config.ROOT / "web/src/pages/Chat.tsx").read_text("utf-8")
    assert f'const DEMO_QUESTION = "{chat.DEMO_QUESTION}";' in page


def test_one_chat_at_a_time(live):
    from irpd.api import chat

    app, fake = live(live.turns())
    with chat._busy:
        ev = _events(app.post("/chat", json={"question": "x"}, headers=SSE))
    assert ev[0][0] == "error" and ev[0][1]["code"] == "busy" and not fake.requests
