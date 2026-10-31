"""`POST /chat`: the agent loop, streamed, with a recorded example transcript for offline use.

**Transcript shape** (the JSON reply, live or demo):

    {
      "question": str,
      "as_of": "2025-06-30",            # the day the agent stands on
      "model": str,
      "steps": [                        # in order
        {"type": "tool_call", "id": str, "name": str,
         "args": {...},                 # JSON args the model sent
         "output": {...}},              # the tool's JSON result (or {"error": ...})
        {"type": "message", "text": str}   # optional text between calls
      ],
      "answer": str,                    # final text; every number in it comes from an output
      "usage": {"input_tokens": int, "output_tokens": int, "cost_usd": float}
    }

**Streaming.** With `Accept: text/event-stream` the run arrives as server-sent
events while it happens:

    tool_call    {id, name, args}                  as a tool starts
    tool_result  {id, name, output, truncated}     as it returns; output cut for display
    message      {text}                            text between calls
    answer       {text}                            the final answer
    done         {model, as_of, stopped, usage}    usage as above
    error        {code, message}                   no_key | busy | budget | failed; ends the stream

The live run uses `loop.run` in a worker thread that feeds a queue, so the
blocking model calls never sit on the event loop. One chat runs at a time per
process. The model is `IRPD_CHAT_MODEL` (default gpt-6-luna) and spend goes to
the ledger at `IRPD_LEDGER` (default `data/spend.jsonl`) under its cap.
"""

import json
import logging
import os
import queue
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from irpd import config, holdout, llm
from irpd.agent import loop

Event = tuple[str, dict[str, Any]]
log = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-6-luna"
SHOW_CHARS = 4_000  # tool output sent to the page; the model sees all of it
HISTORY_TURNS = 3

# A recorded answer, played by `?demo=1` so the page works without a key: gpt-6-luna on
# the eval question `ms_nova_out` (prompt v3), taken from the eval cache. The tool calls,
# outputs and answer are real. Long lists in the outputs are cut to a few rows (the
# `*_left_out` counts include the cut rows); the answer only uses rows that were kept.
TRANSCRIPT: dict = json.loads((Path(__file__).parent / "demo_transcript.json").read_text("utf-8"))
DEMO_QUESTION: str = TRANSCRIPT["question"]  # web/src/pages/Chat.tsx has the same text


def transcript() -> dict:
    return TRANSCRIPT


def demo(t: dict = TRANSCRIPT) -> Iterator[Event]:
    """The fixed transcript as events, in the same order a live run sends them."""
    for step in t["steps"]:
        if step["type"] == "message":
            yield "message", {"text": step["text"]}
            continue
        yield "tool_call", {k: step[k] for k in ("id", "name", "args")}
        yield "tool_result", {"id": step["id"], "name": step["name"], **_shown(step["output"])}
    yield "answer", {"text": t["answer"]}
    yield "done", {"model": t["model"], "as_of": t["as_of"], "stopped": "answer",
                   "usage": t["usage"]}


def prompt(question: str, history: list[tuple[str, str]]) -> str:
    """Fold the last few question/answer pairs into one input."""
    turns = history[-HISTORY_TURNS:]
    if not turns:
        return question
    earlier = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in turns)
    return f"Earlier in this conversation:\n\n{earlier}\n\nNow: {question}"


def model() -> str:
    return os.environ.get("IRPD_CHAT_MODEL", DEFAULT_MODEL)


def ledger() -> Path:
    return Path(os.environ["IRPD_LEDGER"]) if os.environ.get("IRPD_LEDGER") else llm.LEDGER


_busy = threading.Lock()


def live(question: str, history: list[tuple[str, str]], *, agent: Path = holdout.AGENT,
         client: Any = None) -> Iterator[Event]:
    """Run the agent loop and yield its events as they happen."""
    if client is None:
        try:
            llm.load_key()
        except RuntimeError:
            yield _error("no_key", "No OpenAI key is configured, so the agent can't run. "
                                   "Set OPENAI_API_KEY, or try the demo transcript.")
            return
    if not _busy.acquire(blocking=False):
        yield _error("busy", "Another question is running. Try again when it finishes.")
        return
    q: queue.Queue[Event | None] = queue.Queue()
    m = model()

    def work() -> None:
        try:
            with holdout.agent_view(config.CUTOFF, agent) as con:
                t = loop.run(prompt(question, history), as_of=config.CUTOFF, model=m,
                             con=con, client=client, ledger=ledger(), purpose="chat",
                             on_event=lambda kind, data: q.put((kind, data)))
            q.put(("done", {"model": m, "as_of": t.as_of, "stopped": t.stopped,
                            "usage": {"input_tokens": t.input_tokens,
                                      "output_tokens": t.output_tokens,
                                      "cost_usd": round(t.usd, 6)}}))
        except llm.BudgetExceeded as e:
            q.put(_error("budget", f"The spend cap stopped this question: {e}."))
        except Exception:  # details go to the server log; the page gets a plain message
            log.exception("chat failed")
            q.put(_error("failed", "The agent hit an error and stopped. "
                                   "The details are in the server log."))
        finally:
            _busy.release()
            q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not None:
        kind, data = item
        if kind == "tool_call":
            data = {**data, "args": _args(data["args"])}
        elif kind == "tool_result":
            data = {"id": data["id"], "name": data["name"], **_shown(data["output"])}
        yield kind, data


def sse(events: Iterator[Event]) -> Iterator[str]:
    for kind, data in events:
        yield f"event: {kind}\ndata: {json.dumps(data, default=str)}\n\n"


def fold(events: Iterator[Event], question: str) -> dict:
    """The events as one transcript; an `error` key if the run stopped on one."""
    t: dict[str, Any] = {"question": question, "as_of": config.CUTOFF.isoformat(),
                         "model": model(), "steps": [], "answer": "",
                         "usage": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}}
    for kind, data in events:
        if kind == "tool_call":
            t["steps"].append({"type": "tool_call", **data, "output": None})
        elif kind == "tool_result":
            t["steps"][-1]["output"] = data["output"]
        elif kind == "message":
            t["steps"].append({"type": "message", "text": data["text"]})
        elif kind == "answer":
            t["answer"] = data["text"]
        elif kind == "done":
            t.update(model=data["model"], as_of=data["as_of"], usage=data["usage"])
        elif kind == "error":
            t["error"] = data
    return t


def _args(text: Any) -> Any:
    try:
        return json.loads(text) if isinstance(text, str) else text
    except json.JSONDecodeError:
        return text


def _shown(output: Any) -> dict[str, Any]:
    text = json.dumps(output, default=str)
    if len(text) <= SHOW_CHARS:
        return {"output": output, "truncated": False}
    return {"output": json.dumps(output, indent=2, default=str)[:SHOW_CHARS] + "\n…",
            "truncated": True}


def _error(code: str, message: str) -> Event:
    return "error", {"code": code, "message": message}
