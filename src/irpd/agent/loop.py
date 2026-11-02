"""The agent loop: ask the model, run the tools it calls, feed the results back.

One question runs as a list of Responses API requests, each through
`llm.respond`, so every request is checked against the spend cap and logged.
Tools run through `registry.call` on `holdout.agent_view(as_of)`, so the agent
can't see past the day it stands on.

For evals the loop adds one more function tool, `answer`. The model calls it
once at the end with its reply in prose and a structured value; the scorer
reads both. If the model stops with plain text instead, it gets one nudge
that forces the `answer` call.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from irpd import config, holdout, llm
from irpd.agent import registry

PROMPT_VERSION = "v4"

SYSTEM = """\
You are the data analyst for one retail pharmacy in Indonesia. You answer the \
owner's questions about its sales, stock, purchases, suppliers and prices by \
calling tools on the pharmacy's own records.

Today is {as_of}. You can't see anything after today.

- Every number you state must come from a tool result in this conversation. \
Don't estimate, and don't do sums in your head: get totals, shares and \
differences from a tool.
- The records have no costs, no order dates, no expiry dates and no customer \
IDs. So they can't tell profit or margin, true supplier lead times (order to \
delivery), what will expire, or anything about individual or repeat \
customers. For those, say "the data can't tell" and why, in one sentence. \
Don't pass off a proxy as the answer.
- Say what a number counts: units (the stock ledger's base units, such as \
tablets or bottles) or rupiah.
- Owners use loose names: typos, Indonesian, a brand without its strength, or \
an ingredient or kind of medicine rather than a product. Look the name up \
first with find_drug; for an ingredient or kind of medicine, its \
by_ingredient list has the products that contain it. If it fits more than \
one product, don't silently pick one or give only a total: name each \
product with its own figure (the main ones, if there are many), or ask which \
one is meant.
- If the question states a fact about the data (a stock-out, a date, a \
quantity), check it with a tool before answering. If it's wrong, say so and \
give the right figure.
- Demand forecasts come from two forecasters, shown side by side: the \
current method (the default) and an ML model. Tested on 2025, the ML model's \
weekly forecasts were slightly more accurate in every tier, though not by a \
clear margin, and its reorder plans ran out of stock on about 10% fewer days, \
but they lost about the same revenue. When they differ, give both figures \
and let the owner choose. Plan reorders from the current method unless the \
owner asks for the ML model.
- Keep it short and plain."""

ANSWER_RULES = """

When you're done, call `answer` once with your reply and its value:
- a plain number, no separators or units (3997275634; a percentage as 12.5);
- an ISO date (2025-06-30); true or false; one name; or a list of names, best first.
Set cant_tell to true when the data can't tell, with value null."""

ANSWER_TOOL = {
    "type": "function", "name": "answer", "strict": True,
    "description": "Give the final answer: your reply in prose and its structured value.",
    "parameters": {
        "type": "object", "additionalProperties": False,
        "required": ["text", "value", "cant_tell"],
        "properties": {
            "text": {"type": "string", "description": "Your reply to the owner."},
            "value": {"anyOf": [{"type": "number"}, {"type": "string"}, {"type": "boolean"},
                                {"type": "array", "items": {"type": "string"}},
                                {"type": "null"}]},
            "cant_tell": {"type": "boolean"},
        },
    },
}
NUDGE = "Now call `answer` with your reply and its value."


@dataclass
class Transcript:
    question: str
    model: str
    as_of: str
    steps: list[dict[str, Any]] = field(default_factory=list)  # tokens and cost per request
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # name, args, output
    text: str = ""
    answer: dict[str, Any] | None = None  # {"value": ..., "cant_tell": bool}
    stopped: str = ""  # answer | text | max_steps
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0

    @property
    def tools(self) -> list[str]:
        return [c["name"] for c in self.tool_calls]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def instructions(as_of: date, structured: bool) -> str:
    return SYSTEM.format(as_of=as_of.isoformat()) + (ANSWER_RULES if structured else "")


def _item(x: Any) -> dict[str, Any]:
    """An output item as a plain dict, to send back as input."""
    if hasattr(x, "model_dump"):
        return x.model_dump(exclude_none=True)
    return dict(x) if isinstance(x, dict) else dict(vars(x))


def _message_text(items: list[dict[str, Any]]) -> str:
    return "\n".join(c.get("text", "") for it in items if it.get("type") == "message"
                     for c in it.get("content", []) if c.get("type") == "output_text").strip()


def run(question: str, *, as_of: date = config.CUTOFF, model: str = llm.AGENT_MODEL,
        effort: str = "medium", structured: bool = False, max_steps: int = 8,
        con: Any = None, client: Any = None, ledger: Path | None = None,
        cap: float | None = None, purpose: str = "agent",
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        system: str | None = None, tools: dict[str, registry.Tool] | None = None,
        final_tools: tuple[str, ...] = (), max_output_tokens: int = 6_000) -> Transcript:
    """Answer one question. `structured` adds the `answer` tool for evals.

    `on_event(kind, data)`, if given, sees the run as it happens: `tool_call`
    {id, name, args (JSON text)} before a tool runs, `tool_result` {id, name, output} after,
    `message` {text} for text between calls, and `answer` {text} at the end.

    `system` and `tools` replace the analyst prompt and the registered tools
    (the contest planner uses its own). A successful call to one of
    `final_tools` ends the run, with `stopped` set to its name. `max_output_tokens`
    caps each request (and so the worst case the spend cap reserves).
    """
    emit = on_event or (lambda kind, data: None)
    own = con is None
    con = holdout.agent_view(as_of) if own else con
    specs = registry.openai_tools(tools) + ([ANSWER_TOOL] if structured else [])
    system = system or instructions(as_of, structured)
    t = Transcript(question, model, as_of.isoformat())
    items: list[Any] = [{"role": "user", "content": question}]
    choice: Any = "auto"
    try:
        while len(t.steps) < max_steps:
            resp, r = llm.respond(purpose, system, items, tools=specs, model=model,
                                  tool_choice=choice, effort=effort, client=client,
                                  ledger=ledger, cap=cap,
                                  max_output_tokens=max_output_tokens)
            t.steps.append({"input": r.input_tokens, "cached": r.cached_tokens,
                            "output": r.output_tokens, "usd": r.usd})
            t.input_tokens += r.input_tokens
            t.cached_tokens += r.cached_tokens
            t.output_tokens += r.output_tokens
            t.usd += r.usd
            out = [_item(x) for x in resp.output]
            items += out
            calls = [it for it in out if it.get("type") == "function_call"]
            text = _message_text(out)
            final = next((c for c in calls if c["name"] == "answer"), None)
            if final:
                a = json.loads(final["arguments"])
                t.answer = {"value": a.get("value"), "cant_tell": bool(a.get("cant_tell"))}
                t.text, t.stopped = a.get("text") or text, "answer"
                emit("answer", {"text": t.text})
                return t
            if not calls:
                t.text = text or t.text
                if structured and choice == "auto":
                    items.append({"role": "user", "content": NUDGE})
                    choice = {"type": "function", "name": "answer"}
                    continue
                t.stopped = "text"
                emit("answer", {"text": t.text})
                return t
            if text:
                emit("message", {"text": text})
            for c in calls:
                emit("tool_call", {"id": c["call_id"], "name": c["name"],
                                   "args": c["arguments"]})
                result = registry.call(c["name"], c["arguments"], con, tools)
                emit("tool_result", {"id": c["call_id"], "name": c["name"], "output": result})
                t.tool_calls.append({"name": c["name"], "args": c["arguments"],
                                     "output": result})
                items.append({"type": "function_call_output", "call_id": c["call_id"],
                              "output": json.dumps(result, default=str)})
                if c["name"] in final_tools and "error" not in result:
                    t.text, t.stopped = text, c["name"]
                    return t
        t.stopped = "max_steps"
        return t
    finally:
        if own:
            con.close()
