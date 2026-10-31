"""The contract between the agent's tools and its loop.

A tool is a plain function `fn(con, **args) -> dict`. `con` is the agent's
connection from `holdout.agent_view(as_of)`, so a tool can never see past the
day the agent is standing on. Its result must be JSON-serialisable and small
(cap long tables; say how many rows were left out).

Tools register themselves with `@tool(...)`. The loop only uses `TOOLS`,
`openai_tools()` and `call()`, so either side can change without the other.
Parameter schemas must work in OpenAI strict mode: every property listed in
`required`, `additionalProperties: false`, optional values typed as
`["string", "null"]` and so on.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import duckdb

MAX_RESULT_CHARS = 12_000


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., dict[str, Any]]


TOOLS: dict[str, Tool] = {}


def tool(name: str, description: str, parameters: dict[str, Any]):
    """Register the decorated function as an agent tool."""
    def add(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        if name in TOOLS:
            raise ValueError(f"tool {name!r} registered twice")
        TOOLS[name] = Tool(name, description, parameters, fn)
        return fn
    return add


def openai_tools(tools: dict[str, Tool] | None = None) -> list[dict[str, Any]]:
    """The registered tools (or `tools`) in the Responses API function-tool format."""
    return [{"type": "function", "name": t.name, "description": t.description,
             "parameters": t.parameters, "strict": True} for t in (tools or TOOLS).values()]


def call(name: str, args: dict[str, Any] | str, con: duckdb.DuckDBPyConnection,
         tools: dict[str, Tool] | None = None) -> dict[str, Any]:
    """Run one tool call. Errors come back as {"error": ...} for the model to read."""
    tools = tools or TOOLS
    if name not in tools:
        return {"error": f"unknown tool {name!r}"}
    try:
        kwargs = json.loads(args) if isinstance(args, str) else args
        result = tools[name].fn(con, **kwargs)
    except Exception as e:  # the model should see the error and try again
        return {"error": f"{type(e).__name__}: {e}"}
    text = json.dumps(result, default=str)
    if len(text) > MAX_RESULT_CHARS:
        return {"error": f"result too large ({len(text):,} chars); ask for fewer rows"}
    return result
