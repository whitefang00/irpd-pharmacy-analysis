"""OpenAI calls with a hard spend cap.

Every call is logged to a spend ledger (`data/spend.jsonl`) with its tokens
and cost. Before a call, the worst case (the input plus the maximum output)
is added to what has been spent so far; if that would pass the cap, the call
is refused. So the cap holds even if a call runs to its output limit.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from irpd import config

LEDGER = config.ROOT / "data/spend.jsonl"
CAP_USD = 1.00

# USD per million tokens: input, cached input, output (Oct 2026 list prices).
PRICES = {
    "gpt-6.1-sol": (2.00, 0.10, 10.00),
    "gpt-6-luna": (0.10, 0.01, 0.50),
}
AGENT_MODEL = "gpt-6.1-sol"
BULK_MODEL = "gpt-6-luna"


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Result:
    text: str
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    usd: float


def cost(model: str, input_tokens: int, cached: int, output: int) -> float:
    p_in, p_cached, p_out = PRICES[model]
    return ((input_tokens - cached) * p_in + cached * p_cached + output * p_out) / 1e6


def spent(ledger: Path | None = None) -> float:
    ledger = ledger or LEDGER
    if not ledger.exists():
        return 0.0
    return sum(json.loads(line)["usd"] for line in ledger.read_text().splitlines() if line)


def load_key() -> str:
    """The API key from the environment, or from the repo's git-ignored .env."""
    if key := os.environ.get("OPENAI_API_KEY"):
        return key
    env = config.ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")
    raise RuntimeError("OPENAI_API_KEY not set (environment or .env)")


def _check(model: str, prompt_chars: int, max_output_tokens: int, ledger: Path,
           cap: float) -> None:
    """Refuse a call whose worst case (rough input plus full output) would pass the cap."""
    est_input = prompt_chars // 3 + 50  # rough, on the high side
    worst = cost(model, est_input, 0, max_output_tokens)
    so_far = spent(ledger)
    if so_far + worst > cap:
        raise BudgetExceeded(f"spent ${so_far:.4f}; this call could cost up to ${worst:.4f}, "
                             f"cap is ${cap:.2f}")


def _log(purpose: str, model: str, resp: Any, ledger: Path) -> Result:
    """Write the call's tokens and cost to the ledger."""
    usage = resp.usage
    cached = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0
    usd = cost(model, usage.input_tokens, cached, usage.output_tokens)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as f:
        f.write(json.dumps({
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "purpose": purpose, "model": model, "input": usage.input_tokens,
            "cached": cached, "output": usage.output_tokens, "usd": round(usd, 6),
        }) + "\n")
    return Result(resp.output_text, usage.input_tokens, cached, usage.output_tokens, usd)


def _client(client: Any) -> Any:
    if client is None:
        from openai import OpenAI
        client = OpenAI(api_key=load_key())
    return client


def call(
    purpose: str,
    instructions: str,
    prompt: str,
    *,
    model: str = BULK_MODEL,
    schema: dict[str, Any] | None = None,
    effort: str = "medium",
    max_output_tokens: int = 8_000,
    client: Any = None,
    ledger: Path | None = None,
    cap: float | None = None,
) -> Result:
    """One Responses API call, refused if its worst case would pass the cap."""
    ledger = ledger or LEDGER
    _check(model, len(instructions) + len(prompt), max_output_tokens, ledger,
           CAP_USD if cap is None else cap)
    kwargs: dict[str, Any] = {
        "model": model, "instructions": instructions, "input": prompt,
        "reasoning": {"effort": effort}, "max_output_tokens": max_output_tokens,
    }
    if schema:
        kwargs["text"] = {"format": {"type": "json_schema", "name": "result",
                                     "schema": schema, "strict": True}}
    resp = _client(client).responses.create(**kwargs)
    return _log(purpose, model, resp, ledger)


def respond(
    purpose: str,
    instructions: str,
    items: list[Any],
    *,
    tools: list[dict[str, Any]],
    model: str = AGENT_MODEL,
    tool_choice: Any = "auto",
    effort: str = "medium",
    max_output_tokens: int = 6_000,
    client: Any = None,
    ledger: Path | None = None,
    cap: float | None = None,
) -> tuple[Any, Result]:
    """One agent turn: input items and function tools, under the same cap and ledger.

    Returns the raw response (for its output items) and its `Result`.
    """
    ledger = ledger or LEDGER
    chars = len(instructions) + len(json.dumps([items, tools], default=str))
    _check(model, chars, max_output_tokens, ledger, CAP_USD if cap is None else cap)
    resp = _client(client).responses.create(
        model=model, instructions=instructions, input=items, tools=tools,
        tool_choice=tool_choice, reasoning={"effort": effort},
        max_output_tokens=max_output_tokens)
    return resp, _log(purpose, model, resp, ledger)
