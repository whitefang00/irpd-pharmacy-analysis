"""Score an agent's answers against the gold answers. Deterministic, no LLM.

The agent gives a structured answer per question: `{"value": ..., "cant_tell": bool}`.
Values are plain numbers, ISO dates ("2025-06-30"), true/false, a name, or a
list of names. Three things are measured:

- accuracy: `score_answer` gives pass/fail and a credit in [0, 1]. Lists get
  partial credit: 'top' lists the share of the gold top k found in the
  agent's first k; 'set' lists the Jaccard overlap. Everything else is 1 or 0.
  "Can't tell" questions pass only if the agent says it can't tell, and any
  other question fails if it does.
- tool choice: `tool_recall`, the share of the question's expected tools the
  agent called.
- unsupported numbers: `unsupported_numbers`, numbers in the agent's prose
  that appear in no tool output and not in the question.
"""

import json
import re
from datetime import date
from typing import Any

import pandas as pd

from irpd.evals.questions import QUESTIONS, Question

# Relative slack when matching a number in prose to a tool output (rounding).
PROSE_REL = 0.005

_MULT = {"k": 1e3, "rb": 1e3, "ribu": 1e3, "thousand": 1e3,
         "m": 1e6, "jt": 1e6, "juta": 1e6, "million": 1e6,
         "b": 1e9, "bn": 1e9, "miliar": 1e9, "billion": 1e9}
_NUM = re.compile(r"(?:(?<=Rp)|(?<![\w.]))-?\d[\d,]*(?:\.\d+)?"  # Rp3,000 counts
                  r"(?:\s?(" + "|".join(_MULT) + r")\b)?", re.IGNORECASE)


def norm(s: Any) -> str:
    return " ".join(str(s).split()).casefold()


def to_number(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, int | float):
        return float(v)
    try:
        return float(re.sub(r"(?i)rp|[%,\s]", "", str(v)))
    except ValueError:
        return None


def to_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    return {"yes": True, "true": True, "no": False, "false": False}.get(norm(v))


def to_date(v: Any) -> date | None:
    try:
        return v if isinstance(v, date) else date.fromisoformat(str(v).strip()[:10])
    except ValueError:
        return None


def list_credit(gold: list, got: list, mode: str) -> float:
    g = {norm(x) for x in gold}
    if mode == "set":
        a = {norm(x) for x in got}
        return 1.0 if not g and not a else len(g & a) / len(g | a)
    a = {norm(x) for x in got[:len(gold)]}
    return 1.0 if not g else len(g & a) / len(g)


def score_answer(q: Question, gold: Any, answer: dict | None) -> tuple[bool, float]:
    """(passed, credit) for one answer."""
    answer = answer or {}
    said_cant = bool(answer.get("cant_tell"))
    if q.kind == "cant_tell" or said_cant:
        ok = q.kind == "cant_tell" and said_cant
        return ok, float(ok)
    v = answer.get("value")
    if q.kind == "list":
        got = list(v) if isinstance(v, list | tuple) else []
        credit = list_credit(gold or [], got, q.list_mode)
        return credit == 1.0, credit
    if q.kind == "number":
        a, g = to_number(v), to_number(gold)
        ok = a is not None and g is not None and abs(a - g) <= max(q.tol, q.rel * abs(g))
    elif q.kind == "date":
        a, g = to_date(v), to_date(gold)
        ok = a is not None and g is not None and abs((a - g).days) <= q.tol
    elif q.kind == "yes_no":
        ok = to_bool(v) is not None and to_bool(v) == to_bool(gold)
    else:  # label
        ok = v is not None and gold is not None and norm(v) == norm(gold)
    return ok, float(ok)


def tool_recall(q: Question, called: list[str]) -> float:
    return len(set(q.tools) & set(called)) / len(q.tools) if q.tools else 1.0


def numbers_in(text: str) -> list[float]:
    out = []
    for m in _NUM.finditer(text):
        raw = m.group(0)
        mult = _MULT.get((m.group(1) or "").lower(), 1.0)
        digits = raw[:m.start(1) - m.start(0)] if m.group(1) else raw
        out.append(float(digits.replace(",", "").strip()) * mult)
    return out


def unsupported_numbers(text: str, tool_outputs: list[Any], question: str = "") -> list[float]:
    """Numbers stated in `text` that match no tool output and aren't in the question.

    A match allows rounding (PROSE_REL), and a share written as a percent
    (0.425 in a tool, 42.5% in prose).
    """
    sources = set(numbers_in(question))
    for out in tool_outputs:
        s = out if isinstance(out, str) else json.dumps(out, default=str)
        sources.update(numbers_in(s))
    known = sources | {x * 100 for x in sources} | {x / 100 for x in sources}

    def supported(x: float) -> bool:
        return any(abs(x - k) <= max(PROSE_REL * abs(k), 0.5 * _last_digit(x)) for k in known)

    # List markers ("1. Flazen", "2) Teamic") are not stated numbers.
    prose = re.sub(r"(?m)^\s*\d{1,2}[.)]\s", " ", text)
    return [x for x in numbers_in(prose) if not supported(x)]


def _last_digit(x: float) -> float:
    """Size of the last stated digit: 1 for 42, 0.1 for 42.5; 0 for 0."""
    s = repr(x).rstrip("0").rstrip(".")
    return 10 ** -len(s.split(".")[1]) if "." in s else 1.0 if x else 0.0


def score_run(results: dict[str, dict], gold: dict[str, Any]) -> pd.DataFrame:
    """One row per question in `gold`.

    `results[id]` holds what the agent did: `answer` (the structured answer),
    `tools` (names called), `text` (its prose) and `tool_outputs`.
    """
    rows = []
    for q in QUESTIONS:
        if q.id not in gold:
            continue
        r = results.get(q.id, {})
        passed, credit = score_answer(q, gold[q.id], r.get("answer"))
        rows.append({
            "id": q.id, "category": q.category, "passed": passed, "credit": credit,
            "tool_recall": tool_recall(q, r.get("tools", [])),
            "unsupported": len(unsupported_numbers(r.get("text", ""),
                                                   r.get("tool_outputs", []), q.text)),
        })
    return pd.DataFrame(rows)
