"""Rubric scoring for eval v2. Deterministic, no LLM judge.

A v2 question's gold answer is a list of `Check`s with the gold values filled
in. Each check looks at the agent's prose and its structured value together
(the "haystack") and passes or fails; the credit is the weighted share passed,
and a question passes when every check does.

Checks:

- `includes_k_of`: at least `k` of `names` are mentioned (`includes_any` is k=1).
  Names match on whole words, ignoring case, punctuation and "10 mg" vs "10mg".
- `number_near`: at least `k` of `values` are stated, each within
  max(tol, rel * |value|). The structured value counts as stated.
- `says_cant_see_future`: says it can't see past today (keyword sets, EN + ID).
- `says_cant_tell`: says the data can't tell (keywords), or sets cant_tell.
- `corrects_premise`: says the thing the question assumes didn't happen.
- `asks_or_lists_candidates`: asks which one is meant, or names at least `k`
  of the candidates.
- `no_unsupported_numbers`: states no number that no tool produced (dates and
  years aside).

The keyword sets are simple on purpose; `tests/test_rubric.py` checks them
against realistic paraphrases, both ways.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from irpd.evals.scorer import numbers_in, to_number, tool_recall, unsupported_numbers

KINDS = {"includes_any", "includes_k_of", "number_near", "says_cant_see_future",
         "says_cant_tell", "corrects_premise", "asks_or_lists_candidates",
         "no_unsupported_numbers"}


@dataclass(frozen=True)
class Check:
    kind: str
    weight: float = 1.0
    names: tuple[str, ...] = ()     # includes_* and asks_or_lists_candidates
    values: tuple[float, ...] = ()  # number_near
    k: int = 1
    rel: float = 0.0
    tol: float = 0.0

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown check {self.kind!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Short constructors for the question bank.

def includes_k_of(names: list[str], k: int, weight: float = 1.0) -> Check:
    return Check("includes_k_of", weight, names=tuple(names), k=k)


def includes_any(names: list[str], weight: float = 1.0) -> Check:
    return Check("includes_any", weight, names=tuple(names))


def number_near(values: list[float], rel: float = 0.0, tol: float = 0.0, k: int = 1,
                weight: float = 1.0) -> Check:
    return Check("number_near", weight, values=tuple(float(v) for v in values), k=k,
                 rel=rel, tol=tol)


def asks_or_lists(names: list[str], k: int = 2, weight: float = 1.0) -> Check:
    return Check("asks_or_lists_candidates", weight, names=tuple(names), k=k)


def flag(kind: str, weight: float = 1.0) -> Check:
    return Check(kind, weight)


def words(s: Any) -> str:
    """Lowercase words separated by single spaces, "10 mg" joined to "10mg"."""
    s = re.sub(r"[^a-z0-9]+", " ", str(s).lower())
    s = re.sub(r"(\d) (mg|ml|gr|g|mcg|iu|s)\b", r"\1\2", s)
    return " " + " ".join(s.split()) + " "


def haystack(text: str, answer: dict | None) -> str:
    v = (answer or {}).get("value")
    parts = [text] + ([str(x) for x in v] if isinstance(v, list | tuple) else [str(v or "")])
    return "\n".join(parts)


def mentioned(name: str, hay: str) -> bool:
    n = words(name)
    return n.strip() != "" and n in words(hay)


def listed(names: tuple[str, ...], hay: str) -> int:
    """How many candidates are named, in full or by the words only they have: in
    "Amlodipine 10 mg: HJ 100, Medicon 340", HJ and Medicon count."""
    toks = [set(words(n).split()) for n in names]
    h = words(hay)
    count = 0
    for i, n in enumerate(names):
        own = toks[i] - set().union(*(t for j, t in enumerate(toks) if j != i))
        count += mentioned(n, hay) or bool(own) and all(f" {t} " in h for t in own)
    return count


# Dates and years aren't stated quantities: "30 June 2025" shouldn't count as two numbers.
_MONTH = (r"(?:jan(?:uary|uari)?|feb(?:ruary|ruari)?|mar(?:ch|et)?|apr(?:il)?|may|mei|"
          r"jun[ei]?|jul[yi]?|aug(?:ust)?|agustus|sept?(?:ember)?|o[ck]t(?:ober)?|"
          r"nov(?:ember)?|de[cs](?:ember)?)\b\.?")
_DATES = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b"
    rf"|\b\d{{1,2}}(?:\s*[-–]\s*\d{{1,2}})?\s+{_MONTH}(?:\s+\d{{4}})?"
    rf"|\b{_MONTH}\s+\d{{4}}\b"
    rf"|\b{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?\b,?(?:\s+\d{{4}})?"
    r"|\b202[2-6]\b"  # the years the data and questions span
    r"|\bq[1-4]\b|\bh[12]\b", re.IGNORECASE)


def without_dates(text: str) -> str:
    return _DATES.sub(" ", text)


def swap_separators(text: str) -> str:
    """Indonesian number format read the other way: "3.776,5" <-> "3,776.5"."""
    return re.sub(r"\d[\d.,]*\d", lambda m: m.group().translate(str.maketrans(".,", ",.")),
                  text)


def stated_numbers(text: str, answer: dict | None) -> list[float]:
    """Numbers in the value and the prose, read both ways (English and Indonesian format)."""
    v = (answer or {}).get("value")
    if isinstance(v, str):
        text = f"{text}\n{v}"
    out = [x for x in [to_number(v)] if x is not None] if isinstance(v, int | float) else []
    text = without_dates(text)
    return out + numbers_in(text) + numbers_in(swap_separators(text))


# Keyword sets, English and Indonesian.

def _any(patterns: list[str]) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


_NT = r"(?:n[o'’]?t| not)"  # didn't / didnt / did not / didn’t
_CANT = r"(?:can[’']?t|cannot|can not)"
FUTURE = _any([
    rf"{_CANT} see", r"unable to see", r"no visibility",
    r"(?:only|just) (?:goes|go|runs?|extends?|covers?|see|reach(?:es)?|have data)"
    r"(?: data)? (?:up )?(?:to|through|until|till)",
    r"(?:data|records?|information|view)\b.{0,40}\b(?:stops?|ends?|cut ?off)\b",
    r"\b(?:after|beyond|past|later than) (?:the )?(?:cut-?off|today|30 jun|june 30|"
    r"2025-06-30|that date|this date|(?:latest|last|most recent) (?:date|day|data))",
    rf"(?:has|have|had){_NT} (?:happened|come)", r"in the future", r"hasn't happened",
    r"\bno (?:data|records?|sales data|information) (?:yet )?(?:for|on|about|after|beyond)",
    rf"(?:do|does|did){_NT} have (?:any )?(?:data|records?|information|sales data)"
    r" (?:yet )?(?:for|on|about|after|beyond)",
    r"\bnot (?:yet )?(?:available|happened|recorded)", r"\byet to (?:happen|come)",
    r"belum (?:ada|terjadi|tersedia|tercatat|bisa)", r"tidak (?:bisa|dapat) melihat",
    r"masa depan", r"(?:hanya|cuma) sampai", r"(?:setelah|sesudah|lewat) (?:tanggal )?30 jun",
    r"tidak ada data", r"di luar (?:jangkauan|data)",
])
CANT_TELL = _any([
    rf"{_CANT} (?:tell|say|calculate|compute|work out|determine|know|give|estimate)",
    r"(?:unable|not able|no way) to (?:tell|say|calculate|compute|work out|determine|know)",
    r"\bno (?:cost|costs|purchase price|buying price|cost price|supplier price|price paid)",
    rf"(?:does|do|did){_NT} (?:have|include|record|contain|hold|show|track)"
    r" (?:any )?(?:cost|costs|purchase|buying|cost price|supplier price|what we pay|margin)",
    r"(?:has|have|contains?|records?|includes?) no (?:cost|purchase|buying|supplier price)",
    r"without (?:any )?(?:cost|costs|purchase price|buying price)",
    r"\bnot in the (?:data|records)", r"only (?:has|have|records?|shows?|includes?)"
    r" (?:the )?selling prices?",
    r"tidak (?:bisa|dapat|mungkin) (?:dihitung|menghitung|tahu|diketahui|ditentukan|memberi)",
    r"tidak ada (?:data )?(?:harga beli|biaya|modal|harga pokok|hpp)",
    r"tanpa (?:data )?(?:harga beli|biaya|modal|hpp)", r"tidak tercatat",
])
PREMISE = _any([
    rf"(?:did|was|has|had|does|is){_NT} (?:actually |really |ever )?"
    r"(?:run out|go out of stock|stock out|sold out|out of stock|run dry|empty|been out)",
    r"\bnever (?:actually )?(?:ran|run|went|go|was|been) (?:out|empty)",
    r"\bno (?:stock-?outs?|record of (?:a |any )?stock-?out|stock-?out spells?|spells?)",
    r"\b(?:zero|0) (?:stock-?out )?days (?:out|of stock-?out)", r"\b(?:zero|0) stock-?out",
    r"(?:was|stayed|remained|is|it's|has been|had been) (?:fully |well |always )?"
    r"(?:in stock|stocked|available)",
    r"\bnot (?:actually )?(?:out of stock|out|empty|sold out)\b", r"(?:isn't|is not|wasn't) out\b",
    r"\bpremise\b", r"\bin fact\b.{0,60}\b(?:stock|available|sold)",
    r"tidak (?:pernah )?(?:habis|kehabisan|kosong|stok ?out)",
    r"stok(?:nya)? (?:selalu |masih |tetap )?(?:ada|tersedia|aman)", r"tidak ada stok ?out",
])
ASKS = _any([
    r"which (?:one|ones|product|products|brand|brands|of (?:these|them|those|the))",
    r"do you mean", r"did you mean", r"are you (?:asking|referring)",
    r"(?:could|can|would) you (?:clarify|specify|confirm|tell me which|let me know which)",
    r"please (?:clarify|specify|confirm|tell me which|let me know which)",
    r"yang mana", r"\b(?:produk|obat|merek|merk) (?:yang )?mana\b", r"mana yang",
    r"yang (?:anda|bapak|ibu) maksud", r"mohon (?:konfirmasi|perjelas|sebutkan)",
    r"yang dimaksud", r"\b(?:produk|obat|merek|merk) apa\b", r"\bsebutkan\b",
])


def says_cant_see_future(text: str) -> bool:
    return bool(FUTURE.search(text))


def says_cant_tell(text: str) -> bool:
    return bool(CANT_TELL.search(text))


def corrects_premise(text: str) -> bool:
    return bool(PREMISE.search(text))


def asks(text: str) -> bool:
    return bool(ASKS.search(text))


def unsupported(text: str, tool_outputs: list[Any], question: str = "") -> list[float]:
    """`scorer.unsupported_numbers`, with dates and years left out of the prose, and
    the prose read in English or Indonesian number format, whichever fits better."""
    text = without_dates(text)
    return min(unsupported_numbers(text, tool_outputs, question),
               unsupported_numbers(swap_separators(text), tool_outputs, question), key=len)


def passes(c: Check, text: str, answer: dict | None, tool_outputs: list[Any],
           question: str = "") -> bool:
    answer = answer or {}
    hay = haystack(text, answer)
    if c.kind in ("includes_any", "includes_k_of"):
        k = 1 if c.kind == "includes_any" else min(c.k, len(c.names))
        return sum(mentioned(n, hay) for n in c.names) >= k
    if c.kind == "asks_or_lists_candidates":
        return asks(text) or listed(c.names, hay) >= min(c.k, len(c.names))
    if c.kind == "number_near":
        got = stated_numbers(text, answer)
        near = [v for v in c.values
                if any(abs(x - v) <= max(c.tol, c.rel * abs(v)) for x in got)]
        return len(near) >= min(c.k, len(c.values))
    if c.kind == "says_cant_see_future":
        return says_cant_see_future(text)
    if c.kind == "says_cant_tell":
        return bool(answer.get("cant_tell")) or says_cant_tell(text)
    if c.kind == "corrects_premise":
        return corrects_premise(text)
    return not unsupported(text, tool_outputs, question)  # no_unsupported_numbers


def score(checks: list[Check], text: str, answer: dict | None, tool_outputs: list[Any],
          question: str = "") -> tuple[bool, float, list[str]]:
    """(passed, credit, failed check kinds) for one answer."""
    ok = [passes(c, text, answer, tool_outputs, question) for c in checks]
    total = sum(c.weight for c in checks)
    credit = sum(c.weight for c, o in zip(checks, ok, strict=True) if o) / total if total else 1.0
    return all(ok), credit, [c.kind for c, o in zip(checks, ok, strict=True) if not o]


def score_run(questions: list, results: dict[str, dict],
              rubrics: dict[str, list[Check]]) -> pd.DataFrame:
    """One row per question with a rubric, as `scorer.score_run` but graded by rubric.

    `questions` are `QuestionV2`s; `results[id]` holds `answer`, `tools`, `text`
    and `tool_outputs`, as for v1.
    """
    rows = []
    for q in questions:
        if q.id not in rubrics:
            continue
        r = results.get(q.id, {})
        text, outs = r.get("text", ""), r.get("tool_outputs", [])
        passed, credit, failed = score(rubrics[q.id], text, r.get("answer"), outs, q.text)
        rows.append({
            "id": q.id, "category": q.category, "split": q.split, "passed": passed,
            "credit": credit, "tool_recall": tool_recall(q, r.get("tools", [])),
            "unsupported": len(unsupported(text, outs, q.text)), "failed_checks": failed,
        })
    return pd.DataFrame(rows)
