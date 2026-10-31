"""Run the eval questions through the agent loop and score them.

`irpd evals [--set v1|v2] [--split dev|test] [--ids a,b] [--limit n]
[--model sol|luna|name] [--ledger path] [--dry-run]`

v1 is the 60 exact-answer questions (`irpd.evals.questions`); v2 the harder,
rubric-graded set (`irpd.evals.questions_v2`), split into dev and test halves.

Each answer is cached per (question, model, prompt version, tool set), so a
rerun only pays for questions it hasn't answered yet. Before any call, the
run's cost is estimated (from past answers' tokens, or a prior) and the run
refuses if the budget left can't cover it. Results go to
`data/evals/<run_id>.jsonl`, one line per question.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from irpd import config, holdout, llm
from irpd.agent import loop, registry
from irpd.evals import QUESTIONS, Question, gold_answers, rubric, scorer
from irpd.evals.questions_v2 import QUESTIONS_V2, QuestionV2, gold_rubrics

RESULTS = config.ROOT / "data/evals"
MAP = config.ROOT / "data/ingredients/ingredient_map.csv"
# Tokens per question (input, cached input, output) until real answers exist.
PRIOR = (30_000, 12_000, 4_000)
MODELS = {"sol": llm.AGENT_MODEL, "luna": llm.BULK_MODEL}


def select(ids: list[str] | None = None, limit: int | None = None, with_map: bool = False,
           qset: str = "v1", split: str | None = None) -> list[Question] | list[QuestionV2]:
    if qset == "v2":
        bank = qs = [q for q in QUESTIONS_V2 if split in (None, q.split)]
    else:
        bank = QUESTIONS
        qs = [q for q in QUESTIONS
              if q.needs is None or (q.needs == "ingredient_map" and with_map)]
    if ids:
        unknown = set(ids) - {q.id for q in bank}
        if unknown:
            raise ValueError(f"unknown question ids: {sorted(unknown)}")
        qs = [q for q in qs if q.id in ids]
    return qs[:limit] if limit else qs


def cache_key(qid: str, model: str) -> str:
    tools = json.dumps(registry.openai_tools(), sort_keys=True)
    return f"{qid}|{model}|{loop.PROMPT_VERSION}|{hashlib.sha1(tools.encode()).hexdigest()[:8]}"


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    recs = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {r["key"]: r for r in recs}


def estimate(n: int, model: str, cache: dict[str, dict[str, Any]]) -> float:
    """Cost of `n` new questions: past answers' mean tokens (this model's if any), else PRIOR."""
    past = [r for r in cache.values() if r["model"] == model] or list(cache.values())
    if past:
        tokens = [sum(r[k] for r in past) / len(past)
                  for k in ("input_tokens", "cached_tokens", "output_tokens")]
    else:
        tokens = PRIOR
    return n * llm.cost(model, *tokens)


def ask(q: Question | QuestionV2, model: str, **kw: Any) -> dict[str, Any]:
    t = loop.run(q.text, model=model, structured=True, purpose=f"eval:{q.id}", **kw)
    rec = t.to_dict()
    rec.update(id=q.id, tools=t.tools, tool_outputs=[c["output"] for c in t.tool_calls])
    return rec


def run(ids: list[str] | None = None, limit: int | None = None,
        model: str = llm.AGENT_MODEL, *, out: Path = RESULTS, map_path: Path = MAP,
        db: Path = config.WAREHOUSE, agent: Path = holdout.AGENT, qset: str = "v1",
        split: str | None = None, ledger: Path | None = None, cap: float | None = None,
        dry_run: bool = False, **kw: Any) -> tuple[pd.DataFrame, float]:
    """Run and score the questions. Returns the scored table and this run's estimate.

    v1 gold comes from the warehouse `db`; v2 rubrics from the sealed `agent` file.
    """
    model = MODELS.get(model, model)
    imap = pd.read_csv(map_path, dtype=str) if map_path.exists() else None
    qs = select(ids, limit, with_map=imap is not None, qset=qset, split=split)
    cache_path = out / "cache.jsonl"
    cache = load_cache(cache_path)
    new = [q for q in qs if cache_key(q.id, model) not in cache]
    est = estimate(len(new), model, cache)
    left = (llm.CAP_USD if cap is None else cap) - llm.spent(ledger)
    print(f"{len(qs)} questions, {len(new)} not cached; estimated ${est:.3f} on {model}, "
          f"${left:.3f} left under the cap")
    if dry_run:
        return pd.DataFrame(), est
    if est > left:
        raise llm.BudgetExceeded(f"estimated ${est:.3f} but only ${left:.3f} left")

    records: dict[str, dict[str, Any]] = {}
    for q in qs:
        key = cache_key(q.id, model)
        if key in cache:
            records[q.id] = {**cache[key], "cached": True}
            continue
        try:
            rec = ask(q, model, ledger=ledger, cap=cap, **kw)
        except llm.BudgetExceeded as e:
            print(f"stopped at {q.id}: {e}")
            break
        except Exception as e:  # an API error: record it, don't cache it
            records[q.id] = {"id": q.id, "error": f"{type(e).__name__}: {e}", "usd": 0.0}
            print(f"{q.id}: {records[q.id]['error']}")
            continue
        rec["key"] = key
        out.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        records[q.id] = {**rec, "cached": False}
        print(f"{q.id}: {rec['stopped']}, {len(rec['tools'])} tool calls, ${rec['usd']:.4f}")

    if qset == "v2":
        gold = {k: v for k, v in gold_rubrics(agent).items() if k in records}
        df = rubric.score_run(qs, records, gold)
        gold = {k: [c.to_dict() for c in v] for k, v in gold.items()}
    else:
        gold = {k: v for k, v in gold_answers(db, ingredient_map=imap).items() if k in records}
        df = scorer.score_run(records, gold)
    if df.empty:
        return df, est
    # A question the API failed on wasn't answered: no credit, even for checks
    # an empty reply passes (such as "no unsupported numbers").
    failed = df["id"].map(lambda i: "error" in records[i])
    df.loc[failed, ["passed", "credit"]] = [False, 0.0]
    df["usd"] = [records[i].get("usd", 0.0) for i in df["id"]]
    df["cached"] = [records[i].get("cached", False) for i in df["id"]]
    df["gold"] = [gold[i] for i in df["id"]]
    df["got"] = [records[i].get("answer") or records[i].get("error") for i in df["id"]]
    df["said"] = [" ".join((records[i].get("text") or "").split())[:240] for i in df["id"]]

    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{qset}-{model}"
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{run_id}.jsonl").open("w") as f:
        for row in df.to_dict("records"):
            f.write(json.dumps({**records[row["id"]], **row, "run_id": run_id},
                               default=str) + "\n")
    print(f"wrote {out / f'{run_id}.jsonl'}")
    print(report(df))
    return df, est


def _table(df: pd.DataFrame, by: str) -> str:
    return df.groupby(by).agg(n=("id", "size"), passed=("passed", "mean"),
                              credit=("credit", "mean"), recall=("tool_recall", "mean"),
                              unsupported=("unsupported", "mean"),
                              usd=("usd", "mean")).round(4).to_string()


def report(df: pd.DataFrame) -> str:
    n = len(df)
    lines = [
        f"\n{n} questions: passed {df['passed'].mean():.0%}, credit {df['credit'].mean():.2f}, "
        f"tool recall {df['tool_recall'].mean():.2f}, "
        f"unsupported numbers {df['unsupported'].mean():.2f} per answer",
        f"cost ${df['usd'].sum():.4f} (${df['usd'].mean():.4f} per question; "
        f"{int(df['cached'].sum())} cached, free this run)",
        "", "by category:", _table(df, "category"),
    ]
    if "split" in df:
        lines += ["", "by split:", _table(df, "split")]
    lines += ["", "per question:",
              df[["id", "passed", "credit", "tool_recall", "unsupported", "usd"]]
              .round(4).to_string(index=False)]
    failed = df[~df["passed"]]
    if len(failed):
        lines += ["", "failures:"]
        if "failed_checks" in df:
            lines += [f"  {r.id}: failed {r.failed_checks}, value {r.got!r}\n    said: {r.said}"
                      for r in failed.itertuples()]
        else:
            lines += [f"  {r.id}: gold {r.gold!r}, got {r.got!r}" for r in failed.itertuples()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="irpd evals", description="run the agent on the evals")
    ap.add_argument("--ids", help="comma-separated question ids")
    ap.add_argument("--limit", type=int, help="run at most this many questions")
    ap.add_argument("--model", default="sol", help="sol, luna, or a full model name")
    ap.add_argument("--max-steps", type=int, default=8, help="model requests per question")
    ap.add_argument("--set", dest="qset", choices=["v1", "v2"], default="v1",
                    help="question set: v1 (exact answers) or v2 (rubric-graded)")
    ap.add_argument("--split", choices=["dev", "test"], help="v2 only: one half")
    ap.add_argument("--ledger", type=Path, help="spend ledger (default data/spend.jsonl)")
    ap.add_argument("--dry-run", action="store_true", help="print the cost estimate only")
    args = ap.parse_args(argv)
    ids = args.ids.split(",") if args.ids else None
    run(ids, args.limit, args.model, qset=args.qset, split=args.split, ledger=args.ledger,
        dry_run=args.dry_run, max_steps=args.max_steps)
    print(f"spent ${llm.spent(args.ledger):.4f} of ${llm.CAP_USD:.2f}")


if __name__ == "__main__":
    main()
