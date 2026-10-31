"""Map drug names to their active ingredients with an LLM, and score the mapping.

The model gets batches of IRPD product names ("Zetamol tb 500mg") and returns
each one's ingredients (English INN names), strength, form and kind, or
"unknown" when it isn't sure. Answers are cached per drug, so a rerun only
pays for drugs not yet mapped.

The score uses the answer key from `scripts/build_reference_labels.py`, which
comes from public drug lists, not from a model:

- single-ingredient drugs: right only if the model gives exactly that one
  ingredient;
- combinations: the share of the key's ingredients the model found (the key
  can miss ingredients of a combination, so extra ones aren't penalised);
- abstention: how often the model says "unknown" on drugs the key knows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from irpd import config, llm
from irpd.names import same

CACHE = config.ROOT / "data/ingredients/llm_map.jsonl"
OUT = config.ROOT / "data/ingredients/ingredient_map.csv"
KEY = config.ROOT / "data/reference/reference_labels.csv"
PROMPT_VERSION = "v2"
MODEL = llm.BULK_MODEL
# Short names for the CLI. The table prefers the first model with a confident answer.
MODELS = {"sol": llm.AGENT_MODEL, "luna": llm.BULK_MODEL}
BATCH = 100

KINDS = ["drug", "herbal", "supplement", "cosmetic", "device", "food", "service", "unknown"]

INSTRUCTIONS = """You label products sold by a retail pharmacy in Indonesia.
Each product name is as typed into the till: a brand or generic name, often
with strength, form, pack size and a maker abbreviation (e.g. "HJ", "Novel",
"If", "DM", "KF"). Names may be Indonesian.

For each product return:
- ingredients: the active ingredients, as lowercase English INN names without
  salts ("diclofenac", not "diclofenac sodium" or "kalium diklofenak";
  "paracetamol", not "acetaminophen"). List every ingredient of a combination.
  Leave the list empty if you are not confident.
- strength: the strength as written in the name ("500 mg"), or "".
- form: tablet, capsule, syrup, suspension, drops, cream, ointment, gel,
  injection, inhaler, suppository, powder, liquid, other, or "".
- kind: drug, herbal, supplement, cosmetic, device, food, service or unknown.
- confident: true only if you know this product or its ingredients follow
  plainly from the name. If unsure, set confident false and leave ingredients
  empty. A wrong answer is worse than no answer.
"""

# v2: v1 was too cautious (a third of known drugs came back "unknown") and
# left supplements empty. v2 asks for what the model knows about a brand.
INSTRUCTIONS_V2 = INSTRUCTIONS.replace(
    """- confident: true only if you know this product or its ingredients follow
  plainly from the name. If unsure, set confident false and leave ingredients
  empty. A wrong answer is worse than no answer.
""",
    """- For vitamins and supplements, list the vitamins and minerals; for herbal
  products, the main herbs.
- confident: true if you recognise the product (most Indonesian brands are
  well known: use what you know about the brand even when the name doesn't
  say it) or its ingredients follow plainly from the name. Set it false, with
  no ingredients, only if you don't recognise the product. Don't invent.
""")
PROMPTS = {"v1": INSTRUCTIONS, "v2": INSTRUCTIONS_V2}

SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["products"],
    "properties": {"products": {"type": "array", "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "ingredients", "strength", "form", "kind", "confident"],
        "properties": {
            "id": {"type": "integer"},
            "ingredients": {"type": "array", "items": {"type": "string"}},
            "strength": {"type": "string"},
            "form": {"type": "string"},
            "kind": {"type": "string", "enum": KINDS},
            "confident": {"type": "boolean"},
        },
    }}},
}


def drugs_to_map(db: Path) -> pd.DataFrame:
    """Every drug in the ledger, biggest revenue first."""
    with duckdb.connect(str(db), read_only=True) as con:
        return con.execute("""
            SELECT d.drug_code, d.drug_name, d.category,
                   coalesce(sum(s.revenue), 0) AS revenue
            FROM drugs d LEFT JOIN sales_lines s USING (drug_code)
            GROUP BY ALL ORDER BY revenue DESC, d.drug_code""").fetchdf()


def cached(path: Path | None = None, version: str | None = None,
           model: str | None = None) -> dict[str, dict]:
    """Cached answers for one prompt version and model (the defaults if not given)."""
    path = path or CACHE
    version, model = version or PROMPT_VERSION, model or MODEL
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {r["drug_code"]: r for r in rows
            if r.get("prompt") == version and r.get("model") == model}


def map_batch(batch: pd.DataFrame, client: Any = None, cache: Path | None = None,
              version: str | None = None, model: str | None = None) -> float:
    """Map one batch, append the answers to the cache, return the cost in USD."""
    cache = cache or CACHE
    version, model = version or PROMPT_VERSION, model or MODEL
    items = [{"id": i, "name": n, "category": c}
             for i, (n, c) in enumerate(zip(batch["drug_name"], batch["category"], strict=True))]
    result = llm.call(f"ingredient mapping {version}", PROMPTS[version],
                      json.dumps(items, ensure_ascii=False),
                      model=model, schema=SCHEMA, client=client)
    answers = {a["id"]: a for a in json.loads(result.text)["products"]}
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("a") as f:
        for i, code in enumerate(batch["drug_code"]):
            a = answers.get(i)
            if a is None:  # the model skipped it; it will be retried next run
                continue
            f.write(json.dumps({"drug_code": code, "prompt": version,
                                "model": model, **{k: a[k] for k in a if k != "id"}},
                               ensure_ascii=False) + "\n")
    return result.usd


def run(limit: int | None = None, model: str = MODEL, version: str = PROMPT_VERSION,
        client: Any = None, db: Path | None = None) -> pd.DataFrame:
    """Map every drug not yet cached for this model and prompt (or the first `limit`,
    biggest revenue first), then write the merged table."""
    model = MODELS.get(model, model)
    if model not in llm.PRICES:
        raise ValueError(f"unknown model {model!r}; use one of {', '.join(MODELS)} "
                         f"or a full name ({', '.join(llm.PRICES)})")
    drugs = drugs_to_map(db or config.WAREHOUSE)
    todo = drugs[~drugs["drug_code"].isin(cached(version=version, model=model).keys())]
    if limit is not None:
        todo = todo.head(limit)
    total = 0.0
    for start in range(0, len(todo), BATCH):
        total += map_batch(todo.iloc[start:start + BATCH], client, version=version, model=model)
        print(f"  mapped {min(start + BATCH, len(todo)):,}/{len(todo):,} with {model} "
              f"(${total:.4f} this run, ${llm.spent():.4f} total)")
    table = ingredient_table(drugs, version=version)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    return table


def ingredient_table(drugs: pd.DataFrame, version: str | None = None,
                     models: list[str] | None = None) -> pd.DataFrame:
    """One row per drug from the cached answers.

    Takes the first model in `models` (default: sol, then luna) with a
    confident answer; if none is confident, the first answer there is, with no
    ingredients. `source_model` says which model's answer the row uses.
    """
    models = models or list(MODELS.values())
    answers = [(m, cached(version=version, model=m)) for m in models]
    rows = []
    for code, name in zip(drugs["drug_code"], drugs["drug_name"], strict=True):
        found = [(m, a[code]) for m, a in answers if code in a]
        model, a = next((f for f in found if f[1].get("confident")),
                        found[0] if found else ("", {}))
        ingredients = sorted(a.get("ingredients", [])) if a.get("confident") else []
        rows.append({"drug_code": code, "drug_name": name,
                     "ingredients": " + ".join(ingredients),
                     "strength": a.get("strength", ""), "form": a.get("form", ""),
                     "kind": a.get("kind", ""), "mapped": bool(a), "source_model": model})
    return pd.DataFrame(rows)


def score(table: pd.DataFrame, key_path: Path | None = None) -> dict[str, float]:
    """Score the mapping against the high-confidence answer key.

    The key is complete for plain generic names ("Amoxicillin 500mg HJ") but
    often lists only part of a branded combination. So there are two scores:

    - covered: on every drug the model answered, does its answer include all
      of the key's ingredients? Extra ingredients aren't penalised.
    - exact: on generic-name drugs only, is the answer exactly the one ingredient?
    """
    key = pd.read_csv(key_path or KEY, dtype=str).fillna("")
    key = key[key["confidence"] == "high"][["DrugCode", "ingredients", "source"]]
    merged = key.merge(table, left_on="DrugCode", right_on="drug_code", suffixes=("_key", ""))
    merged = merged[merged["mapped"]]

    covered, exact, abstained = [], [], 0
    for ref, got, source in zip(merged["ingredients_key"], merged["ingredients"],
                                merged["source"], strict=True):
        ref_items = [r.strip() for r in ref.split("+") if r.strip()]
        got_items = [g.strip() for g in got.split("+") if g.strip()]
        if not got_items:
            abstained += 1
            continue
        covered.append(all(any(same(r, g) for g in got_items) for r in ref_items))
        if source == "generic_name" and len(ref_items) == 1:
            exact.append(len(got_items) == 1 and same(ref_items[0], got_items[0]))
    n = len(merged)
    return {
        "scored": n,
        "abstained": abstained / n if n else 0.0,
        "covered_n": len(covered),
        "covered": sum(covered) / len(covered) if covered else 0.0,
        "exact_n": len(exact),
        "exact": sum(exact) / len(exact) if exact else 0.0,
    }
