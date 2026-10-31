"""Drug lookup, same-ingredient alternatives and price history."""

import difflib
from typing import Any

import duckdb
import pandas as pd

from irpd import lost
from irpd.agent.registry import tool
from irpd.agent.tools.common import (
    RECENT_DAYS,
    _norm,
    as_of,
    candidates,
    clean,
    d,
    recent,
    records,
    resolve,
)

MAX_ALTERNATIVES = 15
MAX_PRICE_CHANGES = 15

DRUG_ARG = {"type": "string",
            "description": "A drug code, or a drug name (fuzzy matched, e.g. 'zetamol 500')."}


def one_drug(description: str) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "required": ["drug"],
            "properties": {"drug": {**DRUG_ARG, "description": description}}}


# Query words that name a form family (as `lost.FORM_FAMILY`), not an ingredient.
FORM_WORDS = {
    "oral liquid": {"syr", "syrup", "sirup", "susp", "suspension", "suspensi", "liquid",
                    "cair", "elixir"},
    "drops": {"drop", "drops", "tetes"},
    "oral solid": {"tab", "tablet", "tablets", "kaplet", "caplet", "caps", "capsule",
                   "kapsul"},
    "topical": {"cream", "krim", "salep", "ointment", "gel", "lotion", "zalf"},
    "suppository": {"supp", "suppository"},
}
# Indonesian words for a kind of medicine, and spellings fuzzy matching misses.
ALIASES = {
    "maag": ["aluminium hydroxide", "aluminum hydroxide", "magnesium hydroxide",
             "hydrotalcite", "sucralfate", "omeprazole", "lansoprazole", "ranitidine"],
    "lambung": ["aluminium hydroxide", "aluminum hydroxide", "magnesium hydroxide",
                "hydrotalcite", "sucralfate", "omeprazole", "lansoprazole", "ranitidine"],
    "mefenamat": ["mefenamic acid"],
}
STOP_WORDS = {"obat", "the", "and", "untuk", "yang", "anak", "dewasa", "forte"}
MAX_BY_INGREDIENT = 12


def _like(word: str, term: str) -> bool:
    """A query word fits an ingredient word: a prefix ('amlodipin') or a near
    spelling ('parasetamol')."""
    return term.startswith(word) or difflib.SequenceMatcher(None, word, term).ratio() >= 0.85


def by_ingredient(con: duckdb.DuckDBPyConnection, query: str) -> dict[str, Any] | None:
    """Products whose ingredients fit a word of `query`, in the form it names, if any.

    None if no query word names an ingredient. Single-ingredient products come first,
    then by revenue over the last 28 days.
    """
    words = [w for w in _norm(query).split() if not w.isdigit()]
    forms = {f for f, ws in FORM_WORDS.items() if ws & set(words)}
    words = [w for w in words if len(w) >= 4 and w not in STOP_WORDS
             and not any(w in ws for ws in FORM_WORDS.values())]
    g = lost.groups(con)
    names = pd.Series(g.ingredients.str.split(" + ", regex=False).explode().unique())
    # Each ingredient counts the query words it fits; keep those that fit the most.
    fits = {n: sum(n in ALIASES.get(w, []) or any(_like(w, t) for t in _norm(n).split()
                                                  if len(t) >= 4) for w in words)
            for n in names}
    best = max(fits.values(), default=0)
    if not best:
        return None
    hit = {n for n, k in fits.items() if k == best}
    sets = g.ingredients.str.split(" + ", regex=False)
    g = g[sets.map(lambda s: bool(hit & set(s)))].copy()
    if forms:
        g = g[g.family.isin(forms)]
    g["single"] = sets[g.index].map(len).eq(1)
    day = as_of(con)
    df = con.execute(f"""
        SELECT dr.drug_code, dr.drug_name, dr.stock_unit AS unit,
               (SELECT max(date) FROM daily_demand x WHERE x.drug_code = dr.drug_code)
                   AS last_sale,
               (SELECT coalesce(sum(revenue), 0) FROM daily_demand x
                 WHERE x.drug_code = dr.drug_code
                   AND x.date > {d(day)} - {RECENT_DAYS}) AS revenue_28d_rp
        FROM drugs dr WHERE dr.drug_code IN (SELECT unnest($codes))""",
                     {"codes": g.index.tolist()}).df().set_index("drug_code")
    form = con.execute("SELECT drug_code, form FROM ingredient_map").df().set_index("drug_code")
    df = df.join(g[["ingredients", "single"]]).join(form)
    st = recent(con, day, df.index.tolist())
    df = df.join(st[["total", "units_sold"]]).rename(
        columns={"total": "stock_total", "units_sold": "units_sold_28d"})
    df = df.sort_values(["single", "revenue_28d_rp", "last_sale"],
                        ascending=False).reset_index().drop(columns="single")
    r = records(df, MAX_BY_INGREDIENT)
    return {"ingredients_matched": sorted(hit), "form_family": sorted(forms) or None,
            "products": r["rows"], "products_left_out": r["rows_left_out"],
            "note": "Single-ingredient products first. For the full list, use run_sql on "
                    "ingredient_map: ingredients is the sorted set joined by ' + ' (so "
                    "'paracetamol' alone means a single-ingredient product); form is "
                    "e.g. syrup, suspension, liquid, drops, tablet."}


@tool("find_drug",
      "Look up drugs by name. Returns the best name matches with their drug_code, so other "
      "tools can be called with an exact code. 'match' is set when one drug clearly fits. "
      "If the name contains an active ingredient or a kind of medicine (English or "
      "Indonesian, e.g. 'paracetamol syrup', 'amlodipin', 'obat maag'), 'by_ingredient' "
      "also lists the products that contain it, in the form named, with stock and 28-day "
      "sales: brands such as Sanmol or Tempra that the name alone would miss. Use it "
      "whenever the owner names an ingredient or kind of medicine rather than a product.",
      {"type": "object", "additionalProperties": False, "required": ["name"],
       "properties": {"name": {"type": "string",
                               "description": "Part or all of a name, or an ingredient "
                                              "with an optional form."}}})
def find_drug(con: duckdb.DuckDBPyConnection, name: str) -> dict[str, Any]:
    c = candidates(con, name)
    try:
        match = resolve(con, name)[0]
    except LookupError:
        match = None
    c = c[c.score > 0.4].rename(columns={"stock_unit": "unit", "score": "match_score"})
    out = {"match": match, "candidates": clean(c.to_dict("records"))}
    found = by_ingredient(con, name)
    if found:
        out["by_ingredient"] = found
    return out


@tool("alternatives",
      "Drugs with the same active ingredient set as this drug (from the ingredient map), "
      "with form, strength, stock now and sales over the last 28 days. Ingredient mapping "
      "was done by a language model, so treat it as good but not perfect.",
      one_drug("The drug to find alternatives for: code or name."))
def alternatives(con: duckdb.DuckDBPyConnection, drug: str) -> dict[str, Any]:
    code, name = resolve(con, drug)
    day = as_of(con)
    g = lost.groups(con)
    me = con.execute("SELECT ingredients, strength, form FROM ingredient_map "
                     "WHERE drug_code = $1", [code]).fetchone()
    head = {"drug_code": code, "drug_name": name, "as_of": day.isoformat()}
    if code not in g.index:
        return {**head, "mapped": False, "alternatives": [],
                "note": "This drug has no confident ingredient mapping (often a device, "
                        "cosmetic or multi-ingredient product), so alternatives are unknown."}
    ingredients, family = g.at[code, "ingredients"], g.at[code, "family"]
    df = con.execute(f"""
        SELECT m.drug_code, m.drug_name, m.strength, m.form, dr.stock_unit AS unit,
               (SELECT max(date) FROM daily_demand x WHERE x.drug_code = m.drug_code)
                   AS last_sale,
               (SELECT coalesce(sum(revenue), 0) FROM daily_demand x
                 WHERE x.drug_code = m.drug_code
                   AND x.date > {d(day)} - {RECENT_DAYS}) AS revenue_28d_rp
        FROM ingredient_map m JOIN drugs dr USING (drug_code)
        WHERE m.ingredients = $1 AND m.drug_code <> $2""", [ingredients, code]).df()
    st = recent(con, day, df.drug_code.tolist())
    df = df.join(st[["floor", "store", "total", "units_sold"]], on="drug_code")
    df = df.rename(columns={"total": "stock_total", "floor": "stock_floor",
                            "store": "stock_store", "units_sold": "units_sold_28d"})
    df["same_form_family"] = df.drug_code.map(g["family"]).eq(family)
    df = df.sort_values(["same_form_family", "revenue_28d_rp"], ascending=False)
    r = records(df, MAX_ALTERNATIVES)
    return {**head, "mapped": True, "ingredients": ingredients, "strength": me[1],
            "form": me[2], "form_family": family,
            "units": "stock and units sold are in each drug's own base unit; revenue in rupiah",
            "alternatives": r["rows"], "alternatives_left_out": r["rows_left_out"]}


@tool("price_history",
      "List-price history of a drug in rupiah per sales unit (Tab, Strip, Box...), "
      "newest first, up to the current day, with the base units each sales unit holds.",
      one_drug("Drug code or name."))
def price_history(con: duckdb.DuckDBPyConnection, drug: str) -> dict[str, Any]:
    code, name = resolve(con, drug)
    day = as_of(con)
    df = con.execute(f"""
        SELECT p.unit AS sales_unit, k.base_per_unit, p.price AS price_rp,
               p.prev_price AS prev_price_rp, p.valid_from, p.valid_to,
               p.lines AS sale_lines, p.likely_unit_change
        FROM prices p LEFT JOIN pack_sizes k
          ON k.drug_code = p.drug_code AND k.sales_unit = p.unit
        WHERE p.drug_code = $1 AND p.valid_from <= {d(day)}
        ORDER BY p.valid_from DESC, p.unit""", [code]).df()
    current = df[df.valid_to.isna()][["sales_unit", "base_per_unit", "price_rp", "valid_from"]]
    changes = df[df.prev_price_rp.notna() & ~df.likely_unit_change.astype(bool)]
    r = records(df, MAX_PRICE_CHANGES)
    return {"drug_code": code, "drug_name": name, "as_of": day.isoformat(),
            "current": clean(current.to_dict("records")),
            "price_changes_by_unit": clean(changes.groupby("sales_unit").size().to_dict()),
            "history": r["rows"], "history_left_out": r["rows_left_out"],
            "note": "valid_to empty = still the price. One-off blips are already left "
                    "out. price_changes_by_unit counts rows with a previous price, leaving "
                    "out likely_unit_change (a jump to about double or half, likely a "
                    "change of pack, not a price rise)."}
