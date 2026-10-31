"""Label IRPD drugs with their active ingredient from a public Indonesian drug list.

These labels are the answer key for the LLM ingredient mapping. They come from
the Indonesian Pharmaceutical Dataset (Mendeley 10.17632/z45bptxptf.2,
CC BY 4.0), not from a model, so the mapping is never scored against itself.

Three ways a drug gets a label, strongest first:

1. generic_name: the IRPD name starts with a generic name, from the list or
   from the FDA's NDC product file (public domain), which fills gaps such as
   paracetamol ("Amoxicillin 500mg HJ" -> amoxicillin).
2. brand_list: the brand appears in a generic's list of trade names. A brand
   listed under several generics is a combination.
3. brand_text: the brand has its own entry, and its description names a
   generic from the list (lower confidence).

Strength and form are parsed from the IRPD name, because the list has neither.
Ingredient names are put into one English form (INN), so "asam mefenamat",
"mefenamic acid" and US names like "acetaminophen" compare equal.

Known limit: a combination brand can come out with only some of its
ingredients, when the list files it under one generic only. Score
combinations on overlap, not exact match.

Only 'high' rows are the answer key. A label is 'low' (reason in `why_low`)
when the name has words the match doesn't cover, when the trade-name list and
the brand's own description name different ingredients, when the brand is
also filed under a mixed group (hormones, multivitamins), or when the brand is
a family of variants with different contents (Komix, Siladex). Precision
matters more than coverage here.

Usage:
    uvx --with duckdb --with pandas python -I scripts/build_reference_labels.py

Inputs (not committed):
    data/raw/irpd/processed_dataset/{sales,stock}_processed.csv
    data/reference/indo_pharma/processed_data_obat.csv
    data/reference/fda_ndc/product.txt (from https://www.accessdata.fda.gov/cder/ndctext.zip)
Output:
    data/reference/reference_labels.csv
"""

import re
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IRPD = ROOT / "data/raw/irpd/processed_dataset"
REF = ROOT / "data/reference/indo_pharma/processed_data_obat.csv"
FDA = ROOT / "data/reference/fda_ndc/product.txt"
OUT = ROOT / "data/reference/reference_labels.csv"

# Words in IRPD names that are packaging or form, not part of the brand.
FORMS = {
    "tab": "tablet", "tb": "tablet", "tablet": "tablet", "tabl": "tablet",
    "kap": "capsule", "kaps": "capsule", "kapsul": "capsule", "caps": "capsule", "cap": "capsule",
    "syr": "syrup", "sir": "syrup", "sirup": "syrup", "syrup": "syrup",
    "susp": "suspension", "suspensi": "suspension",
    "drop": "drops", "drops": "drops", "tetes": "drops",
    "krim": "cream", "cream": "cream", "cr": "cream",
    "salep": "ointment", "zalf": "ointment", "ointment": "ointment",
    "gel": "gel", "inj": "injection", "injeksi": "injection",
    "cair": "liquid", "sachet": "sachet", "supp": "suppository",
    "kaplet": "tablet", "kapl": "tablet", "caplet": "tablet", "oint": "ointment",
}
# Pack words and counts ("strip", "botol", "10s", "5cc"), dropped like forms.
PACKAGING = re.compile(r"^(strip|botol|btl|box|tube|blister|\d+(s|cc|pcs))$")
# Words that name a different product in a brand family ("Transpulmin Kids",
# "Lapisiv T", "Selsun Blue"). A name whose only extra word is one of these,
# or a single letter, isn't trusted to share the bare brand's ingredients.
VARIANT_WORDS = {
    "plus", "forte", "extra", "kids", "kid", "anak", "junior", "baby", "dewasa", "flu",
    "batuk", "pilek", "cool", "hot", "fever", "neuro", "gold", "max", "dm", "dmp", "pe",
    "obh", "expect", "expectorant", "sr", "xr", "mr", "od", "retard", "ds", "blue", "minidose",
    "bb",  # "bayi"/baby
}
STRENGTH = re.compile(r"(\d+(?:[.,]\d+)?)\s*(mg|mcg|g|gr|ml|%|iu)\b", re.I)
BARE_NUMBER = re.compile(r"\b(\d+(?:[.,]\d+)?)\b")
# Phrases that appear inside brand lists but aren't brand names.
LIST_NOISE = re.compile(
    r"\b(bentuk|sirop|sirup|drops?|tablet|kapsul|injeksi|dan|merek|lain)\b", re.I)


# Salt and form words dropped to get a base ingredient ("amlodipine besylate").
SALTS = {
    "hydrochloride", "hcl", "sodium", "potassium", "calcium", "besylate", "besilate",
    "maleate", "mesylate", "succinate", "tartrate", "citrate", "sulfate", "sulphate",
    "phosphate", "acetate", "bromide", "chloride", "hydrobromide", "fumarate",
    "trihydrate", "monohydrate", "dihydrate", "anhydrous", "disodium", "magnesium",
    # esters that don't change the ingredient ("betamethasone dipropionate")
    "propionate", "dipropionate", "valerate", "furoate", "acetonide", "butyrate",
}
# Form or route words that trail an ingredient name in the list
# ("ketoprofen supositoria", "docusate tetes telinga", "levonorgestrel
# kontrasepsi darurat"). The ingredient is the same whatever the route.
FORM_NOISE = {
    "tetes", "topikal", "oral", "injeksi", "suntik", "salep", "krim", "sirup", "tablet",
    "kapsul", "supositoria", "rektal", "inhalasi", "semprot", "ophthalmic", "enema",
    "ovula", "sampo", "mata", "telinga", "hidung", "mulut", "kontrasepsi", "darurat",
}
# Indonesian names put the salt first ("kalium diklofenak" = diclofenac
# potassium). A leading salt word is only dropped when an ingredient follows.
SALT_FIRST = {"kalium", "natrium", "sodium", "potassium", "kalsium", "calcium", "magnesium"}
ANIONS = {
    "klorida", "karbonat", "bikarbonat", "sitrat", "fosfat", "glukonat", "laktat", "sulfat",
    "hidroksida", "oksida", "asetat", "iodida", "fluorida", "carbonate", "bicarbonate",
    "gluconate", "lactate", "hydroxide", "oxide", "iodide", "fluoride",
}
# US names and Indonesian spellings -> the INN used here.
ALIASES = {
    "acetaminophen": "paracetamol", "parasetamol": "paracetamol",
    "albuterol": "salbutamol", "glyburide": "glibenclamide", "epinephrine": "adrenaline",
    "kalsium": "calcium", "vitamin c": "ascorbic acid",
    "aluminium hidroksida": "aluminium hydroxide", "magnesium hidroksida": "magnesium hydroxide",
    "kafein": "caffeine", "metil salisilat": "methyl salicylate",
    "kotrimoksazol": "co-trimoxazole", "zat besi": "iron",
    "methampyrone": "metamizole", "antalgin": "metamizole",
}
# Single vitamins are ingredients; name them as on labels.
ALIASES.update({
    "vitamin a": "retinol", "vitamin b1": "thiamine", "vitamin b2": "riboflavin",
    "vitamin b3 niacin": "niacin", "vitamin b5": "pantothenic acid", "vitamin b6": "pyridoxine",
    "vitamin b9 asam folat": "folic acid", "asam folat": "folic acid",
    "vitamin b12": "cyanocobalamin", "vitamin d": "colecalciferol", "vitamin d3": "colecalciferol",
    "vitamin e": "tocopherol",
})
# List entries that are a drug class or method, not one ingredient. Brands
# filed under them would get a wrong label (Amoxsan -> "penicillin").
CLASS_ENTRIES = re.compile(
    r"^(penicillin|vitamin$|vitamin b$|kb |pil kb|antasida|antibiotik|probiotik|multivitamin|"
    r"suplemen|agonis|antagonis|penghambat|obat |kortikosteroid|muscle relaxant|bile acid|"
    r"anti(ansietas|depresan|emetik|histamin|jamur|malaria|mania|platelet|retroviral)|"
    r"serotonin|statin$|arb$|makrolid|carbapenem|benzodiazepine|barbiturat|bisfosfonat|"
    r"phenothiazine|retinoid|ekspektoran|progesteron$|"
    r"msg|sebamed)", re.I)  # the last two are non-drug entries
# Class entries whose brands mix different ingredients: hormone groups,
# contraceptives, probiotics, multivitamins. A brand filed under one of these
# as well as under a generic has an ingredient (or the real one) the label
# would miss (Andalan, Microgest), so it isn't trusted.
MIXED_CLASSES = re.compile(
    r"^(progesteron$|pil kb|kb |probiotik|multivitamin|suplemen|vitamin b?$)", re.I)

# Phrases that introduce a brand's ingredients in its description.
CONTAINS = re.compile(
    r"(?:mengandung|berisi|kandungan|bahan aktif|terdiri dari|kombinasi)([^.]*)", re.I)


def norm(text: str) -> str:
    """Lowercase, keep letters and digits, collapse spaces."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split())


def strip_route(n: str) -> str:
    """Drop salt and route words after the first word ('ketoprofen supositoria' -> 'ketoprofen')."""
    words = n.split()
    return " ".join(words[:1] + [w for w in words[1:] if w not in SALTS and w not in FORM_NOISE])


def canonical(name: str) -> str:
    """One English (INN) form per ingredient: 'asam mefenamat' -> 'mefenamic acid'."""
    n = norm(name)
    n = ALIASES.get(n, n)
    # Salts come after the base name ("diclofenac potassium"); a first word such
    # as "magnesium" in "magnesium hydroxide" is the ingredient itself.
    n = strip_route(n)
    if n.startswith("asam "):  # Indonesian "asam X" = "X acid"
        n = re.sub(r"at$", "ic", n[5:]) + " acid"
    return ALIASES.get(n, n)


def load_fda_generics() -> set[str]:
    """Single-ingredient generic names from the FDA NDC file, with and without salts."""
    fda = pd.read_csv(FDA, sep="\t", dtype=str, usecols=["SUBSTANCENAME"], encoding="latin1")
    names = set()
    for sub in fda["SUBSTANCENAME"].dropna():
        if ";" in sub:
            continue
        full = norm(sub)
        base = strip_route(full)  # keep a leading "magnesium" or "calcium"
        names.update(n for n in (full, base) if len(n) >= 5)
    # Accept the INN as well as the US name (paracetamol, not only acetaminophen).
    names.update(ALIASES[n] for n in list(names) if n in ALIASES)
    return names


def list_brands(name: str, field: str) -> list[str]:
    """Brand keys from a generic entry's trade-name field.

    The field starts with the generic's name, often shortened or misspelt
    ("sorbitol Microlax, LLax" under "Sorbitol Enema", "variconazole Vfend").
    Drop the exact name if it's there, otherwise the leading lowercase words,
    so the first brand isn't glued to the generic.
    """
    text = field.strip()
    if text.lower().startswith(name.lower()):
        text = text[len(name):]
    else:
        words = text.split(" ")
        while words and words[0][:1].islower():
            words.pop(0)
        text = " ".join(words)
    keys = [norm(LIST_NOISE.sub(" ", b)) for b in text.split(",")]
    return [k for k in keys if k and len(k.split()) <= 4]


def load_reference(fda: set[str]) -> dict:
    """Read the drug list into lookups.

    generics:     normalised generic name -> generic name
    from_lists:   brand -> generics whose trade-name list has it
    from_text:    brand -> generics its own entry's description names
    class_brands: brands also listed under a mixed group (hormones, multivitamins)
    families:     brand stems whose own entry covers several variants (Komix)
    fda:          FDA generic names, passed through
    """
    ref = pd.read_csv(REF, dtype=str).fillna("")
    ref.columns = ["name", "desc", "warning", "dose", "side_effects", "disease", "brands"]
    ref["name"] = ref["name"].str.strip()

    # A row whose trade-name field is a comma list is a generic (ingredient) entry.
    is_list = ref["brands"].str.contains(",")
    is_class = ref["name"].str.match(CLASS_ENTRIES)
    generics = {}
    for n in ref.loc[is_list & ~is_class, "name"]:
        generics[norm(n)] = n.lower()
        generics.setdefault(strip_route(norm(n)), n.lower())  # "levonorgestrel" alone too

    from_lists: dict[str, set[str]] = {}
    class_brands: set[str] = set()
    for i, row in ref[is_list].iterrows():
        for key in list_brands(row["name"], row["brands"]):
            if key in generics:
                continue
            if not is_class[i]:
                from_lists.setdefault(key, set()).add(row["name"].lower())
            elif MIXED_CLASSES.match(row["name"]):
                class_brands.add(key)

    # Brand entries. Prefer the "mengandung ..." clause of the description, where
    # FDA names count too; without one, any list generic the description names.
    # Then the warning's "alergi terhadap ..." clause. Other warning text often
    # names related drugs, so it isn't used.
    vocab = sorted(generics, key=len, reverse=True)
    wide = sorted(set(vocab) | fda, key=len, reverse=True)

    def names_in(text: str, words: list[str]) -> set[str]:
        text = " " + norm(text) + " "
        found = set()
        for g in words:  # longest first, so "asam mefenamat" beats "asam"
            if f" {g} " in text:
                found.add(generics.get(g, g))
                text = text.replace(f" {g} ", " | ")
        return found

    from_text: dict[str, set[str]] = {}
    families: set[str] = set()
    for _, row in ref[~is_list & ~is_class].iterrows():
        if re.search(r"\bvarian\b", row["desc"], re.I):
            families.add(norm(row["name"]))
        clause = CONTAINS.search(row["desc"])
        found = names_in(clause.group(1), wide) if clause else names_in(row["desc"], vocab)
        if not found:
            m = re.search(r"alergi terhadap (?:kandungan )?([^.,;]{3,60})", row["warning"].lower())
            if m:
                found = names_in(m.group(1), vocab)
        if found:
            from_text[norm(row["name"])] = found
    return dict(generics=generics, from_lists=from_lists, from_text=from_text,
                class_brands=class_brands, families=families, fda=fda)


def parse_name(name: str) -> tuple[list[str], str, str]:
    """Split an IRPD name into brand words, strength and form."""
    m = STRENGTH.search(name)
    strength = f"{m.group(1).replace(',', '.')} {m.group(2).lower()}" if m else ""
    if not strength:
        bare = BARE_NUMBER.search(name)
        strength = bare.group(1).replace(",", ".") if bare else ""
    words = norm(STRENGTH.sub(" ", name)).split()
    form = next((FORMS[w] for w in words if w in FORMS), "")
    words = [w for w in words
             if w not in FORMS and not PACKAGING.match(w) and not w.replace(".", "").isdigit()]
    return words, strength, form


def lookup(words: list[str], ref: dict):
    """Longest leading run of words (up to 4) that names a generic or brand."""
    for k in range(min(4, len(words)), 0, -1):
        key = " ".join(words[:k])
        if len(key) < 4:  # too short to trust ("de", "p", "sr")
            continue
        if key in ref["generics"]:
            return k, key, {ref["generics"][key]}, "generic_name"
        if key in ref["fda"]:
            return k, key, {key}, "generic_name"
        if key in ref["from_lists"]:
            return k, key, ref["from_lists"][key], "brand_list"
        if key in ref["from_text"]:
            return k, key, ref["from_text"][key], "brand_text"
    return None


def label(name: str, ref: dict) -> dict:
    words, strength, form = parse_name(name)
    out = {"ingredients": "", "source": "", "matched": "", "confidence": "", "why_low": "",
           "strength": strength, "form": form}
    hit = lookup(words, ref)
    # Salt first ("Kalium Diklofenak"): a match on the salt word alone means
    # the ingredient is the next word, unless only anions follow ("Kalsium Laktat").
    if (hit and hit[0] == 1 and words[0] in SALT_FIRST and len(words) > 1
            and words[1] not in SALTS | ANIONS):
        words = words[1:]
        hit = lookup(words, ref)
    if not hit:
        return out
    k, key, generics, source = hit
    reasons = []
    # A brand match is only trusted when it covers the whole name, apart from
    # one extra word such as a maker suffix ("HJ", "Novel"). Otherwise
    # "Andalan Folamom Multi DHA" would take the label of brand "Andalan".
    # Generic names are trusted with a maker name or a salt after them.
    extra = words[k:]
    if len(extra) > (2 if source == "generic_name" else 1):
        reasons.append("extra words")
    elif source != "generic_name" and any(w in VARIANT_WORDS or len(w) == 1 for w in extra):
        reasons.append("variant of the brand")
    if source == "brand_list":
        text = ref["from_text"].get(key, set())
        if text and not {canonical(g) for g in text} & {canonical(g) for g in generics}:
            # The list and the brand's own entry disagree (Postinor); trust neither.
            reasons.append("list and description disagree")
        elif text:
            generics = generics | text  # the description often names the other half
    if source != "generic_name":
        if key in ref["class_brands"]:
            reasons.append("also filed under a mixed group")
        if key in ref["families"] or words[0] in ref["families"]:
            reasons.append("brand family with several variants")
    out.update(ingredients=" + ".join(sorted({canonical(g) for g in generics})),
               source=source, matched=key, confidence="low" if reasons else "high",
               why_low="; ".join(reasons))
    return out


def main() -> None:
    ref = load_reference(load_fda_generics())
    print(f"reference: {len(ref['generics']):,} generics, {len(ref['from_lists']):,} brands "
          f"from lists, {len(ref['from_text']):,} brands from text, "
          f"{len(ref['fda']):,} FDA generic names")

    con = duckdb.connect()
    drugs = con.execute(f"""
        WITH names AS (
            SELECT DrugCode, any_value(DrugName) DrugName, any_value(Category) Category
            FROM read_csv_auto('{IRPD / "stock_processed.csv"}', sample_size=-1)
            GROUP BY 1),
        revenue AS (
            SELECT DrugCode, sum(Subtotal) revenue, count(*) lines
            FROM read_csv_auto('{IRPD / "sales_processed.csv"}', sample_size=-1)
            GROUP BY 1)
        SELECT n.*, coalesce(r.revenue, 0) revenue, coalesce(r.lines, 0) lines
        FROM names n LEFT JOIN revenue r USING (DrugCode)
        ORDER BY revenue DESC""").fetchdf()

    labels = pd.DataFrame([label(n, ref) for n in drugs["DrugName"]])
    out = pd.concat([drugs, labels], axis=1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(out):,} drugs")

    # Coverage, by drugs and by revenue.
    total = out["revenue"].sum()
    selling = out[out["lines"] > 0]
    cov = (out.assign(source=out["source"].replace("", "unlabelled"))
              .groupby(["source", "confidence"])
              .agg(drugs=("DrugCode", "count"), revenue=("revenue", "sum")))
    cov["revenue_share"] = (cov["revenue"] / total).round(3)
    print("\ncoverage by source and confidence")
    print(cov.drop(columns="revenue").to_string())
    high = out["confidence"] == "high"
    print(f"\nanswer key (high confidence): {high.sum():,} drugs, "
          f"{out.loc[high, 'revenue'].sum() / total:.1%} of revenue")
    print(f"high confidence: {(selling['confidence'] == 'high').mean():.1%} of the "
          f"{len(selling):,} drugs that sell, "
          f"{(out.head(300)['confidence'] == 'high').mean():.1%} of the top 300 by revenue")
    combos = out["ingredients"].str.contains(" + ", regex=False).sum()
    print(f"combinations: {combos:,} drugs")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    cols = ["DrugName", "Category", "ingredients", "source", "strength", "form"]
    print("\nlargest unlabelled drugs by revenue (to check the gaps)")
    gaps = out[out["source"] == ""].head(20)
    print(gaps[["DrugName", "Category", "lines"]].to_string(index=False))
    print("\nlow-confidence matches (kept out of the answer key), sample")
    print(out[out["confidence"] == "low"].head(10)[cols + ["matched"]].to_string(index=False))
    print("\nrandom sample of the answer key to spot-check by hand")
    print(out[high].sample(30, random_state=0)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
