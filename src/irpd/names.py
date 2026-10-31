"""Compare ingredient names written in different ways.

Labels, drug lists and product pages name the same ingredient differently:
English or Indonesian ("diclofenac", "diklofenak"), with or without a salt
("kalium diklofenak 50 mg"), as a synonym ("methampyrone" for metamizole) or
as a formula ("Mg(OH)2"). `same()` treats all of these as one ingredient.
"""

import difflib
import re

# Salt words, English and Indonesian, dropped before comparing ingredients.
SALTS = {
    "hydrochloride", "hcl", "sodium", "potassium", "besylate", "besilate", "maleate",
    "mesylate", "succinate", "tartrate", "citrate", "sulfate", "sulphate", "phosphate",
    "acetate", "bromide", "chloride", "fumarate", "trihydrate", "monohydrate", "dihydrate",
    "kalium", "natrium", "hidroklorida", "besilat", "maleat", "mesilat", "suksinat",
    "tartrat", "sitrat", "sulfat", "fosfat", "asetat", "bromida", "klorida", "fumarat",
    "trihidrat", "monohidrat", "dihidrat", "anhidrat", "anhydrous",
    # esters and similar suffixes that don't change the ingredient
    "propionate", "propionat", "furoate", "furoat", "acetonide", "asetonid", "butylbromide",
    "butilbromida", "hbr", "hydrobromide", "mononitrate", "valerate", "dipropionate",
}
# Route and form words that trail an ingredient name ("ketoprofen supositoria",
# "docusate tetes telinga"). The ingredient is the same whatever the route.
ROUTE_WORDS = {
    "tetes", "topikal", "oral", "injeksi", "suntik", "salep", "krim", "sirup", "tablet",
    "kapsul", "supositoria", "rektal", "inhalasi", "semprot", "ophthalmic", "enema",
    "ovula", "sampo", "mata", "telinga", "hidung", "mulut", "kontrasepsi", "darurat",
}
# Other names for the same ingredient, and chemical formulas used on labels.
SYNONYMS = {
    "methampyrone": "metamizole", "metampiron": "metamizole", "antalgin": "metamizole",
    "dipyrone": "metamizole", "acetaminophen": "paracetamol", "albuterol": "salbutamol",
    "glyceryl guaiacolate": "guaifenesin", "gliseril guaiakolat": "guaifenesin",
    "dimethylpolysiloxane": "simethicone", "dimetilpolisiloksan": "simethicone",
    "ctm": "chlorpheniramine", "klorfeniramin": "chlorpheniramine", "peg": "macrogol",
    "polyethylene glycol": "macrogol", "plantago ovata": "psyllium", "ispaghula": "psyllium",
    # single vitamins, by number or by name
    "vitamin a": "retinol", "vitamin b1": "thiamine", "vitamin b2": "riboflavin",
    "vitamin b3": "niacin", "nicotinic acid": "niacin", "vitamin b5": "pantothenic acid",
    "vitamin b6": "pyridoxine", "vitamin b9": "folic acid", "asam folat": "folic acid",
    "vitamin b12": "cyanocobalamin", "vitamin d3": "colecalciferol",
    "cholecalciferol": "colecalciferol", "vitamin e": "tocopherol",
    "vitamin c": "ascorbic acid", "asam askorbat": "ascorbic acid", "vitamin d": "colecalciferol",
    "acetosal": "aspirin", "asetosal": "aspirin", "acetylsalicylic acid": "aspirin",
    "asam asetilsalisilat": "aspirin", "allium sativum": "garlic", "oleum allium sativum": "garlic",
    "zat besi": "iron", "besi": "iron", "fe": "iron", "ferrous": "iron", "ferric": "iron",
}
# "Vitamin B complex" on a label stands for these.
B_COMPLEX = re.compile(r"\b(vit(amin)?\s*)?b\s*-?\s*(komplek|kompleks|complex)\b", re.I)
B_VITAMINS = "thiamine, riboflavin, niacin, pantothenic acid, pyridoxine, cyanocobalamin"
# Micronised purified flavonoid fraction is diosmin with hesperidin.
MPFF = re.compile(r"\bmicroni[sz]ed purified flavonoid fraction\b|\bmpff\b", re.I)
UNITS = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|g|ml|iu|%)?(?:\s*/\s*\d*\s*(?:ml|g|gram))?\b", re.I)


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split())


def key(word: str) -> str:
    """Spelling-neutral form, so 'diclofenac' and 'diklofenak' compare equal."""
    w = word.replace("ph", "f").replace("x", "ks").replace("c", "k").replace("y", "i")
    return re.sub(r"(.)\1", r"\1", w)


def ingredient_words(name: str) -> list[str]:
    # Chemical formulas first, before "Mg" can be read as milligrams.
    name = re.sub(r"(?i)mg\s*\(\s*oh\s*\)\s*2", "magnesium hydroxide", name)
    name = re.sub(r"(?i)al\s*\(\s*oh\s*\)\s*3", "aluminium hydroxide", name)
    n = norm(UNITS.sub(" ", name))
    n = re.sub(r"^(per \d+ ml|tiap \d+ ml|tiap|per)\s+", "", n)
    n = re.sub(r"^vit ", "vitamin ", n)
    # A synonym can lead a longer name ("vitamin b1 thiamine mononitrate").
    syn = next((k for k in sorted(SYNONYMS, key=len, reverse=True)
                if n == k or n.startswith(k + " ")), "")
    n = SYNONYMS[syn] if syn else n
    if n.startswith("asam "):  # "asam mefenamat" = "mefenamic acid"
        n = re.sub(r"at$", "ic", n[5:]) + " acid"
    n = SYNONYMS.get(n, n)
    words = [w for w in n.split() if not w.isdigit() and w not in {"mg", "ml", "mcg", "g", "iu"}]
    words = words[:1] + [w for w in words[1:] if w not in ROUTE_WORDS]
    kept = [w for w in words if w not in SALTS] or words  # "kalium" alone is still kalium
    return [key(w) for w in kept]


def same(a: str, b: str) -> bool:
    wa, wb = ingredient_words(a), ingredient_words(b)
    if not wa or not wb:
        return False
    return all(any(difflib.SequenceMatcher(None, x, y).ratio() >= 0.8 for y in wb) for x in wa)
