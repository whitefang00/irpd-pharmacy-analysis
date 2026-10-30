"""EDA route 5: switching to an alternative (pillar 3 go/no-go).

Question: when a drug is out of stock, do customers buy another drug with the
same active ingredient and strength instead?

Steps
1. Group drugs by ingredient + strength + form, using two transparent rules:
   generic names that start with an ingredient ("Amlodipine 10mg Novel"), and a
   small hand-written brand dictionary ("Zetamol tb 500mg" -> paracetamol 500).
2. Rebuild end-of-day stock per drug from the raw ledger (per location, then
   summed) and find stock-out spells of drugs that normally sell.
3. For each spell, compare the alternatives' daily sale lines with a weekday-
   matched baseline from the 8 weeks before. Pool across spells, bootstrap the
   spells for uncertainty, and compare with two placebos.
4. Whole-group stock-outs, when no same-ingredient alternative was in stock.
5. Batch and expiry: how close to expiry stock is sold.

The outcome is sale *lines* (roughly, customers), not units, because brands of
the same ingredient sell in different units (tablet vs strip vs box).

Run from the repo root:
    uvx --with duckdb --with pandas python -I eda/switching.py
"""

import re
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

T0 = time.time()
DATA = Path(__file__).resolve().parent.parent / "data" / "raw" / "irpd"
SALES = DATA / "processed_dataset" / "sales_processed.csv"
STOCK = DATA / "raw_dataset" / "stock_raw.csv"

MIN_LINES = 50          # a drug "sells" if it has at least this many sale lines in 3 years
BASELINE_DAYS = 56      # baseline window before a spell (calendar days)
MIN_BASELINE = 20       # in-stock baseline days needed
MIN_FOCAL_RATE = 0.3    # focal drug must sell >= 0.3 lines/day in its baseline
MAX_SPELL = 28          # only the first 28 trading days of a long spell are used
N_BOOT = 2000
N_PLACEBO = 200
rng = np.random.default_rng(7)

pd.set_option("display.width", 160)
con = duckdb.connect()


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------------------
# 1. Ingredient grouping
# ---------------------------------------------------------------------------

# Generic (INN) names as they are spelled in this pharmacy's catalogue.
# Spelling variants map to one ingredient. Diclofenac sodium and potassium are
# merged on purpose: at the counter they are treated as the same painkiller.
GENERICS = {
    "acetylcysteine": "acetylcysteine", "acyclovir": "acyclovir",
    "allopurinol": "allopurinol", "ambroxol": "ambroxol",
    "aminophylin": "aminophylline", "amlodipine": "amlodipine",
    "amoxicillin": "amoxicillin", "antalgin": "metamizole",
    "asam mefenamat": "mefenamic acid", "asam traneksamat": "tranexamic acid",
    "atorvastatin": "atorvastatin", "betahistine": "betahistine",
    "bisoprolol": "bisoprolol", "candesartan": "candesartan",
    "captopril": "captopril", "cefadroxil": "cefadroxil",
    "cefixime": "cefixime", "cetirizine": "cetirizine",
    "cimetidine": "cimetidine", "ciprofloxacin": "ciprofloxacin",
    "clindamycin": "clindamycin", "clobetasol": "clobetasol",
    "clopidogrel": "clopidogrel", "desloratadine": "desloratadine",
    "desoximetasone": "desoximetasone", "dexamethason": "dexamethasone",
    "diclofenac": "diclofenac", "digoxin": "digoxin",
    "domperidon": "domperidone", "fenofibrate": "fenofibrate",
    "flunarizine": "flunarizine", "furosemide": "furosemide",
    "gabapentin": "gabapentin", "gemfibrozil": "gemfibrozil",
    "glibenclamide": "glibenclamide", "glimepiride": "glimepiride",
    "glucosamine": "glucosamine", "griseofulvin": "griseofulvin",
    "guaifenesin": "guaifenesin", "hydrocortison": "hydrocortisone",
    "ibuprofen": "ibuprofen", "isosorbide dinitrate": "isosorbide dinitrate",
    "ketoconazole": "ketoconazole", "lansoprazole": "lansoprazole",
    "lisinopril": "lisinopril", "loratadine": "loratadine",
    "mecobalamin": "mecobalamin", "meloxicam": "meloxicam",
    "metformin": "metformin", "methylprednisolone": "methylprednisolone",
    "metronidazole": "metronidazole", "miconazole": "miconazole",
    "mupirocin": "mupirocin", "nifedipin": "nifedipine",
    "omeprazole": "omeprazole", "ondansetron": "ondansetron",
    "paracetamol": "paracetamol", "permethrin": "permethrin",
    "piracetam": "piracetam", "piroxicam": "piroxicam",
    "ramipril": "ramipril", "ranitidin": "ranitidine",
    "salbutamol": "salbutamol", "simvastatin": "simvastatin",
    "spironolactone": "spironolactone", "triamcinolone": "triamcinolone",
}

# Brand -> (ingredient, strength, sure). Keyed by the exact catalogue name,
# lower-cased. Strength is only used for oral solids (see form rules below).
# sure=False marks brands I am less certain about; the analysis is rerun
# without them as a sensitivity check.
BRANDS = {
    # paracetamol
    "zetamol tb 500mg": ("paracetamol", "500", True),
    "sanmol tablet 4s": ("paracetamol", "500", True),
    "pamol tab": ("paracetamol", "500", True),
    "panadol biru": ("paracetamol", "500", True),
    "fasidol tab": ("paracetamol", "500", False),
    "sanmol syr": ("paracetamol", "", True),
    "tempra syr 60ml": ("paracetamol", "", True),
    "tempra syr 30ml": ("paracetamol", "", True),
    "praxion 120mg syr 60ml": ("paracetamol", "", True),
    "pamol syr 60ml": ("paracetamol", "", True),
    "fasidol syr 60ml": ("paracetamol", "", True),
    "sanmol forte syr": ("paracetamol", "", True),
    "tempra forte syr 60ml": ("paracetamol", "", True),
    "tempra forte 30ml": ("paracetamol", "", True),
    "praxion forte 250mg syr 60ml": ("paracetamol", "", True),
    "fasidol forte syr 60ml": ("paracetamol", "", True),
    "sanmol drops": ("paracetamol", "", True),
    "tempra drop 15ml": ("paracetamol", "", True),
    "praxion 100mg drops 15ml": ("paracetamol", "", True),
    # mefenamic acid
    "ponstan 500mg": ("mefenamic acid", "500", True),
    "mefinal 500mg": ("mefenamic acid", "500", True),
    "teamic 500mg": ("mefenamic acid", "500", False),
    "anastan f": ("mefenamic acid", "500", False),
    # amoxicillin
    "bufamoxy 500": ("amoxicillin", "500", True),
    "yusimox 500mg kapl": ("amoxicillin", "500", True),
    "hufanoxil 500mg": ("amoxicillin", "500", True),
    "etamox 500": ("amoxicillin", "500", True),
    "amoxsan 500mg": ("amoxicillin", "500", True),
    "zemoxil tab": ("amoxicillin", "500", False),
    "bufamoxy dry syr 60ml": ("amoxicillin", "", True),
    "amoxsan dry syr 60ml": ("amoxicillin", "", True),
    # diclofenac
    "cataflam 50mg": ("diclofenac", "50", True),
    "cataflam fast 50mg": ("diclofenac", "50", True),
    "voltadex 50mg": ("diclofenac", "50", True),
    "voltaren 50mg tab": ("diclofenac", "50", True),
    "renadinac": ("diclofenac", "50", False),
    "zelona tb 50mg": ("diclofenac", "50", False),
    "voltadex gel 1% 20g": ("diclofenac", "", True),
    "voltaren emulgel 5gr": ("diclofenac", "", True),
    "voltaren emulgel 10gr": ("diclofenac", "", True),
    "voltaren emulgel 20gr": ("diclofenac", "", True),
    # steroids
    "dexa 0,5 harsen": ("dexamethasone", "0.5", True),
    "dexa 0,75 harsen": ("dexamethasone", "0.75", True),
    "dextamine tab": ("dexamethasone+dexchlorpheniramine", "0.5", True),
    "dexteem plus": ("dexamethasone+dexchlorpheniramine", "0.5", True),
    "carmeson 4mg": ("methylprednisolone", "4", True),
    "lameson 4mg": ("methylprednisolone", "4", True),
    "fametson 4": ("methylprednisolone", "4", False),
    "carmeson 8 mg": ("methylprednisolone", "8", True),
    "lameson 8mg": ("methylprednisolone", "8", True),
    "lameson 16mg": ("methylprednisolone", "16", True),
    # antifungals and other topicals
    "mycoral tab 200mg": ("ketoconazole", "200", True),
    "zoralin tab": ("ketoconazole", "200", True),
    "zoralin cream 10gr": ("ketoconazole", "", True),
    "mycoral zalf 5gr": ("ketoconazole", "", True),
    "canesten 5gr": ("clotrimazole", "", True),
    "kalpanax krim 5gr": ("miconazole", "", False),
    "kloderma cream 5gr": ("clobetasol", "", True),
    "kloderma cream 10gr": ("clobetasol", "", True),
    "kloderma oint 5gr": ("clobetasol", "", True),
    "kloderma oint 10gr": ("clobetasol", "", True),
    "inerson oint 0.25% 15gr": ("desoximetasone", "", True),
    "esperson 5gr": ("desoximetasone", "", True),
    "scabimite 5% 10gr": ("permethrin", "", True),
    "scabimite 5% 30gr": ("permethrin", "", True),
    "azovir cr 5gr": ("acyclovir", "", True),
    "betason n": ("betamethasone+neomycin", "", True),
    "bravoderm n cream 5gr": ("betamethasone+neomycin", "", False),
    # cardio-metabolic
    "norvask 5mg": ("amlodipine", "5", True),
    "lipitor 20mg": ("atorvastatin", "20", True),
    "lipitor 40mg": ("atorvastatin", "40", True),
    "concor 2,5mg": ("bisoprolol", "2.5", True),
    "alofar 100mg": ("allopurinol", "100", True),
    "alofar 300mg": ("allopurinol", "300", True),
    "zyloric 100mg": ("allopurinol", "100", True),
    "zyloric 300mg": ("allopurinol", "300", True),
    # other oral
    "pirocam 20": ("piroxicam", "20", True),
    "mertigo": ("betahistine", "6", True),
    "dramamine 50mg": ("dimenhydrinate", "50", True),
    "antimo tablet": ("dimenhydrinate", "50", True),
    "flagyl forte 500mg": ("metronidazole", "500", True),
    "omepros 30s": ("omeprazole", "20", False),
    "proris 200mg caplet": ("ibuprofen", "200", True),
    "proris syr": ("ibuprofen", "", True),
    "proris forte syr": ("ibuprofen", "", True),
    "incidal od": ("cetirizine", "10", True),
    "lerzin tab": ("cetirizine", "10", False),
    "mucos drop": ("ambroxol", "", True),
    "mucos syr 60ml": ("ambroxol", "", True),
    "biropyron": ("metamizole", "500", False),
    "defalgin antalgin los": ("metamizole", "500", False),
    "kalnex": ("tranexamic acid", "500", False),
    "zeniflox 500": ("ciprofloxacin", "500", False),
    "imodium": ("loperamide", "2", True),
    "ardium tb 500mg": ("diosmin+hesperidin", "500", True),
    "venaron 10s": ("diosmin+hesperidin", "500", False),
    "microgynon": ("levonorgestrel+ethinylestradiol", "0.15", True),
    "andalan pil kb": ("levonorgestrel+ethinylestradiol", "0.15", False),
    "ventolin nebules 2,5mg": ("salbutamol", "", True),
    "lasal nebu": ("salbutamol", "", False),
}

FORMS = [  # first match wins; anything else is an oral solid (tablet/capsule)
    ("liquid", r"\b(syr|sirup|syrup|susp|elixir|dry syr)\b"),
    ("drops", r"\b(drop|drops)\b"),
    ("topical", r"\b(cr|cream|krim|salep|oint|ointment|zalf|gel|emulgel|emugel|lotion)\b|\d+(?:[.,]\d+)?\s*%"),
    ("suppository", r"\b(supp|sup)\b"),
    ("inhaled", r"\b(inhaler|nebu|nebules|sol)\b"),
    ("injection", r"\b(inj|infus)\b"),
]


def form_of(name):
    for form, pat in FORMS:
        if re.search(pat, name):
            return form
    return "oral"


def strength_of(name):
    """First '<number>mg', else the first bare number (e.g. 'Mecobalamin 500')."""
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*mg", name) or re.search(r"\b(\d+(?:[.,]\d+)?)\b", name)
    return m.group(1).replace(",", ".") if m else ""


def map_drug(name):
    """Return (ingredient, strength, form, source, sure) or None."""
    n = re.sub(r"\s+", " ", name.lower()).strip()
    form = form_of(n)
    if n in BRANDS:
        ing, strength, sure = BRANDS[n]
        source = "brand"
    else:
        hit = next((k for k in GENERICS if n.startswith(k)), None)
        if hit is None:
            return None
        ing, strength, sure, source = GENERICS[hit], strength_of(n), True, "generic"
    if form == "oral":
        strength = strength.rstrip("0").rstrip(".") if "." in strength else strength
    elif form in ("liquid", "drops"):
        # Liquids differ in concentration (120 vs 160 mg/5ml); keep only "forte".
        strength = "forte" if "forte" in n else ""
    else:
        strength = ""  # topicals, suppositories etc.: group by ingredient + form
    return ing, strength, form, source, sure


con.execute(f"""
    create table sales as
    select Date as d, DrugCode as code, DrugName as name, Quantity as qty,
           Subtotal as revenue, BatchNumber as batch, ExpiryDate as expiry
    from read_csv('{SALES}')
""")

drugs = con.sql("""
    select code, mode(name) as name, count(*) as lines, sum(revenue) as revenue
    from sales group by code
""").df().sort_values("revenue", ascending=False).reset_index(drop=True)

mapped = drugs["name"].map(map_drug)
drugs["mapped"] = mapped.notna()
for i, col in enumerate(["ingredient", "strength", "form", "source", "sure"]):
    drugs[col] = mapped.map(lambda m: m[i] if m else None)
drugs["group"] = np.where(
    drugs["mapped"],
    drugs["ingredient"] + " | " + drugs["strength"].fillna("") + " | " + drugs["form"].fillna(""),
    None,
)
drugs["sells"] = drugs["lines"] >= MIN_LINES

section("1. Ingredient grouping")
top200 = drugs.head(200)
print(f"Drugs sold: {len(drugs):,}. Mapped to an ingredient: {drugs['mapped'].sum():,} "
      f"({drugs['mapped'].sum() / len(drugs):.0%}).")
print(f"Top 200 drugs by revenue: {top200['mapped'].sum()} mapped "
      f"({(top200['source'] == 'brand').sum()} by brand dictionary, "
      f"{(top200['source'] == 'generic').sum()} by generic name). "
      f"The rest are mostly combinations, vitamins, herbals, devices and milk.")
print(f"Brand dictionary entries: {len(BRANDS)} "
      f"({sum(not v[2] for v in BRANDS.values())} marked uncertain); "
      f"found in catalogue: {(drugs['source'] == 'brand').sum()}.")
missing = set(BRANDS) - set(drugs["name"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip())
if missing:
    print(f"Dictionary names not found in catalogue: {sorted(missing)}")

sellers = drugs[drugs["mapped"] & drugs["sells"]]
gsize = sellers.groupby("group").size()
multi = gsize[gsize >= 2].index
print(f"Mapped drugs that sell (>= {MIN_LINES} lines): {len(sellers)}; "
      f"groups: {len(gsize)}; groups with 2+ selling members: {len(multi)} "
      f"({gsize[gsize >= 2].sum()} drugs).")
show = (sellers[sellers["group"].isin(multi)]
        .groupby("group").agg(members=("name", lambda s: "; ".join(s[:5])), n=("name", "size"),
                              lines=("lines", "sum"))
        .sort_values("lines", ascending=False))
print("\nLargest multi-member groups (by sale lines):")
print(show.head(15).to_string())

# ---------------------------------------------------------------------------
# 2. End-of-day stock and stock-out spells
# ---------------------------------------------------------------------------

focus = sellers[sellers["group"].isin(multi)].copy()
con.register("focus", focus[["code"]])

# Numbers look like '1.250 Tab' (dot = thousands separator, no decimals seen).
NUM = "cast(replace(regexp_extract({c}, '^\\s*(-?[0-9.]+)', 1), '.', '') as double)"
con.execute(f"""
    create table ledger as
    select DrugCode as code, Warehouse as wh, cast(RowNumber as bigint) as rn,
           cast(strptime(DateTime, '%d %b %Y %H:%M:%S') as date) as d,
           {NUM.format(c='InitialStock')} as init, {NUM.format(c='RemainingStock')} as rem
    from read_csv('{STOCK}', all_varchar = true)
    where DrugCode in (select code from focus)
""")
# Trading days = days with any sale. Stock at end of each trading day per
# location = closing balance of the last ledger row so far; before a
# location's first row, its opening balance. Then sum the locations.
stock = con.sql("""
    with days as (select distinct d from sales),
    eod as (select code, wh, d, arg_max(rem, rn) as rem from ledger group by all),
    first_row as (select code, wh, arg_min(init, rn) as init from ledger group by all),
    grid as (select days.d, f.code, f.wh, f.init from days cross join first_row f)
    select g.d, g.code, sum(coalesce(e.rem, g.init)) as stock
    from grid g asof left join eod e
      on g.code = e.code and g.wh = e.wh and g.d >= e.d
    group by all
""").df()

days = np.array(sorted(con.sql("select distinct d from sales").df()["d"]), dtype="datetime64[D]")
weekday = (days.astype("datetime64[D]").view("int64") - 4) % 7  # 0 = Monday
codes_all = drugs.loc[drugs["sells"], "code"].tolist()
col = {c: i for i, c in enumerate(codes_all)}

# Daily sale lines for every drug that sells (the placebo pool needs them all).
lines_df = con.sql("select d, code, count(*) as n from sales group by all").df()
lines_df = lines_df[lines_df["code"].isin(col)]


def row_of(dates):
    return np.searchsorted(days, dates.values.astype("datetime64[D]"))


L = np.zeros((len(days), len(codes_all)))
L[row_of(lines_df["d"]), lines_df["code"].map(col).values] = lines_df["n"].values

S = np.full((len(days), len(codes_all)), np.nan)
S[row_of(stock["d"]), stock["code"].map(col).values] = stock["stock"].values
prev = np.vstack([S[:1], S[:-1]])
out = (S <= 0) & (prev <= 0)          # at zero the whole day
instock = (S > 0) & (prev > 0)        # stocked the whole day

section("2. Stock-out spells")
grp_members = focus.groupby("group")["code"].apply(list).to_dict()
code_group = dict(zip(focus["code"], focus["group"]))
sure_code = dict(zip(drugs["code"], drugs["sure"]))
src_code = dict(zip(drugs["code"], drugs["source"]))


def baseline_days(j, start):
    lo = np.searchsorted(days, days[start] - np.timedelta64(BASELINE_DAYS, "D"))
    b = np.arange(lo, start)
    return b[instock[b, j]]


def expected(cols, window, base):
    """Weekday-matched expected lines for a set of drugs over a window."""
    y = L[:, cols].sum(axis=1)
    overall = y[base].mean()
    means = {w: y[base][weekday[base] == w].mean() if (weekday[base] == w).any() else overall
             for w in range(7)}
    return sum(means[w] for w in weekday[window]), y[window].sum()


first_sale = (L > 0).argmax(axis=0)  # first trading day each drug sold
spells = []
for code in focus["code"]:
    j = col[code]
    alts = [col[c] for c in grp_members[code_group[code]] if c != code]
    o = out[:, j]
    t = 1
    while t < len(days):
        if not o[t]:
            t += 1
            continue
        end = t
        while end < len(days) and o[end]:
            end += 1
        window = np.arange(t, min(end, t + MAX_SPELL))
        base = baseline_days(j, t)
        rec = dict(code=code, group=code_group[code], start=days[t], length=end - t,
                   used=len(window), n_base=len(base))
        if len(base) >= MIN_BASELINE and L[base, j].mean() >= MIN_FOCAL_RATE:
            alt_ok = [a for a in alts if instock[window, a].mean() >= 0.5]
            if alt_ok:
                rec["e_alt"], rec["a_alt"] = expected(alt_ok, window, base)
                rec["e_focal"], rec["a_focal"] = expected([j], window, base)
                rec["alts"] = alt_ok
                # Trap check: a "new" alternative may be the same product under a
                # new code (a replacement), not a customer choosing another brand.
                rec["alt_new"] = any(first_sale[a] > t - 80 for a in alt_ok)
                rec["restocked"] = end < len(days)
        spells.append(rec)
        t = end
sp = pd.DataFrame(spells)
print(f"Stock-out spells (whole days at zero) among {len(focus)} grouped drugs: {len(sp):,}; "
      f"median length {sp['length'].median():.0f} trading days.")
tested = sp.dropna(subset=["e_alt"]).reset_index(drop=True)
tested[["alt_new", "restocked"]] = tested[["alt_new", "restocked"]].astype(bool)
print(f"Spells tested (focal sold >= {MIN_FOCAL_RATE} lines/day over >= {MIN_BASELINE} in-stock "
      f"baseline days, and an alternative was in stock): {len(tested)} spells, "
      f"{tested['code'].nunique()} drugs, {tested['group'].nunique()} groups, "
      f"{tested['used'].sum():,} spell-days.")
print(f"Sanity check: focal drug's own sale lines during tested spells = {tested['a_focal'].sum():.0f} "
      f"(expected {tested['e_focal'].sum():.0f} if it had been in stock).")

# ---------------------------------------------------------------------------
# 3. Uplift, bootstrap and placebos
# ---------------------------------------------------------------------------


def summarise(df):
    """Pooled ratio of actual/expected alternative lines, and switch share:
    extra alternative lines per expected focal line lost."""
    ea, aa, ef = df["e_alt"].sum(), df["a_alt"].sum(), df["e_focal"].sum()
    return aa / ea, (aa - ea) / ef


def bootstrap(df):
    idx = rng.integers(0, len(df), size=(N_BOOT, len(df)))
    ea, aa, ef = (df[c].values[idx].sum(axis=1) for c in ("e_alt", "a_alt", "e_focal"))
    return np.percentile(aa / ea, [2.5, 97.5]), np.percentile((aa - ea) / ef, [2.5, 97.5])


def placebo_time(df):
    """Same drugs and alternatives, random windows when the focal drug was in stock."""
    out_r, out_s = [], []
    cands = {}
    for _ in range(N_PLACEBO):
        rows = []
        for r in df.itertuples():
            j = col[r.code]
            key = (j, r.used)
            if key not in cands:
                ok = np.convolve(instock[:, j].astype(int), np.ones(r.used, int), "valid") == r.used
                starts = np.flatnonzero(ok)
                cands[key] = starts[starts >= 60]
            if len(cands[key]) == 0:
                continue
            s = rng.choice(cands[key])
            window, base = np.arange(s, s + r.used), baseline_days(j, s)
            if len(base) < MIN_BASELINE:
                continue
            ea, aa = expected(r.alts, window, base)
            ef, _ = expected([j], window, base)
            rows.append((ea, aa, ef))
        ea, aa, ef = np.array(rows).sum(axis=0)
        out_r.append(aa / ea)
        out_s.append((aa - ea) / ef)
    return np.array(out_r), np.array(out_s)


tot_lines = drugs.set_index("code")["lines"]
pool = np.array([col[c] for c in codes_all])
pool_lines = tot_lines.loc[codes_all].values
ing_of = dict(zip(drugs["code"], drugs["ingredient"]))


def placebo_unrelated(df):
    """Same spells and windows, but the 'alternatives' are random unrelated drugs
    of similar overall volume. Catches store-wide swings (season, traffic)."""
    out_r = []
    for _ in range(N_PLACEBO):
        rows = []
        for r in df.itertuples():
            j = col[r.code]
            ing = ing_of[r.code]
            fake = []
            for a in r.alts:
                lo, hi = tot_lines[codes_all[a]] / 1.5, tot_lines[codes_all[a]] * 1.5
                ok = pool[(pool_lines >= lo) & (pool_lines <= hi)]
                ok = [k for k in ok if ing_of.get(codes_all[k]) != ing and k != j]
                fake.append(rng.choice(ok))
            t = np.searchsorted(days, np.datetime64(r.start, "D"))
            window, base = np.arange(t, t + r.used), baseline_days(j, t)
            ea, aa = expected(fake, window, base)
            rows.append((ea, aa))
        ea, aa = np.array(rows).sum(axis=0)
        out_r.append(aa / ea)
    return np.array(out_r)


def report(df, label, placebos=True):
    r, s = summarise(df)
    (r_lo, r_hi), (s_lo, s_hi) = bootstrap(df)
    up = (df["a_alt"] > df["e_alt"]).mean()
    print(f"\n[{label}] {len(df)} spells, {df['code'].nunique()} drugs, {df['group'].nunique()} groups")
    print(f"  Alternatives sold {df['a_alt'].sum():.0f} lines vs {df['e_alt'].sum():.0f} expected: "
          f"ratio {r:.3f} (95% bootstrap CI {r_lo:.3f}-{r_hi:.3f}); "
          f"{up:.0%} of spells above expected.")
    print(f"  Switch share (extra alternative lines / focal lines lost): {s:.1%} "
          f"(95% CI {s_lo:.1%} to {s_hi:.1%}).")
    if placebos:
        pr, ps = placebo_time(df)
        print(f"  Placebo A, random in-stock windows: ratio mean {pr.mean():.3f} "
              f"(95% range {np.percentile(pr, 2.5):.3f}-{np.percentile(pr, 97.5):.3f}); "
              f"switch share {ps.mean():.1%} ({np.percentile(ps, 2.5):.1%} to {np.percentile(ps, 97.5):.1%}). "
              f"Share of placebo rounds >= real: {(pr >= r).mean():.0%}.")
        pu = placebo_unrelated(df)
        print(f"  Placebo B, unrelated drugs in the same spells: ratio mean {pu.mean():.3f} "
              f"(95% range {np.percentile(pu, 2.5):.3f}-{np.percentile(pu, 97.5):.3f}). "
              f"Share >= real: {(pu >= r).mean():.0%}. Real minus placebo B: {r - pu.mean():+.3f}.")


section("3. Do alternatives sell more during a stock-out?")
print("Ratio = alternatives' actual / weekday-matched expected sale lines. 1.0 = no change.")
report(tested, "All tested spells")
sure_only = tested[tested.apply(lambda r: sure_code[r.code] and all(sure_code[codes_all[a]] for a in r.alts),
                                axis=1)]
report(sure_only, "Confident mappings only")
gen_only = tested[tested.apply(lambda r: src_code[r.code] == "generic"
                               and all(src_code[codes_all[a]] == "generic" for a in r.alts), axis=1)]
report(gen_only, "Generic-vs-generic groups only (e.g. Amlodipine Novel vs Medicon)")
long_ = tested[tested["used"] >= 3]
report(long_, "Spells of 3+ trading days")
report(tested[tested["restocked"] & ~tested["alt_new"]],
       "Focal drug restocked later and no alternative new in the 80 trading days before")
report(tested[~tested["restocked"] | tested["alt_new"]],
       "Possible replacements: focal never restocked, or an alternative was new", placebos=False)

print("\nBy group (top 12 by expected focal lines lost):")
by = (tested.groupby("group")
      .agg(spells=("code", "size"), alt_actual=("a_alt", "sum"), alt_expected=("e_alt", "sum"),
           focal_lost=("e_focal", "sum"))
      .assign(ratio=lambda x: x.alt_actual / x.alt_expected,
              switch=lambda x: (x.alt_actual - x.alt_expected) / x.focal_lost)
      .sort_values("focal_lost", ascending=False))
print(by.head(12).round(2).to_string())

# ---------------------------------------------------------------------------
# 4. Whole-group stock-outs: no same-ingredient alternative on the shelf
# ---------------------------------------------------------------------------

section("4. Whole-group stock-outs (every live member at zero)")
# A member is "live" between its first and last sale in the data, so a drug
# that was dropped from the range does not count as an alternative.
last_sale = len(days) - 1 - (L[::-1] > 0).argmax(axis=0)
t_idx = np.arange(len(days))[:, None]
live = (t_idx >= first_sale) & (t_idx <= last_sale)

rev_per_line = con.sql("select code, avg(revenue) as v from sales group by code").df().set_index("code")["v"]
ing_groups = focus.groupby("ingredient")["group"].unique().to_dict()
gout, gspells = {}, []
for g, members in grp_members.items():
    m = [col[c] for c in members]
    n_live = live[:, m].sum(axis=1)
    all_out = (n_live >= 2) & np.all(out[:, m] | ~live[:, m], axis=1)
    gout[g] = all_out
    t = 1
    while t < len(days):
        if not all_out[t]:
            t += 1
            continue
        end = t
        while end < len(days) and all_out[end]:
            end += 1
        window = np.arange(t, min(end, t + MAX_SPELL))
        # Baseline: days in the 8 weeks before when at least one member was stocked.
        lo = np.searchsorted(days, days[t] - np.timedelta64(BASELINE_DAYS, "D"))
        b = np.arange(lo, t)
        b = b[np.any(instock[b][:, m], axis=1)]
        rec = dict(group=g, start=days[t], length=end - t, used=len(window))
        if len(b) >= MIN_BASELINE:
            rec["e_grp"], rec["a_grp"] = expected(m, window, b)
            # Same ingredient, other strength or form (e.g. amoxicillin syrup).
            other = [col[c] for og in ing_groups[ing_of[members[0]]] if og != g
                     for c in grp_members.get(og, [])]
            other += [col[c] for c in sellers.loc[(sellers["ingredient"] == ing_of[members[0]])
                                                  & ~sellers["group"].isin(multi), "code"]]
            if other:
                rec["e_oth"], rec["a_oth"] = expected(other, window, b)
                rec["other"] = other
            rec["rev_line"] = float(np.average(rev_per_line.loc[members], weights=L[b][:, m].sum(axis=0) + 1e-9))
            rec["base"] = b
        gspells.append(rec)
        t = end
gs = pd.DataFrame(gspells)
gdays = sum(int(v.sum()) for v in gout.values())
print(f"Groups with 2+ selling members: {len(grp_members)}. Groups that ever went fully out: "
      f"{gs['group'].nunique() if len(gs) else 0}. Group-out days: {gdays:,} in {len(gs)} spells; "
      f"spell length median {gs['length'].median():.0f}, p90 {gs['length'].quantile(.9):.0f}, "
      f"max {gs['length'].max():.0f} trading days.")
print(gs.groupby("group").agg(spells=("length", "size"), days=("length", "sum"))
      .sort_values("days", ascending=False).head(12).to_string())

single_days = int((out[:, [col[c] for c in focus["code"]]]).sum())
print(f"\nFor comparison, single-drug stock-out days across the same {len(focus)} drugs: {single_days:,}.")
spell_days = [(r.group, np.searchsorted(days, np.datetime64(r.start, "D")) + k)
              for r in sp.itertuples() for k in range(r.length)]
both = np.mean([gout[g][t] for g, t in spell_days])
tested_days = [(r.group, np.searchsorted(days, np.datetime64(r.start, "D")) + k)
               for r in tested.itertuples() for k in range(r.used)]
both_t = np.mean([gout[g][t] for g, t in tested_days])
print(f"Share of single-drug stock-out days that were also group-out days: {both:.1%} of all "
      f"{len(spell_days):,} days; {both_t:.1%} of the {len(tested_days):,} days in the 189-style tested "
      f"spells (low by design: those needed an alternative in stock).")

gt = gs.dropna(subset=["e_grp"]).reset_index(drop=True)
eg, ag = gt["e_grp"].sum(), gt["a_grp"].sum()
print(f"\nGroup-out spells with a usable baseline: {len(gt)} ({gt['used'].sum():,} days used).")
print(f"Group sale lines during group-out days: {ag:.0f} vs {eg:.0f} expected "
      f"(any sales here come from stock the ledger shows as zero).")
go = gt.dropna(subset=["e_oth"])
if len(go):
    r = go["a_oth"].sum() / go["e_oth"].sum()
    extra = go["a_oth"].sum() - go["e_oth"].sum()
    idx = rng.integers(0, len(go), size=(N_BOOT, len(go)))
    rb = go["a_oth"].values[idx].sum(axis=1) / go["e_oth"].values[idx].sum(axis=1)
    # Placebo: the same other-strength drugs in random windows when the group was stocked.
    pr = []
    for _ in range(N_PLACEBO):
        ea = aa = 0.0
        for row in go.itertuples():
            ok = np.flatnonzero(~gout[row.group][60:len(days) - row.used]) + 60
            s = rng.choice(ok)
            lo = np.searchsorted(days, days[s] - np.timedelta64(BASELINE_DAYS, "D"))
            b = np.arange(lo, s)
            e_, a_ = expected(row.other, np.arange(s, s + row.used), b)
            ea, aa = ea + e_, aa + a_
        pr.append(aa / ea)
    pr = np.array(pr)
    print(f"Same ingredient, other strength or form ({len(go)} spells with such drugs): "
          f"ratio {r:.2f} (95% CI {np.percentile(rb, 2.5):.2f}-{np.percentile(rb, 97.5):.2f}), "
          f"{extra:+.0f} lines = {extra / go['e_grp'].sum():.0%} of the group's expected lines. "
          f"Placebo (random group-stocked windows): {pr.mean():.2f} "
          f"(95% range {np.percentile(pr, 2.5):.2f}-{np.percentile(pr, 97.5):.2f}).")
    if len(go) < 10:
        print(f"  Only {len(go)} spells: too few to read either way.")
print("Combination products and other ingredients in the same class are not mapped, so substitution "
      "to them cannot be measured here; whatever is not picked up above is an upper bound on truly lost.")
lost_rev = ((gt["e_grp"] - gt["a_grp"]).clip(lower=0) * gt["rev_line"]).sum()
print(f"ESTIMATE of revenue lost on group-out days: {lost_rev / 1e6:,.1f}m rupiah "
      f"(expected minus actual group lines x the group's average revenue per line; first {MAX_SPELL} "
      f"days of each spell only; before any substitution credit).")

# ---------------------------------------------------------------------------
# 5. Batch and expiry
# ---------------------------------------------------------------------------

section("5. Batch and expiry (from sale lines)")
e = con.sql("""
    select count(*) as lines,
           count(expiry) as with_expiry,
           count(*) filter (where batch in ('SW', 'sw', '-') or batch is null) as placeholder_batch,
           count(*) filter (where expiry > d + interval 10 year) as expiry_over_10y,
           count(*) filter (where day(expiry) = 1) as expiry_first_of_month,
           count(*) filter (where month(expiry) = 1 and day(expiry) = 1) as expiry_jan_1
    from sales
""").df().iloc[0]
print(f"Sale lines: {e.lines:,}; with an expiry date: {e.with_expiry:,} ({e.with_expiry / e.lines:.1%}). "
      f"Placeholder batch ('SW', '-', empty): {e.placeholder_batch:,} ({e.placeholder_batch / e.lines:.1%}).")
print(f"Expiry on the 1st of a month: {e.expiry_first_of_month / e.with_expiry:.1%} (month precision only); "
      f"on 1 January: {e.expiry_jan_1 / e.with_expiry:.1%} (likely year-only entries); "
      f"more than 10 years out: {e.expiry_over_10y:,} (typos such as 2123).")

x = con.sql("""
    select date_diff('day', d, expiry) as days_left, qty, revenue
    from sales where expiry is not null and expiry <= d + interval 10 year
""").df()
q = np.percentile(x["days_left"], [5, 25, 50, 75])
print(f"Days from sale to expiry (plausible dates): p5 {q[0]:.0f}, p25 {q[1]:.0f}, "
      f"median {q[2]:.0f}, p75 {q[3]:.0f}.")
for label, cond in [("already expired (< 0 days)", x.days_left < 0),
                    ("within 3 months (0-90 days)", x.days_left.between(0, 90)),
                    ("within 6 months (0-182 days)", x.days_left.between(0, 182))]:
    print(f"  Sold {label}: {cond.sum():,} lines ({cond.mean():.2%}), "
          f"{x.loc[cond, 'revenue'].sum() / 1e6:,.1f}m rupiah.")
expired = con.sql("""
    select name, count(*) as lines, min(date_diff('day', d, expiry)) as worst_days,
           median(date_diff('day', d, expiry)) as median_days
    from sales where expiry < d group by name order by lines desc limit 8
""").df()
print("\nDrugs with the most sale lines after the recorded expiry date:")
print(expired.to_string(index=False))
# If expiry is month-precision and means 'end of month', sales in the expiry
# month itself are not truly expired. Also count placeholder batches.
same_month = con.sql("""
    select count(*) filter (where expiry < d) as expired,
           count(*) filter (where expiry < d and date_trunc('month', d) = expiry) as in_expiry_month,
           count(*) filter (where expiry < d and (batch in ('SW', 'sw', '-') or batch is null)) as placeholder
    from sales
""").df().iloc[0]
print(f"Of {same_month.expired:,} 'expired' lines, {same_month.placeholder:,} carry a placeholder batch, and "
      f"{same_month.in_expiry_month:,} are in the expiry month itself (fine if the date means end of month).")
print("A long tail of 'expired' sales on the same few drugs suggests the POS reused an old batch "
      "record, so treat sale-line expiry as a hint, not proof that expired stock was sold.")

print(f"\nRuntime: {time.time() - T0:.0f}s")
