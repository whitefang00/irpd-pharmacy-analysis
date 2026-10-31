"""Check the reference ingredient labels against K24Klik product pages.

`build_reference_labels.py` labels drugs from a public drug list. This script
checks a random sample of those labels against a second, independent source:
the "Komposisi" (composition) line on K24Klik, an Indonesian online pharmacy.
It reports how often the label is right, partly right or wrong, with the page
link for each verdict so any of them can be checked by hand. A page whose
composition doesn't match its own product name is set aside as page_conflict.

Politeness: product pages come from the site's own sitemap (no search), every
URL is checked against the site's robots.txt before it is fetched, there is
one request every few seconds, and every page is cached so nothing is fetched
twice.

Usage:
    uv run python -I scripts/check_labels_k24.py [--sample 60] [--delay 3]

Input:  data/reference/reference_labels.csv (from build_reference_labels.py)
Output: data/reference/k24_check.csv, cache in data/reference/k24/
"""

import argparse
import hashlib
import html
import re
import subprocess
import time
from pathlib import Path
from urllib.robotparser import RobotFileParser

import pandas as pd

from irpd.names import B_COMPLEX, B_VITAMINS, MPFF, ingredient_words, norm, same

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data/reference/reference_labels.csv"
CACHE = ROOT / "data/reference/k24"
OUT = ROOT / "data/reference/k24_check.csv"
SITEMAP = "https://www.k24klik.com/sitemap.xml"
ROBOTS = "https://www.k24klik.com/robots.txt"
AGENT = "irpd-analysis research script (github.com/whitefang00)"

FORM_WORDS = {"tab", "tablet", "capl", "kaplet", "caps", "kaps", "syr", "sirup", "drop", "drops",
              "susp", "cr", "krim", "salep", "gel", "inj", "sach", "strip", "s", "box", "btl",
              "botol", "tube", "blister", "pot", "inh", "supp", "ovula"}
# First words too common to identify a product on their own ("Asam
# Traneksamat" must not match an "asam mefenamat" page).
COMMON_LEADS = {"asam", "vitamin", "kalium", "natrium", "kalsium", "calcium", "magnesium", "zinc"}


_robots: RobotFileParser | None = None


def allowed(url: str) -> bool:
    """True if the site's robots.txt lets any user agent fetch this URL."""
    global _robots
    if _robots is None:
        text = subprocess.run(["curl", "-sL", "-m", "30", "-A", AGENT, ROBOTS],
                              capture_output=True, text=True, errors="ignore").stdout
        # The file writes "User-agent :" with a space, which the parser doesn't read.
        text = re.sub(r"(?im)^user-agent\s*:", "User-agent:", text)
        # The parser doesn't know "*" wildcards. Cut each rule at its first "*",
        # which blocks at least as much as the site asks.
        text = re.sub(r"(?im)^((?:dis)?allow\s*:\s*[^*\s]*)\*.*$", r"\1", text)
        _robots = RobotFileParser()
        _robots.parse(text.splitlines())
    return _robots.can_fetch("*", url)


def fetch(url: str, delay: float) -> str:
    """Fetch a page once and cache it. curl is used because it handles the site's TLS."""
    path = CACHE / "pages" / (hashlib.sha1(url.encode()).hexdigest()[:16] + ".html")
    if path.exists():
        return path.read_text(errors="ignore")
    if not allowed(url):
        raise PermissionError(f"robots.txt disallows {url}")
    time.sleep(delay)
    out = subprocess.run(["curl", "-sL", "-m", "30", "-A", AGENT, url],
                         capture_output=True, text=True, errors="ignore")
    if out.returncode == 0 and out.stdout:  # don't cache a failed fetch
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(out.stdout)
    return out.stdout


def product_urls(delay: float) -> list[str]:
    xml = fetch(SITEMAP, delay)
    return [u for u in re.findall(r"<loc>([^<]+)</loc>", xml) if "/p/" in u]


def slug_words(url: str) -> list[str]:
    slug = url.rsplit("/p/", 1)[1]
    return re.sub(r"-\d+$", "", slug).split("-")


def find_product(name: str, index: dict[str, list[tuple[list[str], str]]]) -> str:
    """Best product page for an IRPD drug name, or '' if no good match."""
    words = [w for w in norm(name).split() if w not in FORM_WORDS]
    if not words or len(words[0]) < 4:
        return ""
    numbers = set(re.findall(r"\d+", name))
    need = 2 if words[0] in COMMON_LEADS and len(words) > 1 else 1
    best, best_score = "", 0.0
    for slug, url in index.get(words[0], []):
        # Leading brand words must match in order; strength and form add a little.
        lead = 0
        for a, b in zip(words, slug, strict=False):
            if a != b:
                break
            lead += 1
        if lead < need:
            continue
        # Both name a different variant next ("Thermolyte Plus" vs "Thermolyte
        # Sweetener"): not the same product.
        nxt = [w for w in (words[lead:lead + 1] + slug[lead:lead + 1])
               if w.isalpha() and w not in FORM_WORDS]
        if len(nxt) == 2:
            continue
        score = 2 * lead + sum(0.5 for w in slug
                               if w[:1].isdigit() and re.sub(r"\D", "", w) in numbers)
        # Prefer the plain product over variants with extra name words: "Feminax"
        # is the tablet, not "Feminax Lancar Haid Cair".
        extra = [w for w in slug[lead:] if w.isalpha() and w not in FORM_WORDS and w not in words]
        score -= 0.1 * len(extra)
        if score > best_score:
            best, best_score = url, score
    return best


def composition(page: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", "\n", page))
    m = re.search(r"Komposisi\s*:?\s*\n\s*(.+?)\n", text)
    return " ".join(m.group(1).split()) if m else ""


def verdict(ref: str, comp: str, title: str = "") -> str:
    if not comp:
        return "no_composition"
    comp = MPFF.sub("diosmin, hesperidin", B_COMPLEX.sub(B_VITAMINS, comp))
    ref_items = [r.strip() for r in ref.split("+") if r.strip()]
    # Split on commas that start a new ingredient, not decimal commas ("0,25 mcg").
    comp_items = [c for c in re.split(r",(?!\d)|\+|;|\bdan\b|\n", comp) if ingredient_words(c)]
    found = [r for r in ref_items if any(same(r, c) for c in comp_items)]
    if len(found) == len(ref_items) and len(comp_items) <= len(ref_items):
        return "right"
    if found:
        return "partly_right"  # some ingredients right, or a combination with some missing
    # The page's own product name has the labelled ingredient but its
    # composition line doesn't ("Gemfibrozil 300mg" listing gentamicin): the
    # page contradicts itself, so it can't judge the label.
    if title and all(same(r, title) for r in ref_items):
        return "page_conflict"
    return "wrong"


def page_title(page: str) -> str:
    m = re.search(r"<title>([^<]*?)(?: - Manfaat|<)", page)
    return html.unescape(m.group(1)) if m else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=60, help="labels to check")
    ap.add_argument("--delay", type=float, default=3.0, help="seconds between requests")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    labels = pd.read_csv(LABELS, dtype=str).fillna("")
    key_rows = labels[(labels["confidence"] == "high") & (labels["lines"].astype(int) > 0)]
    sample = key_rows.sample(min(args.sample, len(key_rows)), random_state=args.seed)

    urls = product_urls(args.delay)
    index: dict[str, list[tuple[list[str], str]]] = {}
    for u in urls:
        words = slug_words(u)
        index.setdefault(words[0], []).append((words, u))
    print(f"K24Klik sitemap: {len(urls):,} product pages")

    rows = []
    for _, r in sample.iterrows():
        url = find_product(r["DrugName"], index)
        page = fetch(url, args.delay) if url else ""
        comp = composition(page)
        rows.append({
            "DrugName": r["DrugName"], "source": r["source"], "label": r["ingredients"],
            "k24_composition": comp,
            "verdict": verdict(r["ingredients"], comp, page_title(page)) if url else "no_page",
            "url": url,
        })
        print(f"  {rows[-1]['verdict']:<14} {r['DrugName'][:34]:<34} "
              f"{r['ingredients'][:30]:<30} | {comp[:50]}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    checked = out[out["verdict"].isin(["right", "partly_right", "wrong"])]
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print(f"scored {len(checked)} of {len(out)} labels (right, partly right or wrong)")
    print(out["verdict"].value_counts().to_string())
    if len(checked):
        print(f"\nright: {(checked['verdict'] == 'right').mean():.0%}, "
              f"right or partly right: {(checked['verdict'] != 'wrong').mean():.0%}")
        by_source = checked.groupby("source")["verdict"].value_counts(normalize=True)
        print(by_source.round(2).to_string())
        print("\nwrong labels (check these links):")
        pd.set_option("display.width", 220)
        pd.set_option("display.max_colwidth", 70)
        cols = ["DrugName", "label", "k24_composition", "url"]
        print(checked[checked["verdict"] == "wrong"][cols].to_string(index=False))
    conflict = out[out["verdict"] == "page_conflict"]
    if len(conflict):
        print("\npages whose composition contradicts their own product name (not scored):")
        print(conflict[["DrugName", "label", "k24_composition", "url"]].to_string(index=False))


if __name__ == "__main__":
    main()
