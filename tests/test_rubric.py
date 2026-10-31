"""Eval v2 rubric checks and keyword detectors, on realistic paraphrases both ways."""

import pytest

from irpd.evals import rubric as r
from irpd.evals.rubric import Check, asks_or_lists, flag, includes_k_of, number_near, score

# Detectors

FUTURE_YES = [
    "I can't see anything after 30 June 2025, so I can't tell you August's sales.",
    "My data only goes up to 30 Jun 2025; August hasn't happened yet from where I stand.",
    "I don't have any sales data for August 2025 — the records stop at the cutoff.",
    "That's in the future for me: today is 2025-06-30.",
    "There is no data beyond 30 June 2025.",
    "The records end on 30 June 2025, so July stock isn't available.",
    "Saya hanya bisa melihat data sampai 30 Juni 2025, jadi Agustus belum ada datanya.",
    "Data penjualan Agustus 2025 belum tersedia.",
    "The data can't tell: August 2025 is after the latest date available, 2025-06-30.",
]
FUTURE_NO = [
    "In August 2025 we sold Rp 312,000,000.",
    "Flazen 50 had 4,016 tablets in stock on 15 July 2025.",
    "Sales in August were strong, about the same as July.",
    "Penjualan bulan Agustus sekitar Rp 300 juta.",
]
CANT_TELL_YES = [
    "The data can't tell: it has selling prices but no costs.",
    "I can't calculate margin because there are no purchase prices in the records.",
    "The records don't include what we pay suppliers, so I can't say which is cheapest.",
    "Without cost data there's no way to work out profit.",
    "The data only has selling prices, not costs.",
    "Margin tidak bisa dihitung karena tidak ada data harga beli.",
    "Data kita tanpa harga beli, jadi saya tidak dapat menghitung margin.",
]
CANT_TELL_NO = [
    "Your margin on Flazen 50 is about 25%.",
    "Amapharm.PT is the cheapest supplier for Amlodipine 10mg Novel.",
    "Patent drugs usually carry the best margins, roughly 30-40%.",
    "Margin Flazen 50 sekitar 20 persen.",
]
PREMISE_YES = [
    "Teamic 500mg didn't actually run out in May 2025 — it had stock every day.",
    "There was no stock-out of Teamic in May 2025.",
    "Teamic never ran out in May; it stayed in stock the whole month.",
    "Flazen 50 isn't out of stock: there are 4,016 tablets on hand.",
    "Actually, Flazen 50 is in stock right now.",
    "The records show zero stock-out days for Teamic in May.",
    "Teamic tidak pernah habis di bulan Mei 2025; stoknya selalu ada.",
    "The records show 4,016 tablets in stock, not out of stock.",
]
PREMISE_NO = [
    "Teamic ran out in May 2025 because the supplier delivered late.",
    "The stock-out was caused by a demand spike after Eid.",
    "Flazen 50 ran out last week; the next delivery is likely in 7 days.",
    "Teamic habis karena permintaan naik.",
]
ASKS_YES = [
    "Which one do you mean: Amlodipine 10mg Novel, Medicon or Promed?",
    "Did you mean Sanmol Syr or Tempra Syr 60ml?",
    "Could you clarify which product you're asking about?",
    "Obat maag yang mana yang Bapak maksud?",
    "“Obat maag” mencakup banyak produk. Produk mana yang ingin dicek stoknya?",
]
ASKS_NO = [
    "We have 280 tablets of Amlodipine 10mg Novel.",
    "Stock is fine.",
    "Stok obat maag aman.",
]


@pytest.mark.parametrize("text", FUTURE_YES)
def test_future_detector_finds_paraphrases(text):
    assert r.says_cant_see_future(text)


@pytest.mark.parametrize("text", FUTURE_NO)
def test_future_detector_ignores_answers(text):
    assert not r.says_cant_see_future(text)


@pytest.mark.parametrize("text", CANT_TELL_YES)
def test_cant_tell_detector_finds_paraphrases(text):
    assert r.says_cant_tell(text)


@pytest.mark.parametrize("text", CANT_TELL_NO)
def test_cant_tell_detector_ignores_guesses(text):
    assert not r.says_cant_tell(text)


@pytest.mark.parametrize("text", PREMISE_YES)
def test_premise_detector_finds_corrections(text):
    assert r.corrects_premise(text)


@pytest.mark.parametrize("text", PREMISE_NO)
def test_premise_detector_ignores_acceptance(text):
    assert not r.corrects_premise(text)


@pytest.mark.parametrize("text", ASKS_YES)
def test_asks_detector(text):
    assert r.asks(text)


@pytest.mark.parametrize("text", ASKS_NO)
def test_asks_detector_ignores_plain_answers(text):
    assert not r.asks(text)


# Checks

def ok(check: Check, text: str, value=None, cant_tell=False, tools=()) -> bool:
    return r.passes(check, text, {"value": value, "cant_tell": cant_tell}, list(tools))


def test_names_match_whole_words_and_spacing():
    assert r.mentioned("Amlodipine 10mg Novel", "Order amlodipine 10 mg Novel now.")
    assert r.mentioned("Gastrucid tb ", "Move 21 Gastrucid tb down.")
    assert not r.mentioned("De", "We decided to order Demacolin.")
    assert r.mentioned("De", "1. De - Rp 1.1M")
    assert not r.mentioned("Flazen 50", "Flazen 500")


def test_includes_k_of_counts_prose_and_value():
    c = includes_k_of(["A drug", "B drug", "C drug", "D drug"], 2)
    assert ok(c, "Reorder A drug.", value=["C drug"])
    assert not ok(c, "Reorder A drug and X drug.")
    assert ok(includes_k_of(["A drug"], 3), "A drug")  # k is capped at the list's length
    assert ok(Check("includes_any", names=("B drug", "Z")), "b DRUG")


def test_number_near_reads_value_and_prose():
    c = number_near([1962.1], rel=0.15)
    assert ok(c, "About 2,100 tablets.")
    assert ok(c, "See value.", value=1800)
    assert ok(c, "", value="1,900 tablets")
    assert not ok(c, "About 2,300 tablets.")
    two = number_near([100, 50], rel=0.01, k=2)
    assert ok(two, "Lost Rp 100, of which Rp 50 switched.")
    assert not ok(two, "Lost Rp 100.")
    assert ok(number_near([941911.59], rel=0.05), "Betason N lost about Rp 0.94 million.")


def test_number_near_ignores_dates():
    assert not ok(number_near([30], tol=0.5), "As of 30 June 2025, nothing.")
    assert not ok(number_near([2025]), "In 2025 we lost a lot.")


def test_cant_tell_flag_counts():
    assert ok(flag("says_cant_tell"), "No idea.", cant_tell=True)
    assert not ok(flag("says_cant_tell"), "Margin is 25%.")


def test_asks_or_lists_candidates():
    c = asks_or_lists(["Amlodipine 10mg Novel", "Amlodipine 10mg Promed", "Amlodipine 10mg HJ"])
    assert ok(c, "Amlodipine 10mg Novel has 280 and Amlodipine 10mg Promed 350.")
    assert ok(c, "Which brand do you mean?")
    assert not ok(c, "Amlodipine 10mg Novel has 280 left.")
    # brands named by the words only they have
    assert ok(c, "Amlodipine 10 mg stock by brand: HJ 100; Novel 280.")
    assert not ok(c, "Amlodipine 10 mg: 1,310 tablets across all brands.")
    syrups = asks_or_lists(["Sanmol Syr", "Sanmol Drops", "Tempra Syr 60ml"])
    assert not ok(syrups, "Syr: 3 bottles.")  # no word is Sanmol Syr's own


def test_no_unsupported_numbers_allows_dates_and_tool_numbers():
    c = flag("no_unsupported_numbers")
    assert ok(c, "I can only see up to 30 June 2025.")
    assert ok(c, "Data ends 2025-06-30; Q3 isn't in it.")
    tool = {"drug_name": "Flazen 50", "stock_total": 4016}
    assert ok(c, "Flazen 50 has 4,016 tablets.", tools=[tool])
    assert not ok(c, "A typical pharmacy margin is 20-30%.")


def test_score_is_the_weighted_share():
    checks = [flag("says_cant_tell", weight=2), flag("no_unsupported_numbers")]
    assert score(checks, "The data can't tell: no costs.", {}, []) == (True, 1.0, [])
    passed, credit, failed = score(checks, "Can't tell exactly, maybe 25%.", {}, [])
    assert not passed and credit == pytest.approx(2 / 3)
    assert failed == ["no_unsupported_numbers"]
    with pytest.raises(ValueError):
        Check("nonsense")


def test_words_that_look_like_months_are_not_dates():
    assert r.without_dates("maybe 25% in March 2025").split() == ["maybe", "25%", "in"]
    assert ok(number_near([1962.1], rel=0.15), "We expect 1962 tablets.")


def test_indonesian_number_format():
    tools = [{"stock_total": 3776, "cover_days": 81.13}]
    assert ok(flag("no_unsupported_numbers"), "Stok 3.776 tablet (81,13 hari).", tools=tools)
    assert ok(number_near([3776]), "Stok teamic 3.776 tablet.")
    assert not ok(flag("no_unsupported_numbers"), "Stok 3.900 tablet.", tools=tools)


def test_asks_in_indonesian_with_produk_apa():
    from irpd.evals.rubric import asks
    assert asks("Obat maag yang dimaksud produk apa? Sebutkan nama produknya.")
    assert not asks("Stok obat maag aman: Promag 120 tablet.")
