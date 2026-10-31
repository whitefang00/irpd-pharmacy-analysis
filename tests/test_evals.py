import csv
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from irpd import config, warehouse
from irpd.evals import QUESTIONS, gold_answers
from irpd.evals.questions import Question
from irpd.evals.scorer import (
    list_credit,
    numbers_in,
    score_answer,
    score_run,
    tool_recall,
    unsupported_numbers,
)

TOOLS = {"run_sql", "forecast", "stock_position", "stockout_history", "alternatives",
         "reorder_plan", "transfer_plan", "price_history"}
KINDS = {"number", "date", "yes_no", "label", "list", "cant_tell"}

# A tiny pharmacy, 1 May – 10 Jul 2025, with things going on across the cutoff:
# Flazen 50 is out of stock from 24 Jun to 1 Jul and gets a price rise on 3 Jul;
# Teamic 500mg's floor runs empty on 29 Jun with stock upstairs until 2 Jul.
D = date(2025, 5, 1)
END = date(2025, 7, 10)
DRUGS = {  # code: name, category, units sold a day
    "F": ("Flazen 50", "Patent", 2),
    "T": ("Teamic 500mg", "Generik", 3),
    "A": ("Amlodipine 10mg Novel", "Generik", 1),
}
BUYS = {  # (day, code): qty, location, supplier
    (D, "F"): (60, "floor", "Amapharm.PT"),
    (D, "T"): (200, "store", "Antar Mitra Sembada.PT"),
    (D, "A"): (20, "floor", "Enseval Putra Megatrading.PT"),
    (date(2025, 5, 25), "A"): (30, "floor", "Enseval Putra Megatrading.PT"),
    (date(2025, 6, 5), "F"): (40, "floor", "Amapharm.PT"),
    (date(2025, 7, 2), "F"): (50, "floor", "Amapharm.PT"),
    (date(2025, 7, 2), "A"): (30, "floor", "Enseval Putra Megatrading.PT"),
}
MOVES_DOWN = {D + timedelta(days=10 * k): 30 for k in range(6)} | {date(2025, 7, 2): 20}
WAREHOUSES = {"floor": "Apotek", "store": "Gudang Obat 1"}


def price(code: str, day: date) -> int:
    if code == "F":
        return 3500 if day < date(2025, 5, 15) else 3600 if day < date(2025, 7, 3) else 3800
    return {"T": 2000, "A": 500}[code]


def write_fixture(root: Path, until: date = END) -> Path:
    """Simulate the ledger and sales day by day, and write the IRPD files up to `until`."""
    stock = {(c, loc): 0 for c in DRUGS for loc in WAREHOUSES}
    moves, raw, sales = [], [], []

    def move(day, time, doc, kind, code, loc, qty_in, qty_out):
        name, cat, _ = DRUGS[code]
        before = stock[code, loc]
        stock[code, loc] = before + qty_in - qty_out
        n = len(moves) + 1
        moves.append([n, day, time, doc, kind, code, name, cat, before, qty_in, qty_out,
                      stock[code, loc], "Tab"])
        raw.append([n, WAREHOUSES[loc], code])

    day = D
    while day <= until:
        for code in DRUGS:
            if (day, code) in BUYS:
                qty, loc, supplier = BUYS[day, code]
                move(day, "08:00:00", f"PB{day:%m%d}{code}", f"Pembelian ({supplier})",
                     code, loc, qty, 0)
        if day in MOVES_DOWN:
            qty, doc = MOVES_DOWN[day], f"MT{day:%m%d}"
            move(day, "08:30:00", doc, "Mutasi", "T", "store", 0, qty)
            move(day, "08:30:00", doc, "Mutasi", "T", "floor", qty, 0)
        for code, (name, cat, rate) in DRUGS.items():
            qty = min(rate, stock[code, "floor"])
            if qty > 0:
                inv = f"PJ{day:%m%d}{code}"
                move(day, "19:00:00", inv, "Penjualan", code, "floor", 0, qty)
                p = price(code, day)
                sales.append([len(sales) + 1, day, "19:00:00", inv, "Cashier_002", code, name,
                              cat, qty, "Tab", p, 0, qty * p, "TUNAI"])
        day += timedelta(days=1)

    def write(path, header, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerows([header, *rows])

    write(root / "processed_dataset/sales_processed.csv",
          ["RowNumber", "Date", "Time", "InvoiceNumber", "CashierID", "DrugCode", "DrugName",
           "Category", "Quantity", "Unit", "UnitPrice", "DiscountAmount", "Subtotal",
           "PaymentType"], sales)
    write(root / "processed_dataset/stock_processed.csv",
          ["RowNumber", "Date", "Time", "DocumentNumber", "TransactionType", "DrugCode",
           "DrugName", "Category", "InitialStock", "StockIn", "StockOut", "RemainingStock",
           "Unit"], moves)
    write(root / "raw_dataset/stock_raw.csv", ["RowNumber", "Warehouse", "DrugCode"], raw)
    return root


IMAP = pd.DataFrame({"drug_code": ["F", "T", "A"],
                     "ingredients": ["diclofenac", "", "amlodipine"]})


@pytest.fixture(scope="module")
def full_db(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("evals_full")
    return warehouse.build(write_fixture(root / "data"), root / "full.duckdb")


@pytest.fixture(scope="module")
def cut_db(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("evals_cut")
    return warehouse.build(write_fixture(root / "data", config.CUTOFF), root / "cut.duckdb")


def test_question_bank_is_well_formed():
    ids = [q.id for q in QUESTIONS]
    assert len(ids) == len(set(ids))
    assert 40 <= len(ids) <= 60
    for q in QUESTIONS:
        assert q.kind in KINDS, q.id
        assert q.tools and set(q.tools) <= TOOLS, q.id
        assert (q.gold is None) == (q.kind == "cant_tell"), q.id
        assert q.list_mode in {"top", "set"}
        assert q.needs in {None, "ingredient_map"}


def test_gold_functions_run(full_db):
    gold = gold_answers(full_db, IMAP)
    assert set(gold) == {q.id for q in QUESTIONS}
    assert gold["so_flazen_any_h1"] is True
    assert gold["pos_teamic_total"] == 20  # all of it upstairs
    assert gold["pos_flazen_store"] == 0
    assert gold["sh_move_down"] == ["Teamic 500mg"]
    assert gold["so_out_at_cutoff_fast"] == ["Amlodipine 10mg Novel", "Flazen 50"]
    assert gold["pur_flazen_last"] == "2025-06-05"
    assert gold["sup_main_flazen"] == "Amapharm.PT"
    assert gold["ing_bufamoxy_alts_in_stock"] == 0
    assert all(gold[q.id] is None for q in QUESTIONS if q.kind == "cant_tell")


def test_ingredient_questions_skip_without_a_map(full_db):
    gold = gold_answers(full_db)
    assert not any(q.id in gold for q in QUESTIONS if q.needs)


def test_no_gold_answer_reads_past_the_cutoff(full_db, cut_db):
    # The same answers whether or not the warehouse holds data after the cutoff,
    # even though Flazen's stock-out and Teamic's shelf-out run past it and
    # Flazen's price changes after it.
    assert gold_answers(full_db, IMAP) == gold_answers(cut_db, IMAP)


def q(kind: str, **kw) -> Question:
    return Question("t", "test", "?", kind, ("run_sql",), **kw)


def test_numbers_pass_within_absolute_or_relative_tolerance():
    assert score_answer(q("number", tol=0.5), 42.0, {"value": 42.4})[0]
    assert not score_answer(q("number", tol=0.5), 42.0, {"value": 42.6})[0]
    assert score_answer(q("number", rel=0.01), 1000, {"value": "Rp 1,009"})[0]
    assert not score_answer(q("number", rel=0.01), 1000, {"value": 1011})[0]
    assert not score_answer(q("number"), 7, {"value": None})[0]


def test_dates_yes_no_and_labels():
    assert score_answer(q("date"), "2025-06-08", {"value": "2025-06-08"})[0]
    assert score_answer(q("date", tol=1), "2025-06-08", {"value": "2025-06-09"})[0]
    assert not score_answer(q("date"), "2025-06-08", {"value": "2025-06-09"})[0]
    assert score_answer(q("yes_no"), True, {"value": "Yes"})[0]
    assert not score_answer(q("yes_no"), False, {"value": True})[0]
    assert score_answer(q("label"), "Amapharm.PT", {"value": " amapharm.pt "})[0]


def test_lists_get_partial_credit():
    gold = ["A", "B", "C", "D"]
    assert score_answer(q("list"), gold, {"value": ["d", "c", "b", "a"]}) == (True, 1.0)
    assert score_answer(q("list"), gold, {"value": ["A", "B", "X", "Y", "C"]}) == (False, 0.5)
    assert list_credit(["A", "B"], ["A", "B", "C"], "set") == pytest.approx(2 / 3)
    assert list_credit([], [], "set") == 1.0
    assert score_answer(q("list"), gold, {"value": "A, B"})[1] == 0.0


def test_cant_tell_must_be_said_and_only_when_true():
    assert score_answer(q("cant_tell"), None, {"cant_tell": True})[0]
    assert not score_answer(q("cant_tell"), None, {"value": 0.3})[0]
    assert not score_answer(q("number"), 5, {"value": 5, "cant_tell": True})[0]


def test_unsupported_numbers_are_caught():
    assert numbers_in("Rp 3,997,275,634 and 42.5% and 1.2 juta") == [3997275634, 42.5, 1.2e6]
    assert numbers_in("Rp3,997,275,634 and rp12,443,759.23") == [3997275634, 12443759.23]
    tools = [{"revenue": 3997275634, "share": 0.4251}]
    text = "Revenue was Rp 4.0 billion, 42.5% of it Patent, across 7 shops."
    assert unsupported_numbers(text, tools) == [7]
    assert unsupported_numbers("The top 5 drugs.", [], "Which 5 drugs?") == []


def test_score_run_reports_each_metric():
    gold = {"rev_2024_total": 1_000_000.0, "ct_margin_flazen": None}
    results = {"rev_2024_total": {"answer": {"value": 1_002_000}, "tools": ["run_sql"],
                                  "text": "Rp 1,002,000", "tool_outputs": ["1002000"]},
               "ct_margin_flazen": {"answer": {"value": 0.25}, "tools": [],
                                    "text": "about 25%", "tool_outputs": []}}
    df = score_run(results, gold).set_index("id")
    assert df.loc["rev_2024_total", "passed"] and not df.loc["ct_margin_flazen", "passed"]
    assert df.loc["rev_2024_total", "tool_recall"] == 1.0
    assert df.loc["ct_margin_flazen", "unsupported"] == 1
    assert tool_recall(QUESTIONS[0], []) == 0.0


def test_list_markers_are_not_stated_numbers():
    from irpd.evals.scorer import unsupported_numbers
    text = "1. Flazen 50 - Rp35,340,700\n2. Teamic - Rp24,956,000\n3) Other 99"
    assert unsupported_numbers(text, [{"Flazen 50": 35340700, "Teamic": 24956000}]) == [99.0]
