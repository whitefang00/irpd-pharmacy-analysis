"""A small synthetic IRPD warehouse for panel and leak tests (public, no private data).

Six drugs over 2 Jan 2023 to 30 Jun 2024, simulated day by day: customers
buy from the floor, the shop restocks unless a drug is in an outage window
(stock-outs), one drug is restocked through the upstairs store with transfer
gaps (shelf-outs), one sells in strips of 10 tablets (pack sizes), prices
change on set days, and one drug has a one-off price blip on 3 Mar 2024 that
only hindsight removes. Eid weeks sell more.

Draws are seeded per day, so `perturb_after` changes demand, prices and
restocking only on later days and leaves everything up to it identical.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np

from irpd import forecast, warehouse

START, END = date(2023, 1, 2), date(2024, 6, 30)
BLIP_DAY = date(2024, 3, 3)  # a leak-test origin: only hindsight knows it is a blip


@dataclass(frozen=True)
class Drug:
    code: str
    name: str
    category: str
    stock_unit: str
    sales_unit: str
    pack: int            # stock units per sales unit
    rate: float          # customers per day
    price: tuple         # ((from_day, price per sales unit), ...)
    first: date = START
    outages: tuple = ()  # (start, stop): no restock
    via_store: bool = False
    shelf_gaps: tuple = ()  # (start, stop): no transfers to the floor


DRUGS = (
    Drug("W1", "Paracetamol 500mg", "Generik", "Tab", "Tab", 1, 6.0,
         ((START, 700), (date(2024, 3, 15), 750)),
         outages=((date(2023, 11, 10), date(2023, 11, 30)),
                  (date(2024, 2, 20), date(2024, 3, 5)), (date(2024, 5, 1), date(2024, 5, 20)))),
    Drug("W2", "Flazen 50", "Patent", "Tab", "Strip", 10, 0.6,
         ((START, 30000), (date(2023, 9, 1), 31500), (date(2024, 4, 1), 33000))),
    Drug("W3", "Cough Syrup", "OTC", "Btl", "Btl", 1, 2.0, ((START, 18000),),
         via_store=True,
         shelf_gaps=((date(2023, 12, 1), date(2023, 12, 12)),
                     (date(2024, 2, 24), date(2024, 3, 2)),
                     (date(2024, 4, 20), date(2024, 4, 28)))),
    Drug("W4", "Paracetamol Drops", "Generik", "Btl", "Btl", 1, 0.15, ((START, 9000),)),
    Drug("W5", "Flazen 100", "Patent", "Tab", "Tab", 1, 3.0, ((START, 5000),),
         first=date(2023, 7, 3), outages=((date(2024, 1, 10), date(2024, 2, 15)),)),
    Drug("W6", "Herbal Tea", "-", "Box", "Box", 1, 1.0, ((START, 12000),)),
)
# drug_code -> (ingredients, strength, form, kind); W3 and W6 are singletons.
INGREDIENTS = {
    "W1": ("paracetamol", "500 mg", "tablet", "drug"),
    "W2": ("diclofenac", "50 mg", "tablet", "drug"),
    "W3": ("dextromethorphan", "15 mg/5 ml", "syrup", "drug"),
    "W4": ("paracetamol", "100 mg/ml", "drops", "drug"),
    "W5": ("diclofenac", "100 mg", "tablet", "drug"),
    "W6": ("", "", "", "herbal"),
}


def _in(day: date, windows) -> bool:
    return any(a <= day <= b for a, b in windows)


def _price(d: Drug, day: date, perturb_after: date | None) -> float:
    p = [v for since, v in d.price if since <= day][-1]
    if d.code == "W6" and day == BLIP_DAY:
        p *= 1.2
    if perturb_after and day > perturb_after:
        p *= 1.5
    return p


def write_csvs(out: Path, seed: int = 0, perturb_after: date | None = None) -> Path:
    """Write the three IRPD files under `out` (the layout `warehouse.build` reads)."""
    sales, stock, raw = [], [], []
    level = {d.code: {"floor": 0, "store": 0} for d in DRUGS}
    n_inv = n_doc = 0

    def move(day, time, doc, kind, d, loc, qty_in, qty_out):
        before = level[d.code][loc]
        level[d.code][loc] = before + qty_in - qty_out
        row = len(stock) + 1
        stock.append([row, day.isoformat(), time, doc, kind, d.code, d.name, d.category,
                      before, qty_in, qty_out, level[d.code][loc], d.stock_unit])
        place = "Apotek" if loc == "floor" else "Gudang Obat 1"
        raw.append([row, day.strftime("%d %b %Y ") + time, place, d.code, d.name, doc, kind,
                    f"{before} {d.stock_unit}", f"{qty_in} {d.stock_unit}",
                    f"{qty_out} {d.stock_unit}", f"{level[d.code][loc]} {d.stock_unit}", "-",
                    d.category, "NON KONSINYASI", "Aktif"])

    day = START
    while day <= END:
        rng = np.random.default_rng([seed, (day - START).days])
        late = perturb_after is not None and day > perturb_after
        eid = any(-7 <= (day - e).days <= 28 for e in forecast.EID)
        for d in DRUGS:
            if day < d.first:
                continue
            total = level[d.code]["floor"] + level[d.code]["store"]
            target = max(int(d.rate * d.pack * 21), 2 * d.pack)
            outage = _in(day, d.outages) or (late and rng.random() < 0.3)
            if total < target / 3 and not outage:
                n_doc += 1
                move(day, "07:00:00", f"PB{n_doc:05d}", "Pembelian (Mitra Sehat)", d,
                     "store" if d.via_store else "floor", target - total, 0)
            if d.via_store and not _in(day, d.shelf_gaps):
                need = max(int(d.rate * 3) - level[d.code]["floor"], 0)
                qty = min(need, level[d.code]["store"])
                if qty > 0:
                    n_doc += 1
                    doc = f"MT{n_doc:05d}"
                    move(day, "07:30:00", doc, "Mutasi", d, "store", 0, qty)
                    move(day, "07:30:00", doc, "Mutasi", d, "floor", qty, 0)
            rate = d.rate * (1.6 if eid else 1.0) * (2.5 if late else 1.0)
            customers = int(rng.poisson(rate))
            if d.code == "W6" and day in (BLIP_DAY - timedelta(days=1), BLIP_DAY):
                customers = 1  # one line a day, so the blip is a one-line spell
            for c in range(customers):
                want = int(rng.integers(1, 3))
                qty = min(want, level[d.code]["floor"] // d.pack)
                if qty <= 0:
                    break
                n_inv += 1
                inv, time = f"PJ{n_inv:06d}", f"{8 + c % 12:02d}:{c % 60:02d}:00"
                price = _price(d, day, perturb_after)
                sales.append([len(sales) + 1, day.isoformat(), time, inv, "Cashier_001",
                              d.code, d.name, d.category, qty, d.sales_unit, price, 0,
                              qty * price, "TUNAI"])
                move(day, time, inv, "Penjualan", d, "floor", 0, qty * d.pack)
        day += timedelta(days=1)

    (out / "processed_dataset").mkdir(parents=True, exist_ok=True)
    (out / "raw_dataset").mkdir(parents=True, exist_ok=True)
    _write(out / "processed_dataset/sales_processed.csv",
           ["RowNumber", "Date", "Time", "InvoiceNumber", "CashierID", "DrugCode", "DrugName",
            "Category", "Quantity", "Unit", "UnitPrice", "DiscountAmount", "Subtotal",
            "PaymentType"], sales)
    _write(out / "processed_dataset/stock_processed.csv",
           ["RowNumber", "Date", "Time", "DocumentNumber", "TransactionType", "DrugCode",
            "DrugName", "Category", "InitialStock", "StockIn", "StockOut", "RemainingStock",
            "Unit"], stock)
    _write(out / "raw_dataset/stock_raw.csv",
           ["RowNumber", "DateTime", "Warehouse", "DrugCode", "DrugName", "DocumentNumber",
            "TransactionDescription", "InitialStock", "StockIn", "StockOut", "RemainingStock",
            "DrugClass", "Category", "DrugType", "Status"], raw)
    return out


def _write(path: Path, header: list[str], rows: list) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def build(out: Path, seed: int = 0, perturb_after: date | None = None) -> Path:
    """Write the files and build the warehouse, with an ingredient map, under `out`."""
    path = warehouse.build(write_csvs(out / "data", seed, perturb_after), out / "wh.duckdb")
    with duckdb.connect(str(path)) as con:
        con.executemany(
            "INSERT INTO ingredient_map VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [[d.code, d.name, *INGREDIENTS[d.code], bool(INGREDIENTS[d.code][0]),
              "synthetic" if INGREDIENTS[d.code][0] else ""] for d in DRUGS])
    return path
