"""A small made-up pharmacy for the agent tool tests, written as IRPD-shaped CSV files.

3 Mar - 13 Jul 2025 (past the cutoff, so tools must not see the end), five products:

- P1 Zetamol tb 500mg (paracetamol tablet): 10 Tab a day at Rp 550, Rp 600 from
  1 May. PT Satu delivers 150 every other Monday into the store; the floor is
  topped up from the store. The 14 Apr delivery is skipped, so it is out of
  stock 16-27 Apr, until the 28 Apr delivery.
- P2 Sanmol tab 500 (paracetamol tablet): 4 Tab a day; PT Dua delivers 30 every
  Monday onto the floor.
- P3 Zetamol Syr 60ml (paracetamol syrup): 1 Fls every other day; one delivery.
- T1 Teamic 500mg (tranexamic acid): 6 Tab a day, delivered to the store by PT
  Dua; nobody refills the floor from 24 Jun, so it is shelf-out from 25 Jun.
- U1 Kasa steril: 2 Pcs a day, no ingredient mapping.
"""

import csv
from datetime import date, datetime, timedelta
from pathlib import Path

FIRST, LAST = date(2025, 3, 3), date(2025, 7, 13)

DRUGS = {  # code: name, category, unit, ingredients, strength, form
    "P1": ("Zetamol tb 500mg", "Generik", "Tab", "paracetamol", "500mg", "tablet"),
    "P2": ("Sanmol tab 500", "Patent", "Tab", "paracetamol", "500mg", "tablet"),
    "P3": ("Zetamol Syr 60ml", "Patent", "Fls", "paracetamol", "", "syrup"),
    "T1": ("Teamic 500mg", "Patent", "Tab", "tranexamic acid", "500mg", "tablet"),
    "U1": ("Kasa steril", "Alkes", "Pcs", "", "", ""),
}
PLACE = {"floor": "Apotek", "store": "Gudang Obat 1"}


class Book:
    def __init__(self):
        self.stock: list[dict] = []
        self.sales: list[dict] = []
        self.bal = {(c, loc): 0 for c in DRUGS for loc in PLACE}
        self.t = None
        self.n = 0

    def _row(self, day, code, loc, doc, kind, qin, qout):
        self.t = max(self.t or datetime.min, datetime.combine(day, datetime.min.time())
                     .replace(hour=8)) + timedelta(minutes=1)
        before = self.bal[code, loc]
        self.bal[code, loc] = before + qin - qout
        name, cat, unit = DRUGS[code][:3]
        self.stock.append(dict(t=self.t, code=code, name=name, cat=cat, unit=unit, loc=loc,
                               doc=doc, kind=kind, opening=before, qin=qin, qout=qout,
                               closing=self.bal[code, loc]))

    def buy(self, day, code, qty, supplier, loc):
        self.n += 1
        self._row(day, code, loc, f"PB{self.n:04d}", f"Pembelian ({supplier})", qty, 0)

    def move(self, day, code, qty, src, dst):
        self.n += 1
        doc = f"MT{self.n:04d}"
        self._row(day, code, src, doc, "Mutasi", 0, qty)
        self._row(day, code, dst, doc, "Mutasi", qty, 0)

    def sell(self, day, code, qty, price):
        qty = min(qty, self.bal[code, "floor"])
        if qty <= 0:
            return
        self.n += 1
        inv = f"PJ{self.n:04d}"
        self._row(day, code, "floor", inv, "Penjualan", 0, qty)
        name, cat, unit = DRUGS[code][:3]
        self.sales.append(dict(t=self.t, inv=inv, code=code, name=name, cat=cat, qty=qty,
                               unit=unit, price=price))


def simulate() -> Book:
    b = Book()
    day = FIRST
    while day <= LAST:
        i = (day - FIRST).days
        monday = day.weekday() == 0
        if monday and (i // 7) % 2 == 0 and day != date(2025, 4, 14):
            b.buy(day, "P1", 150, "PT Satu", "store")
        if monday:
            b.buy(day, "P2", 30, "PT Dua", "floor")
            b.buy(day, "T1", 45, "PT Dua", "store")
        if day == FIRST:
            b.buy(day, "P3", 60, "PT Tiga", "floor")
            b.buy(day, "U1", 300, "PT Tiga", "floor")
        for code, low, fill in (("P1", 20, 60), ("T1", 12, 30)):
            if code == "T1" and day >= date(2025, 6, 24):
                continue
            if b.bal[code, "floor"] < low and b.bal[code, "store"] > 0:
                b.move(day, code, min(fill, b.bal[code, "store"]), "store", "floor")
        b.sell(day, "P1", 10, 550 if day < date(2025, 5, 1) else 600)
        b.sell(day, "P2", 4, 500)
        if i % 2 == 0:
            b.sell(day, "P3", 1, 20000)
        b.sell(day, "T1", 6, 2000)
        b.sell(day, "U1", 2, 1000)
        day += timedelta(days=1)
    return b


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write(dst: Path) -> Path:
    """Write the fixture files (and the ingredient map, as ingredient_map.csv) to `dst`."""
    b = simulate()
    proc, raw = [], []
    for n, r in enumerate(b.stock, 1):
        proc.append({"RowNumber": n, "Date": r["t"].date(), "Time": r["t"].time(),
                     "DocumentNumber": r["doc"], "TransactionType": r["kind"],
                     "DrugCode": r["code"], "DrugName": r["name"], "Category": r["cat"],
                     "InitialStock": r["opening"], "StockIn": r["qin"], "StockOut": r["qout"],
                     "RemainingStock": r["closing"], "Unit": r["unit"]})
        u = f" {r['unit']}"
        raw.append({"RowNumber": n, "DateTime": r["t"].strftime("%d %b %Y %H:%M:%S"),
                    "Warehouse": PLACE[r["loc"]], "DrugCode": r["code"], "DrugName": r["name"],
                    "DocumentNumber": r["doc"], "TransactionDescription": r["kind"],
                    "InitialStock": f"{r['opening']}{u}", "StockIn": f"{r['qin']}{u}",
                    "StockOut": f"{r['qout']}{u}", "RemainingStock": f"{r['closing']}{u}",
                    "DrugClass": "-", "Category": r["cat"], "DrugType": "NON KONSINYASI",
                    "Status": "Aktif"})
    sales = [{"RowNumber": n, "Date": s["t"].date(), "Time": s["t"].time(),
              "InvoiceNumber": s["inv"], "CashierID": "Cashier_001", "DrugCode": s["code"],
              "DrugName": s["name"], "Category": s["cat"], "Quantity": s["qty"],
              "Unit": s["unit"], "UnitPrice": s["price"], "DiscountAmount": 0,
              "Subtotal": s["qty"] * s["price"], "PaymentType": "TUNAI"}
             for n, s in enumerate(b.sales, 1)]
    _write(dst / "processed_dataset/stock_processed.csv", proc)
    _write(dst / "raw_dataset/stock_raw.csv", raw)
    _write(dst / "processed_dataset/sales_processed.csv", sales)
    _write(dst / "ingredient_map.csv", [
        {"drug_code": c, "drug_name": v[0], "ingredients": v[3], "strength": v[4],
         "form": v[5], "kind": "drug" if v[3] else "device", "mapped": True,
         "source_model": "fixture"} for c, v in DRUGS.items()])
    return dst
