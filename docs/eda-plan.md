# EDA plan

Goal: know every column, how far each can be trusted, and which patterns the
three pillars in [PLAN.md](../PLAN.md) depend on. Each route is a script in
`eda/` that reads `data/raw/irpd/` and prints its results, so every number in
[FINDINGS.md](../FINDINGS.md) can be reproduced.

Run any script with:

```bash
uvx --with duckdb --with pandas python -I eda/<script>.py
```

## Routes

**1. Columns and quality** (`eda/columns.py`)
- Every column in raw and processed: type, fill rate, distinct values, top
  values, constants.
- What processing changed: row counts dropped, categories excluded, columns
  split.
- Keys and joins: invoice to stock document, drug code across files, product
  name and unit consistency per code.
- Anomalies: duplicates, negative or huge stock, `Test Entry` rows, odd units.

**2. Stock ledger and stock-outs** (`eda/stock.py`)
- Rebuild end-of-day stock per drug per location from the raw ledger, and
  check it reconciles.
- Stock-out spells: how often, how long, which drugs, and what ends them
  (purchase, transfer, stock-take).
- The pharmacist's behaviour: how often each supplier delivers, how much
  stock there is when a purchase arrives (implied reorder point), order size.
- Stock-takes (shrinkage) and transfers between the shop floor and the store.

**3. Demand and time patterns** (`eda/demand.py`)
- Trend 2023–2025, weekday, hour, shift, Ramadan and Eid, illness seasons.
- Concentration: how few drugs make most units and revenue.
- Demand classes per drug: ADI (average days between sales) and CV² (how much
  order sizes vary), sorted into smooth, intermittent, erratic and lumpy
  (Syntetos-Boylan cut-offs 1.32 and 0.49). Done first on in-stock days only,
  so stock-out zeros don't count as no demand.
- Baskets: lines per invoice and common pairs. New and discontinued drugs.

**4. Prices and revenue** (`eda/prices.py`)
- How often prices change, by how much, and in which direction.
- Does demand move when price moves? A first look on drugs with several price
  changes.
- Revenue by category, manufacturer, drug class and payment type. Discounts.

**5. Switching to an alternative** (`eda/switching.py`)
- Group drugs by active ingredient with simple name rules, on the top drugs
  first.
- Around each stock-out of a drug, do same-ingredient alternatives sell more
  than usual? Compare with the same weekdays before the stock-out.
- Batch and expiry: how close to expiry drugs are sold.
- This is the go/no-go check for pillar 3.

## Rules

- Read-only on the data. Scripts write nothing into the repo.
- Every finding gives its number and the script that produced it.
- Revenue is `Subtotal` in rupiah. There are no costs.
