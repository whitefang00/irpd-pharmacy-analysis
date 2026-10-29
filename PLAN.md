# Plan

## Goal

Build an AI analyst for a real retail pharmacy's sales and stock. It answers
questions about demand, revenue and inventory by calling tools: SQL, a
forecaster, a reorder planner and a policy replay. Every number it states must
come from a tool result.

The data has a full stock ledger but no costs, so the focus is
**inventory**: stock-outs, lost revenue and reorder decisions.

Target headline:

> *An AI pharmacy analyst that read 3 years of a real pharmacy's stock ledger,
> recovered the sales hidden by stock-outs, worked out which customers switched
> to a generic, and then out-ordered the real pharmacist on 6 months it never
> saw.*

A chat agent over SQL isn't new on its own. What makes this one stand out is that the data has a **real human to
compete against**: the pharmacist's actual stock decisions are in the ledger.
The project rests on three pillars, plus an eval suite that checks the agent's
answers.

### 1. Agent vs the real pharmacist

- The agent only sees data up to a hard cutoff (provisionally 30 June 2025).
  It sets reorder points for the rest of 2025.
- The replay runs that policy against real demand, and against what the
  pharmacy actually did over the same months.
- Fair comparison: the agent's policy has to keep about the same average
  stock value (at selling price) as the pharmacist, or the result is shown on
  a curve of stock held vs stock-outs.
- Scores: stock-out days, units and revenue lost (net of switching, see
  pillar 3), and average stock held.
- The holdout lives in a table only the scorer can open. A test proves no tool
  can reach it.
- **The result is published whichever way it goes.** If the agent loses, the
  write-up says where and why.

### 2. Recovering the sales nobody saw

- On days a drug is at zero, recorded sales show no demand. A naive forecaster
  learns from that and under-orders.
- Treat sales on stock-out days as censored (a floor, not the truth) and
  estimate true demand.
- Proof: on drugs that rarely run out, hide some periods as fake stock-outs
  and check the model recovers what really sold.
- Score against a naive forecaster on raw sales, on held-out weeks.
- Visual: the stock line hits zero, sales flatline, and the estimate of
  missed sales is shaded in.

### 3. Did they leave, or buy the generic?

- When a branded drug runs out, a customer may buy another brand of the same
  medicine, so the sale isn't lost.
- An LLM maps each drug name to its active ingredient, strength and form (for
  example, Zetamol tb 500mg → paracetamol 500mg tablet). A hand-labelled
  sample of about 200 drugs scores the mapping: accuracy, and "unknown"
  instead of a guess.
- Then, around each stock-out, compare sales of same-ingredient alternatives
  with their normal level. That splits lost revenue into **switched** and
  **truly lost**.
- **First check before building it:** whether switching shows up at all. If
  the signal is weak, it becomes a measured finding instead of a feature.

### Evals

Questions whose correct answers are computed in code. They measure accuracy,
tool choice, and how often the agent states a number no tool produced.

### Demo

The agent answers "what should I reorder this week and why?" with its tool
calls shown live, next to the replay chart of agent vs pharmacist.

## Data

**IRPD: Indonesian Retail Pharmacy Sales and Stock Transaction Dataset
(2023–2025).** One privately owned retail pharmacy in Central Java, exported
from its POS system. CC BY 4.0, with an accompanying Data in Brief article.
[Mendeley 10.17632/w6zjtdjk9m](https://data.mendeley.com/datasets/w6zjtdjk9m/1).
It lives in `data/raw/irpd/` and is not committed.

| File | Rows | Notes |
|---|---|---|
| `sales_raw.csv` / `sales_processed.csv` | 649,656 / 627,389 sale lines | Timestamped receipt lines: drug, batch, expiry, qty, unit, price, discount, fees, cashier, shift, payment |
| `stock_raw.csv` / `stock_processed.csv` | 804,027 / 775,560 movements | Every stock movement: sale, purchase (with supplier), transfer, stock-take, return. Opening, in, out, closing |

**First checks (Oct 2026)**

- 2 Jan 2023 – 31 Dec 2025, 1,088 trading days. It's closed on New Year's Day
  and Eid, and Sundays are the busiest day.
- 349K invoices, 1.8 lines each. 2,972 drugs sold and 4,051 in the stock ledger.
- Opening + in − out = closing on every stock row. Within each storage
  location (`Warehouse` in the raw file), each row's opening equals the
  previous closing 100% of the time.
- Units sold (4.64M) roughly match sale stock-outs in the ledger (4.74M).
- Revenue = `Subtotal` = qty × price − discount + fees on every row. Discounts
  and fees are tiny. About 12.6bn rupiah in total: 4.44bn in 2023, 4.00bn in
  2024 and 4.16bn in 2025.
- 47% of drugs changed price at least once.
- The top 10% of drugs make up 84% of units.
- 4.4% of stock movements end at zero, and 3,949 drugs hit zero at some point.

**Decisions so far**

| Finding | Decision |
|---|---|
| The processed stock file merges the sales floor and the upstairs store, which breaks the running balance on transfers | Build stock from the **raw** file, per location, then sum per drug per day |
| No cost or purchase-price data | Track **revenue**, not profit. Value stock at selling price, labelled as such |
| Days at zero stock hide demand | Treat sales on stock-out days as a floor (censored), not as true demand |
| About 40% of drug-days sit at zero, mostly slow drugs that ran out and weren't restocked | Measure stock-outs on drugs that actually sell (demand tiers) |
| 0.3% of rows have negative stock, plus a few huge balances | Flag them, keep them, and exclude them from the replay scoring |
| The processed files drop medical equipment, baby products and drinks | Start from the processed category scope, and note it |

**What the data can't tell us** (the README and the agent will say this)

- No costs, so no margin or profit.
- No purchase order dates, so no true lead times. Lead time is a parameter,
  defaulting to each supplier's observed restock cadence.
- The stock ledger has no batch, so expiry risk only comes from batches seen
  in sales.
- One pharmacy and 3 years. Illness seasons are only seen twice before the
  cutoff, so rolling backtests are reported alongside the holdout.

## Phases

1. **Audit and pipeline.** Load raw sales and stock into DuckDB. Build tables:
   `sales_lines`, `daily_stock` (per drug, per location and total),
   `stockouts` (spells at zero), `purchases` (supplier, qty), `stock_takes`
   (adjustments), `prices` (price history) and `batches` (expiry from sales).
   Also the pharmacist baseline: what they held and when they reordered.
2. **Switching check (pillar 3, go/no-go).** Rough ingredient grouping on the
   top drugs, then check whether same-ingredient alternatives sell more during
   stock-outs. Decide whether pillar 3 is a feature or a finding.
3. **Ingredient mapping (pillar 3).** LLM mapping of drug names to ingredient,
   strength and form, the hand-labelled sample, and its score.
4. **Stock-out-aware demand (pillar 2).** Demand tiers, the censored-demand
   forecaster, the fake stock-out test, and the score against the naive
   forecaster.
5. **Agent tools.** `run_sql`, `forecast`, `stock_position` (stock and days of
   cover), `stockout_history` (lost units and revenue, switched vs truly
   lost), `alternatives` (same-ingredient drugs), `reorder_plan`,
   `price_history`, `shrinkage` (stock-take adjustments) and `expiry_risk`.
   All tools read through an `as_of` date.
6. **Evals.** 40–60 questions with correct answers computed in code: stock-outs,
   revenue by drug, category, manufacturer, shift and hour, supplier mix,
   price changes, expiry, alternatives.
7. **Agent vs pharmacist replay (pillar 1).** Hard cutoff and sealed holdout.
   The agent sets reorder points, and the replay scores it against the
   pharmacist at matched stock value.
8. **App and demo.** A web app: a drug page with its stock line
   (stock-outs shaded, purchases marked, missed sales shaded), forecast
   playback, the agent vs pharmacist chart, and chat with the agent's tool
   calls shown live. Then record a demo.

## Open decisions

- Model budget and default backend.
