# Findings

What the exploratory analysis found. Each section names the script in `eda/`
that reproduces its numbers. The scripts read `data/raw/irpd/`, print their
results and write nothing. Run one with:

```sh
uvx --with duckdb --with pandas python -I eda/<script>.py
```

Revenue is in rupiah (Rp). There are no costs. The modelling that came later
changed a few of these numbers; see [Corrections after modelling](#corrections-after-modelling).

## The data is sound, with a few traps (`columns.py`, `stock.py`)

- The stock ledger is internally consistent. Opening + in − out = closing on
  every row, and per drug and location each row starts where the last ended
  (188 exceptions, nearly all stock-takes booked in the same second).
- Sales and stock join cleanly on invoice + drug: 645,912 sale lines match.
  The 3,738 that don't are almost all services, which carry no stock.
- The processed files are the raw files minus three categories (medical
  equipment, baby, drinks), row for row.

Traps to handle in the pipeline:

- Units. 197 drugs are sometimes sold by the strip or box while the
  ledger counts tablets or sachets. Demand should come from the ledger's
  `StockOut` on sale rows, in base units.
- Locations. The processed stock file merges the shop floor and the
  upstairs store. Its end-of-day balance is wrong on 36% of drug-days for
  drugs that use the store. Build stock from the raw file per location.
- Opening errors. A few drugs started with years of stock on the books
  (one held 68 years of sales) until counts in July–September 2023. Start
  stock analysis after September 2023 and flag these drugs.
- Negative stock is mostly a sale rung up before the stock was booked
  in, fixed within minutes. Only 30 drug-location-days end the day negative.
- Placeholders. `-` means none: `DrugClass` is 99.7% `-` and
  `Manufacturer` 67%. Six drugs have names of 1–3 letters ("De" is a top
  seller). 12% of lines have a placeholder batch.
- Returns (825 rows) are only in the ledger, so sales revenue is
  slightly overstated.

## Demand: steady, concentrated, with one big calendar effect (`demand.py`)

- Traffic fell and baskets got pricier. Invoices per trading day went
  361 → 303 → 299 (2023–2025), while revenue per invoice rose
  Rp 33.9K → 36.2K → 38.5K.
- Two daily peaks: 9–10am and 4–8pm, with 7pm alone at 12.8% of invoices.
  Sunday is busiest (1.16× the average day).
- Eid is the biggest calendar effect. The week after Eid runs 1.44–1.64×
  normal, the week before 1.24–1.39×. The fasting weeks themselves are flat.
- No illness season repeats from year to year, so season isn't worth modelling.
- 699 drugs make 80% of revenue. Units mix tablets and bottles, so rank by
  revenue.
- Weekly is the right grain. Daily, almost every drug is intermittent or
  lumpy (46 are smooth). Weekly, 506–536 drugs are smooth and make about 60%
  of revenue.
- 1,803 of the drugs selling after 30 June 2025 have at least a year of
  history before it.

## Prices: small rises after deliveries, no visible demand response (`prices.py`)

- Prices are clean list prices. Price tier can be ignored.
- 2,552 real price changes over 3 years, 90% of them rises (median +6%).
  68% come within 14 days of a delivery of that drug.
- Shelf prices rose 10.1% over 3 years (5.6%, 2.2%, 1.5% a year).
- No detectable price response. After a rise, demand fell no more than on
  random dates. Report it as a finding, but don't build on it.
- The 2024 revenue dip (−10%) is fewer customers (invoices −17%), partly
  offset by prices and mix. The 2025 recovery (+4%) is bigger baskets,
  prices and new products, with invoices still down 2%.

## Stock-outs: rare on fast drugs, common on slow ones (`stock.py`)

- Inside each drug's selling period, stock is at zero on 0.8% of days for
  fast drugs, 3.1% regular, 8.7% slow and 20.9% rare.
- Spells are short on fast drugs (median 2 days) and longer on regular ones
  (median 4, 1 in 10 over 25 days). 94% end with a delivery.
- On stock-out days sales drop to 3% of normal, so the censoring is clean.
- The pharmacist's policy: deliveries mostly on weekdays from many
  suppliers. Fast drugs are reordered about every 14 days, with a median of
  18 days of cover left when the delivery arrives. Only 2% arrive at zero.
- Shelf-outs: on fast drugs the shop floor is empty while the store
  upstairs has stock more often (2,577 days) than the whole pharmacy is out
  (1,864). Next-day sales are 0.59× normal, so some sales are lost to
  restocking the floor, not to ordering.
- A rough upper bound on demand lost to stock-outs is about Rp 450M over 3
  years (about 3.6% of revenue), most of it on slow drugs.
- Stock-takes mostly correct booking errors. Shrinkage is near zero.

## Switching: customers do buy the alternative (`switching.py`)

- With a hand-written brand dictionary plus rules for generic names, 59
  ingredient groups have 2+ drugs that sell (213 drugs).
- Across 189 stock-out spells, same-ingredient alternatives sold 1.18×
  normal (95% CI 1.12–1.26). Random windows for the same drugs gave 0.985×,
  and unrelated drugs 0.974×.
- About 84% of the lost sale lines show up as extra alternative sales
  (95% CI 63–102%), and close to 100% on confident mappings. So most
  stock-outs cost little revenue when an alternative is on the shelf.
- It varies by ingredient: strong for allopurinol, amlodipine and
  omeprazole, little for paracetamol and mefenamic acid.
- Whole groups almost never run out. Only 4 of 59 groups were ever fully
  out at once: 19 days in 7 short spells (salbutamol 4mg, glibenclamide 5mg,
  furosemide 40mg, triamcinolone 4mg). That's 0.2% of single-drug stock-out
  days. When it happened, the group sold nothing, so those sales were lost,
  but only about Rp 0.2M in total.
- Expiry dates on sale lines are month-level at best and partly stale. Use
  them as a hint only.

## What this changed in the build

- Agent vs pharmacist. The pharmacist is already good on fast
  drugs, so the contest is on regular and slow drugs and on shelf-outs. The
  baseline is their actual total stock. Matched stock value for fast and
  regular drugs in H2 2025 is about Rp 318M. Score the 820 fast and regular
  drugs still selling then. Watch two holdout events: the August 2025 full
  count, and the store coming back into use in July 2025.
- Lost demand. Stock-outs barely change how drugs are
  classified, so the value is measuring lost demand, not fixing the
  forecaster. Use spells on total stock, and partly censor shelf-out days.
- Switching. Go. The signal is clear and passes both
  placebos. The LLM mapping should output ingredient, strength and form
  separately, allow "unknown", and treat combinations as their own sets.
  About half of revenue is combinations and non-drugs, so expect many unknowns.

## Corrections after modelling

The modelling that followed changed some of the numbers above. The EDA
sections are left as they were. These are the figures to use.

- Switching is about 30%, not 84%. The EDA measured it on 189 stock-outs in
  59 hand-built ingredient groups, mostly fast drugs. Measured over every drug
  the LLM mapping covers (1,641 stock-outs with an alternative in stock), the
  alternatives pick up 29.5% of the lost sale lines (95% CI 11-50%). The two
  placebos give -1.7% and +1.4%. Amlodipine, omeprazole and simvastatin still
  switch strongly, and paracetamol still barely does. Allopurinol does not
  hold up: over 6 stock-outs its alternatives sold less than normal.
- Matched stock value is Rp 312M, not 318M. Under the replay's definitions
  (834 drugs, stock valued at each drug's median selling price), the
  pharmacist's average stock for fast and regular drugs in H2 2025 is Rp 312M.
- The scope is 834 drugs, not 820. These are the drugs that were fast or
  regular at the cutoff and sold at least once after it.
- Lost demand is smaller than the rough bound. The EDA's upper bound was
  about Rp 450M over 3 years (3.6% of revenue). The estimator, tested on fake
  stock-outs, gives Rp 194M from October 2023 to June 2025 (2.75% of revenue),
  or Rp 163M (2.3%) net of switching. It counts at most 28 days per spell;
  counting every day of every spell gives Rp 535M, mostly slow and rare drugs
  that were never restocked.
- The pharmacist's ledger is not a fair opponent. The EDA said to use the
  pharmacist's actual stock as the baseline. But 64% of their H2 2025
  stock-out days are in spells over 14 days, which a simulator that delivers
  on time can't reproduce. A copy of their habits in the same simulator, at
  its own stock level, has about 6x fewer stock-out days (717 vs 4,246). The
  contest is against that copy, and the ledger is a reference.
- Ingredient mapping covers more than expected. The EDA expected many
  unknowns, since about half of revenue is combinations and non-drugs. The
  mapping, which handles combinations as their own sets, gives a confident
  answer for drugs making 84% of revenue.
