# Follow-up on the ML results: summary

Date: 1 Nov 2026. After the ML results, we tried to make sense of the demand
patterns behind them: why the forecasts are flat, how forecasts turn into reorder
decisions, and which fixes are worth testing. This file pulls that work together.
Nothing was trained for production and no model choice was made. Numbers marked "exploratory" come from
small checks and are not validated results. Anything measured on 2025 data is
diagnostic, because 2025 has already been inspected in this project.

## The short answer

The flat 8-week forecasts are not a bug, and they are not hiding much signal.
For this store, a flat level is the statistically right forecast outside the
Eid weeks. The useful work is elsewhere: the reorder policy is set once per
half-year, the forecast spread is too narrow for multi-week totals, the Eid
features have fixable defects, and the decision test cannot detect the size
of improvement that is realistically available.

Recommendation: keep the one global LightGBM for the mean forecast. Do not add
an ensemble or a per-segment model zoo. Fix the evaluation first, then make
four targeted repairs (Eid features, lead-time spread, censoring-aware
targets, rules for fading and sporadic drugs). Treat a clean test on 2026 data
as the only way to settle "ML vs current" for good.

## Why the forecasts are flat, and why that is fine

Three separate lines of work reached this: the literature review
([literature.md](literature.md)), the data analysis
([demand-patterns.md](demand-patterns.md)) and the code diagnosis
([why-forecasts-are-flat.md](why-forecasts-are-flat.md)).

**By construction.** In a drug's 8 rows (h = 1..8), only `h` and four Eid
calendar features change (`src/irpd/ml/features/calendar.py:43-52`). Every
history feature repeats. Away from Eid those calendar features are constant,
and `h` carries 0.01% of the model's gain. SES, Croston and TSB are flat by
definition too (`forecast.py:311-313`), and weeks past 8 repeat week 8
(`forecast.py:344-345`). The saved boosters reproduce `ml_forecast.csv`
exactly, so nothing is broken in serving.

**The data supports it.** On 2024 non-Eid blocks, week-to-week variation
matches what constant-rate demand would produce (observed/expected dispersion
0.94-0.98 by tier). Units come in packs, so the noise floor is compound
Poisson, not plain Poisson. For a flat SES forecast on selection data, about
82% of the error is irreducible noise and 18% is getting the level wrong.
Change within the 8-week window explains none of it. Recent changes partly
reverse rather than continue (correlation -0.2 to -0.34). ML's reaction to a
recent level shift (slope 0.20-0.35) matches what actually follows
(0.17-0.30). The baseline overreacts (0.46-0.59). WAPE barely grows with the
horizon (0.451 at h1, 0.455-0.479 at h2-h8).

**What a flat path would miss, and how much it's worth.**

- Eid weeks: they carry 21% of the error inside Eid windows. But the
  store-wide Eid factor removes only 1.5% of it. Per-drug Eid responses don't
  repeat from one year to the next (r = 0.12). Even a perfect store-level
  calendar model is worth at most 0.017 WAPE in Eid weeks.
- Yearly seasonality by drug: year-to-year month profiles don't correlate
  (-0.04), so it isn't worth testing.
- Payday weeks: a small maybe (-5.8% in one exploratory store-level test).

## Defects found

These are worth fixing whatever else happens.

1. **Eid lands a week late.** Both forecasters put the 2025 Eid peak one week
   after the real one. The real peak was the week of 24 Mar, with 37.8k units.
2. **`forecast.EID` ends at 2025** (`forecast.py:45`). Eid 2026 is invisible
   to both forecasters, and `c_eid_days` leaves its training range after
   about Sep 2025.
3. **`c_eid_days` fits one-off dips.** Far from Eid, the model learned a
   Jan 2024 dip, so ML ran 14-16% low at the 5 Jan 2025 origin.
4. **Lead-time totals are under-covered.** Weekly errors are positively
   correlated (0.13-0.24), but the policy sums weekly variances as if
   independent. Single weeks are calibrated (p95 covers ~0.96). Four-week
   totals cover only 0.90-0.93 at p95. The variance needs about 1.5-2x.
5. **Stock-out weeks are dropped** from training and scoring. These are the
   high-demand weeks, so censored drugs are forecast low (ML bias 0.91 on
   them). In 2024, dropping them flips the rare-tier ranking.
6. **Tier averages hide the slow tier.** ML's whole gain in the slow tier
   comes from its top 10% of drugs. The other 90% do slightly worse
   (1.362 vs 1.326).

## The evaluation doesn't reward the right things

This is the most important finding (details in
[forecasts-to-decisions.md](forecasts-to-decisions.md)). We confirmed it in the code:
`contest.to_policy` and `replay.simulate`/`match_stock_value`.

- **The policy is set once and held for six months.** Only the first 4 weeks
  of the forecast enter the policy. So a perfect forecast of the next 28 days
  makes lost revenue worse in this replay (+28% in H2 2024, +52% in H1 2025),
  while a perfect half-year average helps (-4% to -14%). The replay rewards a
  good six-month rate. It cannot reward Eid, short-horizon accuracy or fast
  reaction.
- **Matching to the pharmacist's stock value hides bias.** Every drug's
  reorder levels are scaled by about 1.9x, so the nominal 95% service level
  actually runs at 99.3-99.8%. Multiplying all forecasts by 0.8 changes
  matched lost revenue by under 1%, but unmatched lost revenue (what the app
  actually recommends) by +37% to +91%.
- **The test has very little power.** A forecaster 30% worse by WAPE was
  flagged as worse in only 1 of 10 bootstrap runs. Shelf-outs make up 33-40%
  of lost revenue and barely depend on the forecaster. The 0.5% tolerance in
  stock matching alone explains ML's +0.4% lost revenue in H1 2025: it held
  0.6% less stock. The intervals resample drugs over a single policy day.
- **Re-planning every 14 days** cut lost revenue 23% in H2 2024 (5.60 to
  4.33 Rp M, at matched stock). This is bigger than any forecast change
  tested. Caveat: almost all of it comes from refreshing shelf (floor)
  levels: shelf-out loss fell from 2.27 to 0.65, while stock-out loss rose
  from 3.34 to 3.68. The simulated shelf has no space limit, so the real gain
  is probably smaller.

So "ML vs current" was decided by a test that could not have told them apart.
The decision to keep the current forecaster as the default was the right call
under the pre-registered rule. But the rule's verdict says "no detectable
difference", not "ML isn't better".

## Where the reports disagreed, and what won

| Question | Claim | Evidence that settled it | Verdict |
|---|---|---|---|
| Ensemble of ML + SES+Eid | Literature review ranked it #2 | Errors correlate 0.93-0.98 on this data; ensemble WAPE 0.4627 vs ML 0.4603 on selection origins | Store data wins: no ensemble |
| Young drugs over-forecast 27-41% | Demand-pattern analysis | True for SES only. ML's bias on drugs under 13 weeks is 1.06, against 1.36 for the baseline | Only a problem for the current forecaster |
| "ML is 9% low" | Earlier journal note | It's relative to the current forecaster, which runs 5-8% high. ML vs actual sales: 0.985-0.99 | ML is close to unbiased, except on censored drugs |
| Variance inflation helps | Literature review and inventory analysis | Unmatched: lost revenue -26% for +6% stock. At equal stock, no gain | Calibration fix, not an allocation fix: judge on coverage |

## Demand segments

Only about 12% of revenue behaves in a way that needs special handling.
Shares are of the last 52 weeks, using selection data to 8 Dec 2024.

| Segment | Drugs | Revenue | Treatment |
|---|---|---|---|
| steady-stable | 362 | 46.9% | Global model as is |
| steady-shifted | 125 | 14.6% | Global model; level is the error source |
| intermittent | 823 | 26.7% | Global model is near the noise floor (1.09x) |
| censored | 206 | 4.5% | Censoring-aware targets |
| fading | 340 | 2.6% | TSB-style decay plus a stop-reorder flag |
| bulk | 74 | 1.9% | Separate large single orders from base demand |
| sporadic | 179 | 1.9% | Min-stock rule, not a forecast |
| new | 186 | 1.0% | ML already handles these better than SES |

Representative drugs, with plots in `fig/demand/` (see
[demand-patterns.md](demand-patterns.md)):

- SW2106000809 (Teamic 500mg): steady.
- SW2106000902 (Zentra): its level halved.
- OBT2108210081 (Berotec): intermittent.
- SW2106001787 (Atorvastatin 10mg DM): bulk orders plus censoring.
- OBT2306190002 (SGM Eksplor): 37% of weeks censored.

## Experiment plan

The full pre-registered plan, with a register, is in
[experiment-plan.md](experiment-plan.md). The wider options behind it are in
[options.md](options.md). In short, in order:

- **E0 Evaluation harness.** Build a rolling replay that re-plans every 14
  days and compares at equal stock across a curve of stock values. Add
  unmatched service level and fill rate, revenue-weighted pinball/CRPS of
  lead-time totals, and a bootstrap over origins as well as drugs. Measure
  the harness's own noise level before using it.
- **E1 Re-planning.** Confirm the 23% gain, with a shelf-capacity limit
  added so the floor gain isn't overstated.
- **E2 Lead-time spread.** Variance x(1 + ρ_tier(w-1)). Judged on 2- and
  4-week coverage, not lost revenue.
- **E3 Eid repair.** Add 2026-27 dates, cap `c_eid_days` beyond ±45 days, and
  count Eid days within each target week. Judged on whether the peak lands in
  the right week and nothing else gets worse. It cannot show a WAPE gain at
  this sample size.
- **E4 Censoring-aware targets.** Judged on bias for censored drugs.
- **E5 Lifecycle rules** for fading and sporadic drugs.
- **E6 Re-decide the default forecaster** under the new harness.
- **E7 One-shot test on 2026 data.** No data after 31 Dec 2025 exists, so
  this needs a new export from the pharmacy.

**Power is the limit throughout.** The smallest WAPE difference this data can
detect is about 0.02 with 11 origins, or 0.014 with 26. The ML-vs-baseline
gap is about that size. So gate forecast changes on scores of lead-time
totals, and use the replay only to show nothing got worse.

## What happened when we ran the experiments

E0-E6 were run afterwards on stored predictions; full results are in
[experiments.md](experiments.md), with working notes in
[experiment-notes.md](experiment-notes.md). In short, no change passed its
pre-registered rule. The harness (E0) reproduces the known numbers (5.60 and
4.33 Rp M), but the replay turned out unable to rank forecasters. With net-of-switching
lost revenue, 14-day re-planning (E1) cut loss by 10% in H2 2024, short of the 15%
bar. The variance factor (E2), the Eid repairs (E3), the censoring-aware targets
(E4) and the lifecycle rules (E5) each failed at least one condition. ML has 5.1%
lower pooled pinball than the current forecaster (CI −7.9% to −2.5%), but the E6 rule
for making it the default is not met. Some of these numbers differ from the
estimates above, which came from exploratory checks; the experiment results are the
ones to rely on. The 2025 numbers there are diagnostic too.

## Files and how to run them

See [README.md](README.md) for the file map and how to run each script.
