# Architecture options and how to test them

The last of the five follow-up reports. It reconciles the other four
([literature](literature.md), [demand patterns](demand-patterns.md),
[why forecasts are flat](why-forecasts-are-flat.md),
[forecasts to decisions](forecasts-to-decisions.md)), sizes the candidate
changes with a few small checks, and recommends an architecture. The experiment
plan is in [experiment-plan.md](experiment-plan.md).

Scripts are in [`scripts/architecture/`](scripts/architecture/), outputs in
`data/followup/architecture/` (not committed):

- `explore.py` → printed tables and `A_*.csv` to `E_*.csv`. Covers ensembles,
  lead-time variance calibration, power, the ceiling for a store-level calendar
  model, and young-drug bias. It only reads existing prediction files and takes 40 s.
- `power_rolling.py` → printed tables. Rolling 14-day re-plan replay,
  unmatched, with perturbations.
- `frontier_rolling.py` → printed tables and `frontier_rolling_H2-2024.csv`. The same variants compared at
  equal average stock value.

Everything here is **exploratory**. Results on the selection period (11 origins in
2024, H2 2024 replay) are the clean evidence. Results on the 2025 eval origins are
**diagnostic**, because 2025 has already been inspected. Nothing was trained.

## The short version

The point forecast is close to the noise floor, and the two forecasters make almost
the same errors. Their errors correlate 0.93-0.98 by tier. So the gains left are
small and come from four places: fixing the Eid defects, making the lead-time spread
honest, fixing level bias where stock-outs censor the data, and a few lifecycle
rules. None of these needs a new forecasting architecture.

The evaluation is the bigger problem. A forecast change of the size we can still make
cannot be seen by any lost-revenue test we can run on this data. At a fixed stock
budget, mean noise that adds 0.01 to WAPE moves lost revenue anywhere from −11% to
+10%, depending only on which drugs the noise lands on. Decisions about the forecast
should therefore rest on proper forecast scores of lead-time totals, computed over
many origins. The rolling replay should act as a non-inferiority guard and as the
judge of policy changes such as re-plan cadence and floor rules.

**Recommendation.** Keep one global LightGBM for the mean. Put a calibrated lead-time
distribution on top of it, and add a thin layer of rules for specific segments. Do
not build a router or an ensemble for the mean. Keep SES+Eid as the fallback and as
a reference. Fix the evaluation first.

## 1. Where the reports conflict, and which evidence wins

**Equal-weight ensemble (the literature review ranks it #2).** The data says no. The ensemble case
rests on the two forecasters making different errors. They don't. On the selection
origins, the correlation between ML and baseline errors is 0.93 (fast), 0.98
(regular), 0.96 (slow) and 0.94 (rare). On 2025 (diagnostic), ML against the current
forecaster is 0.97 / 0.98 / 0.98 / 0.87. Some results from `explore.py`:

| WAPE h1-8 | ML | baseline / current | equal-weight ensemble |
|---|---|---|---|
| selection, all tiers | **0.4603** | 0.4763 | 0.4627 |
| selection, revenue-weighted 4-week | **0.364** | 0.381 | 0.365 |
| 2025 diag., ML + current | 0.4746 | 0.4792 | **0.4741** |
| 2025 diag., revenue-weighted 4-week | **0.382** | 0.402 | 0.386 |

On the clean period the ensemble sits between its members and never beats ML. Its
one virtue is bias: ML runs 0.985, the ensemble 1.004. That bias can be fixed more
directly (item 3 below). The literature result is correct in general but does not
apply when the members are this alike. Evidence from this store wins.

**Young drugs (demand patterns: SES over-forecasts them by 27-41%).** This holds for SES and
for the current forecaster, but not for ML. Drug-origins in the selection period
where the drug is under 13 weeks old:

| | ML | SES-family baseline |
|---|---|---|
| bias | 1.06 | 1.36 |
| WAPE | 1.08 | 1.26 |

ML already pools across drugs, so a pooled prior for new drugs only matters if the
current forecaster stays the default. On segment labels (hindsight for "new"), ML
bias is 1.21 against 1.65.

**"ML is 9% low" (JOURNAL; the forecasts-to-decisions report asks for a check).** The 9% is relative to the
current forecaster, not to actual sales. Against actuals:

- ML bias is 0.985 on selection and 0.989 on 2025.
- On the H2 2025 rolling forecasts, ML runs 3% over raw sales and current runs 8%
  over (JOURNAL). The flatness diagnosis finds median ML/actual 1.008, with current 5-8% high after
  the July store change.

So the gap is mostly the current forecaster running high. ML has one real bias: it
runs low on the censored segment (0.91 on selection, scored on uncensored weeks
only). True demand there is higher still, so the real shortfall is larger. That is a
censoring problem, not a global level problem.

**How big is the censoring bias?** The literature review treats it as major. The forecasts-to-decisions report's `lost.py`
evidence says the selection bias from stock-outs is small (pre-stock-out demand is
0.95-0.99x the post-restock level). The flatness diagnosis finds that scoring dropped weeks at the
sales floor adds 0.06-0.13 WAPE on slow drugs and 0.12-0.4 on rare drugs. Both are
right about different things. Across the store the bias is small. Inside the
censored segment (206 drugs, 4.5% of revenue) and the slow and rare tiers it is
material, and ML is 9% low there. The fix belongs to that segment, not to the whole
model.

**Lead-time variance multiplier (forecasts-to-decisions says x1.5-2).** Confirmed, with one caveat.
I fitted a per-tier window factor, 1 + ρ(w−1), on selection origins 1-6 and tested
it on origins 7-11 (Jul-Oct 2024) and on 2025.

| 4-week p95 coverage (ML) | recipe | ρ-factor |
|---|---|---|
| fast | 0.933 sel / 0.897 2025 | 0.967 sel / 0.935 2025 |
| regular | 0.934 / 0.933 | 0.957 / 0.954 |
| slow | 0.960 / 0.942 | 0.970 / 0.957 |

The factor helps at 2 and 4 weeks and leaves 1 week alone. The caveat: the needed
size changes with the season. Over the quiet test origins of H2 2024 it over-covers,
and fast-tier p95 pinball gets slightly worse (0.0291 → 0.0297). Over the Eid season
of 2025 it is still a little short. A conformal multiplier fitted directly on 4-week
totals over-shoots more (x1.95-2.2 on fast). So use the simpler ρ-factor, and test
whether ρ should differ inside and outside Eid windows.

**Does the variance fix cut lost revenue?** Not at equal stock. In the rolling
replay at the app's own scale (unmatched), variance x1.5 cut lost revenue by 26% but
raised average stock value by 6%. On the equal-stock frontier the ratios are
0.89-1.07: no clear gain. The fix makes the stated 0.95 service level true, so what
the pharmacist is told matches what happens. It does not allocate stock better. It
should be judged on coverage and on achieved service against target, not on
lost-revenue superiority.

**What re-planning is worth.** The forecasts-to-decisions report found that re-planning every 14 days cut
lost revenue by 23%, mostly from refreshed floor levels. The flatness diagnosis's "Eid peak a week
late" and the decisions report's "Eid never enters the static replay" are both true. Eid timing
can only matter in a replay that re-plans. Neither finding contradicts the other.

## 2. How much each option could gain: ceilings

**A store-level calendar model (Eid, payday, holidays).** I rescaled every forecast
so that each (origin, week) store total equals the actual store total. This is an
oracle: no store-level calendar model can do better. It gives:

| WAPE | non-Eid weeks | Eid weeks |
|---|---|---|
| ML, selection | 0.4542 → 0.4510 | 0.4401 → 0.4389 |
| ML, 2025 diag. | 0.4632 → 0.4624 | 0.4677 → **0.4509** |
| current, 2025 diag. | 0.4713 → 0.4638 | 0.4625 → 0.4519 |

Even a perfect Eid model is worth about 0.017 WAPE in 2025 Eid weeks. That is roughly
0.002-0.003 averaged over the year, below anything we can detect (section 4). The
Eid work is still worth doing, as a defect repair: the peak lands in the wrong week,
`c_eid_days` fits one-off dips and runs outside its training range, and Eid 2026 is
invisible. It should be judged on peak placement and non-inferiority, not on
expecting a WAPE win.

**Level estimation.** The level miss is about 18% of flat-forecast error outside Eid,
and recent changes mostly reverse (demand patterns). An oracle that knew each drug's true
half-year mean cut matched lost revenue by 4-14% (forecasts to decisions). That is the ceiling for
any better level model. A realistic method would capture a small part of it.

**Segment rules.** These segments cover about 12% of revenue: censored 4.5%, fading
2.6%, bulk 1.9%, sporadic 1.9% and new 1.0%. Each has a clear failure. Fading drugs
are over-forecast 1.26-1.29 by both models. Censored drugs are under-forecast by ML.
Sporadic drugs have no forecast worth making. The gains are real for those drugs but
small for the store as a whole.

## 3. The options compared

Gains are best guesses from the evidence above. "Detectable?" asks whether the
gain can be told apart from noise with the data we have (section 4).

| Option | Segments | Expected gain | Risk | Effort | Detectable? | Verdict |
|---|---|---|---|---|---|---|
| **Keep global LightGBM for the mean** | all | already 0.016 WAPE ahead of baseline on selection, 0.005 ahead of current on 2025 | low | none | gap is at the MDE | **keep** |
| Equal-weight ensemble ML + SES+Eid | fast/regular/slow | −0.002 to +0.0005 WAPE; bias moves to 1.00 | low | very low | no | **drop** (keep as a bias reference only) |
| Segment-specialist router for the mean (TSB/ADIDA for intermittent, etc.) | intermittent, slow, rare | ≈0: intermittent is at 1.6x the noise floor and all flat methods tie within 0.01-0.02 | medium: more parts, hindsight labels | medium | no | **drop** for the mean; use rules for policy instead |
| Rules: fading decay / stop-reorder flag; min-stock rule for sporadic | fading 2.6%, sporadic 1.9% | removes 1.26-1.37 over-forecast; frees stock | low | low | as stock value at equal service | **do** |
| Censoring-aware targets (impute lost demand in stock-out weeks; treat shelf-out days) | censored 4.5%, slow, rare | ML bias on censored 0.91 → ~1.0 | medium: unobservable truth | medium | bias yes; lost revenue no | **do, judged on bias** |
| Pooled / Gamma-Poisson level for new and slow drugs | new 1%, slow | large for SES (bias 1.36 → ~1.05); ML already 1.06 | low | low | — | **only if current stays default** |
| Full hierarchical Bayesian NB state-space | slow, rare | small; ML already pools | high (fit time, maintenance) | high | no | **drop** |
| Bulk-order decomposition | bulk 1.9% | mean already fine (ML 0.91, baseline 0.96); spread matters | medium | medium | no | **later**, spread only |
| **Eid repair**: 2026+ dates, cap `c_eid_days` at \|d\| > 45, day-count week composition, unit-based day profile applied as a multiplier | all, 5-6 weeks a year | ≤0.017 WAPE in Eid weeks (ceiling), right peak week | low | low | peak placement yes; WAPE no | **do (defect fix)** |
| **Lead-time variance**: window factor 1 + ρ(w−1) per tier (maybe also by Eid flag) | fast/regular/slow | 4-week p95 coverage 0.90-0.93 → ~0.95; achieved service matches target | low | very low | coverage yes | **do** |
| Conformal on lead-time totals | all | similar coverage, over-shoots in quiet periods | low | low | coverage yes | fallback to the ρ-factor |
| Quantile / GAMLSS spread model | fast/regular | unknown; per-drug dispersion ranking is worth 8-21% at fixed stock (forecasts to decisions) | medium | medium | maybe | **later**, after the ρ-factor |
| Payday / week-of-month feature | all | store effect −5.8% in payday-share weeks; drug-level gain below the oracle ceiling of ~0.003 | low | low | no | one ablation at most |

## 4. Power: what the data can and cannot detect

**Forecast scores.** Across the 11 selection origins, the per-origin WAPE difference
between ML and baseline has a standard deviation of 0.018 (fast and regular) and
0.044 (slow). Adjacent windows overlap by half, so I inflate the variance (x1.4 on
the SD). That gives a minimum detectable effect at 80% power, two-sided 5%, of:

| | 11 origins | 26 origins | 52 origins |
|---|---|---|---|
| fast / regular | 0.022 | 0.014 | 0.010 |
| slow | 0.052 | 0.034 | 0.024 |

The real ML-vs-baseline gap (0.018 fast, 0.012 regular, 0.008 slow) sits at or below
these. Resampling origins as well as drugs widens the 90% interval by a factor of
1.5-2. The regular-tier interval becomes −0.022..−0.0003, nearly touching zero. The
slow-tier interval spans zero either way. The project's drug-only bootstrap
overstates how sure we are.

**Decision replay.** With a single 6-month window, lost revenue is dominated by a few
high-revenue drugs.

- Forecasts to decisions: in the static, matched replay, mean noise of σ = 0.1 or 0.25 was flagged
  in only 1 of 10 runs.
- My rolling run, unmatched: persistent per-drug noise of σ = 0.1 (about +0.01 WAPE)
  moved lost revenue by 0.87-1.07x across five seeds. That is a range of ±10% from
  noise that makes the forecast worse. σ = 0.25 (about +0.09 WAPE) moved it 0.96-1.26x
  and was flagged in 2 of 5 runs.
- At equal stock value the σ = 0.1 ratios run 0.89-1.10.

The detection limit for lost revenue is therefore roughly a WAPE change of 0.05-0.1.
Every forecast option in the table is 10-50 times smaller than that. Stock-out days
are more sensitive (flagged 5/5 at σ = 0.25), but they can move the opposite way from
lost revenue, as the forecasts-to-decisions report showed. The consequences:

1. Gate forecast changes on proper scores of lead-time totals over many origins
   (revenue-weighted pinball at 0.95, CRPS, bias), with two-way bootstrap intervals.
2. Use the rolling replay as a **non-inferiority** check, comparing curves of lost
   revenue against stock value. Do not use it as a superiority test for forecasts.
3. Use the replay as a superiority test only for policy changes, where effects are
   20%+: re-plan cadence, floor rule, segment rules that release stock.
4. Pool replay windows (H1 2024 if the stock history allows it, H2 2024, and later
   2026) and several start days, to raise the effective sample above one origin per
   window.

## 5. Recommended architecture

```
weekly units ──► global LightGBM mean (Tweedie 1.3, A2 features)
                  + repaired Eid features (dates through 2027, capped c_eid_days,
                    day-count composition of the target week)
                  + censoring-aware training targets (lost-demand imputation)
             ──► segment rules: fading → decay + stop-reorder flag
                                sporadic → min-stock rule, no forecast
                                (new → pooled start, only on the SES fallback)
             ──► per-drug NB dispersion (as now) × window factor 1 + ρ_tier (w − 1)
             ──► (s, S) re-planned every 14 days, floor levels refreshed
fallback / reference: SES + Eid (current), with the same Eid and variance repairs
```

Why this shape. The mean model is already near the floor, and it beats or ties the
SES family in every large segment: steady, intermittent, bulk, censored and new.
What remains wrong is systematic and local. Eid weeks are misplaced. Spread is too
narrow over windows of more than one week. ML runs low on censored drugs. Both
models over-forecast fading drugs. Each of these has a targeted fix that can be
checked on its own. A router or an ensemble adds moving parts in exchange for
changes this data cannot detect.

What this does **not** settle is whether ML or current should be the app's default.
On the evidence, ML is at least as good everywhere it matters and less biased in
2025. The old decision rule could not have detected that, and its replay could not
see the difference. Re-deciding belongs at the end of the roadmap, under the new
harness, with non-inferiority as the rule.

## Caveats

- The ensemble, calibration and power numbers use existing prediction files. The
  selection-period ML is the wave3 A2 config (3aafa0367da7), not the frozen
  36ad7c615b96, though they share features and loss.
- The ceiling for a store-level calendar model uses actual store totals. It bounds
  any store-wide shape model, but not drug-specific Eid responses. The demand-patterns report found
  those unrepeatable (r ≈ 0.12).
- The rolling replays perturb the current forecaster's bases on one window (H2 2024).
  They measure the harness's sensitivity, not which forecaster is better.
- Segment labels come from 8 Dec 2024 statistics, so they are hindsight for "new"
  and "fading".
- Eid 2026: Kemenag fixed 1 Syawal 1447 H at about 20-21 March 2026. Confirm the
  exact date before adding it to `forecast.EID`.
