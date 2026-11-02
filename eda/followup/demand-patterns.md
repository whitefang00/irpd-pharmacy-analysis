# Demand segmentation and forecastable signal

Part of our follow-up after the ML results: what the demand actually looks like,
and how much of it a forecast can hope to capture. Everything that could inform a
model choice uses the selection file (`selection.duckdb` through `split.selection_con()`, every table as of
8 Dec 2024). Anything from 2025 is marked **diagnostic**. Scripts and outputs
(outputs land in `data/followup/demand/`, which is not committed):

- [`scripts/demand/drug_stats.py`](scripts/demand/drug_stats.py) → printed tables, `drug_stats.parquet`, `active_stats.parquet`
- [`scripts/demand/signal_decomp.py`](scripts/demand/signal_decomp.py) → printed tables, `insample_decomp_by_tier.csv`, `oos_decomp_by_tier.csv`
- [`scripts/demand/segment.py`](scripts/demand/segment.py) → printed tables, `segments.csv`, `segments_summary.csv`, `segment_flat_wape.csv`
- [`scripts/demand/representative.py`](scripts/demand/representative.py) → `representative_stats.csv`, [`fig/demand/representative_drugs.png`](fig/demand/representative_drugs.png)

Each runs in a few seconds from the repo root, in this order
(`uv run python eda/followup/scripts/demand/<script>.py`).
Scope: 2,295 drugs that sold in the 52 weeks to 8 Dec 2024, Rp 3,945M revenue in
those weeks. Censored weeks are those `history.load` marks (a total stock-out day,
or 4+ shelf-out days), and they are masked throughout.

## The short answer

Flat forecasts are statistically right for this data outside Eid. Over the 11
selection origins, the part of a flat forecast's error that comes from demand
changing *within* the 8-week window is −0.7% of the error once noise is taken out:
nothing. About 82% of the squared error is irreducible sale-by-sale noise and 18% is
a level miss (the 8-week mean differs from the forecast level). The level miss is
the only real target, and it does not come from trends: recent changes reverse
rather than continue. In Eid windows there is real within-window shape (21% of
error), but the store-wide Eid factor captures little of it per drug, and per-drug
responses don't repeat between years.

The key correction to the "Poisson noise floor" framing: units come in packs.
A fast drug sold by the strip has sale sizes whose E[q²]/E[q] is about 11, so its
irreducible weekly variance is about 11× its mean, not 1×. Measured against that
compound-Poisson floor, the median drug is barely over-dispersed at all (1.23×
fast, 1.07× regular, 1.00× slow).

## 1. Syntetos-Boylan class × tier

Same counts as `lab/ml-eda-notes.md` §1 (as expected; same data). New here: units
shares. Drugs active in the last 52 weeks, revenue and units in those 52 weeks.

| tier | smooth | erratic | intermittent | lumpy | too few | all |
|---|---|---|---|---|---|---|
| fast: drugs | 217 | 14 | 2 | 2 | 0 | 235 |
| regular | 263 | 107 | 172 | 81 | 0 | 623 |
| slow | 2 | 0 | 704 | 160 | 0 | 866 |
| rare | 7 | 1 | 379 | 31 | 153 | 571 |
| **revenue %** | 61.6 | 6.4 | 26.7 | 4.9 | 0.3 | 100 |
| **units %** | 76.6 | 10.7 | 7.8 | 4.9 | 0.1 | 100 |

Revenue by tier: fast 43.1%, regular 35.4%, slow 18.3%, rare 3.2%.

## 2. Behaviour beyond the tiers

Per-tier spread, p10 / median / p90 over drugs:

| | fast | regular | slow | rare |
|---|---|---|---|---|
| units/week | 5.3 / 45 / 229 | 1.2 / 3.0 / 33 | 0.21 / 0.64 / 6.8 | 0.00 / 0.15 / 1.5 |
| var/mean (raw) | 1.5 / 12.4 / 40 | 0.96 / 2.2 / 27 | 0.86 / 1.3 / 16 | 0.87 / 1.0 / 10 |
| sale-size factor E[q²]/E[q] | 1.15 / 11.4 / 28.6 | 1.01 / 2.05 / 20.1 | 1.00 / 1.19 / 14.3 | 1.00 / 1.00 / 10.0 |
| var/mean ÷ size factor | 0.75 / **1.23** / 2.16 | 0.73 / **1.07** / 1.72 | 0.70 / **1.00** / 1.52 | 0.70 / **0.97** / 1.26 |
| zero weeks | 0 / 0 / 6% | 6 / 21 / 42% | 41 / 65 / 83% | 67 / 90 / 100% |
| censored weeks | 0 / 0 / 6% | 0 / 2 / 22% | 0 / 8 / 50% | 0 / 16 / 92% |

**Volatility is mostly pack size.** Raw var/mean of 12 on fast drugs looks heavily
over-dispersed, but dividing by the sale-size factor brings it to about 1. Weekly
demand is close to compound Poisson around a slowly moving level. This matters for
the NB spread: a dispersion estimated from units would be right; one assuming
Poisson on units would be far too narrow for strip-sold drugs.

**Autocorrelation** (Eid weeks and censored weeks masked). Share of drugs whose
lag-1 ACF beats 2/√n, and mean ACF:

| tier | lag-1 significant | acf1 | acf2 | acf3 | acf4 | acf52 | lag-52 significant |
|---|---|---|---|---|---|---|---|
| fast | 41% | 0.18 | 0.15 | 0.14 | 0.13 | 0.03 | 4.7% |
| regular | 25% | 0.13 | 0.11 | 0.09 | 0.09 | 0.02 | 4.0% |
| slow | 14% | 0.07 | 0.06 | 0.06 | 0.04 | 0.00 | 3.3% |
| rare | 5% | 0.03 | 0.02 | 0.01 | 0.02 | 0.00 | 1.4% |

Lag 52 is significant for no more drugs than chance (5%), so no yearly pattern
beyond Eid, as the EDA found. The flat ACF across lags 1-4 is a drifting level,
not short-term dynamics.

**Level shifts.** Best single break in mean (on the variance-stabilised series,
drugs with 40+ observed weeks):

| tier | drugs | sup-F > 9 | big break (F > 15, ratio > 1.5× or < 0.67×) | trend \|t\| > 3 | median after/before |
|---|---|---|---|---|---|
| fast | 227 | 65% | 33% | 40% | 0.72 |
| regular | 582 | 44% | 23% | 21% | 0.64 |
| slow | 752 | 31% | 14% | 12% | 0.51 |
| rare | 304 | 15% | 5% | 2% | 0.48 |

Breaks are common, mostly downward, and sit around a third of the way into the
history (median position 0.37, so mid-to-late 2023). That matches the store-wide
traffic fall in FINDINGS. Level shifts are real but they are *past* events; §3 shows
they don't carry forward as trends.

**Lifecycle.** 186 drugs first sold in the last 26 weeks (1.0% of revenue). 374
drugs sold in the year but not in the last 13 weeks (2.6% of revenue). 67 drugs
with 1+ unit a week halved their level between the two halves of the last year.

**Eid heterogeneity.** For 456 fast and regular drugs with 3+ units a week, the
per-drug ratio of Eid weeks 0..+2 to weeks −9..−3 spread from 0.62 to 2.71 (p10-p90)
in 2023 and 0.58 to 2.12 in 2024. The year-to-year correlation of log ratios is
0.12. Consistent with the EDA (0.04 / 0.01 on its definition): the spread is noise.

**Bulk lines.** A "bulk week" here has units above 3× the drug's median sale week,
with one sale document holding at least half of them. 16% of fast, 30% of regular
and 25% of slow drugs have at least one. For 74 drugs these weeks carry 20%+ of all
units (1.9% of revenue, 2.7% of units). Fast drugs are not affected (0 drugs over 20%).

**Censoring.** Share of drugs with more than 20% / 40% of weeks censored: fast
4.7% / 2.1%, regular 10.6% / 4.2%, slow 28.3% / 14.9%, rare 46.8% / 32.0%.

## 3. How much signal is there beyond a flat level?

**In-sample, per drug.** Within-drug variance of weekly units, history from first
sale to 8 Dec 2024 (drugs with 40+ observed weeks). Components are added in order:
the store-wide Eid index (no fitted coefficient), a linear trend, then a
leave-one-out 13-week centred mean for extra drift. The noise floor is compound
Poisson (level × size factor). Pooled over drugs (volume-weighted):

| tier | drugs | Eid | trend | extra drift | noise floor | left over |
|---|---|---|---|---|---|---|
| fast | 228 | 4.7% | 10.8% | −2.3% | 67.2% | 14.0% |
| regular | 592 | 0.0% | 5.6% | 1.1% | 80.9% | 5.6% |
| slow | 805 | 0.2% | 9.5% | −2.1% | 56.6% | 31.2% |
| rare | 391 | 0.3% | 2.4% | −0.8% | 88.4% | 2.3% |
| all | 2,016 | 3.3% | 9.5% | −1.5% | 69.7% | 13.2% |

For the median drug the noise floor is 73% (fast), 87% (regular), 94% (slow) and
93% (rare) of its variance; trend 7%, 3%, 2%, 3%; Eid 3%, 1%, 0%, 0%. A plain
Poisson floor would claim only about 2% is noise: that is the pack-size error.
The trend share is in-sample hindsight (it fits a line through the whole history,
mostly the 2023 drop). The slow tier's 31% "left over" is driven by a few
high-volume slow drugs with bulk lines and breaks; the median slow drug has
almost nothing left over. Across drugs, 81% of the pooled variance of weekly units
is simply which drug it is (its level).

**Out of sample, the test that matters.** A flat SES (α 0.15) forecast at each of
the 11 selection origins, scored on the next 8 weeks, drugs sold in the 13 weeks
before (18,441 drug-origins). Its squared error splits exactly into a noise floor
(compound Poisson at the realised 8-week mean), a level miss beyond what noise in
that mean explains, and within-window shape beyond noise. Shape is the most *any*
non-flat path over h1..h8 could ever win, even with perfect knowledge.

| | drug-origins | noise floor | level miss | within-8-week shape | MSE of SES×Eid index / SES |
|---|---|---|---|---|---|
| fast, non-Eid | 1,861 | 79.7% | 18.5% | 1.9% | 1.000 |
| fast, Eid window | 698 | 58.0% | 14.2% | 27.8% | 0.979 |
| regular, non-Eid | 4,697 | 93.7% | 15.2% | −8.8% | 1.000 |
| regular, Eid | 1,786 | 98.0% | 10.3% | −8.3% | 1.009 |
| slow, non-Eid | 5,101 | 72.2% | 24.5% | 3.3% | 1.000 |
| slow, Eid | 1,777 | 73.8% | 10.2% | 16.0% | 0.984 |
| rare, non-Eid | 1,879 | 35.9% | 56.0% | 8.1% | 1.000 |
| **all, non-Eid** | 13,538 | **82.3%** | **18.4%** | **−0.7%** | 1.000 |
| **all, Eid windows** | 4,903 | 64.6% | 14.2% | **21.1%** | 0.985 |

(Negative shares mean the realised variance was below the floor estimate; the
floor slightly overstates noise for some regular drugs. Treat ±5% as zero.)

Outside Eid there is no within-window shape to forecast. A flat path is the
correct expected value, so flat ML output (e.g. OBT2106180001: 2.4536, 2.4584 ...
from 30 Jun 2025, no Eid in range) is not a bug. In Eid windows, about a fifth of
the error is shape, but the store-wide Eid index only removes 1.5% of the error,
because per-drug Eid responses are mostly unrepeatable noise (§2).
*Diagnostic:* on the 2025 eval origins, which include the 2025 Eid, the ML paths for
the representative drugs do move: their h1-h8 range is a median 15-26% of the mean.

**Is the level miss predictable?** Not by trend. In non-Eid windows, the change
from the previous 13 weeks to the last 13 weeks predicts the *opposite* change over
the next 8 weeks (correlation −0.34 fast, −0.27 regular, −0.30 slow, −0.20 rare;
−0.23 to −0.07 against the SES residual). Part of that is regression to the mean
from noise in the 13-week mean, but there is no sign of momentum. The best linear
weights for the next 8-week mean are:

| tier | weight on last 13 weeks | weight on previous 13 | sum |
|---|---|---|---|
| fast | 0.59 | 0.33 | 0.92 |
| regular | 0.71 | 0.27 | 0.98 |
| slow | 0.69 | 0.33 | 1.01 |
| rare | 0.91 | 0.19 | 1.10 |

A trend would show as weight above 1 on recent and negative on older weeks. Instead
it is a long-memory average, roughly what SES at α 0.1-0.15 does. The fast sum of
0.92 is the store-wide decline. So the 18% level miss is mostly a random-walk drift
in each drug's level that nobody could know in advance; better level estimation
can shave some of it, not most.

Rough budget, for a fast or regular drug outside Eid: of the flat forecast's error,
~80-90% is irreducible, ~15-18% is level drift (partly reducible), ~0% is shape.
In Eid windows add ~20% of shape, of which the store factor captures a small part.

## 4. A segmentation that changes the forecast

Mutually exclusive, assigned in priority order from statistics to 8 Dec 2024:

1. **new**: first sale in the last 26 weeks.
2. **fading**: no sale in the last 13 weeks.
3. **censored**: 30%+ of weeks censored since stock was known.
4. **bulk**: weeks driven by one large document carry 20%+ of units.
5. **steady-stable / steady-shifted**: smooth or erratic, 2+ units a week; shifted if it has a big break (sup-F > 15 and level ratio beyond 1.5×).
6. **sporadic**: rare tier, "too few", or 85%+ zero weeks.
7. **intermittent**: everything else.

| segment | drugs | revenue % | units % | median units/wk | median zero weeks | median censored | median size factor | median acf1 |
|---|---|---|---|---|---|---|---|---|
| steady-stable | 362 | 46.9 | 64.5 | 13.8 | 4% | 0% | 7.4 | 0.07 |
| steady-shifted | 125 | 14.6 | 19.7 | 19.0 | 4% | 0% | 8.5 | 0.27 |
| intermittent | 823 | 26.7 | 8.1 | 0.98 | 53% | 3% | 1.16 | 0.06 |
| censored | 206 | 4.5 | 2.4 | 0.60 | 67% | 45% | 1.17 | 0.08 |
| fading | 340 | 2.6 | 1.6 | 0.20 | 90% | 37% | 1.18 | 0.07 |
| bulk | 74 | 1.9 | 2.7 | 2.9 | 56% | 0% | 7.6 | −0.01 |
| sporadic | 179 | 1.9 | 0.3 | 0.14 | 89% | 8% | 1.00 | −0.03 |
| new | 186 | 1.0 | 0.6 | 0.52 | 67% | 0% | 1.00 | −0.09 |

Steady drugs are fast (221) or regular (266); intermittent are regular (256) and
slow (564). The segments cut across tiers for the small groups: censored, fading,
bulk and new each draw from regular, slow and rare.

**Exploratory check: simple flat methods per segment** (11 selection origins,
WAPE; labels use 8 Dec 2024 statistics, so "new" and "fading" carry hindsight):

| segment | SES .05 | SES .15 | SES .30 | mean 13 | mean 52 | Croston SBA | TSB | median 13 | MSE/floor (SES .15) | bias (SES .15) |
|---|---|---|---|---|---|---|---|---|---|---|
| steady-stable | 0.377 | 0.377 | 0.388 | 0.381 | 0.382 | 0.370 | 0.376 | 0.375 | 1.30 | 1.02 |
| steady-shifted | 0.437 | 0.414 | 0.421 | 0.423 | 0.472 | 0.406 | 0.414 | 0.420 | 1.60 | 1.05 |
| intermittent | 1.004 | 0.967 | 0.981 | 0.979 | 1.017 | 0.976 | 0.967 | 0.901 | 1.61 | 1.04 |
| censored | 1.018 | 0.942 | 0.942 | 0.997 | 0.998 | 0.980 | 0.949 | 0.941 | 2.45 | 1.00 |
| bulk | 0.965 | 0.945 | 0.961 | 0.954 | 0.981 | 0.938 | 0.949 | 0.832 | 0.77 | 0.96 |
| fading | 1.453 | 1.150 | 1.080 | 1.262 | 1.318 | 1.420 | 1.172 | 1.000 | 2.41 | 1.31 |
| new | 1.668 | 1.476 | 1.332 | 1.506 | 1.503 | 1.730 | 1.472 | 1.289 | 6.92 | 1.75 |
| sporadic | 2.373 | 1.937 | 1.881 | 1.901 | 1.948 | 2.373 | 1.969 | 1.055 | 1.49 | 1.29 |

Median forecasts win WAPE on intermittent, bulk and sporadic drugs only because
they under-forecast (bias 0.07-0.71); they are useless for stock policy.

Young drugs measured at each origin, no hindsight (first sale under 26 weeks
before the origin, 1,586 drug-origins): SES .15 WAPE 1.04 with bias **1.27**;
SES .50 WAPE 0.92, bias 1.10. Under 13 weeks old: SES .15 bias 1.41. Older drugs:
SES .15 WAPE 0.46, bias 1.02. SES starts at the first sale, which is by definition
non-zero, and launch weeks run high, so new drugs are over-forecast by a quarter
to two fifths.

What each segment should change:

- **steady-stable** (47% of revenue). Flat level + store Eid factor, as now. All
  flat methods tie within 0.02 WAPE; MSE is 1.3× the noise floor. Little left to win
  on the point forecast. Any spread model must reflect the size factor (strip sales): a per-drug NB fitted to units does; a Poisson assumption would not.
- **steady-shifted** (15%). Same flat form, but don't use long windows (mean 52 loses
  0.06 WAPE) and don't use slower smoothing. α 0.15 is about right; a changepoint
  reset after a confirmed break is the one level idea worth testing. Highest lag-1
  ACF (0.27), so lag features help here most.
- **intermittent** (27%). Rate forecast (SES/TSB, they tie); the point forecast is
  ~1 unit a week and its error is noise (1.6× floor). Effort belongs in the
  reorder quantile, not the mean.
- **censored** (4.5%). Error 2.5× the floor even with censored targets dropped: the
  level is learned from a few observed weeks. Needs censoring-aware level
  estimation (lost-demand imputation) and service-level attention.
- **bulk** (1.9%). Split base retail demand from occasional large orders; cap spikes
  in the level; widen spread (size factor ~7.6). The mean is fine (bias 0.96).
- **new** (1.0%) and **fading** (2.6%). Lifecycle. New drugs need a decaying or
  pooled start rather than SES from the first sale (bias 1.27-1.41 at origin). Fading
  drugs need decay toward zero and a "stop reordering" flag.
- **sporadic** (1.9%). No forecast worth making; a min-stock rule.

Segments that change the method cover about 12% of revenue (censored, bulk, new,
fading, sporadic). The 88% in steady and intermittent drugs is already near the
noise floor with a flat level.

## 5. Representative drugs

Stats to 8 Dec 2024 (selection), with 2025 diagnostics from the existing
prediction files. Plot: [`fig/demand/representative_drugs.png`](fig/demand/representative_drugs.png) (weekly
units, censored weeks red, partial shelf-out weeks yellow, Eid dashed, ML and
current eval paths, ML served forecast, rolling forecasts as weekly averages).

| | SW2106000809 | SW2106000902 | OBT2108210081 | SW2106001787 | OBT2306190002 |
|---|---|---|---|---|---|
| name | Teamic 500mg | Zentra tab | Berotec 10mg | Atorvastatin 10mg DM | SGM Eksplor gain optigrow 400gr |
| segment | steady-stable | steady-shifted | intermittent | bulk | censored |
| tier / SB class | fast / smooth | fast / smooth | regular / intermittent | regular / lumpy | regular / intermittent |
| revenue, last 52 wk | Rp 48.7M | Rp 15.8M | Rp 15.0M | Rp 3.8M | Rp 14.5M |
| units/week | 486 | 169 | 1.56 | 26.4 | 3.38 |
| zero weeks | 0% | 0% | 19% | 21% | 12% |
| var/mean ; size factor | 23.3 ; 12.5 | 27.7 ; 12.0 | 1.09 ; 1.01 | 36.1 ; 35.4 | 1.95 ; 1.89 |
| censored weeks | 0% | 0% | 0% | 19% | 37% |
| bulk unit share | 0 | 0 | 0 | 33% | 0 |
| acf1 / acf4 / acf52 | 0.30 / 0.18 / 0.11 | 0.74 / 0.61 / −0.25 | −0.21 / −0.09 / 0.04 | 0.10 / 0.01 / −0.14 | 0.20 / 0.37 / – |
| best break: ratio, position | 0.86, 0.30 | 0.50, 0.68 | 1.38, 0.54 | 3.08, 0.53 | 2.63, 0.35 |
| Eid ratio 2023 ; 2024 | 1.41 ; 1.52 | 2.67 ; 1.04 | 1.69 ; 1.15 | 4.11 ; 0.67 | – |
| in-sample: noise floor / trend / Eid share | 49% / 10% / 26% | 22% / 28% / 11% | 94% / 1% / 0% | 108% / 9% / 0% | 86% / 7% / 2% |
| SES .15 MSE / noise floor (selection origins) | 2.39 | 1.95 | 1.09 | 1.02 | 1.12 |
| *diag. 2025 WAPE: ML / current / g1 baseline* | 0.18 / 0.19 / 0.21 | 0.37 / 0.38 / 0.44 | 0.77 / 0.76 / 0.80 | 0.92 / 0.89 / 0.91 | 0.76 / 0.81 / 0.83 |

Why each:

- **Teamic 500mg**: the top-revenue clean steady drug. Strip sales (size factor 12.5)
  explain most of its "volatility"; the clearest store-following Eid response of the
  five, the one place a store Eid factor earns its keep.
- **Zentra tab**: the clearest drifting level (acf1 0.74, halved in 2024, then, in the
  diagnostic 2025 data, rose again). Its error is mostly level, the one kind of error
  a better level could reduce.
- **Berotec 10mg**: high-revenue intermittent drug at ~1.5 units a week. Error is at
  the noise floor (1.09×); nothing to forecast beyond the rate.
- **Atorvastatin 10mg DM**: bulk-driven, a third of units from single large orders,
  plus censoring. The weekly series is mostly a few big documents; spread matters
  more than the mean.
- **SGM Eksplor gain optigrow 400gr**: 37% of weeks censored by stock-outs, and
  shelf-outs on top. Its observed level is built from the gaps between stock-outs.

## Caveats

- The noise floor uses each drug's sale-document sizes, including bulk orders, so for
  bulk drugs it counts bulk orders as noise (hence MSE/floor < 1 there). For a few
  regular drugs the floor overstates noise (negative shape shares).
- Segment labels use statistics to 8 Dec 2024; slicing 2024 origins by them is
  slightly hindsighted, badly so for "new" and "fading". The at-origin young-drug
  check avoids that.
- The method table is exploratory: eight simple methods, no tuning, no selection run.
- 2025 numbers are diagnostic only and come from files already inspected.
