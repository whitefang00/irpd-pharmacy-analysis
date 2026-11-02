# Why the forecasts are flat across horizons 1-8

Part of our follow-up after the ML results: a diagnosis of the existing models.
Code paths are cited against the repo's `src/`. Scripts are in
[`scripts/diagnostics/`](scripts/diagnostics/) (numbered 01-13), their outputs in
`data/followup/diagnostics/` (not committed), figures in
[`fig/diagnostics/`](fig/diagnostics/).

Evidence comes from three sources, kept apart:

- **Selection period** (preferred): wave3 predictions of the A2 feature set
  (`data/ml/wave3/predictions/3aafa0367da7_*`, LightGBM Tweedie 1.3, mask
  censoring. Its fast-tier selection WAPE 0.3301 matches A2 in
  `data/ml/wave3/ablation.csv`). Also the re-selected baseline (`baseline-g1`),
  and history from `selection.duckdb`. 11 origins, 7 Jan to 13 Oct 2024.
- **2025 eval predictions** (diagnostic): frozen ML 36ad7c615b96, the current
  forecaster (`eval-old`) and the re-selected baseline, 9 origins Jan-Apr 2025.
- **Served and rolling files** (diagnostic): `ml_forecast.csv`, `ml_rolling.csv`,
  the saved boosters in `data/warehouse/ml_forecast_models/` (loaded read-only),
  and H2 2025 actuals from `irpd.duckdb`.

No model was trained. The saved boosters reproduce `ml_forecast.csv` to its
4-decimal rounding (max abs diff 5e-5).

## Short answer

The flat paths are not a bug. They are what the model is built to produce,
and on this data a flat path is the statistically right answer for almost
every drug-week outside Eid. Inside the 8-week window, weekly variation at the
drug level is purchase noise: random customer arrivals, each buying a random
quantity. Even an oracle that knew each drug's true mean would forecast a
nearly flat line there. Where a real, shared calendar signal exists (Eid), the
ML path is not flat: it rises about 24% at store level. The served forecast
(30 Jun to 18 Aug 2025) falls in a stretch with no calendar event the model
knows about, so it is flat to within 0.3%.

There are real weaknesses, but they concern the Eid shape and one spurious
calendar effect, not flatness itself.

## 1. How the code makes the path

**Rows.** Each (drug, origin) gets eight rows, h = 1..8 (`src/irpd/ml/panel.py:90-92`,
`:166-168`). The target week is the Monday h weeks after the Sunday origin
(`src/irpd/ml/schema.py:160-162`).

**Only five inputs differ between a drug's eight rows.** These are `h`
(`schema.py:94`, `MODEL_COLUMNS`) and the four calendar features of the target
week: `c_eid_days`, `c_eid`, `c_ramadan` and `c_eid_uplift`
(`src/irpd/ml/features/calendar.py:43-52`). Everything else is computed once
per origin and copied to all eight rows:

- the demand features (`features/demand.py:190-192`, `np.repeat(..., HORIZON)`);
- the store features (`features/store.py:41-43`, `by_h=False`);
- price and drug (`features/price.py:60`, `features/drug.py:27`).

That is correct practice, since target-week sales and traffic aren't known at
the origin. But it means the path can only bend where the calendar features
change.

**The calendar features are almost constant away from Eid.**

- `c_eid` and `c_ramadan` are 0, and `c_eid_uplift` is 1, unless the target
  week is within -30..+28 days of an Eid (`calendar.py:37-40`,
  `forecast.py:142-150`).
- `c_eid_days` is the distance from the week's Thursday to the nearest Eid in
  `forecast.EID` (`calendar.py:32-34`). It changes steadily by 7 a week. The
  model can split on it, which gives tiny wiggles.

There is no week-of-year, month, payday or holiday feature. The docstrings say
the calendar group has "target week of year" (`features/__init__.py:14`,
`schema.py:9-10`), but `calendar.py` has none. Week of year lives in the
`season` add-on, which was dropped after it made forecasts worse
(`features/__init__.py:24-25`). This is a doc mismatch, not a code bug.

**Objective and seeds.** The objective is Tweedie 1.3 with default parameters
(`src/irpd/ml/models/lgbm.py:262-273`), so the model fits a log-link mean. The
prediction is the average of three seeds (`lgbm.py:242-244`). Seed averaging
does not flatten anything: the median per-path CV on the 2025 eval is 0.056 to
0.062 for single seeds and 0.058 for the ensemble.

**Serving.**

- `serve.train` fits on `trainable(p, origin)` and predicts the origin's eight
  rows (`src/irpd/ml/serve.py:70-83`).
- `forecast.forecast(..., forecaster="ml")` repeats week 8 for any later week
  (`src/irpd/forecast.py:344-345`). Past h = 8 the path is flat by
  construction, even if Eid falls in weeks 9 and up.
- The rolling file stores only 4-week totals (`src/irpd/ml/rolling.py:75-78`).

**The current forecaster.** SES+Eid gives one level per drug, multiplied by the
target week's Eid factor (`forecast.py:311-313`). Outside an Eid window that
factor is 1, so the path is exactly flat. Seasonal naive, used for rare drugs,
is the only method whose path moves (median path CV 1.18), and that movement
is noise copied from last year.

**What the model uses** (saved boosters, `05_served_model_decomp.py`). Mean
|SHAP| on the log scale is 0.88 for `d_ewm_15`, 0.56 for `d_mean_52`, 0.12 for
`d_mean_26` and 0.10 for `p_price`. Of the h-varying inputs, only
`c_eid_days` / `c_eid` / `c_eid_uplift` matter, and only near Eid. `h` itself
has gain share 0.0001 (`lab/ml-results/full/importance_gain.csv`).

Counterfactuals on the served model:

| Case | Store-level path | Median per-drug path CV |
|---|---|---|
| Served origin, 29 Jun 2025 | 26,861 → 26,630 (−0.9% over 8 weeks) | 0.26% |
| Same, varying only `h` | about half of that | |
| Same, varying only the calendar | the other half | |
| Origin 2 Mar 2025 (Eid ahead), full | 29.5k → 36.6k → 29.7k (+24%) | 7.6% |
| Same, varying only the calendar | 29.5k → 36.7k → 29.8k | 7.5% |
| Same, varying only `h` | flat | 0.13% |

So `h` adds no shape. The calendar features are the only source of shape, and
they work when there is something to encode.

## 2. How flat, against what

**Served file.** Per-drug path CV has median 0.26% and 99th percentile 5.8%.
The h8/h1 ratio is 0.98-1.01 for 80% of drugs. The example drug
OBT2106180001, which moves 2.4536 → 2.4636 → 2.4516, is typical.

**Selection origins, non-Eid windows** (`06_selection_period.py`,
[`fig/diagnostics/path_cv_vs_actual_cv_selection.png`](fig/diagnostics/path_cv_vs_actual_cv_selection.png)):

| Tier | ML path CV (median) | Actual CV over the 8 weeks | Poisson CV at that mean |
|---|---|---|---|
| fast | 0.9% | 43% | 16% |
| regular | 0.9% | 84% | 62% |
| slow | 0.9% | 151% | 127% |

In Eid windows the ML path CV rises to 4-10%. The baseline is exactly 0.

**Store level, 2025** (`03_eid_and_aggregate.py`,
[`fig/diagnostics/aggregate_paths_eid2025.png`](fig/diagnostics/aggregate_paths_eid2025.png)). Summed over drugs, the ML
path CV is 5-11% against 6-15% for actuals. The ML path tracks the Eid rise
and the post-Eid hangover in May, which the current forecaster misses: at the
27 Apr origin, ML/actual averages about 1.02 over the 8 weeks, against about
1.06 for the current forecaster and about 1.15 for the baseline. On the selection origins, the
within-origin store-level ML deviations correlate 0.48 with actual deviations.

## 3. Is flatness justified? Yes, outside Eid

**Oracle test** (`10_compound_poisson.py`, selection data only,
[`fig/diagnostics/oracle_compound_poisson_ratio.png`](fig/diagnostics/oracle_compound_poisson_ratio.png)). If customers arrive
at a constant rate and each buys a random quantity, the weekly units have
index of dispersion E[s²]/E[s], where s is units per sale line. For every
drug I compared that with the observed within-block dispersion of weekly
units, using non-overlapping 8-week blocks of 2024 with stock-out weeks
dropped.

| Tier | Drugs | Observed / compound-Poisson, all blocks (median) | Same, Eid blocks excluded | Weekly lines dispersion, non-Eid |
|---|---|---|---|---|
| fast | 232 | 1.14 | 0.97 | 1.07 |
| regular | 612 | 1.03 | 0.98 | 1.03 |
| slow | 767 | 0.97 | 0.94 | 0.97 |
| rare | 147 | 0.93 | 0.91 | 0.91 |

Outside Eid, the median drug shows no variance beyond constant-rate noise.
Customer counts per week are Poisson (dispersion about 1), and the extra
spread in units comes from purchase sizes. A fast drug's median sale line is
about 10 units, which is why fast drugs look so "lumpy" (window dispersion
about 8). An oracle that knew the true weekly mean would draw a nearly flat
line. The weighted "signal share" of 12-21% comes from a minority of large
drugs, and is inflated by clipping noisy ratios at zero.

**Two more checks agree.**

- Within-window deviations are almost white noise
  (`02_window_dispersion.py`). Lag-1 autocorrelation is −0.093 against a
  shuffle null of −0.143 for fast drugs, a small real persistence. For
  regular and slow it is −0.136 / −0.145 against −0.148 / −0.151.
- Realized within-window variance is below the forecaster's own NB variance:
  the median ratio is 0.73-0.89 in every tier, on both periods.

**h-dependence** (`13_h_dependence.py`). On the selection data, the realized
response of week-h demand to a recent shift (m4/m52) does not decay with h.
For fast drugs the slope is about 0.25 at h1 and 0.19-0.29 later. Regular is
0.25-0.37 and slow 0.09-0.26, with no trend. Error barely grows with
horizon: ML WAPE is 0.451 at h1 and 0.455-0.479 at h2-h8 (selection), 0.464
at h1 and 0.481 at h8 (2025). Without real decay or momentum, an h-shaped
path would only add error.

**Level response is not sluggish** (`06_selection_period.py`, level-response
table). A future 8-week mean loads mostly on the 52-week mean (weight
0.47-0.67) and little on the last 4 weeks (0.10-0.27), so recent shifts
mostly revert. The ML follows a recent shift with slope 0.20-0.35 against a
realized 0.17-0.30, which is about right. The mean8/SES baseline overreacts,
at 0.46-0.59. The flat level is not lagging behind real shifts.

## 4. Weaknesses found (not flatness bugs)

1. **The Eid peak lands a week late in 2025.** Actual units summed over fast,
   regular and slow drugs peaked in the week of 24 Mar (37.8k), the seven days
   before Eid. Both forecasters put the peak in the week of 31 Mar. At the
   2 Mar origin, ML gave 32.3k then 35.1k and the current forecaster gave
   33.7k then 34.2k, against actuals of 37.8k then 33.1k. The cause is
   `12_eid_daily_profile.py`: days −7..−1 and +2..+5 are both high, Eid day is
   closed, and how they fall into Monday weeks changes every year. The peak
   week was days +2..+8 in 2023, −2..+4 in 2024 and −7..−1 in 2025.
   `c_eid_days` is a single Thursday distance (`calendar.py:32-34`). Two
   training Eids can't teach a tree the right weekly split for a new offset.
   `c_eid_uplift` (`forecast.py:121-139`) is built on store lines, shrunk by
   0.6 (`forecast.py:55`), and also rates the Eid week (1.21) above the week
   before (1.19).
2. **`c_eid_days` learns spurious dips far from Eid.** The store had a 3-week
   dip in Jan 2024 (relative level 0.87-0.91, `store_weekly_calendar.csv`),
   at 69-83 days before Eid. No such dip appears at the same Eid distance in
   2023. The model learned it anyway. At the 5 Jan 2025 origin, ML put weeks 1
   to 3 at 24.4k, 23.0k and 23.0k (h1-h2 14-16% under the actual), while the
   current forecaster's flat 26.8k was closer. Far from Eid, `c_eid_days`
   acts as a "position in the Hijri year" proxy that is fitted on one year.
   It also produces the small wiggles in the served path.
3. **`forecast.EID` stops at 2025** (`forecast.py:45`). After about 24 Sep
   2025, `c_eid_days` keeps falling below the training range of about ±177,
   so late-2025 weeks are treated like Oct 2023 rather than Nov-Dec. Ramadan
   and Eid 2026 (about 20 Mar 2026) would be invisible to both forecasters.
   This doesn't matter inside the data, which ends 31 Dec 2025. It would fail
   silently in any later use.
4. **Weeks past 8 repeat week 8** (`forecast.py:344-345`). This is a design
   choice that is fine for a 14-day cycle, but it would hide an Eid that
   falls in weeks 9-12.
5. **H2 2025 store shock** (`09_rolling.py`,
   [`fig/diagnostics/rolling_store_level_h2_2025.png`](fig/diagnostics/rolling_store_level_h2_2025.png), diagnostic). Over 23
   rolling origins, ML 4-week totals follow the store level well (median
   ML/actual 1.008). The current forecaster runs 5-8% high after the July
   store change. Neither forecaster saw the mid-Sep to mid-Oct dip coming:
   actual was 98-101k, ML 113-115k. No feature encodes it. The dip had
   reverted by late October, so reacting slowly cost little. Drug-level
   past/future correlations in this file are dominated by a demeaning
   artifact (iid null −0.18, `09b_rolling_null.py`), so I draw no drug-level
   lag conclusion from them.

## 5. Evaluation methodology

`08_eval_review.py`:

- **WAPE pooled across tiers is a fast-drug number.** Fast drugs are 69-73%
  of scored units and about half the absolute error. Within a tier, the top
  10% of drugs by volume hold 40% of units (fast), 50% (regular), 59-64%
  (slow) and 75-85% (rare). The per-tier reporting in
  `lab/ml-results/eval/scores.csv` is the right call.
- **ML's tier gains are concentrated.** On the 2025 eval in the slow tier, ML
  beats the current forecaster overall (1.081 against 1.088). The whole gain,
  about 3x the net, comes from the top-10% drugs. The other 90% of slow drugs
  are worse (1.362 against 1.326), and ML wins on only 51% of slow drugs. The
  selection period shows the same pattern (slow rest 1.348 against 1.329;
  rare 45% of drugs better). On fast drugs against the current forecaster,
  ML's gain is in the smaller fast drugs, and the top-10% are slightly worse.
- **Dropping stock-out weeks.** For fast drugs it removes 1.8-2.9% of rows,
  which is harmless. For slow it removes 12-14% of rows (9% of units), and
  for rare 26% of rows (13% of units). In those dropped weeks, all
  forecasters forecast 1.4-4.9x the recorded sales floor. Between 14% and 30%
  of dropped weeks still have the forecast below the floor. Scoring dropped
  weeks at the floor (a lower bound on demand) raises WAPE by 0.005-0.02 for
  fast and regular, 0.06-0.13 for slow, and 0.12-0.4 for rare. It doesn't
  change the ML-against-baseline order, with one exception: rare on
  selection, where ML goes from 1.499 against 1.568 to 1.894 against 1.823.
  Rare-tier conclusions are fragile to the censoring rule.
- **Same rows for every method.** The n's match in `scores.csv`, and
  `stats._paired` joins on shared rows with a known actual
  (`src/irpd/ml/stats.py:25-37`), so the comparison is like for like.
- **No leakage found.** Features cut at the origin (`demand.py:183`),
  prices, tiers and categories read `connection_at(origin)`, and `fit`
  asserts the purge (`lgbm.py:170-176`).
- **The bootstrap resamples drugs only** (`stats.py:8-13`). With overlapping
  8-week windows (step 2 or 4 weeks) and shared store shocks, the intervals
  are too narrow along the time axis. The module says so. "Better in every
  tier" rests on 5 to 9 correlated origins, all from Jan-Apr 2025, which is
  mostly Eid season.

## 6. Missing signals worth testing (ranked)

1. **Day-resolution Eid composition of the target week.** Count of target-week
   days in −7..−1 (pre-Eid shopping), closed days (Eid day, plus cuti bersama
   if listed), and days in +1..+7. Optionally a unit-based (not line-based),
   unshrunk day-factor sum. This is the one place real, shared within-window
   signal exists and both forecasters misplace it.
2. **Restrict `c_eid_days` to near Eid** (for example, NaN or a cap when
   |days| > 45), to stop it fitting one-off store dips such as Jan 2024. Add
   2026 to `forecast.EID`.
3. **Payday / week-of-month and national holidays** at store level.
   Exploratory: on 2024 non-Eid weeks, payday-share weeks ran −5.8% (HAC
   p = 0.003, n = 80, R² 0.11). Holiday days were not significant. This is a
   small effect at best. Worth one ablation, not a campaign.

Not worth chasing:

- Drug-specific yearly seasonality: the 2023 against 2024 month profiles,
  after removing the store pattern, have median correlation −0.04 against a
  null of −0.01.
- h-specific dynamics, momentum, or more recent-lag weight: realized slopes
  don't decay with h, and recent shifts revert.
- Anything that adds week-to-week shape at the drug level: it would be
  fitting purchase noise.
