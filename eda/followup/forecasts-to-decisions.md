# From forecast to reorder decision, and whether the evaluation rewards the right thing

Part of our follow-up after the ML results: how forecasts turn into reorder decisions.
Scripts are in [`scripts/inventory/`](scripts/inventory/), their outputs in
`data/followup/inventory/` (not committed), the figure in [`fig/inventory/`](fig/inventory/). All replay experiments use the project's own engine (`contest.load_window`,
`decide.baseline_basis`, `contest.to_policy`, `replay.simulate`, `replay.match_stock_value`) on
in-memory, read-only connections. Nothing was trained.

Labels: the main replay window here is **H2 2024** (policy day 30 Jun 2024), which sits almost
entirely before the 8 Dec 2024 selection boundary. **H1 2025** is used as a second window and is
**diagnostic**. Calibration uses the 2024 selection-period predictions (`data/ml/wave3`) as the
clean set and the 2025 evaluation predictions as **diagnostic**. All of it is exploratory.

## Short answer

The current evaluation does not reward the behaviour the business needs, for three reasons.

1. **The replay sets one policy and holds it for six months.** `decide.compare` calls
   `contest.to_policy` once on the policy day; (s, S) never change. Only the first 4 weeks of the
   forecast enter the policy (lead up to 14 days plus the 14-day cycle), and they are then used as
   the drug's rate for 26 weeks. So the replay tests "how well does the policy-day forecast of
   weeks 1-4 stand in for the drug's half-year rate", not short-horizon accuracy. Fast reaction to
   level shifts and Eid uplift cannot be rewarded at all. A perfect forecast of the first 28 days
   makes the replay much *worse* (below).
2. **Matching to the pharmacist's stock value removes most of what a forecast does.** It
   multiplies every s and S by one store-wide factor (1.7-2.0 in every run). That cancels any
   overall bias, and it turns the nominal 0.95 service level into an effective 0.993-0.998 at
   lead time. What is left to score is how the forecast *ranks* drugs relative to each other.
3. **The lost-revenue test has very little power.** About a third to two fifths of lost revenue
   is shelf-out loss driven by the floor rule, which barely depends on the forecast. The matching
   tolerance alone moves lost revenue by about ±0.5%. A forecast made 30% worse by WAPE (random
   noise) was flagged as worse by the bootstrap in only 1 of 10 tries.

Calibration: the NB recipe is well calibrated for single weeks but **under-covers lead-time
totals**, because weekly errors are positively correlated (lag-1 correlation 0.13-0.23 on fast and
regular drugs). At 4 weeks, p95 covers about 0.90-0.91 on fast drugs and 0.93 on regular drugs;
the error variance of 4-week totals is 1.2-1.8x what the recipe assumes.

What the decision depends on, in order: the drug's **sustained rate over the policy's life**
(not the next week), its **relative dispersion between drugs**, and, once policies are refreshed,
**fast reaction** to level changes. A slightly better pooled WAPE (0.01-0.02) is far below what
these replays can detect.

## 1. How the policy turns a forecast into s and S

- `decide.basis_from_weekly` takes weekly means for h = 1..8 (weeks past 8 repeat week 8) and
  sets variance = mean x dispersion per drug. `plan._daily` spreads each week evenly over 7 days
  (mean/7 and var/7 per day).
- `contest.to_policy` sums the daily mean and variance over the lead time (supplier cadence,
  clipped to 1-14 days) for s and over lead + 14 days for S, then takes the NB 0.95 quantile with
  `forecast.demand_quantile`.
- Summing variances is right only if days and weeks are independent given the mean. The plan
  (10.2) says so and asks for 2- and 4-week coverage as the check. It was run
  (`lab/ml-results/full/coverage.csv`) and shows the shortfall, but the replay still uses the
  baseline recipe unchanged.
- Dispersion (`forecast.dispersion`) is the drug's variance/mean over its last 52 observed,
  Eid-adjusted weeks, floored at 1. That is the *unconditional* spread of history, not the spread
  of forecast errors. It includes slow level drift, which inflates it, and it ignores correlation
  between weeks, which deflates the window total. The two partly cancel at 1 week and stop
  cancelling as the window grows.

## 2. Calibration of lead-time totals

[`scripts/inventory/calibration.py`](scripts/inventory/calibration.py), output `calibration_windows.csv`,
`error_autocorr.csv`, `dispersion_multiplier.csv`. Coverage of the NB p95 for
consecutive-week totals (uncensored windows only):

| set | tier | 1 wk | 2 wk | 4 wk | 4-wk error var / model var |
|---|---|---|---|---|---|
| selection 2024, baseline-g1 | fast | 0.947 | 0.932 | 0.905 | 1.79 |
| | regular | 0.957 | 0.944 | 0.928 | 1.24 |
| | slow | 0.955 | 0.948 | 0.929 | 1.44 |
| selection 2024, ML 3aafa0367da7 | fast | 0.949 | 0.938 | 0.914 | 1.65 |
| | regular | 0.959 | 0.949 | 0.934 | 1.34 |
| 2025 diag., frozen ML | fast | 0.940 | 0.925 | 0.897 | 1.59 |
| | regular | 0.956 | 0.947 | 0.933 | 1.53 |
| 2025 diag., current | fast | 0.943 | 0.928 | 0.899 | 1.58 |
| | regular | 0.953 | 0.944 | 0.923 | 1.54 |
| | rare | 0.908 | 0.863 | 0.822 | 3.45 |

- Standardised weekly errors within one origin and drug are positively correlated: lag 1-3
  correlation 0.18-0.24 on fast drugs and 0.12-0.15 on regular drugs, for both forecasters. This
  is what you expect when the level estimate itself is wrong: the same level error hits every
  week of the window. Prak, Teunter and Syntetos show this happens even when demand itself has no
  autocorrelation, and that the textbook safety stock is too small as a result
  ([EJOR 2017](https://orca.cardiff.ac.uk/id/eprint/92130/)). With correlation ρ over 4 weeks the
  variance grows by about 1 + 3ρ, which matches the 1.6x seen on fast drugs.
- To bring 4-week p95 coverage to 0.95 the dispersion would need about x1.5 on regular and slow
  drugs and about x2 on fast drugs (`dispersion_multiplier.csv`). A window-length factor
  (variance x (1 + ρ(w - 1)) with ρ fitted per tier) is the simple fix; U2-style direct window
  quantiles or a conformal correction on window totals (plan 10.3) are the heavier ones.
- U1 (the ML-residual dispersion) over-covers at 1 week (0.976-0.982 at p95 on fast and regular)
  and is closer to target at 4 weeks, so it partly fixes the window problem by being wide
  everywhere. It was never used in the replay.
- The coverage numbers are optimistic. Windows with any stock-out day are dropped: 4-6% of
  4-week windows on fast drugs, 9-12% regular, 20% slow, 34% rare. A stock-out is more likely when
  demand runs high, so the dropped windows are the ones most likely to exceed p95.
- For the replay as it is scored now, absolute calibration hardly matters, because matching
  rescales everything (section 4). It matters a great deal for the app, which recommends the
  unmatched policy to the pharmacist.

## 3. What part of the forecast moves the decision

### Analytic

For a lead + cycle total with mean m and dispersion d, the NB p95 moves about one-for-one with a
mean error (+10% mean gives +7-9% on q95 for m >= 20) and much less with a dispersion error (+10%
dispersion gives +0-3%). See `nb_elasticity.csv` (from `scripts/inventory/analytic.py`). Under the 1.6x window variance shortfall,
q95 is 5-15% too low. So in an unmatched policy a mean error and a tail error of similar size do
similar harm. But the matched replay cancels both overall, and keeps only their between-drug
pattern.

### Replay experiments

[`scripts/inventory/sensitivity.py`](scripts/inventory/sensitivity.py), output `sensitivity_{H2-2024,H1-2025}.csv`, figure
[`fig/inventory/lost_revenue_vs_wape.png`](fig/inventory/lost_revenue_vs_wape.png). The current forecaster's basis, perturbed, through
the plain policy. "Matched" is how the project scores; "unmatched" is the policy the app
would recommend (scale 1). Lost revenue in Rp M; noisy variants are means of 3 seeds.

| variant | H2 2024 matched | H2 2024 unmatched | H1 2025 matched (diag.) | H1 2025 unmatched (diag.) |
|---|---|---|---|---|
| current forecaster | 5.60 | 7.57 | 2.80 | 7.07 |
| oracle mean, first 28 days | **7.16** | 10.71 | **4.25** | 12.69 |
| oracle mean, whole half-year | **4.83** | 6.27 | 2.70 | 4.55 |
| mean noise σ=0.1 (WAPE +0.01) | 5.56 | 7.81 | 2.79 | 8.03 |
| mean noise σ=0.25 (WAPE +0.09) | 5.58 | 9.07 | 2.98 | 11.40 |
| mean noise σ=0.5 (WAPE +0.27) | 6.23 | 16.80 | 5.69 | 25.90 |
| all means x0.8 | 5.64 | 10.34 | 2.80 | 13.51 |
| all means x1.25 | 5.54 | 6.16 | 2.79 | 4.06 |
| dispersion x0.5 | 5.58 | 9.10 | 2.80 | 10.21 |
| dispersion x2 | 5.68 | 6.25 | 2.75 | 4.07 |
| Poisson (d = 1) | 5.86 | 14.93 | 3.03 | 22.31 |
| one dispersion per tier | 6.06 | 8.98 | 3.40 | 10.22 |
| service 0.8 / 0.9 / 0.99 | 6.37 / 5.63 / 7.26 | 18.8 / 10.6 / 8.1 | 3.43 / 2.77 / 3.58 | 25.6 / 11.9 / 5.4 |

What this says:

- **Short-horizon accuracy is punished by the static replay.** Knowing the true mean of the first
  28 days (4-week WAPE 0) raises matched lost revenue by 28% (H2 2024) and 52% (H1 2025), and
  stock-out days from 539 to 3,541. Four weeks of realised demand is a noisy estimate of a slow
  drug's six-month rate, and many drugs that happen not to sell in those weeks get s = 0 for six
  months. So 4-week WAPE and the replay reward different things.
- **The sustained rate is what pays.** Knowing each drug's true half-year mean cuts matched lost
  revenue by 14% (H2 2024) and 4% (H1 2025), and unmatched by 17-36%. That is the ceiling on what a
  better mean can buy under this policy design.
- **Overall bias is invisible when matched, and large when not.** All means x0.8 changes matched
  lost revenue by under 1% but raises the app's unmatched lost revenue by 37-91%. The ML forecast
  runs about 9% below the current one on 4-week totals at 30 Jun 2025 (JOURNAL); the matched
  replay cannot see that, the pharmacist would.
- **Between-drug dispersion matters more than its level.** Scaling every dispersion by 0.5 or 2
  does almost nothing matched. Replacing each drug's own dispersion with its tier's median costs
  8% (H2 2024) and 21% (H1 2025). A Poisson tail costs 4-8%. So the per-drug ranking of how
  volatile drugs are is decision-relevant; the absolute width is not, once matched.
- **Small mean improvements are below the floor.** WAPE +0.01 (σ=0.1) moves matched lost revenue
  by -1% to +0%; WAPE +0.09 moves it 0% to +6%. The ML vs current difference in WAPE is
  0.01-0.02.
- **Service 0.99 is worse than 0.95 at the same stock.** At a fixed budget a very high quantile
  spends stock on the most volatile drugs. With matching, the knob only changes *shape*.
- **Shelf-out loss is a large, forecast-blind block.** It is 2.0-2.4 Rp M of 5.6 (H2 2024) and
  0.75-0.93 of 2.8 (H1 2025) whatever the forecast does, because `to_policy` sets the floor
  levels from the recent floor sales rate (`facts.floor_rate`) and matching scales only s and S.
  In the ML replay log, shelf-out days are 8,075 vs 8,036 against ~271-451 stock-out days, so the
  forecaster comparison is diluted by a component it cannot affect.

### Matched stock value: effective service and fairness

- At the matched scale (1.875 in both windows) the reorder point s covers lead-time demand with
  probability 0.998 on fast drugs and 0.993-0.994 on regular drugs (median 0.9995 and 0.996)
  under the forecast's own NB. The plain policy is being evaluated at a service level nobody
  chose, and the evaluated quantity is the far tail of the NB, where its shape is least checked.
- Matching is a fair *budget* control (same money tied up), and it is the right control for a
  comparison of allocation skill. It is not a fair test of the *forecast*, because it removes the
  thing a pharmacist acts on (the level) and it uses the realised stock value of the period being
  scored, which no one can know in advance.
- Matching precision is a noise source. `replay.match_stock_value` stops within 0.5% of the target.
  Around the matched point lost revenue is flat and not monotone in stock
  (`stock_elasticity.csv` from `scripts/inventory/stock_elasticity.py`: ±1% stock moves lost revenue -1% to +0.4%; H1 2025 local
  elasticity about -0.7). In the ML replay log, ML held 362.7 Rp M against 365.0 Rp M for the
  re-selected baseline in H1 2025 (0.6% less). At that elasticity that alone is worth about +0.4%
  lost revenue, which is the whole reported gap (+0.4%, interval -66k to +97k). Matching to
  within 0.05%, or reporting the lost revenue vs stock curve, would remove this.

## 4. Stock-out censoring

- **Fitting.** The current forecaster drops weeks with any stock-out day (`forecast.predict`); ML
  masks them in lags and drops them as targets (plan 5.3). Weeks with only shelf-out days stay
  in, although next-day sales on fast drugs are 0.59x normal after a shelf-out (FINDINGS). So the
  fitted level is biased low for drugs with frequent shelf-outs, and dropping stock-out weeks
  removes weeks more likely to have had high demand. `lost.py` found demand before stock-outs is
  0.95-0.99x the post-restock level, so the stock-out selection bias looks small; the shelf-out
  bias is not measured.
- **Dispersion.** Dropping censored weeks also trims the high side of the history used for the
  dispersion, so the tail is a little too thin, adding to the window under-coverage.
- **Scoring.** WAPE and coverage drop target weeks/windows with a stock-out. Results describe
  uncensored weeks only (stated in the plan) and are optimistic on the tail, as above. Rare and
  slow drugs lose 20-34% of 4-week windows.
- **Replay.** The ML replay uses the uncensored demand (`lost.uncensored_units`) for both sides:
  lost demand is added back on pharmacist stock-out days (first 28 days of a spell), switched
  demand is taken off alternatives. Shelf-out days stay censored, so demand on the pharmacist's
  shelf-out days is understated; the policy is not penalised for empty shelves there. Lost revenue
  in the replay is gross: it does not net off the ~30% of lost lines that switch to another brand
  of the same ingredient. That over-weights stock-outs of high-switch drugs (amlodipine,
  omeprazole) relative to the business cost. Recorded demand (the default in `replay.load`)
  would favour whichever policy runs out on the same days the pharmacist did.

## 5. Are the bootstrap intervals adequate?

[`scripts/inventory/rolling_policy.py`](scripts/inventory/rolling_policy.py) part 2, output `bootstrap_noise_H2-2024.csv`: the
project's paired bootstrap (`decide.paired_bootstrap_replay`, 90%) of the current forecaster
against itself with random mean noise.

- σ = 0.1 (WAPE +0.01, about the ML-vs-current gap): lost-revenue ratios 0.984-1.014. 1 of 10
  runs gives an interval that excludes zero; stock-out days do in 3 of 10.
- σ = 0.25 (WAPE +0.09, a much worse forecast): ratios 0.987-1.031, mean 1.004. Only 1 of 10
  intervals excludes zero on lost revenue, 8 of 10 on stock-out days.

So the lost-revenue rule has low power for anything near the size of the real differences, and
the decision rule leans on its least sensitive metric. The intervals also count only drug-to-drug
variation. They leave out the single policy day per window (an effective sample of one origin
per window), the demand path, the matching jitter and the uncertainty in the uncensored demand
estimate. A test over several policy days per window (block-resampling days, or one policy day
every 4 weeks) would capture origin-to-origin variation.

## 6. Leakage safety

I found no leak in the replay path. Policy-day facts, prices, lead times and the pharmacist
baseline are computed on `holdout.agent_view(day)` or `ensure_baseline(con, day)`; U1 uses only
origins before O whose target week ended by O; the uncensored demand and the trailing rate are
scorer-side only. Points to keep in mind:

- The matching target is the realised stock value of the scored period (future information),
  identical for both sides. Harmless for a comparison, not usable by the app.
- The H2 2024 window overlaps the ML selection period (labelled "reported only", correctly), and
  H1 2025 is now selection evidence for the app choice (plan 9.2 says so).
- The current forecaster's Eid window and method choice used data to June 2025 (plan 3.4); this
  favours it in H1 2025.
- Switch rates for uncensoring use spells before the window, so no leak there.

## 7. What would actually help, and does a policy that reacts change the picture?

`rolling_policy.py` part 1, `rolling_H2-2024.csv` (H2 2024, matched, a time-varying copy
of `replay.simulate`; the static run reproduces the project's 5.60 Rp M exactly):

| variant | lost revenue | of which stock-out | of which shelf-out | stock-out days |
|---|---|---|---|---|
| static policy, current forecaster | 5.60 | 3.34 | 2.27 | 539 |
| refreshed every 14 days, current forecaster | 4.33 | 3.68 | 0.65 | 539 |
| refreshed every 14 days, oracle next-28-day mean | 3.91 | 3.25 | 0.66 | 1,340 |

- Refreshing the policy every two weeks cuts lost revenue by 23%, almost all of it by refreshing
  the floor levels. This is bigger than any forecast change tested.
- Once the policy refreshes, short-horizon accuracy starts to pay: a perfect next-28-day mean
  cuts stock-out loss by 12% against the current forecaster (compare: +53% in the static
  replay). Stock-out days rise because drugs with no demand in the next 28 days get s = 0 and sit
  at zero on days nobody comes; lost revenue and stock-out days disagree here, which is another
  reason not to gate on stock-out days.
- So "fast reaction to level shifts" and "Eid" can only be valued by a replay that re-plans. In
  the current static design Eid never enters any policy: the H1 2025 policy is set on 31 Dec and
  uses January weeks only.

Ranked by plausible lost-revenue gain under the business's real process (re-ordering every week
or two):

1. **Re-plan often, and score it that way.** Not a forecast change, but the largest effect
   found, and the precondition for any short-horizon forecast gain to show up.
2. **A good sustained rate per drug, robust for slow and erratic drugs.** The half-year oracle
   is worth 4-14% at matched stock and 17-36% unmatched. Shrinkage of slow-drug levels toward
   longer histories or pooled estimates targets this.
3. **No level bias, at least by tier.** Invisible to the matched replay, large in the app (x0.8
   costs +37-91% unmatched). The ML model's 9% lower 4-week totals at 30 Jun 2025 deserve a
   check against the uncensored demand before anyone uses its plan unmatched.
4. **Per-drug tail shape, calibrated over the lead-time window.** Losing per-drug dispersion
   costs 8-21%. Fix the window under-coverage (x1.5-2 on 4-week variance, or ρ per tier).
5. **Fast reaction to level shifts, then Eid,** valued only once the policy re-plans. Eid is 5
   weeks a year; its value is real but small compared with 1-4.
6. A small pooled-WAPE gain (0.01-0.02) on its own: not detectable in these replays.

## 8. Recommended evaluation scheme

1. **Rolling replay as the primary decision test.** Re-plan every 7 or 14 days from data up to
   that day (14 current-forecaster bases for H2 2024 built in a few minutes here; ML needs a model
   per origin, or a model reused for 4 weeks as the plan allows). Floor levels refreshed too. Several windows,
   several start days.
2. **Score at fixed budget and at the stated service level.** Keep matched stock value, but match
   to within 0.05% or, better, report the whole curve of lost revenue vs average stock value
   (scale 0.75x-2x; `replay.tradeoff` already does this) and compare curves. Also report the
   unmatched policy: achieved cycle service and fill rate against the 0.95 target, stock value
   and lost revenue, since that is what the app tells the pharmacist.
3. **Net lost revenue.** Lost revenue net of switching (`lost.apply_switching` rates), with
   stock-out and shelf-out parts reported apart. Decide on the stock-out part; the shelf-out part
   depends on the floor rule.
4. **Forecast-level metrics that match the decision.**
   - Pinball loss at 0.95 (and at the effective level, about 0.99) and coverage of **lead-time
     and lead + cycle totals**, computed per drug at its own lead time, revenue-weighted, by tier.
     CRPS of the window total as a single summary (a proper score:
     [Gneiting and Raftery 2007](https://doi.org/10.1198/016214506000001437)).
   - Bias by tier on 4-week and 13-week totals against uncensored demand.
   - A revenue-weighted WAPE of the 4-week total, rather than unit WAPE pooled over h = 1..8.
     Unit WAPE weights cheap high-volume drugs most; lost revenue weights by price.
   - Report coverage including censored windows scored on uncensored demand, not only clean ones.
5. **Power and intervals.** Before running, simulate (as in section 5) what effect size the test
   can see. Resample policy days as well as drugs. Treat stock-out days as a reported metric,
   not a gate, since they and lost revenue can move in opposite directions.

This follows the line in the literature that forecasts for inventory should be judged by
inventory outcomes and by the distribution of lead-time demand, not by point accuracy alone
([Kourentzes, Trapero and Barrow 2020](https://research.birmingham.ac.uk/en/publications/optimising-forecasting-models-for-inventory-planning/);
[Goltsos, Syntetos, Glock and Ioannou 2022](https://orca.cardiff.ac.uk/id/eprint/143417/);
[Prak and Teunter 2019](https://ideas.repec.org/a/eee/intfor/v35y2019i1p224-238.html) on folding
estimation error into the lead-time demand distribution).

## Caveats

- Perturbations are of the current forecaster, not of ML; they measure the replay's sensitivity,
  not which forecaster is better.
- Two windows and one rolling run. H1 2025 is diagnostic. The rolling simulation is my own copy of
  `replay.simulate` with time-varying levels; it reproduces the static result exactly.
- The oracle runs use the uncensored demand estimate, which has its own error (about 1% on fast,
  6% on regular drugs on the fake stock-out test).
- In `error_autocorr.csv` the slow tier for the two 2025 baseline files shows a degenerate
  value (correlation 1.0 at lag 3 from very few pairs); ignore those cells.
