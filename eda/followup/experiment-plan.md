# Experiment plan (pre-registered)

After the diagnostic work in this folder, we wrote down the experiments worth running
and fixed their order before running any of them. Each entry states
its hypothesis, data, metric, decision rule and power before results exist. If a
rule must change, write the change and the reason here, dated, **before** looking
at the affected result.

Background and sizing: [options.md](options.md). Exploratory checks behind the
numbers quoted here: `scripts/architecture/` (outputs in `data/followup/architecture/`).
The results are in [experiments.md](experiments.md).

## Data reality

- **Selection set**: data up to 8 Dec 2024 (`selection.duckdb`). This is where all
  development and every choice happens. It has 11 four-weekly ML origins in 2024,
  can have about 22 two-weekly ones, and one Eid (2024) with one Eid before it to
  train on.
- **2025**: already inspected by the project and by this follow-up work. It is **not** a
  clean test. Use it once per experiment, as a second development set, labelled
  "diagnostic". It can show a regression. It cannot certify a gain.
- **Fresh data**: none exists. Sales end 31 Dec 2025; the last full week starts
  29 Dec 2025. Checked in `irpd.duckdb` and the raw CSVs. Today is 1 Nov 2026.
- **Getting a clean test.** Ask the pharmacy (the dataset owners) for the same two
  Excel exports for 1 Jan 2026 to the latest date available. Process them with the
  unchanged `generate_raw.py` and `generate_processed.py`. That gives up to about
  44 weeks, including Eid al-Fitr 2026 (about 20-21 March; confirm with Kemenag).
  Protocol:
  1. Finish experiments E0-E6 on selection plus diagnostic 2025.
  2. Freeze the code commit, config hashes, Eid date list and this file. Record their
     hashes in `lab/JOURNAL.md`.
  3. Only then load the 2026 data, and run the E7 comparison once.
  4. If the data cannot be had, no clean test exists. Every adoption below is then
     "adopted on development evidence". The app should say so. It should not claim a
     measured gain.
- A forward shadow log is the backup if the pharmacy can share data going forward but
  not back-fill. Each Sunday, both forecasters and their (s, S) are written with a
  timestamp before the week's sales arrive. Each week of that log is an untouched
  origin.

## Fixed definitions used by every experiment

- **Origins.** Selection: two-weekly Sundays from 7 Jan 2024 to 13 Oct 2024 (22
  origins). The ML retrains at each origin, as the project's backtest does.
  Diagnostic: the 9 eval origins in 2025.
- **Primary forecast score.** Revenue-weighted pinball loss at 0.95 of each drug's
  lead + cycle total. The window is its own lead time (clipped 1-14 days) plus 14
  days, built from the weekly mean and variance the way `contest.to_policy` builds
  them. Price is the drug's average price over the trailing 182 days at the origin.
  Report it by tier. The pooled fast + regular + slow number is the gate. Rare is
  reported but is not a gate, because it is fragile to the censoring rule.
- **Secondary scores.** CRPS of the same total. p95 coverage of lead and of
  lead + cycle totals. Bias on 4- and 13-week totals by tier. Revenue-weighted WAPE
  of 4-week totals. Unit WAPE h1-8, kept so results compare with earlier work.
- **Censored windows.** Score them on uncensored demand (`lost.uncensored_units`) and
  report them separately. Gates use all windows. Clean-only results are kept for
  comparison with earlier work.
- **Intervals.** 90% two-way bootstrap: resample origins in blocks of 2 adjacent
  origins (windows overlap) and drugs independently, 1,000 draws. This replaces the
  drug-only bootstrap in `stats.py` for every decision.
- **Non-inferiority margin** (forecast). The upper 90% bound of (new − reference) is
  below +1% of the reference's pooled pinball. Superiority needs the upper bound
  below 0.
- **Rolling replay.** (s, S) and floor levels re-planned every 14 days from data up
  to that day. Run for each variant at scales k ∈ {0.7, 0.85, 1, 1.2, 1.45, 1.75,
  2.1}, giving a curve of net lost revenue (switching netted, stock-out and shelf-out
  shown apart) against average stock value. Compare at the reference's matched stock
  value, and also at k = 1 (the app's real recommendation). For k = 1, report
  achieved cycle service, fill rate and stock value. Windows: H2 2024 (primary dev),
  H1 2024 if the stock history supports the policy facts, H1 and H2 2025 diagnostic.
  Start day: the window's first day and +7 days.
- **Replay non-inferiority.** At equal stock, net stock-out lost revenue of the new
  arm is no more than 5% above the reference, averaged over windows and start days.
  This margin is about the size of the harness noise; E0 calibrates it (see below).

---

## E0. Evaluation harness (prerequisite for everything)

**What.**

- E0a. Move the rolling `simulate_tv` from the inventory analysis ([forecasts-to-decisions.md](forecasts-to-decisions.md)) into the project's replay as an option,
  with floor refresh, frontier output, matching tolerance 0.05%, and net-of-switching
  lost revenue.
- E0b. A scorer for lead-time totals that implements the definitions above.
- E0c. A two-way bootstrap.

**Hypothesis.** The harness reproduces known results, and its noise level is known.

**Checks, fixed now.**

1. Static mode with k matched reproduces the project's H2 2024 lost revenue of
   5.60 Rp M to within 0.5%. Rolling mode reproduces the inventory analysis's 4.33 Rp M.
2. A/A test: the current forecaster against itself with a different matching start
   point. Under the two-way bootstrap, at most 1 of 10 intervals excludes zero.
3. Power calibration, done once, before any real comparison. Run 10 seeds each of
   persistent per-drug mean noise at σ = 0.1 and 0.25, plus all means x0.9. Record:
   - for the forecast scores: the fraction of seeds flagged worse;
   - for the replay at equal stock: the spread of lost-revenue ratios across seeds.
   Set the replay non-inferiority margin to the 80th percentile of |ratio − 1| at
   σ = 0.1. The exploratory value is about 0.10 (`scripts/architecture/frontier_rolling.py`). If it
   comes out above 10%, the replay is a guard only for gross failures. Say so in
   every report.

**Expected.** Exploratory runs: the drug-only bootstrap misses σ = 0.1 noise
9 times in 10. Two-way intervals are 1.5-2x wider than drug-only intervals. So power
gets worse, not better. That is honest. The point is to stop over-claiming.

**Compute.** About 1 day of coding. Each rolling frontier is 7 runs x ~1.5 s for the
current forecaster. ML bases need one fit per re-plan day, about 13 x 20 s, or about
5 min per window.

**Depends on.** Nothing.

## E1. Re-plan cadence and floor refresh (policy, not forecast)

**Hypothesis.** Re-planning (s, S) and floor levels every 14 days, instead of once per
half-year, cuts net lost revenue at equal stock by at least 15%. Re-planning every
7 days adds less than 5% on top.

**Data.** H2 2024 (primary). H1 2025 and H2 2025 (diagnostic).

**Metric.** Net lost revenue at the static policy's matched stock value. Stock-out
and shelf-out parts are shown apart.

**Decision rule.** Adopt 14-day re-planning as the evaluation default, and recommend
it in the app, if the reduction is ≥ 15% in H2 2024 and points the same way in both
2025 windows. Choose 7-day only if it beats 14-day by > 5% in all three windows.

**Power.** The inventory analysis measured −23%, which is well above the ~10% harness noise.

**Compute.** Minutes.

**Depends on.** E0a.

## E2. Lead-time variance calibration

**Hypothesis.** Scaling each drug's NB variance for a w-week window by
1 + ρ_tier (w − 1) brings p95 coverage of lead + cycle totals to between 0.935 and
0.965 for fast, regular and slow. It does this without making pooled pinball worse
by more than 1%. ρ is fitted on standardised weekly errors. A second arm fits ρ
separately inside and outside Eid windows (target week within −30..+28 days of Eid).

**Data.** Fit ρ on selection origins dated before 1 Jul 2024. Test on the selection
origins from Jul to Oct 2024, then on 2025 (diagnostic). Do this for both forecasters.

**Metrics.**

- Gate: coverage (lead and lead + cycle), and pooled revenue-weighted pinball at 0.95.
- Reported, not gated: the rolling replay at k = 1, showing achieved cycle service
  against the 0.95 target, stock value, and net lost revenue.

**Decision rule.** Adopt the single-ρ arm if coverage is inside [0.935, 0.965] for
fast, regular and slow on the Jul-Oct 2024 test origins, and pinball is
non-inferior. On 2025, coverage must not fall below 0.92. Adopt the Eid-split arm
instead only if it does better on both checks in 2025 and is no worse on Jul-Oct
2024.

**Exploratory expectation.**

- Single ρ: about x1.6 (fast), x1.35 (regular), x1.3 (slow) at 4 weeks.
- 4-week coverage moves from 0.90-0.93 to 0.95-0.97. It slightly over-covers in quiet
  periods; fast pinball goes from 0.0291 to 0.0297.
- In the replay at k = 1 it should buy about 6% more stock and cut lost revenue by
  about 25%. At equal stock there is no detectable change. That is expected: this is
  a service-level honesty fix, not a better way to allocate stock.

**Power.** Coverage on about 6,000 fast and 13,000 regular windows per period has a
standard error of about 0.003. The effect is about 0.03-0.05, so it is easily seen.

**Compute.** Minutes. No training.

**Depends on.** E0b.

## E3. Eid repair (defect fix, judged on placement plus non-inferiority)

**Changes.**

1. Add Eid 2026 and 2027 to `forecast.EID`, with dates confirmed with Kemenag.
2. Set `c_eid_days` to missing when |days| > 45.
3. Add day-count features for the target week: days in −7..−1, closed days (Eid day
   plus cuti bersama), days in +1..+7, and days in +8..+14.
4. In the current forecaster, replace the Thursday-week uplift with a unit-based daily
   Eid profile summed over each target week's days.

**Hypothesis.** The forecast store-level peak lands in the week holding the most
−7..−1 days. The Jan 2024-style dip far from Eid disappears. Pooled pinball and WAPE
are non-inferior.

**Data.** This is a leave-one-Eid-out test on the selection set. The origins of
18 Feb, 3 Mar, 17 Mar and 31 Mar 2024 forecast Eid 2024 (10 Apr) using only Eid 2023
for training. These are the 4 origins in the 22-origin set whose 8 weeks hold the
Eid peak. 2025 Eid origins are diagnostic. Eid 2026 is the only clean test (E7).

**Metrics.**

- (a) Placement: for each origin with Eid in weeks 1-8, is the forecast peak week
  the actual peak week? Also the absolute % error of store-level weekly totals in
  Eid weeks −4..+3.
- (b) Pooled pinball and WAPE over all origins, for non-inferiority.
- (c) Forecasts made after 24 Sep 2025 have `c_eid_days` inside the training range.
  This is a unit test.

**Decision rule.** Adopt if all of these hold:

- (c) passes;
- the peak is placed right at ≥ 3 of the 4 Eid-2024 origins, and at a majority of
  the 2025 eval origins whose 8 weeks contain the actual peak week (24 Mar 2025);
- Eid-week store absolute % error is lower in both years;
- (b) is non-inferior.

The date list and the cap ship regardless. They are correctness fixes.

**Power.** Only two Eids can be tested, and the gain ceiling is 0.017 WAPE in 2025 Eid
weeks (about 0.003 over the year). That is below the MDE of about 0.014-0.022, so no
superiority claim on WAPE is possible. The placement criteria are deliberately
structural.

**Compute.** Retraining ML on 22 origins x 3 seeds at about 10 s per fit is about
11 minutes. Two arms (repair, control).

**Depends on.** E0b.

## E4. Censoring-aware training targets

**Hypothesis.** Two changes:

- replace the dropped target weeks of stock-out weeks with uncensored demand estimates
  (`lost.uncensored_units`, using pre-window switch rates);
- scale shelf-out days by the measured next-day ratio, or flag them.

Together these raise ML's bias on the censored segment from about 0.91 to within
[0.95, 1.05] against uncensored demand, with no tier's pooled pinball worse than the
margin.

**Data.** Selection origins. Censored-segment labels are taken at each origin, using
history up to that origin, which avoids the hindsight in the segment labels of [demand-patterns.md](demand-patterns.md). 2025 is
diagnostic.

**Metrics.**

- Gate: bias of 4-week totals against uncensored demand, for the censored segment
  and for the slow tier, plus pooled pinball non-inferiority.
- Reported: rolling-replay net stock-out loss restricted to censored drugs.

**Decision rule.** Adopt if censored-segment bias is inside [0.95, 1.05], slow-tier
bias is no further from 1 than the control's, and pooled pinball is non-inferior.
The replay on its own does not decide this, because it has no power on 4.5% of
revenue.

**Risk.** The uncensored estimate has its own error: about 6% on regular drugs in
the inventory analysis's fake stock-out test. Substitution means some lost demand moved to a
sibling drug. Report both gross and switch-netted versions.

**Compute.** Same as E3: about 11 min per arm.

**Depends on.** E0b, E2. Pinball depends on the spread, so calibrate first.

## E5. Lifecycle and sporadic rules

**Rules.**

- **Fading**: no sale in the last 13 weeks at the origin. Forecast with TSB
  (α_p = 0.15), whose rate decays with every zero week. Raise a stop-reorder flag
  once the TSB rate × 26 weeks < 1 unit.
- **Sporadic**: rare tier, or ≥ 85% zero weeks. Use s = 0 and S = the largest single
  sale in 52 weeks (a min-stock rule) instead of NB.
- **New** (first sale under 26 weeks before the origin): apply only to the SES
  fallback, shrinking toward ML's tier prior. Not needed for ML, whose bias is 1.06.

All labels are computed at the origin.

**Hypothesis.** At equal net lost revenue, the rules free stock value on these drugs.
Fading drugs' bias drops from about 1.26 to below 1.1.

**Data.** Selection origins and H2 2024 replay. 2025 diagnostic.

**Metrics.** Bias on fading and sporadic drugs. On the rolling replay frontier: stock
value at the reference's lost revenue, restricted to the affected drugs, plus the
pooled guard.

**Decision rule.** Adopt each rule separately if:

- it frees ≥ 5% of the affected drugs' average stock value at equal pooled net lost
  revenue (non-inferior by the E0 margin), and
- it adds no more than 2 net stock-out days per affected drug per half-year.

**Power.** Moderate. The stock released is mechanical and should be visible. The
lost-revenue guard is weak, so also list the affected drugs whose lost revenue rises
by more than Rp 50k, for a pharmacist to review by eye.

**Compute.** Minutes.

**Depends on.** E0a, E1.

## E6. Re-decide the default forecaster under the new harness

**Arms.** All arms carry the E2-E5 changes that were adopted.

- ML.
- Current (SES + Eid).
- Equal-weight ML + current, as a bias hedge only.

**Hypothesis.** ML is non-inferior to current on pooled pinball and on the replay,
and better on pinball.

**Data.** 22 selection origins (gate). 2025 eval and the H1/H2 2025 rolling replays
(diagnostic).

**Decision rule.** ML becomes the default if:

- pooled pinball is superior (two-way 90% upper bound < 0);
- no tier among fast, regular and slow is worse beyond the margin;
- the replay is non-inferior in every window;
- bias on 4- and 13-week totals is within [0.97, 1.03] for fast and regular.

The ensemble replaces ML only if it beats ML on pooled pinball with an upper bound
below 0. The exploratory evidence makes this unlikely: errors correlate 0.93-0.98,
and the ensemble's selection WAPE is 0.4627 against ML's 0.4603.

**Power.** The ML-vs-baseline WAPE gap is 0.018 on fast and 0.012 on regular, against
an MDE of about 0.014 with 26 origins. The pinball gap is unknown. A null result is
likely on regular and slow. If superiority is not met but non-inferiority is, keep
the owner's choice with both shown, as now. Record that the test could not separate
them.

**Compute.** About 11 min of ML retraining. Replays: about 5 min per window.

**Depends on.** E0-E5.

## E7. Confirmation on fresh 2026 data (one shot)

**Precondition.** 2026 exports obtained, and the freeze described under "Data
reality" recorded before loading.

**Test.**

- Origins: every second Sunday from 4 Jan 2026 to the last date minus 8 weeks.
- Replay windows: H1 2026, and H2 2026 if available.
- Report every metric of E2, E3 (Eid 2026 placement) and E6, each with its decision
  rule unchanged.

**Interpretation.** This is a confirmation, not a selection.

- If a component adopted on development data fails its rule here, report the failure.
  Revert it only if the failure is a non-inferiority breach. A missed superiority
  target is expected at this power.
- Eid 2026 gives one more placement check: one event, so "right week, yes or no".

**Compute.** About 30 min end to end.

## Not scheduled (and why)

- **Hierarchical Bayesian NB pooling.** ML already pools. The gain is bounded by the
  level-miss ceiling, and the cost is high. Revisit only if E6 keeps SES as default.
- **Segment router for the mean, ADIDA/iMAPA, iETS.** All flat methods tie within
  0.01-0.02 WAPE on intermittent drugs, which are at 1.6x the noise floor. Nothing
  for the mean to win.
- **Bulk-order decomposition.** 1.9% of revenue. Its mean is fine. If anything, it
  belongs in a later spread model (quantile/GAMLSS), after E2.
- **Payday / holiday features.** The ceiling for a store-level calendar model is
  about 0.003 WAPE outside Eid. At most one ablation, folded into E3's runs.
- **Horizon-specific dynamics, momentum, drug-level yearly seasonality.** Ruled out by
  the demand-pattern and model-diagnosis work: realised slopes don't decay with h, recent changes reverse, and the
  lag-52 correlation is at chance.

## Order and dependencies

```
E0 harness ─┬─► E1 re-plan cadence ──────────────┐
            ├─► E2 variance calibration ─► E4 censoring ─┤
            ├─► E3 Eid repair ───────────────────────────┼─► E6 default decision ─► E7 fresh 2026
            └─► (E1) ─► E5 lifecycle rules ──────────────┘
```

Total compute is under 2 hours of machine time. Most of the effort is building E0
(1-2 days) and getting the 2026 data. The register at the end of this file lists each experiment with its gate metric
and rule.

## Register

| id | name | kind | gate_metric | decision_rule | data_gate | data_diagnostic | expected_effect | mde_or_power | compute | depends_on |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | Evaluation harness (rolling replay + lead-time scorer + two-way bootstrap) | prerequisite | reproduction + A/A + power calibration | reproduce 5.60/4.33 Rp M within 0.5%; A/A false positive <=1/10; set replay NI margin = p80 \|ratio-1\| at sigma 0.1 | H2 2024 | - | n/a | noise ~+-10% lost revenue at sigma 0.1 (exploratory) | 1-2 days coding; minutes to run | - |
| E1 | Re-plan every 14 days (policy) | policy | net lost revenue at equal stock | adopt if >=15% cut in H2 2024 and same sign in H1/H2 2025 | H2 2024 | H1+H2 2025 | -23% (inventory analysis) | well above noise | minutes | E0a |
| E2 | Lead-time variance factor 1+rho(w-1) per tier | distribution | p95 coverage of lead+cycle totals; pooled rev-weighted pinball95 | coverage in [0.935;0.965] fast/regular/slow on Jul-Oct 2024 origins; pinball NI (+1%); 2025 coverage >=0.92 | selection origins (fit <1 Jul 2024; test after) | 2025 eval | 4-wk coverage 0.90-0.93 -> 0.95-0.97 | coverage SE ~0.003 | minutes | E0b |
| E3 | Eid repair (dates 2026-27; cap c_eid_days; day-count features; daily profile for SES) | defect fix | peak-week placement; Eid-week store abs % error; pooled pinball NI | peak right >=3/4 Eid-2024 origins and majority of 2025; store error lower both years; NI; date list + cap ship regardless | 4 Eid-2024 origins (leave-one-Eid-out) | 2025 eval Eid origins | <=0.017 WAPE in Eid weeks (ceiling); ~0.003/yr | below MDE: structural rule only | ~11 min ML per arm | E0b |
| E4 | Censoring-aware targets (impute stock-out weeks; shelf-out days) | bias fix | bias of 4-wk totals vs uncensored demand (censored segment; slow tier); pinball NI | censored bias in [0.95;1.05]; slow bias no worse; pinball NI | selection origins (labels at origin) | 2025 eval | censored ML bias 0.91 -> ~1.0 | bias detectable; replay not | ~11 min ML per arm | E0b E2 |
| E5 | Lifecycle rules (fading TSB + stop flag; sporadic min-stock; new pooled start for SES only) | rules | stock value freed on affected drugs at equal pooled lost revenue | >=5% stock freed; replay NI; <=2 extra stock-out days per affected drug | H2 2024 rolling replay + selection origins | 2025 | fading bias 1.26 -> <1.1 | moderate | minutes | E0a E1 |
| E6 | Default forecaster: ML vs current vs ensemble (all with adopted fixes) | selection | pooled rev-weighted pinball95 (two-way CI); replay NI; bias 0.97-1.03 | ML default if pinball superior + no tier worse + replay NI + bias ok; ensemble only if beats ML | 22 selection origins | 2025 eval + H1/H2 2025 replays | ML-baseline WAPE gap 0.012-0.018 | MDE ~0.014 WAPE at 26 origins: null likely | ~11 min ML + ~5 min/window | E0-E5 |
| E7 | Fresh 2026 confirmation (one shot after freeze) | confirmation | all E2/E3/E6 metrics unchanged rules | report pass/fail; revert only on NI breach | 2026 exports (to obtain) | - | - | one Eid; ~20 origins | ~30 min | E0-E6 + 2026 data |
