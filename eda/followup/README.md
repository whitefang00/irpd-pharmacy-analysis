# Follow-up: making sense of the demand patterns after the ML results

The ML work ended with forecasts that were hard to read. The 8-week paths came out
almost flat, ML and the current forecaster were close, and the replay could not
clearly rank them. This folder is our attempt to understand why. It asks three
questions: why are the forecasts flat, how do forecasts turn into reorder
decisions, and which fixes are worth trying. It then records the experiments we
ran on those fixes.

All of it is exploratory. Some checks were designed after looking at results, and
those are marked **post hoc** where they appear. 2025 data had already been
inspected during the ML work, so every 2025 number is **diagnostic**: it can show
a problem but cannot prove a gain. Decisions lean on the 2024 selection period.
Nothing here was trained for production, and no default was changed.

Some notes cite files under `lab/` (the ML plan, frozen config and raw results).
Those are private working notes and aren't in the public repo. The numbers
quoted from them are given in full here.

## What we found

The flat forecasts are not a bug. Outside the Eid weeks, week-to-week variation in
2024 looks like constant-rate demand with pack-size noise. About 82% of a flat
forecast's error is irreducible noise, and most of the rest is getting the level
wrong, not the shape. Eid is the one real calendar effect. Even a perfect
store-level Eid model would gain at most 0.017 WAPE in Eid weeks.

The bigger levers sit after the forecast. The reorder policy is set once per
half-year, the forecast spread is too narrow for multi-week totals, the Eid
features have fixable defects, and the replay test cannot detect differences of
the size that are realistically available.

We pre-registered seven experiments (E1-E7) plus a harness check (E0), and ran
E0-E6 on stored predictions. No change passed its rule. Re-planning every 14 days
cut net lost revenue by 10% in H2 2024, short of the 15% bar. The variance
factor, Eid repairs, censoring-aware targets and lifecycle rules each failed at
least one condition. ML has 5.1% lower pooled pinball loss than the current
forecaster (CI −7.9% to −2.5%), with the gain in the slow and rare tiers, but the
rule for making it the default is not met. Settling ML vs current for good needs
a clean test on 2026 data (E7), which does not exist yet.

## Reports

Read [summary.md](summary.md) first. The rest, in the order the work was done:

- [literature.md](literature.md): what the forecasting literature says about
  intermittent, low-count demand like ours.
- [demand-patterns.md](demand-patterns.md): demand statistics, segments, and how
  much signal there is beyond a flat level.
- [why-forecasts-are-flat.md](why-forecasts-are-flat.md): diagnosis of the
  existing models and their predictions.
- [forecasts-to-decisions.md](forecasts-to-decisions.md): how forecast errors
  move reorder outcomes, and whether the replay rewards the right thing.
- [options.md](options.md): candidate architecture changes, sized with small
  checks.
- [experiment-plan.md](experiment-plan.md): the pre-registered experiments and
  their decision rules, with the register table.
- [experiments.md](experiments.md): E0-E6 results.
- [experiment-notes.md](experiment-notes.md): working notes behind the results:
  how the evaluation harness works, definitions fixed before the runs, full E0
  and E1 tables, and how the forecast arms were built.

## Running the scripts

Run everything from the repo root. Scripts read the project's own `data/`
(warehouse, selection file, ML predictions) and write tables and intermediate
files to `data/followup/<topic>/`, which git ignores. Figures go to
`fig/<topic>/` here. Shared paths are in
[`scripts/followup_paths.py`](scripts/followup_paths.py).

```bash
uv run python eda/followup/scripts/<topic>/<script>.py
```

Scripts that draw figures need matplotlib, which is not a project dependency, so
run those as `uv run --with matplotlib python ...`. `diagnostics/07_calendar_signals.py`
also needs `--with statsmodels`. The harness and results scripts use the
`irpd.evalx` code in `src/` and cache slow steps under `data/followup/cache/`;
the cache can be deleted at any time.

Order matters within a topic, because later scripts read earlier outputs:

| Topic | Scripts, in order | Notes |
|---|---|---|
| `demand/` | `drug_stats`, `signal_decomp`, `segment`, `representative` | seconds each |
| `diagnostics/` | `01`-`13` in number order; run `10` twice, the second time with `SKIP_BLOCKS=1,2` | seconds each; `05` ~10 s, `02` ~15 s |
| `inventory/` | `calibration`; `sensitivity H2-2024`, `sensitivity H1-2025`; then `rolling_policy`, `analytic`, `stock_elasticity` | `sensitivity` well under a minute per window |
| `architecture/` | `explore` (needs `diagnostics/01`, `06` and `demand/segment`); `power_rolling`, `frontier_rolling` (need `inventory/sensitivity` and `rolling_policy`) | replay-heavy, minutes |
| `arms/` | `run_arms.py panel`, `ml <arm>`, `cur`, `labels`, `actuals`, `validate` | builds `data/followup/preds/`; the ML arms retrain LightGBM at every origin and take hours |
| `harness/` | `build_data`, `repro`, `e0_checks`; `figs` after `replan/e1` | E0 |
| `replan/` | `e1` | E1 |
| `results/` | `e3`, `e2`, `e4`, `e4_diag`, `e5`, `e5_diag`, `e6`, `gates`, `figs` | E2-E6; `e2`, `e4`-`e6` need `harness/build_data` first; `e6` ~50 min |

The forecast arms in `data/followup/preds/` (about 150 MB, not committed) are the
stored output of `arms/run_arms.py`. Rebuilding them is the only step here that
trains models. Everything else reads stored predictions.

Verified to run from the repo root, with outputs matching the original runs:
all four `demand/` scripts; all `diagnostics/` scripts (`01`-`13`, `09b`, `10` both ways);
`inventory/calibration` and `inventory/sensitivity H2-2024`; `architecture/explore`;
`harness/build_data`; `results/e3` and `results/figs` (`store`, `drugs`, `e1` parts,
which read the stored predictions). The remaining scripts were ported the same way
but not rerun here, because they are long replays (`e6`, `e1`, `e0_checks`,
`power_rolling`) or retrain models (`run_arms`).

## Files

```
eda/followup/
  README.md                     this file
  summary.md                    the synthesis
  literature.md  demand-patterns.md  why-forecasts-are-flat.md
  forecasts-to-decisions.md  options.md
  experiment-plan.md  experiments.md  experiment-notes.md
  fig/
    demand/        representative drugs
    diagnostics/   flatness, compound-Poisson check, rolling store level, Eid paths
    inventory/     lost revenue vs forecast error
    harness/       E0 power calibration
    replan/        E1 frontier and matched split
    results/       E0-E6 figures and example drugs
  scripts/
    followup_paths.py           shared paths
    demand/  diagnostics/  inventory/  architecture/
    arms/  harness/  replan/  results/
```
