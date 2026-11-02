"""Evaluation harness (roadmap E0): lead-time-total scores, two-way bootstrap,
rolling replay.

    from irpd.evalx import score_arm, compare_arms, rolling_frontier

- `score_arm(preds, actuals, labels, daily=None, rho=None)`: revenue-weighted
  pinball(0.95) / CRPS / coverage of lead and lead + cycle totals, 4- and
  8-week bias, revenue-weighted 4-week WAPE, unit WAPE, by tier and pooled.
- `compare_arms(a, b)`: paired differences a - b with two-way (origin blocks of
  2 x drugs, 1,000 draws) 90% intervals and the non-inferiority verdicts.
- `rolling_frontier(source, window, cadence)`: the rolling replay; sources are
  `CurrentSource()`, `ArmSource(path)` or `Perturbed(...)`.
- `compare_replay(a, b)`: ratio of matched net lost revenue with intervals.

See `experiments/harness/README.md` in the sandbox for worked usage.
"""

from irpd.evalx.bootstrap import paired_diff, paired_ratio, two_way
from irpd.evalx.rolling import (KS, WINDOWS, ArmSource, CurrentSource, Perturbed,
                                compare_replay, first_grid_monday, grid, load_window,
                                match, make_plan, noise_factors, rolling_frontier, run_at)
from irpd.evalx.scorer import METRICS, Scores, compare_arms, crps_nb, score_arm

__all__ = ["KS", "METRICS", "WINDOWS", "ArmSource", "CurrentSource", "Perturbed", "Scores",
           "compare_arms", "compare_replay", "crps_nb", "first_grid_monday", "grid",
           "load_window", "make_plan", "match", "noise_factors", "paired_diff",
           "paired_ratio", "rolling_frontier", "run_at", "score_arm", "two_way"]
