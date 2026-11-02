"""Two-way bootstrap for paired differences (roadmap, "Intervals").

Each scored unit belongs to an origin (a forecast origin, or a replay period)
and a drug. Windows from adjacent origins overlap, and a drug's errors repeat
across origins, so neither drugs nor origins are independent. A draw:

- resamples origins in blocks of `block` adjacent origins (with replacement,
  as many blocks as there are), and
- resamples drugs independently (with replacement),

then weights each (origin, drug) cell by (times its origin was drawn) x (times
its drug was drawn). Statistics are ratios of weighted sums, so a draw is two
small matrix products per column. `drugs_only=True` gives the old drug-only
bootstrap (`ml.stats`, `ml.decide`) for comparison.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

N_DRAWS = 1000
LEVEL = 0.9
BLOCK = 2

Stat = Callable[[dict[str, np.ndarray]], np.ndarray]


@dataclass
class Interval:
    point: float
    lo: float
    hi: float
    draws: np.ndarray

    def excludes_zero(self) -> bool:
        return bool(self.lo > 0 or self.hi < 0)


def cell_sums(df: pd.DataFrame, cols: Sequence[str], origin: str = "origin",
              drug: str = "drug_code") -> tuple[dict[str, np.ndarray], list, list]:
    """Origins x drugs matrices of the sum of each column."""
    o_codes, o_idx = np.unique(df[origin].to_numpy(), return_inverse=True)
    d_codes, d_idx = np.unique(df[drug].astype(str).to_numpy(), return_inverse=True)
    out = {}
    for c in cols:
        m = np.zeros((len(o_codes), len(d_codes)))
        np.add.at(m, (o_idx, d_idx), df[c].to_numpy(float))
        out[c] = m
    return out, list(o_codes), list(d_codes)


def origin_weights(n_origins: int, n: int, rng: np.random.Generator,
                   block: int = BLOCK) -> np.ndarray:
    """n x origins counts from resampling blocks of `block` adjacent origins."""
    blocks = [list(range(i, min(i + block, n_origins))) for i in range(0, n_origins, block)]
    picks = rng.integers(0, len(blocks), size=(n, len(blocks)))
    w = np.zeros((n, n_origins))
    for b, members in enumerate(blocks):
        cnt = (picks == b).sum(axis=1)
        for o in members:
            w[:, o] = cnt
    return w


def two_way(df: pd.DataFrame, cols: Sequence[str], stat: Stat, origin: str = "origin",
            drug: str = "drug_code", n: int = N_DRAWS, level: float = LEVEL,
            block: int = BLOCK, seed: int = 0, drugs_only: bool = False) -> Interval:
    """Point value and `level` percentile interval of `stat` over two-way draws.

    `stat` takes {column: weighted sums} (arrays, one value per draw, or scalars
    for the point estimate) and returns the statistic, e.g.
    `lambda s: (s["a"] - s["b"]) / s["y"]` for a difference of normalised sums.
    Origins are ordered by their sort order (dates sort in time).
    """
    sums, origins, drugs = cell_sums(df, cols, origin, drug)
    rng = np.random.default_rng(seed)
    n_o, n_d = len(origins), len(drugs)
    wo = np.ones((n, n_o)) if drugs_only else origin_weights(n_o, n, rng, block)
    wd = rng.multinomial(n_d, np.full(n_d, 1 / n_d), size=n).astype(float)
    draws = {c: ((wo @ m) * wd).sum(axis=1) for c, m in sums.items()}
    point = float(np.asarray(stat({c: m.sum() for c, m in sums.items()})))
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.asarray(stat(draws), float)
    a = (1 - level) / 2
    return Interval(point, float(np.nanquantile(x, a)), float(np.nanquantile(x, 1 - a)), x)


def paired_diff(df: pd.DataFrame, a: str, b: str, den: str | None = None, **kw) -> Interval:
    """Interval on (sum a - sum b) / sum den (or on sum a - sum b without `den`)."""
    if den is None:
        return two_way(df, [a, b], lambda s: s[a] - s[b], **kw)
    return two_way(df, [a, b, den], lambda s: (s[a] - s[b]) / s[den], **kw)


def paired_ratio(df: pd.DataFrame, a: str, b: str, **kw) -> Interval:
    """Interval on sum a / sum b."""
    return two_way(df, [a, b], lambda s: s[a] / s[b], **kw)
