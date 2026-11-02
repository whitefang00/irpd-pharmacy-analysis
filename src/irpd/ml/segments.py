"""Demand segment labels per drug, computed at an origin from the history up to it (E5).

`labels(h, tier)` takes a history already cut at the origin (`history.cut`)
and the tier as of the origin. Every label looks only at those weeks, which
avoids the hindsight of labels drawn from the whole series.

- `fading`: sold before, but no sale in the last FADING_WEEKS (13) weeks.
- `sporadic`: rare tier, or at least SPORADIC_ZERO_SHARE (85%) of the
  observed weeks among the last 52 had no sale. Observed means on or after
  the first sale and not censored; with fewer than SPORADIC_MIN_WEEKS (13)
  observed weeks only the tier rule applies, since a share of a few weeks
  says little.
- `new`: the first sale is under NEW_WEEKS (26) weeks before the origin
  (its week started less than 182 days before the origin's Monday-after).
- `censored`: at least CENSORED_SHARE (20%) of the last 26 weeks on or after
  the first sale, and covered by the stock ledger, had a total stock-out
  day. Shelf-outs don't count here: the floor running empty while the store
  holds stock is a fetch problem, not a demand floor of the same kind.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FADING_WEEKS = 13
SPORADIC_WEEKS = 52
SPORADIC_ZERO_SHARE = 0.85
SPORADIC_MIN_WEEKS = 13
NEW_WEEKS = 26
CENSORED_WEEKS = 26
CENSORED_SHARE = 0.20


def labels(h: dict, tier: np.ndarray) -> pd.DataFrame:
    """One row per drug of `h`: the segment flags and the numbers behind them."""
    y = h["y"]
    n = y.shape[1]
    sold = np.nan_to_num(y) > 0
    has_sale = sold.any(axis=1)
    first = np.where(has_sale, sold.argmax(axis=1), n)
    weeks = np.arange(n)[None, :]
    after_first = weeks >= first[:, None]

    recent = sold[:, max(0, n - FADING_WEEKS):]
    fading = has_sale & ~recent.any(axis=1)

    lo = max(0, n - SPORADIC_WEEKS)
    obs = after_first[:, lo:] & ~h["out"][:, lo:] & ~np.isnan(y[:, lo:])
    n_obs = obs.sum(axis=1)
    zeros = (obs & ~sold[:, lo:]).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        zero_share = np.where(n_obs > 0, zeros / n_obs, np.nan)
    sporadic = (tier == "rare") | ((n_obs >= SPORADIC_MIN_WEEKS)
                                   & (zero_share >= SPORADIC_ZERO_SHARE))

    new = has_sale & (n - first <= NEW_WEEKS)

    lo = max(0, n - CENSORED_WEEKS)
    known = h["stock_known"][lo:][None, :] & after_first[:, lo:]
    n_known = known.sum(axis=1)
    outs = (known & h["stockout"][:, lo:]).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out_share = np.where(n_known > 0, outs / n_known, np.nan)
    censored = np.nan_to_num(out_share) >= CENSORED_SHARE

    return pd.DataFrame({
        "drug_code": h["drugs"].astype(str), "tier": tier.astype(str),
        "fading": fading, "sporadic": sporadic, "new": new, "censored": censored,
        "weeks_since_sale": np.where(has_sale, np.argmax(sold[:, ::-1], axis=1), np.nan),
        "weeks_since_first": np.where(has_sale, n - 1 - first, np.nan),
        "zero_share_52": zero_share, "observed_52": n_obs,
        "stockout_share_26": out_share,
    })
