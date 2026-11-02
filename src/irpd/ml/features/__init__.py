"""Feature groups: their column prefix, ablation step, builder module and owner.

Each group lives in its own module with one function:

    def build(ctx: schema.PanelContext) -> pd.DataFrame

returning prefixed feature columns indexed by (drug_code, origin) for features
that don't depend on h, or (drug_code, origin, h) for ones that do. The panel
joins them onto its rows. `h` and `tier` are bookkeeping columns every model
gets (schema.MODEL_COLUMNS); builders don't repeat them.

- demand (d_, A1): own-sales lags and means, intermittency, last year,
  censored-lag flags, shelf-out days.
- calendar (c_, A1): target week of year, distance to Eid, Ramadan, the
  learned Eid uplift.
- store (s_, A2): store sale lines up to the origin, trend.
- price (p_, A2): price at the origin, changes.
- drug (x_, A2): source category from the `drugs` table.
- groups (g_, A3): ingredient groups. g_static_* holds group id, form, kind,
  strength and group size; g_dyn_* holds the leave-one-out siblings' demand
  index and rupiah, siblings in stock, and the drug's share of group revenue.
- stock (k_, A4): stock on hand at the origin, days of cover. Never used by
  the replay forecaster.
- season (y_, add-on): week of year and last year's sales. In no ablation
  step: the EDA found no yearly pattern beyond Eid. Tried once on its own.

The g_static_ / g_dyn_ split lets the ablation compare knowing what a drug
is with seeing its siblings' sales. Leave-one-out applies to every g_dyn_
column except `g_dyn_share_*`, which is the drug's own share by definition.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

# name -> (column prefix, ablation step)
GROUPS: dict[str, tuple[str, int]] = {
    "demand": ("d_", 1),
    "calendar": ("c_", 1),
    "store": ("s_", 2),
    "price": ("p_", 2),
    "drug": ("x_", 2),
    "groups": ("g_", 3),
    "stock": ("k_", 4),
    "season": ("y_", 9),   # add-on only: in no ablation step (EDA, plan section 8)
}
PREFIXES = tuple(p for p, _ in GROUPS.values())
REPLAY_EXCLUDED = ("stock",)
GROUP_STATIC, GROUP_DYNAMIC = "g_static_", "g_dyn_"


def groups_for_step(step: int) -> tuple[str, ...]:
    """The feature groups in ablation step A`step` (1..4)."""
    return tuple(g for g, (_, s) in GROUPS.items() if s <= step)


def builder(name: str) -> Callable:
    """The `build` function of a feature group's module."""
    if name not in GROUPS:
        raise KeyError(f"unknown feature group {name!r}")
    return importlib.import_module(f"irpd.ml.features.{name}").build
