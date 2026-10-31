"""The `irpd replay` table with recorded demand vs demand uncensored by `lost.uncensored_demand`.

Run from the repo root on a built and sealed warehouse:
    uv run --no-sync python scripts/lost_replay.py
"""

import numpy as np
import pandas as pd

from irpd import holdout, lost, policies, replay


def table(h: replay.Holdout) -> pd.DataFrame:
    """What `irpd replay` prints (cli.replay_report), for a loaded holdout."""
    with holdout.agent_view() as con:
        refs = {"mimic": policies.mimic(con), "textbook": policies.textbook(con)}
    runs = dict(refs)
    for name, policy in refs.items():
        matched, k = replay.match_stock_value(policy, h)
        runs[f"{name} at matched stock (x{k:.2f})"] = matched
    t = replay.score(runs, h)
    return t[["stockout_days", "shelfout_days", "lost_revenue", "avg_stock_value",
              "units_sold", "unmet_units_direct"]]


rec = replay.load()
unc = replay.load(demand=lost.uncensored_demand())
d = unc.demand - rec.demand
out = rec.actual_floor + rec.actual_store <= 0
print(f"{len(rec.drugs)} drugs, {len(rec.days)} days. Demand units: recorded "
      f"{rec.demand.sum():,.0f}, uncensored {unc.demand.sum():,.0f} "
      f"(+{d[d > 0].sum():,.0f} added on out days, {d[d < 0].sum():,.0f} taken off "
      f"alternatives). Added on the pharmacist's stock-out days: "
      f"{d[out & (d > 0)].sum():,.0f} units; revenue at replay prices "
      f"Rp {(np.clip(d, 0, None) * rec.price[:, None]).sum() / 1e6:,.1f}M added, "
      f"Rp {(np.clip(d, None, 0) * rec.price[:, None]).sum() / 1e6:,.1f}M removed.")
with pd.option_context("display.width", 160, "display.float_format", "{:,.0f}".format):
    for name, h in (("recorded demand", rec), ("uncensored demand", unc)):
        print(f"\nH2 2025, {name}")
        print(table(h).to_string())
