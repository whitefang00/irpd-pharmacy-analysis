"""Local elasticity of lost revenue to stock value around the matched scale.

Matching stops when stock value is within 0.5% of the target, so two forecasters can
end up holding slightly different stock. This measures how much lost revenue that
slack alone moves. Current forecaster, static plain policy.
Needs sensitivity.py (both windows) first.
Run from the repo root:  uv run python eda/followup/scripts/inventory/stock_elasticity.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from irpd import contest, replay

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402

OUT = fp.out("inventory")
rows = []
for label in ("H2-2024", "H1-2025"):
    h, b = pickle.loads((OUT / f"cache_{label}.pkl").read_bytes())
    pol = contest.to_policy({}, b)
    _, k0 = replay.match_stock_value(pol, h)
    for f in (0.97, 0.99, 1.0, 1.01, 1.03):
        r = replay.simulate(replay.scaled(pol, k0 * f), h)
        s = replay.summarise(replay.metrics(r.floor, r.store, r.sold, h))
        rows.append({"window": label, "k": k0 * f, "stock": s.avg_stock_value,
                     "lost": s.lost_revenue, "stockout_days": s.stockout_days})
df = pd.DataFrame(rows)
for label, g in df.groupby("window"):
    g = g.sort_values("k")
    e = np.polyfit(np.log(g.stock), np.log(g.lost), 1)[0]
    print(label, "elasticity d ln(lost)/d ln(stock) =", round(e, 2))
    print(g.to_string(index=False))
df.to_csv(OUT / "stock_elasticity.csv", index=False)
