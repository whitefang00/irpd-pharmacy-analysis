"""Does the right forecast depend on h? On the selection origins (2024), regress the
realized week-h units (relative to the 52-week mean) on the recent shift (mean of the
last 4 observed weeks / 52-week mean - 1), separately per h and tier, for actuals and
for the ML_A2 forecast. If the realized slope barely changes with h, a flat path is the
right shape (outside calendar events). Non-Eid target weeks only. Selection data only."""
import os
import warnings
from datetime import date, timedelta
import duckdb
import numpy as np
import pandas as pd
from irpd import forecast as fc

warnings.simplefilter("ignore")
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402

DATA = fp.DATA
OUT = fp.out("diagnostics")
d = pd.read_parquet(f"{OUT}/selection_predictions_combined.parquet")
d = d[(d.model == "ml_A2") & ~d.eid & d.actual.notna()]
con = duckdb.connect(f"{DATA}/warehouse/selection.duckdb", read_only=True)
hist = fc.load_history(con, date(2024, 12, 8))
y = np.where(hist["out"], np.nan, hist["y"])
weeks = list(hist["weeks"])
parts = []
for o in sorted(d.origin.unique()):
    od = pd.Timestamp(o).date()
    n = sum(w + timedelta(days=6) <= od for w in weeks)
    parts.append(pd.DataFrame({"drug_code": hist["drugs"], "origin": pd.Timestamp(o),
                               "m4": np.nanmean(y[:, n - 4:n], 1), "m52": np.nanmean(y[:, n - 52:n], 1)}))
z = d.merge(pd.concat(parts), on=["drug_code", "origin"])
z = z[(z.m52 > 0) & z.m4.notna()]
z["x"] = np.clip(z.m4 / z.m52 - 1, -1, 3)
rows = []
for (t, h), g in z.groupby(["tier", "h"], observed=True):
    rows.append({"tier": t, "h": h, "n": len(g),
                 "slope_actual": np.polyfit(g.x, g.actual / g.m52 - 1, 1)[0],
                 "slope_ml": np.polyfit(g.x, g["mean"] / g.m52 - 1, 1)[0]})
r = pd.DataFrame(rows)
print(r.pivot(index="tier", columns="h", values="slope_actual").round(3).to_string())
print(r.pivot(index="tier", columns="h", values="slope_ml").round(3).to_string())
r.to_csv(f"{OUT}/h_dependence_slopes.csv", index=False)
