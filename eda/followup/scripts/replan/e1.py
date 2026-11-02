"""E1: static vs 14-day vs 7-day re-plan, current forecaster, at the static policy's matched stock.

Windows H2 2024 (primary), H1 2025 and H2 2025 (diagnostic); start days: window day 1 and +7.
Each with and without the floor cap. Definitions: eda/followup/experiment-notes.md.
Run from the repo root:  uv run python eda/followup/scripts/replan/e1.py
Then harness/figs.py draws the E1 figures.
"""
import pickle
import sys
import time
from datetime import timedelta
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

from irpd.evalx import rolling as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402
fp.use_cache("harness")

OUT = fp.out("replan")
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


SRC = R.CurrentSource()
WIN = {}
for label, (first, last) in R.WINDOWS.items():
    for off in (0, 7):
        w = R.load_window(label, start=first + timedelta(days=off))
        WIN[(label, off)] = w
        for t in range(0, len(w.h.days), 7):
            SRC.basis(w.h.days[t] - timedelta(days=1))
        log("window", label, off, len(w.h.drugs), "drugs", f"target {w.target / 1e6:.1f}M")


def job(spec):
    label, off, cap = spec
    w = WIN[(label, off)]
    out = {}
    for cad in (None, 14, 7):
        p = R.make_plan(SRC, w, cad)
        front = pd.DataFrame([R.run_at(p, w, k, cap).summary for k in R.KS])
        target = None if cad is None else out[None]["matched"].summary["stock_value"]
        m = R.match(p, w, target=target, floor_cap=cap)
        out[cad] = {"frontier": front, "matched": m}
    return spec, out


specs = [(label, off, cap) for label in R.WINDOWS for off in (0, 7) for cap in (False, True)]
with get_context("fork").Pool(6) as pool:
    res = dict(pool.map(job, specs))
log("runs done")
pickle.dump(res, open(OUT / "e1_runs.pkl", "wb"))

rows, fronts = [], []
for (label, off, cap), out in res.items():
    st = out[None]["matched"]
    for cad, r in out.items():
        m = r["matched"]
        row = {"window": label, "start_offset": off, "floor_cap": cap,
               "cadence": "static" if cad is None else f"{cad}d", "k": m.k,
               **{c: m.summary[c] for c in ("stock_value", "lost_gross", "gross_out",
                                            "gross_shelf", "lost_net", "net_out", "net_shelf",
                                            "stockout_days", "shelfout_days", "fill_rate",
                                            "cycle_service")}}
        if cad is not None:
            for metric in ("lost_net", "net_out", "lost_gross"):
                c = R.compare_replay(m, st, metric)
                row[f"{metric}_ratio"] = c["ratio"]
                row[f"{metric}_lo"], row[f"{metric}_hi"] = c["ratio_lo"], c["ratio_hi"]
            if cad == 7:
                c = R.compare_replay(m, out[14]["matched"], "lost_net")
                row["vs14_ratio"], row["vs14_lo"], row["vs14_hi"] = c["ratio"], c["ratio_lo"], c["ratio_hi"]
        rows.append(row)
        f = r["frontier"].assign(window=label, start_offset=off, floor_cap=cap,
                                 cadence="static" if cad is None else f"{cad}d")
        fronts.append(f)
tab = pd.DataFrame(rows)
tab.to_csv(OUT / "e1_matched.csv", index=False)
pd.concat(fronts).to_csv(OUT / "e1_frontier.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.4g}".format):
    print(tab.drop(columns=["stock_value"]).to_string())
