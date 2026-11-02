"""E0 check 1: reproduce static 5.60 Rp M (project replay) and the earlier rolling-policy run's 4.33 Rp M
(inventory/rolling_policy.py), H2 2024.

Run from the repo root:  uv run python eda/followup/scripts/harness/repro.py
"""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd
from irpd import contest, replay
from irpd.evalx import rolling as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402
fp.use_cache("harness")

t0 = time.time()
w = R.load_window("H2-2024")
print("window", len(w.h.drugs), len(w.h.days), "target", w.target, f"{time.time()-t0:.0f}s", flush=True)
src = R.CurrentSource()
out = {}
# (a) project path: replay.simulate + match_stock_value (tol 0.5%) on the static policy
b0 = src.basis(w.first - pd.Timedelta(days=1))
pol = contest.to_policy({}, b0)
m, k = replay.match_stock_value(pol, w.h)
r = replay.simulate(m, w.h)
s = replay.summarise(replay.metrics(r.floor, r.store, r.sold, w.h))
out["project_static_tol0.5%"] = {"k": k, "lost": float(s.lost_revenue), "stock": float(s.avg_stock_value)}
print(out, flush=True)
# (b) harness static, tol 0.5% (rolling_policy.py's bisection) and 0.05%
for cad, name in ((None, "static"), (14, "rolling14")):
    p = R.make_plan(src, w, cad)
    for tol in (0.005, 0.0005):
        o = R.match(p, w, tol=tol)
        out[f"harness_{name}_tol{tol}"] = {"k": o.k, "lost": o.summary["lost_gross"],
            "out": o.summary["gross_out"], "shelf": o.summary["gross_shelf"],
            "net": o.summary["lost_net"], "stock": o.summary["stock_value"],
            "stockout_days": o.summary["stockout_days"]}
        print(name, tol, out[f"harness_{name}_tol{tol}"], f"{time.time()-t0:.0f}s", flush=True)
# static via simulate_tv at the project's k must equal replay.simulate
p = R.make_plan(src, w, None)
rr = replay.simulate_tv(p, w.h, k)
out["simulate_tv_equals_simulate"] = bool(np.allclose(rr.floor, r.floor) and np.allclose(rr.sold, r.sold))
json.dump(out, open(fp.out("harness") / "repro.json", "w"), indent=1)
print(json.dumps(out, indent=1))
