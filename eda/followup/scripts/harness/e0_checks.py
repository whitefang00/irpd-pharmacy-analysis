"""E0 checks 2-3: A/A test and power calibration (definitions in eda/followup/experiment-notes.md).

Needs build_data.py first.
Run from the repo root:  uv run python eda/followup/scripts/harness/e0_checks.py
"""
import json
import pickle
import sys
import time
from datetime import timedelta
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from irpd.evalx import bootstrap, compare_arms, rolling as R, score_arm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402
fp.use_cache("harness")

OUT = D = fp.out("harness")
SIGMAS, SEEDS = (0.1, 0.25), range(10)
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


# ---------------- replay side ----------------
W = R.load_window("H2-2024")
SRC = R.CurrentSource()
for t in range(0, len(W.h.days), 1):  # every day: covers the phase-offset A/A too
    SRC.basis(W.h.days[t] - timedelta(days=1)) if t % 14 <= 10 else None
log("bases ready")
CODES = sorted(set().union(*[set(SRC.basis(W.h.days[t] - timedelta(days=1)).facts.index)
                             for t in range(0, len(W.h.days), 14)]))
PLAN_A = R.make_plan(SRC, W, 14)
A = R.match(PLAN_A, W)
log("A matched", A.k, A.summary["stock_value"], W.target)


def job(spec):
    kind, x = spec
    if kind == "aa_bracket":
        return spec, R.match(PLAN_A, W, target=W.target, hi0=x)
    if kind == "aa_offset":
        return spec, R.match(R.make_plan(SRC, W, 14, offset=x), W, target=A.summary["stock_value"])
    if kind == "noise":
        sigma, seed = x
        src = R.Perturbed(SRC, R.noise_factors(CODES, sigma, 1000 + seed), f"s{sigma}_{seed}")
    else:  # scale
        src = R.Perturbed(SRC, x, f"x{x}")
    return spec, R.match(R.make_plan(src, W, 14), W, target=A.summary["stock_value"])


specs = ([("aa_bracket", h) for h in (1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 3.0)]
         + [("aa_offset", o) for o in range(1, 11)]
         + [("noise", (s, k)) for s in SIGMAS for k in SEEDS] + [("scale", 0.9)])
with get_context("fork").Pool(8) as pool:
    res = dict(pool.map(job, specs))
log("replay runs done")

rows = []
for spec, o in res.items():
    for metric in ("net_out", "lost_net", "gross_out"):
        if metric == "lost_net":
            for x in (o, A):
                x.periods["lost_net"] = x.periods["net_out"] + x.periods["net_shelf"]
        c2 = R.compare_replay(o, A, metric)
        c1 = R.compare_replay(o, A, metric, drugs_only=True)
        rows.append({"kind": spec[0], "param": str(spec[1]), "metric": metric, "k": o.k,
                     **c2, "excl_zero": c2["diff_lo"] > 0 or c2["diff_hi"] < 0,
                     "ratio_lo_drug": c1["ratio_lo"], "ratio_hi_drug": c1["ratio_hi"],
                     "excl_zero_drug": c1["diff_lo"] > 0 or c1["diff_hi"] < 0})
rep = pd.DataFrame(rows)
rep.to_csv(OUT / "e0_replay_checks.csv", index=False)
pickle.dump({"A": A.summary, "runs": {str(k): v.summary for k, v in res.items()}},
            open(OUT / "e0_replay_summaries.pkl", "wb"))

# ---------------- forecast-score side ----------------
p = pd.read_parquet(D / "cur_harness.parquet")
lb = pd.read_parquet(D / "cur_harness_labels.parquet")
dl = pd.read_parquet(D / "daily_truth_selection.parquet")
clean = score_arm(p, labels=lb, daily=dl)
codes = sorted(p.drug_code.unique())
frows = []
arms = [("noise", (s, k)) for s in SIGMAS for k in SEEDS] + [("scale", 0.9)]
for kind, x in arms:
    e = (R.noise_factors(codes, x[0], 2000 + x[1]) if kind == "noise"
         else pd.Series(x, index=codes))
    f = e.reindex(p.drug_code).to_numpy()
    s = score_arm(p.assign(mean=p["mean"] * f, var=p["var"] * f), labels=lb, daily=dl)
    for drugs_only in (False, True):
        c = compare_arms(s, clean, metrics=["pinball_lc", "crps_lc", "rwape4", "wape", "cov_lc"],
                         tiers=["pooled", "fast", "regular", "slow"], subsets=["all"],
                         drugs_only=drugs_only)
        frows.append(c.assign(kind=kind, param=str(x), boot="drug-only" if drugs_only else "two-way"))
    log("scored", kind, x)
fc_ = pd.concat(frows, ignore_index=True)
fc_.to_csv(OUT / "e0_forecast_power.csv", index=False)
clean.summary.to_csv(OUT / "e0_current_selection_scores.csv")

# ---------------- summaries ----------------
summ = {}
aa = rep[(rep.metric == "net_out") & rep.kind.str.startswith("aa")]
summ["aa"] = aa.groupby("kind").agg(excl=("excl_zero", "sum"), n=("excl_zero", "size"),
                                    ratio_min=("ratio", "min"), ratio_max=("ratio", "max"),
                                    k_min=("k", "min"), k_max=("k", "max")).to_dict("index")
nz = rep[rep.kind.isin(["noise", "scale"])].copy()
nz["sigma"] = nz.param.map(lambda s: s.strip("()").split(",")[0])
nz["absdev"] = (nz.ratio - 1).abs()
summ["replay_power"] = nz.groupby(["metric", "kind", "sigma"]).agg(
    ratio_mean=("ratio", "mean"), ratio_min=("ratio", "min"), ratio_max=("ratio", "max"),
    p80_absdev=("absdev", lambda v: float(np.quantile(v, 0.8))),
    flagged=("excl_zero", "sum"), flagged_drug=("excl_zero_drug", "sum"),
    width=("ratio_hi", lambda v: 0), n=("ratio", "size")).reset_index().to_dict("records")
margin = float(np.quantile(nz[(nz.metric == "net_out") & (nz.kind == "noise")
                              & (nz.sigma == "0.1")].absdev, 0.8))
summ["replay_ni_margin"] = margin
fp = fc_[(fc_.metric == "pinball_lc")]
summ["forecast_power"] = fp.assign(sigma=fp.param.map(lambda s: s.strip("()").split(",")[0])) \
    .groupby(["boot", "kind", "sigma", "tier"]).agg(
        rel_mean=("rel", "mean"), worse=("worse", "sum"), ni_fail=("non_inferior", lambda v: int((~v.astype(bool)).sum())),
        width=("rel_hi", "mean"), n=("rel", "size")).reset_index().to_dict("records")
json.dump(summ, open(OUT / "e0_summary.json", "w"), indent=1, default=str)
log("margin", margin)
print(json.dumps(summ, indent=1, default=str)[:6000])
