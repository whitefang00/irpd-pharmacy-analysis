"""E6: ML vs current vs equal-weight ensemble under the new harness.

No E2-E5 change was adopted by its rule (see eda/followup/experiments.md), so the arms are the
controls: ml_control, cur_control, and ens = 0.5 (ML + current) for the mean and
0.5 (var_ML + var_cur) for the variance (a plain choice; the experiment plan fixes no recipe).

Forecast part: selection origins (gate) and 2025 (diagnostic), common drug set.
Replay part: rolling 14-day, H2 2024 (starts 8 and 22 Jul 2024), H1 2025 diagnostic
(starts 6 and 20 Jan 2025). Reference = cur_control matched to the pharmacist's stock;
ML and ens matched to the reference's stock. ml_censor is also run in H2 2024 (day-1
start) for E4's reported replay on censored drugs.

Run from the repo root:  uv run python eda/followup/scripts/results/e6.py [score] [replay]
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import timedelta

import numpy as np
import pandas as pd

import common as C
import replay_util as U
from irpd.evalx import compare_arms
from irpd.evalx import rolling as R

MARGIN = 0.075
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


def ens_frame() -> pd.DataFrame:
    a = C.preds("ml_control", C.grid())
    b = C.preds("cur_control", C.grid())
    x = a.merge(b[["origin", "drug_code", "h", "mean", "var"]], on=["origin", "drug_code", "h"],
                suffixes=("", "_c"))
    x["mean"] = 0.5 * (x["mean"] + x["mean_c"])
    x["var"] = 0.5 * (x["var"] + x["var_c"])
    return x[["origin", "drug_code", "h", "week_start", "mean", "var", "tier"]]


def score_part():
    ens = ens_frame()
    ens.to_parquet(C.OUT / "ens.parquet")
    rows, cmps = [], []
    for period in ("sel", "diag"):
        s = {"ml": C.score("ml_control", period), "cur": C.score("cur_control", period),
             "ens": C.score("ens", period, df=ens, tag="ens")}
        for name, sc in s.items():
            for sub in ("all", "clean"):
                sm = sc.summary.loc[sub]
                for tier in ("fast", "regular", "slow", "rare", "pooled"):
                    rows.append({"period": period, "arm": name, "subset": sub, "tier": tier,
                                 **sm.loc[tier, ["windows", "pinball_lc", "crps_lc", "cov_lc",
                                                 "bias4", "bias8", "rwape4", "wape"]].to_dict()})
        for a, b in (("ml", "cur"), ("ens", "cur"), ("ens", "ml")):
            c = compare_arms(s[a], s[b], metrics=("pinball_lc", "crps_lc", "rwape4", "wape"),
                             subsets=("all",), tiers=("pooled", "fast", "regular", "slow", "rare"))
            c.insert(0, "ref", b); c.insert(0, "arm", a); c.insert(0, "period", period)
            cmps.append(c)
    r, c = pd.DataFrame(rows), pd.concat(cmps)
    r.to_csv(C.OUT / "e6_scores.csv", index=False)
    c.to_csv(C.OUT / "e6_compare.csv", index=False)
    print(r[r.subset == "all"].round(4).to_string())
    print(c[c.metric == "pinball_lc"][["period", "arm", "ref", "tier", "a", "b", "rel", "rel_lo",
                                         "rel_hi", "non_inferior", "superior"]].round(4).to_string())


def replay_part():
    ens = pd.read_parquet(C.OUT / "ens.parquet")
    srcs = {"cur": R.ArmSource(C.raw("cur_control"), "cur"), "ml": R.ArmSource(C.raw("ml_control"), "ml"),
            "ens": R.ArmSource(ens, "ens"), "ml_censor": R.ArmSource(C.raw("ml_censor"), "ml_censor")}
    res, outs = [], {}
    for win in ("H2-2024", "H1-2025"):
        s0 = R.first_grid_monday(R.WINDOWS[win][0])
        for st in (s0, s0 + timedelta(days=14)):
            w = R.load_window(win, st)
            ref = R.match(U.make_plan_hook(srcs["cur"], w), w)
            outs[(win, str(st), "cur")] = ref
            log(win, st, "cur", ref.k, ref.summary["net_out"])
            arms = ["ml", "ens"] + (["ml_censor"] if (win == "H2-2024" and st == s0) else [])
            for a in arms:
                o = R.match(U.make_plan_hook(srcs[a], w), w, target=ref.summary["stock_value"])
                outs[(win, str(st), a)] = o
                for metric in ("net_out", "lost_net", "lost_gross"):
                    cr = R.compare_replay(o, ref, metric)
                    res.append({"window": win, "start": str(st), "arm": a, "ref": "cur",
                                "k": o.k, "k_ref": ref.k, **cr,
                                "stockout_days": o.summary["stockout_days"],
                                "stockout_days_ref": ref.summary["stockout_days"],
                                "fill_rate": o.summary["fill_rate"],
                                "fill_rate_ref": ref.summary["fill_rate"],
                                "cycle_service": o.summary["cycle_service"],
                                "cycle_service_ref": ref.summary["cycle_service"]})
                log(win, st, a, o.k, o.summary["net_out"], res[-3]["ratio"])
    r = pd.DataFrame(res)
    r.to_csv(C.OUT / "e6_replay.csv", index=False)
    with open(C.OUT / "e6_replay_outcomes.pkl", "wb") as f:
        pickle.dump({k: (v.k, v.summary, v.periods) for k, v in outs.items()}, f)
    # E4 reported replay: censored drugs (labelled censored at any re-plan origin), ml_censor vs ml
    k = ("H2-2024", str(R.first_grid_monday(R.WINDOWS["H2-2024"][0])))
    w = R.load_window("H2-2024", R.first_grid_monday(R.WINDOWS["H2-2024"][0]))
    days = [w.h.days[t] - timedelta(days=1) for t in R.replan_days(w, 14)]
    cen = set()
    for d in days:
        lb = U.labels_at(d)
        cen |= set(lb.index[lb["censored"]])
    rows = []
    for a in ("cur", "ml", "ml_censor"):
        pdd = U.per_drug(outs[k + (a,)])
        x = pdd.loc[pdd.index.isin(cen)]
        rows.append({"arm": a, "censored_drugs": len(x), "net_out": x["net_out"].sum(),
                     "lost_net": x["lost_net"].sum(), "stock_value": x["stock_value"].sum(),
                     "stockout_days": x["stockout_days"].sum()})
    pd.DataFrame(rows).to_csv(C.OUT / "e4_replay_censored_drugs.csv", index=False)
    print(pd.DataFrame(rows).round(1).to_string())
    print(r[r.metric == "net_out"].round(3).to_string())


def verdict():
    c = pd.read_csv(C.OUT / "e6_compare.csv")
    s = pd.read_csv(C.OUT / "e6_scores.csv")
    rp = pd.read_csv(C.OUT / "e6_replay.csv") if (C.OUT / "e6_replay.csv").exists() else None
    sel = c[(c.period == "sel") & (c.metric == "pinball_lc")]
    out = {}
    m = sel[(sel.arm == "ml") & (sel.ref == "cur")].set_index("tier")
    b = s[(s.period == "sel") & (s.arm == "ml") & (s.subset == "all")].set_index("tier")
    bc = s[(s.period == "sel") & (s.arm == "ml") & (s.subset == "clean")].set_index("tier")
    v = {"pooled_rel": m.loc["pooled", "rel"], "pooled_rel_hi": m.loc["pooled", "rel_hi"],
         "pooled_superior": bool(m.loc["pooled", "superior"]),
         "tiers_NI": {t: bool(m.loc[t, "non_inferior"]) for t in ("fast", "regular", "slow")},
         "bias": {t: {"bias4": b.loc[t, "bias4"], "bias8": b.loc[t, "bias8"]} for t in ("fast", "regular")},
         "bias_clean": {t: {"bias4": bc.loc[t, "bias4"], "bias8": bc.loc[t, "bias8"]} for t in ("fast", "regular")}}
    v["bias_ok"] = all(0.97 <= x <= 1.03 for t in v["bias"].values() for x in t.values())
    if rp is not None:
        x = rp[(rp.arm == "ml") & (rp.metric == "net_out")].groupby("window")["ratio"].mean()
        v["replay_net_out_ratio"] = x.to_dict()
        v["replay_NI_H2_2024"] = bool(x.get("H2-2024", np.inf) <= 1 + MARGIN)
        v["replay_NI_all_windows"] = bool((x <= 1 + MARGIN).all())
    v["passes"] = bool(v["pooled_superior"] and all(v["tiers_NI"].values()) and v["bias_ok"]
                       and v.get("replay_NI_H2_2024", False))
    out["ml_vs_cur"] = v
    e = sel[(sel.arm == "ens") & (sel.ref == "ml")].set_index("tier")
    out["ens_vs_ml"] = {"pooled_rel": e.loc["pooled", "rel"], "pooled_rel_hi": e.loc["pooled", "rel_hi"],
                        "superior": bool(e.loc["pooled", "superior"])}
    (C.OUT / "e6_verdict.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:] or ["score", "replay", "verdict"]
    if "score" in a:
        score_part()
    if "replay" in a:
        replay_part()
    if "verdict" in a:
        verdict()
