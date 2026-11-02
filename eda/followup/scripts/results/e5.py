"""E5: lifecycle rules. Fading -> TSB (+ stop flag); sporadic -> min-stock (s = 0, S = max single sale in 52 weeks).

Part 1 (forecast, selection + 2025 diagnostic): bias of 4- and 8-week totals on drugs
labelled fading / sporadic at the origin, base vs base with fading drugs' mean and var
replaced by the TSB arm. Sporadic min-stock is a policy rule, so its forecast bias is
the base's (reported, not gated).

Part 2 (rolling replay, H2 2024, 14-day re-plan, starts 8 Jul and 22 Jul 2024 = the
first G Monday and +14): reference = base arm matched to the pharmacist's stock value;
each rule is run at the reference's k (same store-wide scale), so stock freed on the
affected drugs is read at the reference's pooled lost revenue, and the pooled loss
ratio is checked against the E0 margin (7.5%). Affected drugs = drugs in the replay
labelled fading (resp. sporadic) at any re-plan origin.

Run from the repo root:  uv run python eda/followup/scripts/results/e5.py [bases...]
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
from irpd.evalx.scorer import weekly_from_daily

MARGIN = 0.075
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


def fading_frame(base: str, origins=None) -> pd.DataFrame:
    """Base arm with fading-labelled drugs' mean/var replaced by TSB (all origins)."""
    b = C.raw(base)
    t = C.raw("tsb")
    lb = C.labels()
    fk = lb[lb["fading"]][["origin", "drug_code"]]
    tf = t.merge(fk, on=["origin", "drug_code"])
    key = pd.MultiIndex.from_frame(b[["origin", "drug_code"]])
    drop = key.isin(pd.MultiIndex.from_frame(fk))
    cols = ["origin", "drug_code", "h", "week_start", "mean", "var", "tier"]
    return pd.concat([b.loc[~drop, cols], tf[cols]], ignore_index=True)


def bias_part():
    rows, cmps = [], []
    for base in ("ml_control", "cur_control"):
        ff = fading_frame(base)
        for period in ("sel", "diag"):
            kind = "selection" if period == "sel" else "full"
            wk = weekly_from_daily(C.daily(kind))
            wk["week_start"] = pd.to_datetime(wk["week_start"]).dt.date
            lb = C.labels()[["origin", "drug_code", "fading", "sporadic"]]
            for variant, df in (("base", None), ("fading->tsb", ff)):
                p = C.preds(base, C.origins_for(period), df)
                x = p.merge(wk, on=["drug_code", "week_start"])
                for L in (4, 8):
                    g = x[x.h <= L].groupby(["origin", "drug_code"]).agg(
                        f=("mean", "sum"), y=("uncensored_units", "sum"),
                        out=("censored", "sum"), n=("h", "size")).reset_index()
                    g = g[g.n == L].merge(lb, on=["origin", "drug_code"])
                    for seg in ("fading", "sporadic"):
                        s = g[g[seg]]
                        rows.append({"base": base, "period": period, "variant": variant,
                                     "segment": seg, "weeks": L, "windows": len(s),
                                     "drugs": s.drug_code.nunique(),
                                     "forecast": s.f.sum(), "truth": s.y.sum(),
                                     "bias": s.f.sum() / s.y.sum() if s.y.sum() else np.nan,
                                     "share_windows_all_out": (s.out == L).mean()})
            c = compare_arms(C.score(base, period, df=ff, tag=f"{base}_fadingtsb"),
                             C.score(base, period), metrics=("pinball_lc", "bias4"),
                             subsets=("all",), tiers=("pooled", "fast", "regular", "slow", "rare"))
            c.insert(0, "period", period); c.insert(0, "base", base)
            cmps.append(c)
    b, c = pd.DataFrame(rows), pd.concat(cmps)
    b.to_csv(C.OUT / "e5_bias.csv", index=False)
    c.to_csv(C.OUT / "e5_compare.csv", index=False)
    print(b.round(4).to_string())
    print(c[c.metric == "pinball_lc"][["base", "period", "tier", "a", "b", "rel", "rel_hi",
                                         "non_inferior"]].round(4).to_string())


# ---------------- replay ----------------

def sporadic_hook(pol, day):
    lb = U.labels_at(day)
    sp = set(lb.index[lb["sporadic"]])
    mx = U.max_single_sale(day)
    p = pol.set_index("drug_code")
    m = p.index.isin(sp) & (p["s"] >= 0)  # leave stopped drugs stopped
    S = mx.reindex(p.index[m]).fillna(1.0).clip(lower=1.0).to_numpy()
    p.loc[m, "s"] = 0.0
    p.loc[m, "S"] = S
    p.loc[m, "floor_max"] = np.minimum(p.loc[m, "floor_max"].to_numpy(float), S)
    p.loc[m, "floor_min"] = np.minimum(p.loc[m, "floor_min"].to_numpy(float), S)
    return p.reset_index()


def stop_hook(pol, day):
    t = C.raw("tsb")
    o = max(x for x in t["origin"].unique() if x <= day)
    lb = U.labels_at(day)
    st = t[(t.origin == o) & (t.h == 1) & t.stop_flag]["drug_code"]
    st = set(st) & set(lb.index[lb["fading"]])
    p = pol.set_index("drug_code")
    p.loc[p.index.isin(st), "s"] = -1.0
    return p.reset_index()


def affected(w, seg, start):
    days = [w.h.days[t] - timedelta(days=1) for t in R.replan_days(w, 14)]
    s = set()
    for d in days:
        lb = U.labels_at(d)
        s |= set(lb.index[lb[seg]])
    return sorted(s & set(w.h.drugs))


def replay_part(bases):
    res = []
    for base in bases:
        for start in (R.first_grid_monday(R.WINDOWS["H2-2024"][0]),):
            starts = [start, start + timedelta(days=14)]
            for st in starts:
                w = R.load_window("H2-2024", st)
                src = R.ArmSource(C.raw(base), base)
                p_ref = U.make_plan_hook(src, w)
                ref = R.match(p_ref, w)
                log(base, st, "ref k", round(ref.k, 3), ref.summary["lost_net"])
                srcf = R.ArmSource(fading_frame(base), base + "+fading")
                variants = {"fading": (U.make_plan_hook(srcf, w, hook=stop_hook), "fading"),
                            "sporadic": (U.make_plan_hook(src, w, hook=sporadic_hook), "sporadic")}
                pd_ref = U.per_drug(ref)
                for name, (p, seg) in variants.items():
                    out = R.run_at(p, w, ref.k)
                    pdv = U.per_drug(out)
                    aff = affected(w, seg, st)
                    a_ref, a_new = pd_ref.loc[aff], pdv.loc[aff]
                    rise = (a_new["lost_net"] - a_ref["lost_net"])
                    sv = R.compare_replay(out, ref, "lost_net")
                    so = R.compare_replay(out, ref, "net_out")
                    res.append({
                        "base": base, "start": str(st), "rule": name, "k": ref.k,
                        "affected_drugs": len(aff),
                        "aff_stock_ref": a_ref["stock_value"].sum(),
                        "aff_stock_new": a_new["stock_value"].sum(),
                        "aff_stock_freed": 1 - a_new["stock_value"].sum() / a_ref["stock_value"].sum(),
                        "total_stock_ref": ref.summary["stock_value"],
                        "total_stock_new": out.summary["stock_value"],
                        "lost_net_ref": ref.summary["lost_net"], "lost_net_new": out.summary["lost_net"],
                        "lost_net_ratio": sv["ratio"], "lost_net_ratio_lo": sv["ratio_lo"],
                        "lost_net_ratio_hi": sv["ratio_hi"],
                        "net_out_ratio": so["ratio"],
                        "aff_lost_net_ref": a_ref["lost_net"].sum(), "aff_lost_net_new": a_new["lost_net"].sum(),
                        "extra_stockout_days_per_drug": (a_new["stockout_days"].sum() - a_ref["stockout_days"].sum()) / len(aff),
                        "drugs_lost_rise_gt_50k": int((rise > 50_000).sum()),
                        "drugs_lost_rise_list": ";".join(rise[rise > 50_000].sort_values(ascending=False).index[:15]),
                    })
                    log(base, st, name, {k: res[-1][k] for k in ("affected_drugs", "aff_stock_freed",
                                                                  "lost_net_ratio", "extra_stockout_days_per_drug")})
    r = pd.DataFrame(res)
    r.to_csv(C.OUT / "e5_replay.csv", index=False)
    return r


def verdict(r):
    out = {}
    for (base, rule), g in r.groupby(["base", "rule"]):
        freed = g["aff_stock_freed"].mean()
        ratio = g["lost_net_ratio"].mean()
        days = g["extra_stockout_days_per_drug"].mean()
        out[f"{base}:{rule}"] = {"stock_freed_mean": freed, "lost_net_ratio_mean": ratio,
                                 "extra_stockout_days_per_drug": days,
                                 "freed_ge_5pct": bool(freed >= 0.05),
                                 "loss_NI_7.5pct": bool(ratio <= 1 + MARGIN),
                                 "days_le_2": bool(days <= 2),
                                 "passes": bool(freed >= 0.05 and ratio <= 1 + MARGIN and days <= 2)}
    (C.OUT / "e5_verdict.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    args = sys.argv[1:]
    if "bias" in args or not args:
        bias_part()
    bases = [a for a in args if a.endswith("_control")] or ["ml_control", "cur_control"]
    if "replay" in args or not args:
        r = replay_part(bases)
        verdict(r)
