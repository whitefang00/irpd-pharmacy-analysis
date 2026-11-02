"""E4: censoring-aware training targets (ml_censor vs ml_control; ml_eid_censor vs ml_eid).

Bias = sum of 4-week forecast totals / sum of 4-week uncensored demand, on the
common (origin, drug) set. Segments are taken at the origin (labels.parquet):
`censored` (20%+ of the last 26 weeks with a total stock-out day) and the slow tier.
Truth, gross: daily uncensored units (lost.uncensored_units, as the scorer uses).
Truth, switch-netted: recorded units + lost units x (1 - the drug's stock-out switch
rate), with rates from spells before 30 Jun 2024 (selection) / 31 Dec 2024 (2025);
this is an approximation of the experiment plan's "switch-netted" version, reported only.
E2 adopted no variance factor, so pinball uses the plain recipe.
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

import common as C
from irpd.evalx import compare_arms
from irpd.evalx import rolling as R
from irpd.evalx.scorer import weekly_from_daily

ARMS = ["ml_control", "ml_censor", "ml_eid", "ml_eid_censor", "cur_control"]
PAIRS = [("ml_censor", "ml_control"), ("ml_eid_censor", "ml_eid")]


def totals4(arm, period):
    kind = "selection" if period == "sel" else "full"
    wk = weekly_from_daily(C.daily(kind))
    wk["week_start"] = pd.to_datetime(wk["week_start"]).dt.date
    p = C.preds(arm, C.origins_for(period))
    p = p[p["h"] <= 4].merge(wk, on=["drug_code", "week_start"], how="inner")
    g = p.groupby(["origin", "drug_code", "tier"]).agg(
        f4=("mean", "sum"), y4=("uncensored_units", "sum"), u4=("units", "sum"),
        n=("h", "size")).reset_index()
    g = g[g["n"] == 4]
    day = date(2024, 6, 30) if period == "sel" else date(2024, 12, 31)
    codes = sorted(g["drug_code"].unique())
    sw, _ = R._cached(f"switch_{day}_{len(codes)}", lambda: R.switch_rates(day, codes))
    g["sw"] = g["drug_code"].map(dict(zip(codes, sw))).fillna(0)
    g["y4_net"] = g["u4"] + (g["y4"] - g["u4"]).clip(lower=0) * (1 - g["sw"])
    lb = C.labels()[["origin", "drug_code", "censored", "fading", "sporadic", "new"]]
    return g.merge(lb, on=["origin", "drug_code"], how="left")


def bias_table(period):
    rows = []
    for arm in ARMS:
        g = totals4(arm, period)
        segs = {"censored (all tiers)": g["censored"] == True,
                "censored, fast+regular+slow": (g["censored"] == True) & g["tier"].isin(["fast", "regular", "slow"]),
                "slow tier": g["tier"] == "slow", "fast": g["tier"] == "fast",
                "regular": g["tier"] == "regular", "rare": g["tier"] == "rare",
                "not censored": g["censored"] != True}
        for s, m in segs.items():
            x = g[m]
            rows.append({"period": period, "arm": arm, "segment": s, "windows": len(x),
                         "bias_gross": x["f4"].sum() / x["y4"].sum(),
                         "bias_net": x["f4"].sum() / x["y4_net"].sum(),
                         "bias_recorded": x["f4"].sum() / x["u4"].sum()})
    return pd.DataFrame(rows)


def main():
    b = pd.concat([bias_table("sel"), bias_table("diag")])
    b.to_csv(C.OUT / "e4_bias.csv", index=False)
    cmps = []
    for period in ("sel", "diag"):
        for a, r in PAIRS:
            c = compare_arms(C.score(a, period), C.score(r, period),
                             metrics=("pinball_lc", "bias4", "wape"), subsets=("all", "censored"),
                             tiers=("pooled", "fast", "regular", "slow", "rare"))
            c.insert(0, "ref", r); c.insert(0, "arm", a); c.insert(0, "period", period)
            cmps.append(c)
    cmp_ = pd.concat(cmps)
    cmp_.to_csv(C.OUT / "e4_compare.csv", index=False)
    verdict = {}
    for a, r in PAIRS:
        s = b[(b.period == "sel")].set_index(["arm", "segment"])["bias_gross"]
        cen = s[(a, "censored (all tiers)")]
        slow_new, slow_ref = s[(a, "slow tier")], s[(r, "slow tier")]
        c = cmp_[(cmp_.period == "sel") & (cmp_.arm == a) & (cmp_.metric == "pinball_lc")
                 & (cmp_.subset == "all") & (cmp_.tier == "pooled")]
        v = {"ref": r, "censored_bias": cen, "censored_bias_ref": s[(r, "censored (all tiers)")],
             "censored_in_band": bool(0.95 <= cen <= 1.05),
             "slow_bias": slow_new, "slow_bias_ref": slow_ref,
             "slow_not_further": bool(abs(slow_new - 1) <= abs(slow_ref - 1)),
             "pinball_rel": float(c["rel"].iloc[0]), "pinball_rel_hi": float(c["rel_hi"].iloc[0]),
             "pinball_NI": bool(c["non_inferior"].iloc[0])}
        v["passes"] = v["censored_in_band"] and v["slow_not_further"] and v["pinball_NI"]
        verdict[a] = v
    (C.OUT / "e4_verdict.json").write_text(json.dumps(verdict, indent=1))
    print(b.round(4).to_string())
    print(cmp_[(cmp_.metric == "pinball_lc")][["period", "arm", "subset", "tier", "a", "b", "rel",
                                                "rel_lo", "rel_hi", "non_inferior"]].round(4).to_string())
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
