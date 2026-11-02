"""E3: Eid repair. Placement, Eid-week store error, pooled pinball/WAPE non-inferiority.

Store level = fast + regular + slow drugs of the common (origin, drug) set at each
origin (as in the forecast-arm placement check). Truth = weekly uncensored units
(actuals.parquet) summed over the same drugs. Eid weeks -4..+3 are counted from the
Monday week holding Eid day (8 Apr 2024, 31 Mar 2025). Peak week = the week with
the largest store total among the origin's 8 target weeks.
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

import common as C
from irpd.evalx import compare_arms

ARMS = ["cur_control", "cur_eid", "ml_control", "ml_eid", "ml_eid_censor", "ml_censor",
        "ml_eid_offset"]
PEAK = {2024: date(2024, 4, 8), 2025: date(2025, 3, 24)}
EID_WEEK = {2024: date(2024, 4, 8), 2025: date(2025, 3, 31)}
PAIRS = [("cur_eid", "cur_control", "pre-registered"), ("ml_eid", "ml_control", "pre-registered"),
         ("ml_eid_censor", "ml_censor", "pre-registered"),
         ("ml_eid_offset", "ml_control", "POST HOC")]


def store_paths() -> pd.DataFrame:
    """Store-level weekly forecast and truth per (arm, origin, week), all grid origins."""
    a = C.actuals()[["drug_code", "week_start", "uncensored_units", "units"]]
    rows = []
    for arm in ARMS:
        p = C.preds(arm, C.grid())
        p = p[p["tier"].isin(["fast", "regular", "slow"])]
        x = p.merge(a, on=["drug_code", "week_start"], how="left")
        g = x.groupby(["origin", "h", "week_start"]).agg(
            fc=("mean", "sum"), var=("var", "sum"), actual=("uncensored_units", "sum"),
            recorded=("units", "sum"), n=("uncensored_units", "count")).reset_index()
        g["arm"] = arm
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def main():
    sp = store_paths()
    sp.to_parquet(C.OUT / "store_paths.parquet")
    place, err = [], []
    for yr, peak in PEAK.items():
        origins = [o for o in C.grid() if peak - timedelta(days=56) < o < peak]
        ew = EID_WEEK[yr]
        eid_weeks = {ew + timedelta(days=7 * k): k for k in range(-4, 4)}
        for o in origins:
            r = {"year": yr, "origin": o}
            for arm in ARMS:
                g = sp[(sp.arm == arm) & (sp.origin == o)].set_index("week_start")
                act_peak = g["actual"].idxmax()
                fpk = g["fc"].idxmax()
                r["actual_peak"] = act_peak
                r[arm] = fpk
                r[f"{arm}_ok"] = fpk == act_peak
                r[f"{arm}_ratio"] = g.loc[fpk, "fc"] / g.loc[act_peak, "actual"]
                e = g[g.index.isin(eid_weeks)]
                for wk, row in e.iterrows():
                    err.append({"year": yr, "origin": o, "arm": arm, "week": wk,
                                "rel_week": eid_weeks[wk], "fc": row["fc"],
                                "actual": row["actual"]})
            place.append(r)
    place, err = pd.DataFrame(place), pd.DataFrame(err)
    place.to_csv(C.OUT / "e3_placement.csv", index=False)
    err.to_csv(C.OUT / "e3_eid_week_errors.csv", index=False)
    err["ae"] = (err["fc"] - err["actual"]).abs()
    ape = (err.groupby(["arm", "year"]).apply(lambda d: d["ae"].sum() / d["actual"].sum(),
                                              include_groups=False).unstack())
    ape.to_csv(C.OUT / "e3_eid_week_ape.csv")
    hits = place.groupby("year")[[f"{a}_ok" for a in ARMS]].sum().T
    hits.columns = [f"right_{c}" for c in hits.columns]

    cmps = []
    for period in ("sel", "diag"):
        for a, b, kind in PAIRS:
            c = compare_arms(C.score(a, period), C.score(b, period),
                             metrics=("pinball_lc", "wape", "rwape4"), subsets=("all",),
                             tiers=("pooled", "fast", "regular", "slow"))
            c.insert(0, "kind", kind); c.insert(0, "ref", b); c.insert(0, "arm", a)
            c.insert(0, "period", period)
            cmps.append(c)
    cmp_ = pd.concat(cmps)
    cmp_.to_csv(C.OUT / "e3_compare.csv", index=False)

    verdict = {}
    for a, b, kind in PAIRS:
        h = place.groupby("year")[f"{a}_ok"].sum()
        n = place.groupby("year").size()
        c = cmp_[(cmp_.period == "sel") & (cmp_.arm == a) & (cmp_.tier == "pooled")]
        ni = {m: bool(c[c.metric == m]["non_inferior"].iloc[0]) for m in ("pinball_lc", "wape")}
        lower = {int(y): bool(ape.loc[a, y] < ape.loc[b, y]) for y in ape.columns}
        verdict[a] = {"kind": kind, "ref": b, "unit_test_c": True,
                      "placed_2024": f"{int(h[2024])}/{int(n[2024])}",
                      "placed_2025": f"{int(h[2025])}/{int(n[2025])}",
                      "placement_ok": bool(h[2024] >= 3 and h[2025] > n[2025] / 2),
                      "eid_ape_lower_both_years": all(lower.values()), "ape_lower": lower,
                      "pinball_NI_sel": ni["pinball_lc"], "wape_NI_sel": ni["wape"]}
        v = verdict[a]
        v["passes"] = (v["placement_ok"] and v["eid_ape_lower_both_years"]
                       and v["pinball_NI_sel"] and v["wape_NI_sel"])
    (C.OUT / "e3_verdict.json").write_text(json.dumps(verdict, indent=1, default=str))
    print(place[["year", "origin", "actual_peak"] + ARMS].to_string())
    print(place[["year", "origin"] + [f"{a}_ratio" for a in ARMS]].round(3).to_string())
    print(hits)
    print(ape.round(4))
    print(cmp_[cmp_.tier == "pooled"][["period", "arm", "ref", "metric", "a", "b", "rel", "rel_lo",
                                        "rel_hi", "non_inferior"]].round(4).to_string())
    print(json.dumps(verdict, indent=1, default=str))


if __name__ == "__main__":
    main()
