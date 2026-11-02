"""E2: lead-time variance factor 1 + rho_tier (w - 1).

rho per tier = lag-1 correlation of standardised weekly errors
z = (y - mean) / sqrt(max(var, mean)) between adjacent horizons (h, h+1), as in
the exploratory check (architecture/explore.py), fitted on selection
origins before 1 Jul 2024. Truth: uncensored weekly demand from the selection-cut
daily truth. Eid-split arm: separate rho for pairs whose later week starts within
-30..+28 days of an Eid; a scored window uses the Eid rho if it overlaps that band.
Tested on Jul-Oct 2024 selection origins (gate) and 2025 (diagnostic).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as C
from irpd import forecast as fc
from irpd.evalx import compare_arms
from irpd.evalx.scorer import weekly_from_daily
from rescore import eid_mask, factor, rescore

TIERS = ["fast", "regular", "slow", "rare"]
GATE = ["fast", "regular", "slow"]


def weekly_truth(kind):
    return weekly_from_daily(C.daily(kind))[["drug_code", "week_start", "uncensored_units"]]


def zframe(arm, origins, kind="selection"):
    p = C.preds(arm, origins)
    t = weekly_truth(kind)
    t["week_start"] = pd.to_datetime(t["week_start"]).dt.date
    x = p.merge(t, on=["drug_code", "week_start"], how="inner")
    x["z"] = (x["uncensored_units"] - x["mean"]) / np.sqrt(np.maximum(x["var"], x["mean"]).clip(lower=1e-9))
    return x


def in_eid(week_start):
    d = pd.to_datetime(week_start)
    hit = np.zeros(len(d), bool)
    for e in C.EID:
        off = (d - pd.Timestamp(e)).dt.days
        hit |= ((off >= -30) & (off <= 28)).to_numpy()
    return hit


def fit_rho(x, split=False):
    pv = x.pivot_table(index=["origin", "drug_code", "tier"], columns="h", values="z")
    wk = x.pivot_table(index=["origin", "drug_code", "tier"], columns="h", values="week_start",
                       aggfunc="first")
    out = {}
    for tier in TIERS:
        g = pv[pv.index.get_level_values("tier") == tier]
        gw = wk.loc[g.index]
        a = np.concatenate([g[k].to_numpy() for k in range(1, 8)])
        b = np.concatenate([g[k + 1].to_numpy() for k in range(1, 8)])
        e = np.concatenate([in_eid(gw[k + 1]) for k in range(1, 8)])
        ok = ~np.isnan(a) & ~np.isnan(b)
        if not split:
            out[tier] = max(float(np.corrcoef(a[ok], b[ok])[0, 1]), 0.0)
        else:
            for nm, m in (("eid", e), ("out", ~e)):
                k = ok & m
                out[f"{tier}_{nm}"] = max(float(np.corrcoef(a[k], b[k])[0, 1]), 0.0)
    return out


def split_scores(plain, rs):
    w = plain.windows
    tier = w["tier"].to_numpy(object)
    fl, flc = [], []
    for days in (w["lead"].to_numpy(), w["lead"].to_numpy() + 14):
        e = eid_mask(w["origin"], days, C.EID)
        r = np.array([rs.get(f"{t}_{'eid' if ei else 'out'}", 0.0) for t, ei in zip(tier, e)])
        (fl if not fl else flc).append(factor(days, r))
    return rescore(plain, fl[0], flc[0], "eid_split")


def coverage_by_length(arm, origins, kind, rho):
    """p95 coverage of 1/2/4/8-week totals from the origin, plain vs factor."""
    p = C.preds(arm, origins)
    t = weekly_truth(kind)
    x = p.merge(t, on=["drug_code", "week_start"], how="inner")
    rows = []
    for L in (1, 2, 4, 8):
        g = x[x["h"] <= L].groupby(["origin", "drug_code", "tier"]).agg(
            m=("mean", "sum"), v=("var", "sum"), y=("uncensored_units", "sum"), n=("h", "size"))
        g = g[g["n"] == L].reset_index()
        for meth in ("plain", "rho"):
            f = 1 + g["tier"].map(rho).fillna(0).to_numpy() * (L - 1) if meth == "rho" else 1.0
            v = np.maximum(g["v"].to_numpy() * f, g["m"].to_numpy())
            q = fc.demand_quantile(g["m"].to_numpy(), v, 0.95)
            g["cov"] = g["y"].to_numpy() <= q
            for tier, gg in g.groupby("tier"):
                rows.append({"arm": arm, "weeks": L, "method": meth, "tier": tier,
                             "n": len(gg), "cov": gg["cov"].mean()})
    return pd.DataFrame(rows)


def main():
    res, rhos, cmp_rows, cov_rows = [], {}, [], []
    for arm in ("ml_control", "cur_control"):
        x = zframe(arm, C.origins_for("fit"))
        rho, rs = fit_rho(x), fit_rho(x, split=True)
        rhos[arm] = {"single": rho, "eid_split": rs}
        for period in ("test", "diag"):
            plain = C.score(arm, period)
            single = C.score(arm, period, rho=rho)
            split = split_scores(plain, rs)
            # sanity: rescore with the single rho equals score_arm(rho=...)
            if period == "test":
                w = plain.windows
                t = w["tier"].map(rho).fillna(0).to_numpy()
                chk = rescore(plain, factor(w["lead"], t), factor(w["lead"] + 14, t))
                d = (chk.summary - single.summary).abs().max().max()
                print(arm, "rescore check max abs diff", d)
            for name, s in (("plain", plain), ("rho", single), ("eid_split", split)):
                sm = s.summary.loc["all"]
                for tier in ["fast", "regular", "slow", "rare", "pooled"]:
                    res.append({"arm": arm, "period": period, "variant": name, "tier": tier,
                                **sm.loc[tier, ["windows", "pinball_lc", "cov_l", "cov_lc",
                                                "crps_lc"]].to_dict()})
            for name, s in (("rho", single), ("eid_split", split)):
                c = compare_arms(s, plain, metrics=("pinball_lc",), subsets=("all",),
                                 tiers=("pooled", "fast", "regular", "slow"))
                c.insert(0, "variant", name); c.insert(0, "period", period); c.insert(0, "arm", arm)
                cmp_rows.append(c)
            c = compare_arms(split, single, metrics=("pinball_lc",), subsets=("all",),
                             tiers=("pooled",))
            c.insert(0, "variant", "eid_split_vs_rho"); c.insert(0, "period", period)
            c.insert(0, "arm", arm)
            cmp_rows.append(c)
            kind = "selection" if period == "test" else "full"
            cv = coverage_by_length(arm, C.origins_for(period), kind, rho)
            cv["period"] = period
            cov_rows.append(cv)
    res, cmp_, cov = pd.DataFrame(res), pd.concat(cmp_rows), pd.concat(cov_rows)
    res.to_csv(C.OUT / "e2_scores.csv", index=False)
    cmp_.to_csv(C.OUT / "e2_compare.csv", index=False)
    cov.to_csv(C.OUT / "e2_coverage_by_length.csv", index=False)
    (C.OUT / "e2_rho.json").write_text(json.dumps(rhos, indent=1))

    # decision rule
    verdict = {}
    for arm in ("ml_control", "cur_control"):
        r = res[res.arm == arm]
        out = {}
        for var in ("rho", "eid_split"):
            t = r[(r.period == "test") & (r.variant == var) & r.tier.isin(GATE)]
            d = r[(r.period == "diag") & (r.variant == var) & r.tier.isin(GATE)]
            cov_ok = bool(((t[["cov_l", "cov_lc"]] >= 0.935) & (t[["cov_l", "cov_lc"]] <= 0.965)).all().all())
            c = cmp_[(cmp_.arm == arm) & (cmp_.variant == var) & (cmp_.tier == "pooled")]
            ni_test = bool(c[c.period == "test"]["non_inferior"].iloc[0])
            ni_diag = bool(c[c.period == "diag"]["non_inferior"].iloc[0])
            d_ok = bool((d[["cov_l", "cov_lc"]] >= 0.92).all().all())
            out[var] = {"cov_in_band_test": cov_ok, "pinball_NI_test": ni_test,
                        "cov_ge_0.92_2025": d_ok, "pinball_NI_2025": ni_diag,
                        "passes": cov_ok and ni_test and d_ok}
        p = r[(r.period == "test") & (r.variant == "plain") & r.tier.isin(GATE)]
        out["plain_cov_in_band_test"] = bool(((p[["cov_l", "cov_lc"]] >= 0.935) & (p[["cov_l", "cov_lc"]] <= 0.965)).all().all())
        verdict[arm] = out
    (C.OUT / "e2_verdict.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(rhos, indent=1))
    print(res.round(4).to_string())
    print(cmp_[["arm", "period", "variant", "tier", "a", "b", "rel", "rel_lo", "rel_hi", "non_inferior"]].round(4).to_string())
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
