"""Exploratory sizing checks for the architecture options. EXPLORATORY. No training; reads existing prediction files.

Sets:
- sel  : 11 selection origins 2024 (wave3 ML A2 3aafa0367da7 vs re-selected baseline-g1). Clean.
- ev25 : 9 eval origins Jan-Apr 2025 (frozen ML 36ad7c615b96, current eval-old, baseline-g1).
         DIAGNOSTIC: 2025 is already inspected.

Checks:
 A. Equal-weight ensembles: unit WAPE (h1-8), revenue-weighted WAPE of 4-week totals, bias,
    by tier and by demand segment (demand/segment.py); error correlation between members.
 B. Lead-time variance calibration on selection: fit on first 6 origins, test on last 5,
    (i) window factor 1+rho(w-1) with rho per tier, (ii) per-tier conformal multiplier on 4-wk totals.
 C. Power: per-origin spread of the WAPE difference; MDE in origins; drug-only vs two-way bootstrap.
 D. Ceiling for any store-level calendar (Eid) model: rescale every forecast so that each
    (origin, h) store total equals the actual store total (oracle), score WAPE in Eid / non-Eid weeks.
 E. Young drugs: bias at origin for drugs first sold < 26 weeks before the origin (selection).

Needs diagnostics/01_flatness_eval.py, diagnostics/06_selection_period.py and
demand/segment.py first.
Run from the repo root:  uv run python eda/followup/scripts/architecture/explore.py
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from irpd import forecast as fc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402

OUT = fp.out("architecture")
RNG = np.random.default_rng(7)
TIERS = ["fast", "regular", "slow", "rare"]

sel = pd.read_parquet(fp.out("diagnostics") / "selection_predictions_combined.parquet")
ev = pd.read_parquet(fp.out("diagnostics") / "eval_predictions_combined.parquet")
for d in (sel, ev):
    d["tier"] = d.tier.astype(str)
    d["model"] = d.model.astype(str)

con = duckdb.connect(str(fp.WAREHOUSE / "selection.duckdb"), read_only=True)
wd = con.execute("select drug_code, week_start, units, revenue from weekly_demand").fetchdf()
pr = wd.groupby("drug_code").agg(u=("units", "sum"), r=("revenue", "sum"))
price = (pr.r / pr.u.where(pr.u > 0)).rename("price")
first = wd[wd.units > 0].groupby("drug_code").week_start.min().rename("first_sale")
seg = pd.read_csv(fp.out("demand") / "segments.csv").set_index("drug_code").segment


def wide(d, models):
    """One row per (origin, drug, h) with a mean column per model; actual NaN = censored."""
    k = ["origin", "drug_code", "tier", "h", "eid"]
    w = d[d.model.isin(models)].pivot_table(index=k, columns="model", values="mean").reset_index()
    a = d[d.model == models[0]][k + ["actual"]]
    v = d[d.model.isin(models)].pivot_table(index=k, columns="model", values="var").reset_index()
    v = v.rename(columns={m: "var_" + m for m in models})
    w = w.merge(a, on=k).merge(v, on=k)
    w = w[w.tier != "none"]
    w["price"] = w.drug_code.map(price).fillna(price.median())
    w["segment"] = w.drug_code.map(seg).fillna("unlabelled")
    return w


def add_ens(w, members, name):
    w[name] = w[members].mean(axis=1)
    w["var_" + name] = w[["var_" + m for m in members]].mean(axis=1)


def scores(w, cols, by):
    o = w[w.actual.notna()]
    rows = []
    for key, g in o.groupby(by):
        r = {by: key, "rows": len(g), "units": g.actual.sum()}
        for c in cols:
            r[f"wape_{c}"] = (g[c] - g.actual).abs().sum() / g.actual.sum()
            r[f"bias_{c}"] = g[c].sum() / g.actual.sum()
        rows.append(r)
    # revenue-weighted WAPE of 4-week totals (weeks 1-4), uncensored windows only
    t = w[w.h <= 4].groupby(["origin", "drug_code"]).agg(
        tier=("tier", "first"), segment=("segment", "first"), price=("price", "first"),
        miss=("actual", lambda s: s.isna().any()), actual=("actual", "sum"),
        **({by: (by, "first")} if by not in ("tier", "segment") else {}),
        **{c: (c, "sum") for c in cols}).reset_index()
    t = t[~t.miss]
    rw = {}
    for key, g in t.groupby(by):
        rw[key] = {f"rw4_{c}": (g.price * (g[c] - g.actual).abs()).sum() / (g.price * g.actual).sum()
                   for c in cols}
    res = pd.DataFrame(rows)
    return res.merge(pd.DataFrame(rw).T.rename_axis(by).reset_index(), on=by)


def err_corr(w, a, b):
    o = w[w.actual.notna()]
    out = {}
    for tier, g in o.groupby("tier"):
        out[tier] = np.corrcoef(g[a] - g.actual, g[b] - g.actual)[0, 1]
    return out


lines = []
P = lambda *x: lines.append(" ".join(str(i) for i in x))

# ---------------- A. ensembles ----------------
ws = wide(sel, ["ml_A2", "baseline"])
add_ens(ws, ["ml_A2", "baseline"], "ens")
we = wide(ev, ["ml", "current", "baseline_reselected"])
add_ens(we, ["ml", "current"], "ens_ml_cur")
add_ens(we, ["ml", "baseline_reselected"], "ens_ml_g1")
add_ens(we, ["ml", "current", "baseline_reselected"], "ens3")

cs = ["ml_A2", "baseline", "ens"]
ce = ["ml", "current", "baseline_reselected", "ens_ml_cur", "ens_ml_g1", "ens3"]
A1 = scores(ws, cs, "tier"); A1.to_csv(OUT / "A_ensemble_selection_tier.csv", index=False)
A2 = scores(ws, cs, "segment"); A2.to_csv(OUT / "A_ensemble_selection_segment.csv", index=False)
A3 = scores(we, ce, "tier"); A3.to_csv(OUT / "A_ensemble_eval2025_tier_DIAG.csv", index=False)
A4 = scores(we, ce, "segment"); A4.to_csv(OUT / "A_ensemble_eval2025_segment_DIAG.csv", index=False)
ws["all"] = "all"; we["all"] = "all"
A5 = scores(ws, cs, "all"); A6 = scores(we, ce, "all")
P("A. error correlation ML vs baseline (selection):", err_corr(ws, "ml_A2", "baseline"))
P("A. error correlation ML vs current (eval25 diag):", err_corr(we, "ml", "current"))

# ---------------- B. lead-time variance calibration ----------------
def totals(w, m, wlen):
    x = w.copy()
    x["block"] = (x.h - 1) // wlen
    t = x.groupby(["origin", "drug_code", "block"]).agg(
        tier=("tier", "first"), mean=(m, "sum"), var=("var_" + m, "sum"),
        miss=("actual", lambda s: s.isna().any()), actual=("actual", "sum"), n=("h", "size")).reset_index()
    return t[(t.n == wlen) & ~t.miss]


def zcorr(w, m):
    o = w[w.actual.notna()].copy()
    o["z"] = (o.actual - o[m]) / np.sqrt(np.maximum(o["var_" + m], o[m]).clip(lower=1e-9))
    pv = o.pivot_table(index=["origin", "drug_code", "tier"], columns="h", values="z")
    rho = {}
    for tier, g in pv.groupby(level="tier"):
        a = np.concatenate([g[k].to_numpy() for k in range(1, 8)])
        b = np.concatenate([g[k + 1].to_numpy() for k in range(1, 8)])
        ok = ~np.isnan(a) & ~np.isnan(b)
        rho[tier] = np.corrcoef(a[ok], b[ok])[0, 1]
    return rho


def cover(t, mult):
    m = t["mean"].to_numpy(float)
    v = np.maximum(t["var"].to_numpy(float) * mult, m * 1.000001)
    y = t.actual.to_numpy(float)
    q = fc.demand_quantile(m, v, 0.95)
    pin = np.where(y >= q, 0.95 * (y - q), 0.05 * (q - y))
    return (y <= q).mean(), pin.sum() / y.sum()


origins = sorted(ws.origin.unique())
fit_o, test_o = origins[:6], origins[6:]
Brows = []
for m, w, tag in [("ml_A2", ws, "sel"), ("baseline", ws, "sel"), ("ens", ws, "sel"),
                  ("ml", we, "ev25_DIAG"), ("current", we, "ev25_DIAG"), ("ens_ml_cur", we, "ev25_DIAG")]:
    fit = w[w.origin.isin(fit_o)] if tag == "sel" else ws[ws.origin.isin(origins)]
    fm = m if tag == "sel" else {"ml": "ml_A2", "current": "baseline", "ens_ml_cur": "ens"}[m]
    rho = zcorr(fit, fm)
    # conformal multiplier: smallest k on grid giving 4-wk p95 coverage >= .95 on fit set
    grid = np.round(np.arange(1.0, 4.01, 0.05), 2)
    kconf = {}
    for tier in TIERS:
        tf = totals(fit[fit.tier == tier], fm, 4)
        kconf[tier] = next((k for k in grid if cover(tf, k)[0] >= 0.95), grid[-1])
    test = w[w.origin.isin(test_o)] if tag == "sel" else w
    for wlen in (1, 2, 4):
        for tier in TIERS:
            t = totals(test[test.tier == tier], m, wlen)
            if len(t) < 50:
                continue
            f_rho = 1 + max(rho[tier], 0) * (wlen - 1)
            f_conf = kconf[tier] ** ((wlen - 1) / 3) if wlen > 1 else 1.0  # interpolate to window
            for meth, k in [("recipe", 1.0), ("rho_factor", f_rho), ("conformal4", f_conf)]:
                c, p = cover(t, k)
                Brows.append({"set": tag, "model": m, "tier": tier, "window": wlen, "method": meth,
                              "mult": round(k, 3), "n": len(t), "cov95": c, "pinball95": p})
B = pd.DataFrame(Brows); B.to_csv(OUT / "B_variance_calibration.csv", index=False)

# ---------------- C. power ----------------
def per_origin(w, a, b, tier):
    o = w[(w.tier == tier) & w.actual.notna()]
    g = o.groupby("origin")
    return g.apply(lambda x: ((x[a] - x.actual).abs().sum() - (x[b] - x.actual).abs().sum()) / x.actual.sum(),
                   include_groups=False)


def boot(w, a, b, tier, two_way, B_=400):
    o = w[(w.tier == tier) & w.actual.notna()].copy()
    o["ea"] = (o[a] - o.actual).abs(); o["eb"] = (o[b] - o.actual).abs()
    cell = o.groupby(["origin", "drug_code"]).agg(ea=("ea", "sum"), eb=("eb", "sum"), y=("actual", "sum")).reset_index()
    O = cell.origin.unique(); D = cell.drug_code.unique()
    oi = {k: i for i, k in enumerate(O)}; di = {k: i for i, k in enumerate(D)}
    M = np.zeros((3, len(O), len(D)))
    M[0, cell.origin.map(oi), cell.drug_code.map(di)] = cell.ea
    M[1, cell.origin.map(oi), cell.drug_code.map(di)] = cell.eb
    M[2, cell.origin.map(oi), cell.drug_code.map(di)] = cell.y
    stats = []
    for _ in range(B_):
        dw = np.bincount(RNG.integers(0, len(D), len(D)), minlength=len(D))
        ow = np.bincount(RNG.integers(0, len(O), len(O)), minlength=len(O)) if two_way else np.ones(len(O))
        W = np.outer(ow, dw)
        s = (M * W).sum(axis=(1, 2))
        stats.append((s[0] - s[1]) / s[2])
    return np.percentile(stats, [5, 95])


Crows = []
for tier in ["fast", "regular", "slow"]:
    po = per_origin(ws, "ml_A2", "baseline", tier)
    sd = po.std(ddof=1)
    # MDE for a paired mean over K origins, 80% power, two-sided 5%; inflate by 1.4 for 50% window overlap
    mde = lambda K: 2.8 * sd * 1.4 / np.sqrt(K)
    d1 = boot(ws, "ml_A2", "baseline", tier, False)
    d2 = boot(ws, "ml_A2", "baseline", tier, True)
    Crows.append({"tier": tier, "mean_diff_wape": po.mean(), "sd_origin": sd,
                  "mde_11_origins": mde(11), "mde_26_origins": mde(26), "mde_52_origins": mde(52),
                  "ci90_drug_only": f"{d1[0]:.4f}..{d1[1]:.4f}", "ci90_two_way": f"{d2[0]:.4f}..{d2[1]:.4f}"})
C = pd.DataFrame(Crows); C.to_csv(OUT / "C_power_selection.csv", index=False)

# ---------------- D. oracle store-level calendar ceiling ----------------
def oracle_store(w, m):
    o = w[w.actual.notna() & w.tier.isin(["fast", "regular", "slow"])].copy()
    s = o.groupby(["origin", "h"]).agg(f=(m, "sum"), y=("actual", "sum"))
    o = o.join((s.y / s.f).rename("k"), on=["origin", "h"])
    o["orc"] = o[m] * o.k
    r = []
    for eid, g in o.groupby("eid"):
        r.append({"model": m, "eid_week": eid, "rows": len(g),
                  "wape": (g[m] - g.actual).abs().sum() / g.actual.sum(),
                  "wape_oracle_store": (g.orc - g.actual).abs().sum() / g.actual.sum(),
                  "store_k_p10": g.groupby(["origin", "h"]).k.first().quantile(.1),
                  "store_k_p90": g.groupby(["origin", "h"]).k.first().quantile(.9)})
    return r


D = pd.DataFrame(oracle_store(ws, "ml_A2") + oracle_store(ws, "baseline")
                 + [dict(x, model=x["model"] + "_DIAG25") for x in oracle_store(we, "ml") + oracle_store(we, "current")])
D.to_csv(OUT / "D_oracle_store_calendar.csv", index=False)

# ---------------- E. young drugs ----------------
o = ws[ws.actual.notna()].copy()
o["age_w"] = (o.origin - o.drug_code.map(first)).dt.days / 7
o["age"] = pd.cut(o.age_w, [-1, 13, 26, 1e9], labels=["<13w", "13-26w", "older"])
E = o.groupby("age", observed=True).apply(lambda g: pd.Series({
    "drug_origins": g.groupby(["origin", "drug_code"]).ngroups,
    "bias_ml": g.ml_A2.sum() / g.actual.sum(), "bias_base": g.baseline.sum() / g.actual.sum(),
    "wape_ml": (g.ml_A2 - g.actual).abs().sum() / g.actual.sum(),
    "wape_base": (g.baseline - g.actual).abs().sum() / g.actual.sum()}), include_groups=False)
E.to_csv(OUT / "E_young_drugs_selection.csv")

with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format, "display.max_rows", 500):
    P("\nA. selection by tier\n", A1.to_string(index=False))
    P("\nA. selection all\n", A5.to_string(index=False))
    P("\nA. selection by segment (labels at 8 Dec 2024, hindsight for new/fading)\n", A2.to_string(index=False))
    P("\nA. eval 2025 DIAG by tier\n", A3.to_string(index=False))
    P("\nA. eval 2025 DIAG all\n", A6.to_string(index=False))
    P("\nA. eval 2025 DIAG by segment\n", A4.to_string(index=False))
    P("\nB. calibration (sel: fit origins 1-6, test 7-11; ev25: fit on all sel)\n",
      B.pivot_table(index=["set", "model", "tier", "window"], columns="method", values="cov95").to_string())
    P("\nB. pinball95 (4-week)\n",
      B[B.window == 4].pivot_table(index=["set", "model", "tier"], columns="method", values="pinball95").to_string())
    P("\nB. multipliers (4-week)\n",
      B[B.window == 4].pivot_table(index=["set", "model", "tier"], columns="method", values="mult").to_string())
    P("\nC. power (ML - baseline WAPE, selection)\n", C.to_string(index=False))
    P("\nD. oracle store calendar\n", D.to_string(index=False))
    P("\nE. young drugs (selection)\n", E.to_string())
txt = "\n".join(lines)
(OUT / "explore.out").write_text(txt)
print(txt)
