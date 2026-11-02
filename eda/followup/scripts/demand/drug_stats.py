"""Per-drug demand statistics on selection data (to 8 Dec 2024). Nothing from 2025.

Writes drug_stats.parquet, active_stats.parquet and eid_index.npy to data/followup/demand/
and prints summary tables. Run first of the demand scripts.
Run from the repo root:  uv run python eda/followup/scripts/demand/drug_stats.py
"""
import sys
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from irpd import forecast as fc
from irpd.ml import history, split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", "{:,.3f}".format)
OUT = fp.out("demand")
END = split.SELECTION_END

con = split.selection_con()
H = history.load(con, END)
DRUGS, WEEKS, Y, CENS = H["drugs"], H["weeks"], H["y"], H["out"]
REV = H["revenue"]
SHELF = H["shelfout_days"]
N, T = Y.shape
KNOWN = H["stock_known"]
YC = np.where(CENS, np.nan, Y)

# store-wide Eid uplift per week (unshrunk), from lines to END
f = fc.eid_factors(H["lines"])
raw = {k: 1 + (v - 1) / fc.EID_SHRINK for k, v in f.items()}
EIDX = fc.eid_index(WEEKS, raw)
EIDW = EIDX > 1.05
np.save(OUT / "eid_index.npy", EIDX)

tiers = con.execute("SELECT * FROM demand_tiers").df().set_index("drug_code")

# single-line sizes: per drug-week, max units on one sale document vs the week's units
lines = con.execute(f"""
    WITH s AS (SELECT drug_code, date_trunc('week', date)::DATE AS wk, document,
                      sum(qty_out) AS q
               FROM stock_moves WHERE move_type = 'sale' AND date <= DATE '{END}'
               GROUP BY ALL)
    SELECT drug_code, wk, max(q) AS max_line, sum(q) AS units_lines, count(*) AS docs
    FROM s GROUP BY ALL""").df()
lines["wk"] = pd.to_datetime(lines.wk).dt.date
col = {w: j for j, w in enumerate(WEEKS)}
idx = {c: i for i, c in enumerate(DRUGS)}
lines = lines[lines.drug_code.isin(idx) & lines.wk.isin(col)]
MAXLINE = np.zeros((N, T))
MAXLINE[lines.drug_code.map(idx), lines.wk.map(col)] = lines.max_line
DOCS = np.zeros((N, T))
DOCS[lines.drug_code.map(idx), lines.wk.map(col)] = lines.docs

last52 = np.arange(T) >= T - 52
first = H["first_sale"]


def acf(x, k):
    a, b = x[:-k], x[k:]
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() < 20:
        return np.nan
    a, b = a[m], b[m]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def best_break(z, trim=13):
    """Best single mean break on z (no NaN). Returns (sup-F, index, ratio of means after/before)."""
    n = len(z)
    if n < 2 * trim + 4:
        return np.nan, -1, np.nan
    cs = np.cumsum(z)
    ss0 = ((z - z.mean()) ** 2).sum()
    best, bi = -1, -1
    for k in range(trim, n - trim):
        m1, m2 = cs[k - 1] / k, (cs[-1] - cs[k - 1]) / (n - k)
        gain = k * (m1 - z.mean()) ** 2 + (n - k) * (m2 - z.mean()) ** 2
        if gain > best:
            best, bi = gain, k
    ss1 = ss0 - best
    F = best / (ss1 / (n - 2)) if ss1 > 0 else np.nan
    return F, bi, np.nan


rows = []
for i, code in enumerate(DRUGS):
    y, yc = Y[i], YC[i]
    fs = first[i]
    if fs < 0:
        continue
    live = np.arange(T) >= fs
    obs = live & ~np.isnan(yc)
    known_live = live & KNOWN
    r = dict(drug_code=code)
    r["first_week"] = fs
    sold = np.nan_to_num(y) > 0
    r["last_sale_week"] = int(np.where(sold)[0].max())
    r["weeks_live"] = int(live.sum())
    r["units52"] = np.nansum(y[last52])
    r["rev52"] = np.nansum(REV[i][last52])
    r["rev_all"] = np.nansum(REV[i])
    o52 = obs & last52
    r["obs52"] = int(o52.sum())
    r["mu52"] = np.nanmean(yc[o52]) if o52.any() else np.nan
    r["var52"] = np.nanvar(yc[o52], ddof=1) if o52.sum() > 2 else np.nan
    r["zero_share52"] = np.mean(yc[o52] == 0) if o52.any() else np.nan
    # censoring since stock is known
    r["cens_share"] = CENS[i][known_live].mean() if known_live.any() else np.nan
    r["shelf_week_share"] = (SHELF[i][known_live] > 0).mean() if known_live.any() else np.nan
    r["stockout_week_share"] = (H["out"][i][known_live] & (SHELF[i][known_live] < 4)).mean() \
        if known_live.any() else np.nan
    # bulk: weeks where one document holds >= 50% of units and units > 3x median non-zero week
    nz = y[live & sold]
    med = np.median(nz) if len(nz) else np.nan
    bulk = live & sold & (MAXLINE[i] >= 0.5 * np.nan_to_num(y)) & (np.nan_to_num(y) > 3 * med) \
        & (MAXLINE[i] >= 3)
    r["bulk_weeks"] = int(bulk.sum())
    tot = np.nansum(y[live])
    r["bulk_unit_share"] = np.nansum(y[bulk]) / tot if tot > 0 else np.nan
    r["max_week_share"] = np.nanmax(y[live]) / tot if tot > 0 else np.nan
    r["docs_per_week"] = DOCS[i][live].mean()
    r["units_per_doc"] = tot / DOCS[i][live].sum() if DOCS[i][live].sum() > 0 else np.nan
    # Eid-masked, censored-masked series for ACF and breaks
    z = np.where(EIDW, np.nan, yc)
    z = np.where(live, z, np.nan)
    for k in (1, 2, 3, 4, 52):
        r[f"acf{k}"] = acf(z, k)
    r["n_obs"] = int((~np.isnan(z)).sum())
    zz = z[~np.isnan(z)]
    # level shift: first vs second half of observed history; last 26 vs prior 26
    if len(zz) >= 26:
        h = len(zz) // 2
        r["half_ratio"] = (zz[h:].mean() + 0.1) / (zz[:h].mean() + 0.1)
    w = z[-52:]
    a, b = w[:26], w[26:]
    if np.sum(~np.isnan(a)) >= 13 and np.sum(~np.isnan(b)) >= 13:
        r["last26_ratio"] = (np.nanmean(b) + 0.1) / (np.nanmean(a) + 0.1)
    # structural break on Anscombe-transformed series
    if len(zz) >= 40 and zz.mean() > 0:
        F, k, _ = best_break(2 * np.sqrt(zz + 3 / 8))
        r["break_F"] = F
        r["break_pos"] = k / len(zz)
        r["break_ratio"] = (zz[k:].mean() + 0.1) / (zz[:k].mean() + 0.1)
        t = np.arange(len(zz))
        s = 2 * np.sqrt(zz + 3 / 8)
        beta = np.polyfit(t, s, 1)
        res = s - np.polyval(beta, t)
        se = np.sqrt(res.var(ddof=2) / ((t - t.mean()) ** 2).sum())
        r["trend_t"] = beta[0] / se
        r["disp_all"] = zz.var(ddof=1) / zz.mean()
    # Eid response, 2023 and 2024 weeks 0..+2 over base weeks -9..-3
    for e in fc.EID[:2]:
        wk = e - timedelta(days=e.weekday())
        if wk not in col:
            continue
        j = col[wk]
        base = yc[max(0, j - 9):j - 2]
        eidw = yc[j:j + 3]
        if live[max(0, j - 9)] and np.sum(~np.isnan(base)) >= 4 and np.sum(~np.isnan(eidw)) >= 2:
            r[f"eid{e.year}"] = (np.nanmean(eidw) + 0.1) / (np.nanmean(base) + 0.1)
    rows.append(r)

S = pd.DataFrame(rows).set_index("drug_code")
S = S.join(tiers[["tier", "sb_class", "adi", "cv2", "sale_days_per_year"]])
S["disp52"] = S.var52 / S.mu52
S["weeks_since_last_sale"] = (T - 1) - S.last_sale_week
S["active52"] = S.units52 > 0
names = con.execute("SELECT drug_code, drug_name FROM drugs").df().set_index("drug_code")
S = S.join(names)
S.to_parquet(OUT / "drug_stats.parquet")

A = S[S.active52].copy()
TOTREV = A.rev52.sum()
TOTU = A.units52.sum()
print(f"drugs with a sale ever: {len(S)}, active in last 52 weeks: {len(A)}; "
      f"rev52 total Rp {TOTREV/1e6:,.0f}M")

TIERS = ["fast", "regular", "slow", "rare"]
CL = ["smooth", "erratic", "intermittent", "lumpy", "too few"]


def share_tab(df, by, cols):
    g = df.groupby(by, observed=False)
    return pd.DataFrame({"drugs": g.size(), "units %": 100 * g.units52.sum() / TOTU,
                         "revenue %": 100 * g.rev52.sum() / TOTREV})


print("\n== SB class x tier: drugs (active 52w)")
print(pd.crosstab(A.tier, A.sb_class, margins=True).reindex(index=TIERS + ["All"],
                                                              columns=CL + ["All"]))
print("\n== SB class x tier: revenue %")
print((pd.crosstab(A.tier, A.sb_class, values=A.rev52, aggfunc="sum", margins=True)
       / TOTREV * 100).reindex(index=TIERS + ["All"], columns=CL + ["All"]).round(1))
print("\n== SB class x tier: units %")
print((pd.crosstab(A.tier, A.sb_class, values=A.units52, aggfunc="sum", margins=True)
       / TOTU * 100).reindex(index=TIERS + ["All"], columns=CL + ["All"]).round(1))


def q(s):
    return f"{s.quantile(.1):.2f} / {s.median():.2f} / {s.quantile(.9):.2f}"


print("\n== per-tier distributions (p10 / median / p90), active drugs")
out = {}
for t in TIERS:
    d = A[A.tier == t]
    out[t] = {"drugs": len(d), "mu52": q(d.mu52), "dispersion var/mean": q(d.disp52),
              "cv (sd/mean)": q(np.sqrt(d.var52) / d.mu52), "zero weeks": q(d.zero_share52),
              "acf1": q(d.acf1), "acf2": q(d.acf2), "acf4": q(d.acf4), "acf52": q(d.acf52),
              "half_ratio": q(d.half_ratio), "last26 ratio": q(d.last26_ratio),
              "censored wk share": q(d.cens_share), "bulk unit share": q(d.bulk_unit_share)}
print(pd.DataFrame(out))

# share of drugs with "significant" lag-1 ACF (> 2/sqrt(n))
A["acf1_sig"] = A.acf1 > 2 / np.sqrt(A.n_obs)
A["acf52_sig"] = A.acf52.abs() > 2 / np.sqrt(np.maximum(A.n_obs - 52, 1))
print("\n== share of drugs with acf1 > 2/sqrt(n), |acf52| > 2/sqrt(n-52); median acf by tier")
print(A.groupby("tier")[["acf1_sig", "acf52_sig", "acf1", "acf2", "acf3", "acf4", "acf52"]]
      .agg(["mean"]).reindex(TIERS).round(3))
# null check: 52-lag significance count expected ~5% under noise
# structural break: sup-F 5% crit for 1 break, 15% trimming ~ 8.85 (Andrews 1993); use 9 and 15
A["break_sig"] = A.break_F > 9.0
A["break_big"] = (A.break_F > 15) & ((A.break_ratio > 1.5) | (A.break_ratio < 1 / 1.5))
A["trend_sig"] = A.trend_t.abs() > 3
print("\n== level shifts (drugs with >=40 observed weeks)")
d = A[A.break_F.notna()]
print(d.groupby("tier").agg(n=("break_F", "size"), break_supF_gt9=("break_sig", "mean"),
                            big_break=("break_big", "mean"), trend_abs_t_gt3=("trend_sig", "mean"),
                            median_ratio=("break_ratio", "median"))
      .reindex(TIERS).round(3))
print("break position (share of history) for big breaks:", q(d[d.break_big].break_pos))
print("disp (var/mean, whole history, Eid/cens masked):\n",
      d.groupby("tier").disp_all.describe(percentiles=[.1, .5, .9]).reindex(TIERS).round(2))

print("\n== lifecycle (all drugs ever sold to 8 Dec 2024)")
S2 = S.copy()
S2["new26"] = S2.first_week >= T - 26
S2["dormant13"] = (S2.weeks_since_last_sale >= 13)
print("new (first sale in last 26 weeks):", int(S2.new26.sum()),
      f"rev% {100*S2[S2.new26].rev52.sum()/TOTREV:.1f}")
print("no sale in last 13 weeks but sold in last 52:",
      int((S2.dormant13 & S2.active52).sum()),
      f"rev% {100*S2[S2.dormant13 & S2.active52].rev52.sum()/TOTREV:.2f}")
print("declining strongly (last26 ratio < 0.5, mu52>=1):",
      int(((A.last26_ratio < 0.5) & (A.mu52 >= 1)).sum()))

print("\n== Eid response per drug (fast/regular, base >= 3 units/wk)")
e = A[A.tier.isin(["fast", "regular"])].dropna(subset=["eid2023", "eid2024"])
e = e[e.mu52 >= 3]
print("n", len(e), "corr log ratios 2023 vs 2024:",
      round(np.corrcoef(np.log(e.eid2023), np.log(e.eid2024))[0, 1], 3))
print("2023 p10/med/p90:", q(e.eid2023), " 2024:", q(e.eid2024))

print("\n== bulk / single-line spikes")
print(A.groupby("tier").agg(bulk_drugs=("bulk_weeks", lambda s: (s > 0).mean()),
                            bulk_share_med=("bulk_unit_share", "median"),
                            bulk_share_gt20=("bulk_unit_share", lambda s: (s > .2).mean()),
                            units_per_doc=("units_per_doc", "median")).reindex(TIERS).round(3))
print("\n== censoring")
print(A.groupby("tier").agg(cens=("cens_share", "median"),
                            cens_gt20=("cens_share", lambda s: (s > .2).mean()),
                            cens_gt40=("cens_share", lambda s: (s > .4).mean()),
                            shelf=("shelf_week_share", "mean")).reindex(TIERS).round(3))
A.to_parquet(OUT / "active_stats.parquet")
