"""How much forecastable signal is there beyond a flat level? Selection data only (to 8 Dec 2024).

Part 1 (in-sample): split the variance of weekly units into between-drug level,
store-wide Eid, linear trend, slow drift (leave-one-out 13-week centred mean),
the compound-Poisson noise floor, and what's left.

Part 2 (out-of-sample, 11 selection origins x 8 weeks): split the squared error
of a flat SES (alpha 0.15) forecast into
  noise floor  (compound Poisson at the realised 8-week mean),
  level miss   ((8-week mean - forecast)^2 minus the noise in that mean),
  shape        (within-window variance beyond noise: the most any non-flat
                forecast over h=1..8 could win, before even predicting it).
Plus a momentum test: does the recent trend predict the next 8 weeks?

Needs drug_stats.py first. Writes to data/followup/demand/.
Run from the repo root:  uv run python eda/followup/scripts/demand/signal_decomp.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from irpd import forecast as fc
from irpd.ml import history, split
from irpd.tables.sales import tiers_sql

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", "{:,.3f}".format)
OUT = fp.out("demand")
END = split.SELECTION_END
TIERS = ["fast", "regular", "slow", "rare"]

con = split.selection_con()
H = history.load(con, END)
DRUGS, WEEKS, Y, CENS = H["drugs"], H["weeks"], H["y"], H["out"]
N, T = Y.shape
YC = np.where(CENS, np.nan, Y)
EIDX = np.load(OUT / "eid_index.npy")
EIDW = EIDX > 1.05
idx = {c: i for i, c in enumerate(DRUGS)}

# compound-Poisson size factor E[q^2]/E[q] per drug from sale documents (base units)
sz = con.execute(f"""
    WITH s AS (SELECT drug_code, document, sum(qty_out) AS q FROM stock_moves
               WHERE move_type = 'sale' AND date <= DATE '{END}' GROUP BY ALL)
    SELECT drug_code, sum(q*q)/sum(q) AS size_factor, avg(q) AS mean_q, count(*) AS docs
    FROM s WHERE q > 0 GROUP BY 1""").df().set_index("drug_code")
SF = sz.size_factor.reindex(DRUGS).fillna(1.0).to_numpy()
sz.to_parquet(OUT / "size_factor.parquet")
stats = pd.read_parquet(OUT / "drug_stats.parquet")
tier_end = stats.tier.reindex(DRUGS).to_numpy()

print("size factor E[q^2]/E[q] by tier (p10/med/p90):")
for t in TIERS:
    s = pd.Series(SF[tier_end == t])
    print(f"  {t:8s} {s.quantile(.1):.2f} / {s.median():.2f} / {s.quantile(.9):.2f}")
st = stats[stats.active52]
st = st.join(sz.size_factor)
st["excess_disp"] = st.disp52 / st.size_factor
print("var/mean in last 52 wk divided by size factor (1 = compound-Poisson), p10/med/p90:")
for t in TIERS:
    s = st[st.tier == t].excess_disp
    print(f"  {t:8s} {s.quantile(.1):.2f} / {s.median():.2f} / {s.quantile(.9):.2f}")

# ---------------- Part 1: in-sample variance decomposition --------------------
first = H["first_sale"]
recs = []
for i in range(N):
    if first[i] < 0 or tier_end[i] not in TIERS:
        continue
    live = np.arange(T) >= first[i]
    m = live & ~np.isnan(YC[i])
    if m.sum() < 40:
        continue
    y = YC[i][m]
    e = EIDX[m]
    t = np.arange(T)[m].astype(float)
    mu = y.mean()
    if mu <= 0:
        continue
    ss_tot = ((y - mu) ** 2).sum()
    # Eid: store index, no fitted coefficient
    h1 = mu * e / e.mean()
    # trend: (a + b t) * e by least squares on y/e
    b = np.polyfit(t, y / e, 1)
    h2 = np.polyval(b, t) * e
    # slow drift: leave-one-out centred 13-week mean of y/e (observed weeks only)
    ye = np.full(T, np.nan)
    ye[m] = YC[i][m] / EIDX[m]
    h3 = np.empty(len(y))
    for k, tt in enumerate(np.where(m)[0]):
        lo, hi = max(0, tt - 6), min(T, tt + 7)
        w = np.concatenate([ye[lo:tt], ye[tt + 1:hi]])
        h3[k] = np.nanmean(w) if np.sum(~np.isnan(w)) >= 4 else mu
    h3 = h3 * e
    floor = (np.maximum(h3, 0) * SF[i]).sum()
    floor_p = np.maximum(h3, 0).sum()
    ss1 = ((y - h1) ** 2).sum()
    ss2 = ((y - h2) ** 2).sum()
    ss3 = ((y - h3) ** 2).sum()
    recs.append(dict(drug_code=DRUGS[i], tier=tier_end[i], n=len(y), mu=mu, ss_tot=ss_tot,
                     ss_eid=ss1, ss_trend=ss2, ss_drift=ss3, floor=floor, floor_p=floor_p,
                     # LOO MA's own noise (12 neighbours) adds ~floor/12; keep for reference
                     ))
D = pd.DataFrame(recs).set_index("drug_code")
D.to_parquet(OUT / "insample_decomp.parquet")


def decomp(d):
    tot = d.ss_tot.sum()
    eid = (d.ss_tot - d.ss_eid).sum()
    trend = (d.ss_eid - d.ss_trend).sum()
    drift = (d.ss_trend - d.ss_drift).sum()
    # LOO 13-wk mean carries ~1/12 of a week's noise; correct the residual for it
    resid = d.ss_drift.sum() - d.floor.sum() / 12
    return pd.Series({"drugs": len(d), "Eid %": 100 * eid / tot, "trend %": 100 * trend / tot,
                      "extra drift %": 100 * drift / tot,
                      "noise floor (cP) %": 100 * d.floor.sum() / tot,
                      "noise floor (Poisson) %": 100 * d.floor_p.sum() / tot,
                      "unexplained excess %": 100 * (resid - d.floor.sum()) / tot})


def decomp_scaled(d):
    # each drug weighted equally: divide its sums by its own ss_tot
    r = pd.DataFrame({"eid": (d.ss_tot - d.ss_eid) / d.ss_tot,
                      "trend": (d.ss_eid - d.ss_trend) / d.ss_tot,
                      "drift": (d.ss_trend - d.ss_drift) / d.ss_tot,
                      "floor": d.floor / d.ss_tot})
    return (100 * r.median()).rename(lambda c: c + " % (median drug)")


print("\n== Part 1: within-drug variance of weekly units, censored weeks masked, "
      "history from first sale to 8 Dec 2024 (>=40 observed weeks)")
P1 = pd.DataFrame({t: decomp(D[D.tier == t]) for t in TIERS} | {"all": decomp(D)}).T
print(P1)
print(pd.DataFrame({t: decomp_scaled(D[D.tier == t]) for t in TIERS}).T)
# between-drug share of total variance (pooled over drug-weeks), on log-ish scale
allv = []
for i in range(N):
    if first[i] < 0:
        continue
    live = np.arange(T) >= first[i]
    v = YC[i][live & ~np.isnan(YC[i])]
    if len(v):
        allv.append(v)
cat = np.concatenate(allv)
within = sum(((v - v.mean()) ** 2).sum() for v in allv)
print(f"between-drug share of pooled variance (raw units): {100*(1-within/((cat-cat.mean())**2).sum()):.1f}%")
P1.to_csv(OUT / "insample_decomp_by_tier.csv")

# ---------------- Part 2: out-of-sample flat-forecast error decomposition -----
origins = split.selection_origins()
col = {w: j for j, w in enumerate(WEEKS)}
rows = []
for o in origins:
    n = history.weeks_to(H, o)
    tiers = con.execute(tiers_sql(o)).df().set_index("drug_code").tier
    past = YC[:, :n]
    lvl = fc._ses(past, 0.15)
    tgt = YC[:, n:n + 8]
    e_t = EIDX[n:n + 8]
    e_p = EIDX[:n]
    eid_win = bool((e_t > 1.05).any())
    sold13 = np.nansum(Y[:, n - 13:n], axis=1) > 0
    m13 = np.nanmean(past[:, -13:], axis=1)
    p13 = np.nanmean(past[:, -26:-13], axis=1)
    for i in np.where(sold13)[0]:
        y = tgt[i]
        ok = ~np.isnan(y)
        k = ok.sum()
        if k < 4 or np.isnan(lvl[i]):
            continue
        yy = y[ok]
        f = lvl[i]
        ybar = yy.mean()
        mse_flat = ((yy - f) ** 2).mean()
        # SES with the store Eid index applied (base de-seasonalised roughly by ignoring it)
        fe = f * e_t[ok]
        mse_eid = ((yy - fe) ** 2).mean()
        s2 = yy.var(ddof=1)
        noise = max(ybar, 0) * SF[i]
        rows.append(dict(origin=o, drug_code=DRUGS[i], tier=tiers.get(DRUGS[i], "rare"),
                         eid_win=eid_win, k=k, f=f, ybar=ybar, mse_flat=mse_flat,
                         mse_eid=mse_eid, s2=s2, noise=noise, noise_p=max(ybar, 0),
                         m13=m13[i], p13=p13[i]))
R = pd.DataFrame(rows)
R.to_parquet(OUT / "oos_decomp_rows.parquet")
# per row: mse_flat = (k-1)/k s2 + (ybar-f)^2 ; components (expected):
R["lvl"] = (R.ybar - R.f) ** 2
R["shp"] = (R.k - 1) / R.k * (R.s2 - R.noise)
R["lvl_x"] = R.lvl - R.noise / R.k


def oos(d):
    tot = d.mse_flat.sum()
    return pd.Series({"drug-origins": len(d),
                      "noise floor %": 100 * d.noise.sum() / tot,
                      "noise floor Poisson-only %": 100 * d.noise_p.sum() / tot,
                      "level miss beyond noise %": 100 * d.lvl_x.sum() / tot,
                      "within-8wk shape beyond noise %": 100 * d.shp.sum() / tot,
                      "MSE ratio SESxEid / SES": d.mse_eid.sum() / tot})


print("\n== Part 2: flat SES(0.15) forecast, 11 selection origins x 8 weeks, "
      "drugs sold in 13 weeks before origin; censored targets dropped")
P2 = pd.DataFrame({f"{t}{' Eid' if e else ''}": oos(R[(R.tier == t) & (R.eid_win == e)])
                   for t in TIERS for e in (False, True)}
                  | {"all": oos(R), "all non-Eid": oos(R[~R.eid_win]),
                     "all Eid windows": oos(R[R.eid_win])}).T
print(P2)
P2.to_csv(OUT / "oos_decomp_by_tier.csv")

# momentum: does recent change (last 13 vs previous 13) predict next 8 vs last 13?
c = 0.5
R2 = R[(R.m13 > 0) & (R.p13.notna()) & ~R.eid_win].copy()
R2["past"] = np.log((R2.m13 + c) / (R2.p13 + c))
R2["fut"] = np.log((R2.ybar + c) / (R2.m13 + c))
R2["fut_ses"] = np.log((R2.ybar + c) / (R2.f + c))
print("\n== momentum (non-Eid windows): corr(log last13/prev13, log next8/last13); "
      "slope; and vs SES level")
for t in TIERS:
    d = R2[R2.tier == t]
    b = np.polyfit(d.past, d.fut, 1)[0]
    print(f"  {t:8s} n={len(d):5d} corr={np.corrcoef(d.past, d.fut)[0,1]:+.3f} slope={b:+.3f}"
          f"  corr vs SES residual={np.corrcoef(d.past, d.fut_ses)[0,1]:+.3f}")

# optimal weights of last 13 vs previous 13 weeks for the next-8-week mean (scaled rows)
print("\n== OLS ybar8 ~ a*m13 + b*p13 (no intercept, rows scaled by their 26-wk mean), non-Eid")
for t in TIERS:
    d = R2[R2.tier == t]
    s = (d.m13 + d.p13) / 2 + 0.1
    X = np.column_stack([d.m13 / s, d.p13 / s])
    coef, *_ = np.linalg.lstsq(X, d.ybar / s, rcond=None)
    print(f"  {t:8s} a(last13)={coef[0]:.2f} b(prev13)={coef[1]:.2f}  sum={coef.sum():.2f}")
