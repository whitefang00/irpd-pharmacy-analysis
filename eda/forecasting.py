"""EDA for the ML forecasters: the questions in the ML plan, section 4.

Each section answers one question whose answer fixes a choice in the model
setup (lags, windows, Eid handling, group definition, recency weighting, the
Prophet subset), plus how much sales drop on shelf-out days.

Everything reads the selection file through `irpd.ml.split.selection_con`
(every table as of 8 Dec 2024, locked against other files), so nothing here
can see 2025. Weekly history comes from `irpd.ml.history.load`. Prints its
tables and writes nothing.

Run from the repo root (IRPD_DB points at the warehouse; the selection file
sits next to it):
    uv run python -I eda/forecasting.py
"""

import time
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd

from irpd import config
from irpd import forecast as fc
from irpd.ml import history, schema, split
from irpd.tables.sales import tiers_sql

T0 = time.time()
END = split.SELECTION_END
TIERS = ["fast", "regular", "slow", "rare"]
CLASSES = ["smooth", "erratic", "intermittent", "lumpy", "too few"]
ROUTE = {**dict.fromkeys(["tablet", "capsule", "syrup", "suspension", "drops", "powder",
                          "liquid"], "oral"),
         **dict.fromkeys(["cream", "ointment", "gel"], "topical"),
         "inhaler": "inhaled", "injection": "injected"}
RNG = np.random.default_rng(0)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 80)
pd.set_option("display.float_format", "{:,.3f}".format)
warnings.filterwarnings("ignore", category=RuntimeWarning)  # nanmean of all-NaN rows

con = split.selection_con()
H = history.load(con, END)
DRUGS, WEEKS = H["drugs"], H["weeks"]
Y = H["y"]                              # units, NaN before first sale
OUT = H["out"]                          # week had a total stock-out day (stock known only)
YC = np.where(OUT, np.nan, Y)           # censored weeks masked
N_DRUGS, N_WEEKS = Y.shape
WEEK_IDX = {w: j for j, w in enumerate(WEEKS)}
STOCK_COL = int(np.argmax(H["stock_known"]))
EID_WEEK = fc.eid_index(WEEKS, dict.fromkeys(fc.EID_DAYS, 2.0)) > 1  # overlaps an Eid window
EIDS = [e for e in fc.EID if e <= END]


def header(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def show(title, df, index=True):
    print(f"\n-- {title}")
    print(df.to_string(index=index))


def tiers_at(day):
    """Tier and Syntetos-Boylan class per drug, from daily demand up to `day`."""
    t = con.execute(f"SELECT drug_code, tier, sb_class, revenue_share "
                    f"FROM ({tiers_sql(day)})").df().set_index("drug_code")
    return t.reindex(DRUGS)


def window_mean(y, n, k):
    """Mean of the last k weeks before column n, skipping NaN."""
    return np.nanmean(y[:, max(0, n - k):n], axis=1) if n else np.full(len(y), np.nan)


def corr(a, b):
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 3:
        return np.nan
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return (a * b).sum() / d if d > 0 else np.nan


TIER_END = tiers_at(END)
TIER = TIER_END.tier.fillna("none").to_numpy()
ACTIVE = history.active(H, END, schema.ACTIVE_WEEKS)
print(f"Selection data to {END}: {N_DRUGS:,} drugs, {N_WEEKS} weeks "
      f"({WEEKS[0]} to {WEEKS[-1]}), {ACTIVE.sum():,} sold in the last 52 weeks.")

# ---------------------------------------------------------------------------
header("1. Intermittency: Syntetos-Boylan class by tier (drugs sold in the last 52 weeks)")
# ADI = weeks in the selling window / weeks with a sale; CV² of non-zero weeks.
# Cut-offs 1.32 and 0.49, as in tiers_sql.

for day in (split.FIRST_ORIGIN, END):
    t = tiers_at(day)
    act = history.active(H, day, schema.ACTIVE_WEEKS)
    t = t[act & t.tier.notna().to_numpy()]
    ct = pd.crosstab(t.tier, t.sb_class).reindex(index=TIERS, columns=CLASSES).fillna(0)
    ct["all"] = ct.sum(axis=1)
    ct.loc["all"] = ct.sum()
    show(f"drugs at {day}", ct.astype(int))
    rev = t.groupby("sb_class").revenue_share.sum()
    rev = rev.reindex(CLASSES).fillna(0) / rev.sum()
    show(f"revenue share by class at {day}", rev.to_frame("share").T, index=False)

# ---------------------------------------------------------------------------
header("2. Censoring: drug-weeks with a total stock-out day")
# From the first week the stock ledger covers, weeks after each drug's first
# sale only. Tier as of the end of the selection data.

live = ~np.isnan(Y)
live[:, :STOCK_COL] = False
rows = []
for t in TIERS + ["all"]:
    m = (TIER == t) if t != "all" else np.isin(TIER, TIERS)
    lv = live[m]
    rows.append({"tier": t, "drug_weeks": int(lv.sum()),
                 "stockout_share": OUT[m][lv].mean(),
                 "shelfout_share": (H["shelfout_days"][m][lv] > 0).mean(),
                 "drugs_ever_censored": (OUT[m] & lv).any(axis=1).mean()})
show("by tier", pd.DataFrame(rows), index=False)

q = pd.PeriodIndex(pd.to_datetime(WEEKS), freq="Q").astype(str)
rows = []
for quarter in sorted(set(q[STOCK_COL:])):
    cols = q == quarter
    r = {"quarter": quarter}
    for t in TIERS:
        m = TIER == t
        lv = live[m][:, cols]
        r[t] = OUT[m][:, cols][lv].mean()
    rows.append(r)
show("stock-out share by quarter (tier at the end)", pd.DataFrame(rows), index=False)

# How many days of a censored week were actually out? A week with one day out
# is mostly observed; masking it throws that away.
days_out = con.execute(f"""
    SELECT drug_code, date_trunc('week', date)::DATE AS week_start, count(*) AS days
    FROM daily_stock WHERE location = 'total' AND closing <= 0
      AND date >= DATE '{WEEKS[STOCK_COL]}' AND date <= DATE '{END}'
    GROUP BY ALL""").df()
days_out["tier"] = days_out.drug_code.map(dict(zip(DRUGS, TIER, strict=True)))
days_out["week_start"] = pd.to_datetime(days_out.week_start).dt.date
di = {c: i for i, c in enumerate(DRUGS)}
keep = days_out.drug_code.isin(di) & days_out.week_start.isin(WEEK_IDX)
days_out = days_out[keep]
days_out = days_out[live[days_out.drug_code.map(di), days_out.week_start.map(WEEK_IDX)]]
days_out["bucket"] = pd.cut(days_out.days, [0, 1, 3, 6, 7], labels=["1", "2-3", "4-6", "7"])
show("censored weeks by days out (share of each tier's censored weeks)",
     pd.crosstab(days_out.tier, days_out.bucket, normalize="index").reindex(TIERS))

# ---------------------------------------------------------------------------
header("3. Autocorrelation and which windows predict best")
# Pooled ACF: each drug's Eid-adjusted weekly units, censored weeks masked,
# standardised per drug over its observed weeks; drugs with 26+ observed weeks.

factors = fc.eid_factors(H["lines"])
eid_adj = fc.eid_index(WEEKS, factors)
Z = YC / eid_adj[None, :]
nobs = (~np.isnan(Z)).sum(axis=1)
mu, sd = np.nanmean(Z, axis=1), np.nanstd(Z, axis=1)
okz = (nobs >= 26) & (sd > 0)
Z = (Z - mu[:, None]) / np.where(sd > 0, sd, 1)[:, None]
LAGS = [1, 2, 3, 4, 5, 6, 8, 13, 26, 52]
rows = []
for t in TIERS:
    zt = Z[okz & (TIER == t)]
    r = {"tier": t, "drugs": len(zt)}
    for k in LAGS:
        a, b = zt[:, k:], zt[:, :-k]
        ok = ~np.isnan(a) & ~np.isnan(b)
        r[f"lag{k}"] = (a * b)[ok].sum() / np.sqrt((a[ok] ** 2).sum() * (b[ok] ** 2).sum())
    rows.append(r)
show("pooled ACF of weekly demand", pd.DataFrame(rows), index=False)

# Same thing on week-to-week changes: what is left once the level is known.
D = np.diff(Z, axis=1)
rows = []
for t in TIERS:
    dt = D[okz & (TIER == t)]
    r = {"tier": t}
    for k in [1, 2, 3, 4, 52]:
        a, b = dt[:, k:], dt[:, :-k]
        ok = ~np.isnan(a) & ~np.isnan(b)
        r[f"lag{k}"] = (a * b)[ok].sum() / np.sqrt((a[ok] ** 2).sum() * (b[ok] ** 2).sum())
    rows.append(r)
show("pooled ACF of week-to-week changes", pd.DataFrame(rows), index=False)

# Which plain window predicts the next 8 weeks best? At each selection origin,
# drugs sold in the 13 weeks before it, tier as of the origin, censored target
# weeks dropped. A flat forecast from each window; SES (alpha 0.15) for
# reference. Last year: the same week 52 back, and the mean of the 3 around it.
SEL = split.selection_origins()
TIER_AT = {o: tiers_at(o).tier.fillna("none").to_numpy() for o in SEL}
WINDOWS = [1, 2, 4, 8, 13, 26, 52]
parts = []
for o in SEL:
    n = history.weeks_to(H, o)
    yc = YC[:, :n]
    scored = np.nansum(Y[:, max(0, n - 13):n], axis=1) > 0
    preds = {f"mean{k}": window_mean(yc, n, k) for k in WINDOWS}
    preds["ses"] = fc._ses(yc)
    for h in range(1, 9):
        t_col = n + h - 1
        actual = YC[:, t_col]
        ly1 = YC[:, t_col - 52] if t_col >= 52 else np.full(N_DRUGS, np.nan)
        ly3 = (np.nanmean(YC[:, t_col - 53:t_col - 50], axis=1) if t_col >= 53
               else np.full(N_DRUGS, np.nan))
        p = {**preds, "ly1": ly1, "ly3": ly3}
        df = pd.DataFrame({"origin": o, "h": h, "tier": TIER_AT[o], "actual": actual,
                           **{k: np.nan_to_num(v) for k, v in p.items()},
                           "has_ly": ~np.isnan(ly3)})
        parts.append(df[scored & ~np.isnan(actual)])
BT = pd.concat(parts, ignore_index=True)
BT = BT[BT.tier.isin(TIERS)]
METHODS = [f"mean{k}" for k in WINDOWS] + ["ses"]


def wape_table(bt, methods):
    g = bt.groupby("tier")
    out = pd.DataFrame({m: g.apply(lambda d, m=m: (d[m] - d.actual).abs().sum()
                                   / d.actual.sum()) for m in methods})
    return out.reindex(TIERS)


def bias_table(bt, methods):
    g = bt.groupby("tier")
    return pd.DataFrame({m: g[m].sum() / g.actual.sum() for m in methods}).reindex(TIERS)


show("WAPE of flat window forecasts, 11 selection origins, h 1-8", wape_table(BT, METHODS))
show("bias (forecast / actual)", bias_table(BT, METHODS))

# ---------------------------------------------------------------------------
header("4. Eid: does it hit every drug the same way?")
# For each Eid in the data, each week from the one before Eid's week to four
# after, over the drug's normal level: its mean weekly units in the weeks
# fully inside days -63 to -15 before Eid (the base the store factor uses).

REL = range(-1, 5)


def eid_weeks(eid):
    mon = eid - timedelta(days=eid.weekday())
    win = [WEEK_IDX.get(mon + timedelta(weeks=r)) for r in REL]
    base = [j for j, w in enumerate(WEEKS)
            if w >= eid + timedelta(days=-63) and w + timedelta(days=6) <= eid - timedelta(days=15)]
    return win, base


lines_w = H["lines"].groupby(lambda d: d - timedelta(days=d.weekday())).sum()
EIDDATA = {}
for eid in EIDS:
    win, base = eid_weeks(eid)
    b = np.nanmean(YC[:, base], axis=1)
    W = YC[:, win]
    ok = (b > 0) & (np.isin(TIER, TIERS))
    store_base = lines_w[[WEEKS[j] for j in base]].mean()
    prof = {"store lines": [lines_w[WEEKS[j]] / store_base for j in win]}
    for t in TIERS:
        m = ok & (TIER == t)
        prof[t] = [np.nansum(W[m, i]) / b[m][~np.isnan(W[m, i])].sum() for i in range(len(win))]
    show(f"Eid {eid}: weekly demand over normal, by tier (week 0 holds Eid)",
         pd.DataFrame(prof, index=[f"wk{r:+d}" for r in REL]).T)
    ratio = np.nansum(W, axis=1) / (b * (~np.isnan(W)).sum(axis=1))
    EIDDATA[eid] = {"base": b, "W": W, "ratio": ratio, "win": win}

# Spread of each drug's Eid-window ratio, drugs with a base of 3+ units a week.
rows = []
for eid, d in EIDDATA.items():
    for t in TIERS:
        m = (TIER == t) & (d["base"] >= 3) & np.isfinite(d["ratio"])
        if m.sum() < 10:
            continue
        r = d["ratio"][m]
        rows.append({"eid": eid, "tier": t, "drugs": int(m.sum()),
                     **{f"p{p}": np.percentile(r, p) for p in (10, 25, 50, 75, 90)}})
show("each drug's 6-week Eid ratio (drugs with base >= 3 units/week)",
     pd.DataFrame(rows), index=False)

# Does a drug's own Eid ratio repeat from 2023 to 2024? If it doesn't, a
# per-drug Eid effect can't be learned from one past Eid.
e23, e24 = EIDDATA[EIDS[0]], EIDDATA[EIDS[1]]
lr23, lr24 = np.log(e23["ratio"]), np.log(e24["ratio"])
rows = []
for floor in (1, 3, 10):
    m = (e23["base"] >= floor) & (e24["base"] >= floor) & np.isfinite(lr23) & np.isfinite(lr24)
    rows.append({"base >=": floor, "drugs": int(m.sum()),
                 "corr log ratio 2023 vs 2024": corr(lr23[m], lr24[m]),
                 "spearman": pd.Series(lr23[m]).corr(pd.Series(lr24[m]), method="spearman")})
show("repeatability of each drug's Eid ratio", pd.DataFrame(rows), index=False)

# ---------------------------------------------------------------------------
header("5-6. Ingredient groups: definitions, sizes, form mix")

imap = con.execute("SELECT drug_code, ingredients, form, kind FROM ingredient_map").df()
imap = imap.set_index("drug_code").reindex(DRUGS)
imap["ingredients"] = imap.ingredients.fillna("")
imap["form"] = imap.form.fillna("")
imap["route"] = imap.form.map(ROUTE).fillna("other")
has_ing = imap.ingredients.ne("").to_numpy()
GROUPDEFS = {
    "ingredient": imap.ingredients,
    "ingredient_route": imap.ingredients + " | " + imap.route,
    "ingredient_form": imap.ingredients + " | " + imap.form,
}
rev24 = np.nansum(H["revenue"][:, [j for j, w in enumerate(WEEKS) if w.year == 2024]], axis=1)
rows = []
GROUP_OF = {}
for name, key in GROUPDEFS.items():
    g = pd.Series(np.where(has_ing & ACTIVE, key.to_numpy(), None), index=DRUGS)
    GROUP_OF[name] = g
    size = g.map(g.value_counts())
    multi = (size >= 2).to_numpy()
    gs = g[multi].value_counts()
    r = {"definition": name, "groups": int(g.nunique()), "groups 2+": len(gs),
         "drugs with siblings": int(multi.sum()),
         "share of active drugs": multi.sum() / ACTIVE.sum(),
         "share of 2024 revenue": rev24[multi].sum() / rev24[ACTIVE].sum(),
         "median size (2+)": gs.median(), "max size": gs.max()}
    if name == "ingredient":
        d = pd.DataFrame({"g": g[multi], "form": imap.form[multi], "route": imap.route[multi]})
        r["groups mixing forms"] = (d.groupby("g").form.nunique() > 1).mean()
        r["groups mixing routes"] = (d.groupby("g").route.nunique() > 1).mean()
    rows.append(r)
show("group definitions (drugs sold in the last 52 weeks with a mapped ingredient)",
     pd.DataFrame(rows).set_index("definition").T)
show("active drugs by route", imap[ACTIVE & has_ing].route.value_counts().to_frame().T)

# ---------------------------------------------------------------------------
header("5. Do drugs with the same ingredient move together?")
# Demand index: each week's units over the drug's mean of the 13 weeks before
# it (6+ observed weeks), censored weeks masked, clipped at 10. Store-wide
# movement (traffic, Eid) is removed by dividing by the mean index of all
# fast, regular and slow drugs that week. Each drug is then correlated with
# its siblings' leave-one-out mean index, and with the mean of as many random
# drugs from outside its group (the control). Drugs need 26+ paired weeks.

trail = np.full(Y.shape, np.nan)
cnt = np.zeros(Y.shape)
for t in range(13, N_WEEKS):
    blk = YC[:, t - 13:t]
    cnt[:, t] = (~np.isnan(blk)).sum(axis=1)
    trail[:, t] = np.nanmean(blk, axis=1)
IDX = np.where((cnt >= 6) & (trail > 0), YC / trail, np.nan)
IDX = np.minimum(IDX, 10)
ELIG = np.isin(TIER, ["fast", "regular", "slow"]) & ACTIVE
store_idx = np.nanmean(IDX[ELIG], axis=0)
R = IDX / store_idx[None, :]
R[~ELIG] = np.nan


def block_index(width):
    """Store-adjusted index over non-overlapping blocks of `width` weeks."""
    starts = list(range(13, N_WEEKS - width + 1, width))
    out = np.full((N_DRUGS, len(starts)), np.nan)
    for b, t in enumerate(starts):
        blk = YC[:, t:t + width]
        ok = ((~np.isnan(blk)).sum(axis=1) >= (width + 1) // 2) & (cnt[:, t] >= 6) \
            & (trail[:, t] > 0)
        out[:, b] = np.where(ok, np.minimum(np.nanmean(blk, axis=1) / trail[:, t], 10), np.nan)
    out[~ELIG] = np.nan
    return out / np.nanmean(out, axis=0)[None, :]


def comove(groups, rmat, lag=0, min_pairs=26):
    """Per drug: corr with siblings' LOO mean index, and with a random control set.

    Also returns the centred (drug, sibling, control) values for a pooled
    correlation over every drug-period.
    """
    g = groups.to_numpy()
    out, pooled = [], []
    members: dict[str, list[int]] = {}
    for i in np.flatnonzero(ELIG):
        if g[i] is not None:
            members.setdefault(g[i], []).append(i)
    pool = np.flatnonzero(ELIG)
    for _key, idx in members.items():
        if len(idx) < 2:
            continue
        for i in idx:
            sib = [j for j in idx if j != i]
            s = np.nanmean(rmat[sib], axis=0)
            ctrl = RNG.choice(pool[~np.isin(pool, idx)], size=len(sib), replace=False)
            c = np.nanmean(rmat[ctrl], axis=0)
            a = rmat[i, lag:]
            s, c = s[:len(s) - lag], c[:len(c) - lag]
            ok = ~np.isnan(a) & ~np.isnan(s) & ~np.isnan(c)
            if ok.sum() < min_pairs:
                continue
            out.append({"tier": TIER[i], "sib": corr(a, s), "random": corr(a, c)})
            pooled.append(pd.DataFrame({"drug": i, "tier": TIER[i], "a": a[ok] - a[ok].mean(),
                                        "s": s[ok] - s[ok].mean(), "c": c[ok] - c[ok].mean()}))
    return pd.DataFrame(out), pd.concat(pooled, ignore_index=True)


def pooled_corr(d, col):
    return (d.a * d[col]).sum() / np.sqrt((d.a ** 2).sum() * (d[col] ** 2).sum())


def diff_ci(d, reps=1000):
    """90% interval of pooled corr (siblings) - pooled corr (random), resampling drugs."""
    per = d.assign(as_=d.a * d.s, ac=d.a * d.c, aa=d.a ** 2, ss=d.s ** 2, cc=d.c ** 2
                   ).groupby("drug")[["as_", "ac", "aa", "ss", "cc"]].sum().to_numpy()
    w = RNG.multinomial(len(per), np.full(len(per), 1 / len(per)), size=reps)
    t = w @ per
    diff = t[:, 0] / np.sqrt(t[:, 2] * t[:, 3]) - t[:, 1] / np.sqrt(t[:, 2] * t[:, 4])
    return np.percentile(diff, [5, 95])


GRAINS = {"weekly": (R, 26), "4-week blocks": (block_index(4), 12),
          "13-week blocks": (block_index(13), 5)}
rows = []
for grain, (rmat, min_pairs) in GRAINS.items():
    for name, groups in GROUP_OF.items():
        for lag in ((0, 1) if grain == "weekly" else (0,)):
            cm, pl = comove(groups, rmat, lag, min_pairs)
            for t in ["fast", "regular", "slow", "all"]:
                d = cm if t == "all" else cm[cm.tier == t]
                p = pl if t == "all" else pl[pl.tier == t]
                rows.append({"grain": grain, "definition": name, "lag": lag, "tier": t,
                             "drugs": len(d), "sib median": d.sib.median(),
                             "random median": d.random.median(),
                             "sib pooled": pooled_corr(p, "s"),
                             "random pooled": pooled_corr(p, "c"),
                             "diff 90% CI": "{:+.3f} to {:+.3f}".format(*diff_ci(p)),
                             "sib > random": (d.sib > d.random).mean()})
show("correlation with siblings' leave-one-out index vs random drugs "
     "(lag 1 = siblings one week earlier; pooled = over all drug-periods)",
     pd.DataFrame(rows), index=False)

# Eid by group: does the siblings' 2023 Eid ratio predict a drug's 2024 one?
# Predict each drug's 2024 Eid-window weeks as its 2024 base times a factor,
# and compare factors learned from the 2023 Eid only.
ing = GROUP_OF["ingredient"].to_numpy()
win24 = e24["win"]
b23, b24 = e23["base"], e24["base"]
pooled23 = np.nansum(e23["W"][ELIG & (b23 > 0)]) / (6 * b23[ELIG & (b23 > 0)].sum())
store_f = fc.eid_index([WEEKS[j] for j in win24],
                       fc.eid_factors(H["lines"][H["lines"].index <= split.FIRST_ORIGIN]))
tier23 = {t: np.nansum(e23["W"][(TIER == t) & (b23 > 0)])
          / (6 * b23[(TIER == t) & (b23 > 0)].sum()) for t in TIERS}
grp23 = {}
for i in np.flatnonzero((b23 > 0) & (ing != None)):  # noqa: E711
    grp23.setdefault(ing[i], []).append(i)
use = ELIG & (b23 >= 1) & (b24 >= 1) & ((~np.isnan(e24["W"])).sum(axis=1) >= 4)
mult = {"none": np.full(N_DRUGS, np.nan), "store (current)": np.ones(N_DRUGS),
        "store x tier": np.array([tier23.get(t, pooled23) / pooled23 for t in TIER]),
        "store x drug": e23["ratio"] / pooled23,
        "store x drug, half-shrunk": np.sqrt(e23["ratio"] / pooled23),
        "store x group (LOO)": np.ones(N_DRUGS)}
has_grp = np.zeros(N_DRUGS, bool)
for i in np.flatnonzero(use):
    sib = [j for j in grp23.get(ing[i], []) if j != i]
    if sib:
        gr = np.nansum(e23["W"][sib]) / (6 * b23[sib].sum())
        mult["store x group (LOO)"][i] = gr / pooled23
        has_grp[i] = True
rows = []
for subset, m in (("all eligible", use), ("drugs with siblings", use & has_grp)):
    act = e24["W"][m]
    for name, mm in mult.items():
        f = np.ones(len(win24)) if name == "none" else store_f
        mm = np.nan_to_num(mm[m], nan=1.0, posinf=1.0)
        pred = b24[m][:, None] * f[None, :] * mm[:, None]
        ok = ~np.isnan(act)
        rows.append({"drugs": subset, "factor": name, "n": int(m.sum()),
                     "wape": np.abs(pred - act)[ok].sum() / act[ok].sum(),
                     "bias": pred[ok].sum() / act[ok].sum()})
show("forecasting the 2024 Eid window from factors learned on 2023 "
     "(fast/regular/slow, base >= 1 unit/week both years)", pd.DataFrame(rows), index=False)

# ---------------------------------------------------------------------------
header("7. History length at each selection origin, and the Prophet subset")
# Weeks since first sale, drugs sold in the 52 weeks before the origin. The
# Prophet subset (plan 6.4): fast and regular at the origin, 52+ weeks since
# first sale, under 20% of those weeks censored.

rows = []
for o in SEL:
    c = history.cut(H, o)
    n = len(c["weeks"])
    act = history.active(H, o, schema.ACTIVE_WEEKS)
    first = c["first_sale"]
    age = np.where(first >= 0, n - first, 0)
    cen = np.array([c["out"][i, first[i]:].mean() if first[i] >= 0 else 0
                    for i in range(N_DRUGS)])
    tier = TIER_AT[o]
    fr = np.isin(tier, ["fast", "regular"])
    a = age[act]
    rows.append({"origin": o, "active": int(act.sum()),
                 "<13": int((a < 13).sum()), "13-25": int(((a >= 13) & (a < 26)).sum()),
                 "26-51": int(((a >= 26) & (a < 52)).sum()), "52+": int((a >= 52).sum()),
                 "fast+reg": int((act & fr).sum()),
                 "fast+reg 52+wk": int((act & fr & (age >= 52)).sum()),
                 "prophet subset": int((act & fr & (age >= 52) & (cen < 0.2)).sum())})
show("weeks since first sale (active drugs), and the Prophet subset",
     pd.DataFrame(rows), index=False)

# ---------------------------------------------------------------------------
header("8. Yearly seasonality beyond Eid")
# The store-adjusted demand index R (section 5) against itself 52 weeks back,
# pooled per tier, weeks in an Eid window (now or then) left out. Lags 50-54
# other than 52 are the placebo: a real yearly pattern shows as a peak at 52.

rows = []
for t in ["fast", "regular", "slow"]:
    r = {"tier": t}
    rt = R[TIER == t]
    rt = rt - np.nanmean(rt, axis=1, keepdims=True)
    for k in (48, 50, 51, 52, 53, 54, 56):
        a, b = rt[:, k:], rt[:, :-k]
        noeid = ~EID_WEEK[k:] & ~EID_WEEK[:-k]
        ok = ~np.isnan(a) & ~np.isnan(b) & noeid[None, :]
        r[f"lag{k}"] = (a * b)[ok].sum() / np.sqrt((a[ok] ** 2).sum() * (b[ok] ** 2).sum())
    rows.append(r)
show("pooled correlation of the store-adjusted index with itself k weeks back",
     pd.DataFrame(rows), index=False)

lw = lines_w.reindex(WEEKS).to_numpy()
ltrail = np.array([lw[max(0, t - 13):t].mean() if t >= 13 else np.nan for t in range(N_WEEKS)])
li = lw / ltrail
rows = []
for k in (50, 51, 52, 53, 54):
    ok = ~EID_WEEK[k:] & ~EID_WEEK[:-k]
    rows.append({"lag": k, "weeks": int((ok & ~np.isnan(li[:-k])).sum()),
                 "corr": corr(np.where(ok, li[k:], np.nan), np.where(ok, li[:-k], np.nan))})
show("store sale lines over their trailing 13-week mean, vs k weeks back (Eid weeks out)",
     pd.DataFrame(rows), index=False)

# Direct test: does last year help a flat forecast? Target weeks with a value
# 52 weeks back; 'm13 x ly shape' scales the 13-week mean by last year's
# (3 weeks around the target) over last year's 13 weeks before the origin.
ly_note = ("m13 x ly shape needs 13 weeks before the same origin a year back, so "
           "origins from Apr 2024 only")
parts = []
for o in SEL:
    n = history.weeks_to(H, o)
    if n - 52 < 13:
        continue
    base_ly = window_mean(YC, n - 52, 13)
    m13 = window_mean(YC[:, :n], n, 13)
    scored = np.nansum(Y[:, n - 13:n], axis=1) > 0
    for h in range(1, 9):
        tc = n + h - 1
        act = YC[:, tc]
        ly3 = np.nanmean(YC[:, tc - 53:tc - 50], axis=1)
        shape = np.where(base_ly > 0, ly3 / base_ly, np.nan)
        df = pd.DataFrame({"tier": TIER_AT[o], "actual": act, "mean13": np.nan_to_num(m13),
                           "ly3": ly3,
                           "m13 x ly shape": m13 * shape,
                           "m13 x sqrt(ly shape)": m13 * np.sqrt(shape)})
        parts.append(df[scored & ~np.isnan(act) & ~np.isnan(shape) & ~EID_WEEK[tc]])
LYT = pd.concat(parts, ignore_index=True)
LYT = LYT[LYT.tier.isin(TIERS)].fillna(0)
cols = ["mean13", "ly3", "m13 x ly shape", "m13 x sqrt(ly shape)"]
show(f"WAPE, non-Eid target weeks ({ly_note})", wape_table(LYT, cols))

# ---------------------------------------------------------------------------
header("9. How stable is the store?")

first_sale = H["first_sale"]
last_sale = np.array([np.flatnonzero(np.nan_to_num(Y[i]) > 0).max()
                      if first_sale[i] >= 0 else -1 for i in range(N_DRUGS)])
rows = []
for quarter in sorted(set(q)):
    cols_q = np.flatnonzero(q == quarter)
    sold = (np.nan_to_num(Y[:, cols_q]) > 0).any(axis=1)
    rows.append({"quarter": quarter, "weeks": len(cols_q),
                 "lines/week": lw[cols_q].mean(),
                 "units/week": np.nansum(Y[:, cols_q]) / len(cols_q),
                 "drugs selling": int(sold.sum()),
                 "new drugs": int(np.isin(first_sale, cols_q).sum()),
                 "stopped": (int(np.isin(last_sale, cols_q).sum())
                             if cols_q[-1] + 13 < N_WEEKS else None)})
show("by quarter (last runs to 8 Dec; 'stopped' = last sale in the quarter, "
     "shown once 13 weeks have passed)",
     pd.DataFrame(rows), index=False)

# Level drift of drugs that sold through the whole period: H2 2024 mean over
# H1 2023 mean, per tier.
h1_23 = np.array([w < date(2023, 7, 1) for w in WEEKS])
h2_24 = np.array([w >= date(2024, 7, 1) for w in WEEKS])
m1, m2 = np.nanmean(YC[:, h1_23], axis=1), np.nanmean(YC[:, h2_24], axis=1)
rows = []
for t in ["fast", "regular", "slow"]:
    m = (TIER == t) & (m1 > 0) & (m2 > 0)
    lr = np.log(m2[m] / m1[m])
    rows.append({"tier": t, "drugs": int(m.sum()), "median ratio": np.exp(np.median(lr)),
                 "median |log ratio|": np.median(np.abs(lr)),
                 "share moved > 2x either way": (np.abs(lr) > np.log(2)).mean()})
show("drug level, H2 2024 over H1 2023 (drugs selling in both)",
     pd.DataFrame(rows), index=False)

u24q4 = np.nansum(Y[:, q == "2024Q4"], axis=1)
newer = first_sale >= int(np.flatnonzero(q == "2023Q3")[0])
print(f"\nShare of 2024Q4 units from drugs first sold after June 2023: "
      f"{u24q4[newer].sum() / u24q4.sum():.3f}")

# Does the error of a fixed recipe drift over 2024? WAPE of mean13 per origin.
per_origin = BT.groupby(["origin", "tier"]).apply(
    lambda d: (d.mean13 - d.actual).abs().sum() / d.actual.sum()).unstack()[TIERS]
show("WAPE of the 13-week mean at each selection origin", per_origin)

# ---------------------------------------------------------------------------
header("10. How much do sales drop on shelf-out days?")
# Shelf-out: shop floor empty while the upstairs store has stock. Daily units
# on shelf-out days over the drug's normal daily units: the mean of the 28
# days before the spell, leaving out shelf-out and total stock-out days
# (14+ such days needed). Tier as of the end of the selection data.

d0 = config.STOCK_START - timedelta(days=28)
days = pd.date_range(d0, END).date
DAY = {d: i for i, d in enumerate(days)}
dd = con.execute(f"SELECT drug_code, date, units FROM daily_demand "
                 f"WHERE date >= DATE '{d0}'").df()
dd["date"] = pd.to_datetime(dd.date).dt.date
dd = dd[dd.drug_code.isin(di)]
U = np.zeros((N_DRUGS, len(days)))
U[dd.drug_code.map(di), dd.date.map(DAY)] = dd.units
so = con.execute(f"""SELECT drug_code, date FROM daily_stock
    WHERE location = 'total' AND closing <= 0 AND date >= DATE '{d0}'""").df()
so["date"] = pd.to_datetime(so.date).dt.date
so = so[so.drug_code.isin(di)]
SO = np.zeros(U.shape, bool)
SO[so.drug_code.map(di), so.date.map(DAY)] = True
spells = con.execute("SELECT drug_code, start, stop FROM shelfouts").df()
spells = spells[spells.drug_code.isin(di)]
SH = np.zeros(U.shape, bool)
spell_list = []
for code, a, b in zip(spells.drug_code, pd.to_datetime(spells.start).dt.date,
                      pd.to_datetime(spells.stop).dt.date, strict=True):
    i, ia, ib = di[code], DAY[a], DAY[b]
    SH[i, ia:ib + 1] = True
    spell_list.append((i, ia, ib))
rows = []
for i, ia, ib in spell_list:
    lo = max(0, ia - 28)
    ok = ~SH[i, lo:ia] & ~SO[i, lo:ia]
    if ok.sum() < 14:
        continue
    base = U[i, lo:ia][ok].mean()
    if base <= 0:
        continue
    for k, j in enumerate(range(ia, ib + 1)):
        if SO[i, j]:
            continue
        rows.append({"tier": TIER[i], "day": k + 1, "units": U[i, j], "base": base})
    if ib + 1 < len(days) and not SH[i, ib + 1] and not SO[i, ib + 1]:
        rows.append({"tier": TIER[i], "day": 0, "units": U[i, ib + 1], "base": base})
SD = pd.DataFrame(rows)
SD = SD[SD.tier.isin(TIERS)]
SD["when"] = pd.cut(SD.day, [-1, 0, 1, 2, 7, 10_000],
                    labels=["day after", "day 1", "day 2", "days 3-7", "day 8+"])
g = SD.groupby(["tier", "when"], observed=True)
tab = (g.units.sum() / g.base.sum()).unstack()
tab["all days"] = SD[SD.day > 0].groupby("tier").apply(lambda d: d.units.sum() / d.base.sum())
tab["drug-days"] = SD[SD.day > 0].groupby("tier").size()
show("sales on shelf-out days over normal (1.0 = no drop)", tab.reindex(TIERS))
alld = SD[SD.day > 0]
print(f"\nAll tiers: {alld.units.sum() / alld.base.sum():.3f} of normal over "
      f"{len(alld):,} shelf-out drug-days.")

# The same at weekly grain, as the panel sees it: uncensored weeks' units over
# the drug's trailing 13-week mean, by shelf-out days in the week.
sh = H["shelfout_days"]
rows = []
for t in ["fast", "regular", "slow"]:
    m = (TIER == t)[:, None] & ~np.isnan(YC) & ~np.isnan(trail) & (trail > 0) & live
    r = {"tier": t}
    for lab, lo, hi in (("0", 0, 0), ("1", 1, 1), ("2-3", 2, 3), ("4-6", 4, 6), ("7", 7, 7)):
        mm = m & (sh >= lo) & (sh <= hi)
        r[lab] = np.nansum(YC[mm]) / trail[mm].sum()
        r[f"n{lab}"] = int(mm.sum())
    rows.append(r)
show("weekly units over trailing 13-week mean, by shelf-out days in the week "
     "(uncensored weeks)", pd.DataFrame(rows), index=False)

con.close()
print(f"\nDone in {time.time() - T0:.0f}s.")
