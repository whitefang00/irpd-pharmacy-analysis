"""Scores of lead-time totals (roadmap, "Fixed definitions"; E0b).

Input is an arm's predictions in the shared format (IMPL-SPEC): one row per
(origin, drug_code, h) with the weekly `mean` and `var` and the drug's `tier`
at the origin. For each (origin, drug) the scorer builds totals the way
`contest.to_policy` builds them: each week spread evenly over its 7 days,
summed from the Monday after the origin over

- the lead window: the drug's lead time (`lead_days`, clipped 1-14; 7 when
  unknown), and
- the lead + cycle window: lead + 14 days,

and the NB quantile of a total is `forecast.demand_quantile` on the summed
mean and variance. An optional variance factor 1 + rho_tier (w - 1), with w
the window in weeks (at least 1), scales a window's summed variance.

Truth is uncensored demand (`lost.uncensored_units`): a daily frame gives
exact window totals; with weekly actuals only, partial weeks are pro-rated.
A window is censored if it holds a stock-out day (daily), or overlaps a week
flagged censored (weekly). Censored windows are scored on uncensored demand
and reported apart; the headline uses all windows.

Scores, each by tier and for the pooled fast + regular + slow set (the gate):

- `pinball_lc`, `pinball_l`: revenue-weighted pinball loss at 0.95,
  sum(price x loss) / sum(price x truth). `pinball_lc` is the primary score.
- `crps_lc`: revenue-weighted CRPS of the NB lead + cycle total, same scaling.
- `cov_l`, `cov_lc`: share of windows with truth <= the p95.
- `bias4`, `bias8`: sum forecast / sum truth on 4- and 8-week totals. The
  roadmap's 13-week bias is not possible from h <= 8, so it is 8 weeks here.
- `rwape4`: revenue-weighted WAPE of 4-week totals.
- `wape`: unit WAPE over weekly rows h 1-8 on uncensored truth;
  `wape_clean`: the same on recorded units, uncensored weeks only (as
  earlier work scored it).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy import stats

from irpd import forecast as fc
from irpd.contest import window_var_factor
from irpd.evalx import bootstrap

Q = 0.95
CYCLE_DAYS = 14
LEAD_CLIP = (1, 14)
LEAD_DEFAULT = 7
GATE_TIERS = ("fast", "regular", "slow")
TIERS = ("fast", "regular", "slow", "rare")
PRICE_COLS = ("price", "avg_price", "price_182d", "avg_price_182d")
# Revenue-weighted scores: (numerator column, denominator column) in `windows`.
RATIO = {"pinball_lc": ("w_pin_lc", "w_y_lc"), "pinball_l": ("w_pin_l", "w_y_l"),
         "crps_lc": ("w_crps_lc", "w_y_lc"), "cov_l": ("cov_l", "one"),
         "cov_lc": ("cov_lc", "one"), "bias4": ("f4", "y4"), "bias8": ("f8", "y8"),
         "rwape4": ("w_ae4", "w_y4")}
WEEKLY = {"wape": ("ae", "y"), "wape_clean": ("ae_clean", "y_clean")}
METRICS = (*RATIO, *WEEKLY)


# ---- distributional pieces ----------------------------------------------


def pinball(y: np.ndarray, q: np.ndarray, tau: float = Q) -> np.ndarray:
    """Pinball (quantile) loss of quantile forecast q at level tau."""
    d = np.asarray(y, float) - np.asarray(q, float)
    return np.maximum(tau * d, (tau - 1) * d)


def _crps_normal(m, sd, y):
    z = (y - m) / sd
    return sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def crps_nb(mean, var, y, max_cells: int = 2_000_000, exact_up_to: float = 20_000
            ) -> np.ndarray:
    """CRPS of a negative binomial (mean, var >= mean) at real-valued y >= 0.

    Exact on the integer support: CRPS = integral of (F(x) - 1{x >= y})^2,
    with F constant on [z, z + 1). The sum runs from below both y and the
    mass (mean - 10 sd) to above both (mean + 10 sd); outside it the integrand
    is below 1e-20. Mean 0 is a point mass at 0 (CRPS = y). Totals above
    `exact_up_to` use the normal CRPS with the same mean and sd.
    """
    m = np.asarray(mean, float).ravel()
    y = np.asarray(y, float).ravel()
    d = np.maximum(np.asarray(var, float).ravel() / np.where(m > 0, m, 1), 1 + 1e-9)
    sd = np.sqrt(m * d)
    out = np.where(m > 0, 0.0, y)
    lo = np.floor(np.maximum(0, np.minimum(y, m - 10 * sd))).astype(int)
    hi = np.ceil(np.maximum(y, m + 10 * sd)).astype(int) + 2
    big = (m > 0) & (m + 10 * sd > exact_up_to)
    if big.any():
        out[big] = _crps_normal(m[big], sd[big], y[big])
    idx = np.flatnonzero((m > 0) & ~big)
    size = hi - lo
    idx = idx[np.argsort(size[idx])]
    while len(idx):
        cells = size[idx] * np.arange(1, len(idx) + 1)
        n = max(1, int(np.searchsorted(cells, max_cells, side="right")))
        rows, idx = idx[:n], idx[n:]
        z = lo[rows][:, None] + np.arange(size[rows].max())[None, :]
        r = (m[rows] / (d[rows] - 1))[:, None]
        F = stats.nbinom.cdf(z, r, 1 / d[rows][:, None])
        below = np.clip(y[rows][:, None] - z, 0, 1)  # length of [z, z+1) left of y
        inside = z < hi[rows][:, None]
        out[rows] = (np.where(inside, below * F ** 2 + (1 - below) * (1 - F) ** 2, 0)).sum(1)
    return out


def window_weights(days: np.ndarray, weeks: int = 8) -> np.ndarray:
    """rows x weeks: the share of each week inside the first `days` days."""
    h = np.arange(weeks)[None, :]
    return np.clip(np.asarray(days, float)[:, None] - 7 * h, 0, 7) / 7


# ---- inputs ---------------------------------------------------------------


def _wide(preds: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    p = preds.copy()
    p["origin"] = pd.to_datetime(p["origin"]).dt.date
    p["drug_code"] = p["drug_code"].astype(str)
    p["h"] = p["h"].astype(int)
    p = p[p["h"].between(1, 8)]
    w = p.pivot_table(index=["origin", "drug_code"], columns="h", values=["mean", "var"],
                      aggfunc="first")
    tier = p.groupby(["origin", "drug_code"])["tier"].first()
    full = w["mean"].notna().all(axis=1) & w["var"].notna().all(axis=1)
    w = w[full]
    out = pd.DataFrame(index=w.index)
    out["tier"] = tier.reindex(w.index).astype(str)
    return out, w["mean"][list(range(1, 9))].to_numpy(float), \
        w["var"][list(range(1, 9))].to_numpy(float)


def _labels(idx: pd.MultiIndex, labels: pd.DataFrame | None,
            prices: pd.DataFrame | None) -> pd.DataFrame:
    out = pd.DataFrame(index=idx)
    out["lead"], out["price"] = np.nan, np.nan
    if labels is not None:
        lb = labels.copy()
        lb["origin"] = pd.to_datetime(lb["origin"]).dt.date
        lb["drug_code"] = lb["drug_code"].astype(str)
        lb = lb.drop_duplicates(["origin", "drug_code"]).set_index(["origin", "drug_code"])
        if "lead_days" in lb:
            out["lead"] = lb["lead_days"].reindex(idx).to_numpy(float)
        pc = next((c for c in PRICE_COLS if c in lb), None)
        if pc:
            out["price"] = lb[pc].reindex(idx).to_numpy(float)
    if prices is not None and out["price"].isna().any():
        pr = prices.copy()
        pr["origin"] = pd.to_datetime(pr["origin"]).dt.date
        pr = pr.set_index(["origin", "drug_code"])["price"]
        out["price"] = out["price"].fillna(pd.Series(pr.reindex(idx).to_numpy(), index=idx))
    out["lead"] = out["lead"].fillna(LEAD_DEFAULT).clip(*LEAD_CLIP).round().astype(int)
    out["price"] = out["price"].fillna(0.0)
    return out


@dataclass
class Truth:
    """Daily (or pro-rated weekly) uncensored demand, as cumulative sums per drug."""

    drugs: dict[str, int]
    first: object                    # date of column 0
    cum_y: np.ndarray                # drugs x (days + 1)
    cum_out: np.ndarray              # cumulative stock-out days
    daily: bool

    def total(self, rows: np.ndarray, start_off: np.ndarray, days: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sum over days [start, start + days) per row: truth, censored, available."""
        end = start_off + days
        ok = (rows >= 0) & (start_off >= 0) & (end <= self.cum_y.shape[1] - 1)
        r, a, b = np.where(ok, rows, 0), np.where(ok, start_off, 0), np.where(ok, end, 0)
        y = self.cum_y[r, b] - self.cum_y[r, a]
        c = (self.cum_out[r, b] - self.cum_out[r, a]) > 0
        return np.where(ok, y, np.nan), c & ok, ok


def truth_from_daily(daily: pd.DataFrame) -> Truth:
    """Truth from (drug_code, date, uncensored_units, stockout)."""
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.date
    first, last = min(d["date"]), max(d["date"])
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    codes = sorted(d["drug_code"].astype(str).unique())
    y = (d.pivot_table(index="drug_code", columns="date", values="uncensored_units",
                       aggfunc="sum").reindex(index=codes, columns=days).fillna(0).to_numpy())
    o = (d.pivot_table(index="drug_code", columns="date", values="stockout", aggfunc="max")
         .reindex(index=codes, columns=days).fillna(0).to_numpy(float))
    z = np.zeros((len(codes), 1))
    return Truth({c: i for i, c in enumerate(codes)}, first,
                 np.concatenate([z, np.cumsum(y, 1)], 1),
                 np.concatenate([z, np.cumsum(o, 1)], 1), True)


def truth_from_weekly(actuals: pd.DataFrame) -> Truth:
    """Truth from weekly actuals, each week spread evenly over its days."""
    a = actuals.copy()
    a["week_start"] = pd.to_datetime(a["week_start"]).dt.date
    col = "uncensored_units" if "uncensored_units" in a else "units"
    daily = pd.DataFrame({
        "drug_code": np.repeat(a["drug_code"].astype(str).to_numpy(), 7),
        "date": [w + timedelta(days=i) for w in a["week_start"] for i in range(7)],
        "uncensored_units": np.repeat(a[col].to_numpy(float) / 7, 7),
        "stockout": np.repeat(a["censored"].astype(bool).to_numpy()
                              if "censored" in a else np.zeros(len(a), bool), 7)})
    t = truth_from_daily(daily)
    t.daily = False
    return t


def weekly_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Weekly actuals (Monday weeks) from a daily truth frame."""
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["week_start"] = (d["date"] - pd.to_timedelta(d["date"].dt.dayofweek, unit="D")).dt.date
    return (d.groupby(["drug_code", "week_start"])
             .agg(units=("units", "sum"), uncensored_units=("uncensored_units", "sum"),
                  censored=("stockout", "max"), days=("date", "size"))
             .reset_index().query("days == 7").drop(columns="days"))


# ---- scoring ----------------------------------------------------------------


@dataclass
class Scores:
    """`score_arm` output."""

    windows: pd.DataFrame   # one row per (origin, drug): totals, quantiles, losses
    weekly: pd.DataFrame    # one row per (origin, drug, h): errors for unit WAPE
    summary: pd.DataFrame   # (subset, tier) x METRICS
    rho: dict[str, float] | None


def score_arm(preds: pd.DataFrame, actuals: pd.DataFrame | None = None,
              labels: pd.DataFrame | None = None, daily: pd.DataFrame | None = None,
              rho: dict[str, float] | None = None, q: float = Q,
              prices: pd.DataFrame | None = None) -> Scores:
    """Score an arm's predictions on lead-time totals (see the module docstring).

    preds: origin, drug_code, h, week_start, mean, var, tier (IMPL-SPEC format).
    actuals: weekly truth (drug_code, week_start, units, censored,
        uncensored_units); optional if `daily` is given.
    labels: origin, drug_code, lead_days and the trailing-182-day price
        (`price` or `avg_price`); missing lead -> 7 days, missing price -> 0.
    daily: daily truth (drug_code, date, units, uncensored_units, stockout);
        preferred, since lead windows rarely end on a Sunday.
    rho: tier -> rho for the variance factor 1 + rho (w - 1); None = plain.
    """
    if actuals is None and daily is None:
        raise ValueError("pass weekly actuals or a daily truth frame")
    meta, M, V = _wide(preds)
    lab = _labels(meta.index, labels, prices)
    truth = truth_from_daily(daily) if daily is not None else truth_from_weekly(actuals)
    weekly_truth = weekly_from_daily(daily) if actuals is None else actuals

    origin = np.array([o for o, _ in meta.index], dtype=object)
    drug = np.array([d for _, d in meta.index], dtype=object)
    rows = np.array([truth.drugs.get(c, -1) for c in drug])
    off = np.array([(o - truth.first).days + 1 for o in origin])  # Monday after origin
    tier = meta["tier"].to_numpy(object)
    lead, price = lab["lead"].to_numpy(int), lab["price"].to_numpy(float)

    w = pd.DataFrame({"origin": origin, "drug_code": drug, "tier": tier,
                      "lead": lead, "price": price})
    for name, days in (("l", lead), ("lc", lead + CYCLE_DAYS), ("4", np.full(len(w), 28)),
                       ("8", np.full(len(w), 56))):
        wt = window_weights(days)
        m, v = (M * wt).sum(1), (V * wt).sum(1)
        v = np.maximum(v * window_var_factor(days, tier, rho), m)
        y, cens, ok = truth.total(rows, off, days)
        w[f"m_{name}"], w[f"v_{name}"], w[f"y_{name}"] = m, v, y
        w[f"cens_{name}"], w[f"ok_{name}"] = cens, ok
    for name in ("l", "lc"):
        qq = fc.demand_quantile(w[f"m_{name}"].to_numpy(), w[f"v_{name}"].to_numpy(), q)
        y = w[f"y_{name}"].to_numpy()
        w[f"q_{name}"] = qq
        w[f"pin_{name}"] = pinball(y, qq, q)
        w[f"cov_{name}"] = (y <= qq).astype(float)
        w[f"w_pin_{name}"] = w[f"pin_{name}"] * price
        w[f"w_y_{name}"] = y * price
    w = w[w["ok_lc"]].copy()
    w["crps_lc"] = crps_nb(w["m_lc"], w["v_lc"], w["y_lc"])
    w["w_crps_lc"] = w["crps_lc"] * w["price"]
    w = w.rename(columns={"m_4": "f4", "m_8": "f8", "y_4": "y4", "y_8": "y8"})
    for c in ("f4", "y4", "f8", "y8"):  # windows without 4 or 8 weeks of truth drop out
        ok = w["ok_4"] if c.endswith("4") else w["ok_8"]
        w[c] = np.where(ok, w[c], 0.0)
    w["w_ae4"] = np.abs(w["f4"] - w["y4"]) * w["price"]
    w["w_y4"] = w["y4"] * w["price"]
    w["one"] = 1.0
    w["censored"] = w["cens_lc"]

    wk = _weekly_errors(preds, weekly_truth)
    return Scores(w.reset_index(drop=True), wk, summarise(w, wk), rho)


def _weekly_errors(preds: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    p = preds[["origin", "drug_code", "h", "week_start", "mean", "tier"]].copy()
    p["origin"] = pd.to_datetime(p["origin"]).dt.date
    p["week_start"] = pd.to_datetime(p["week_start"]).dt.date
    p["drug_code"] = p["drug_code"].astype(str)
    a = actuals.copy()
    a["week_start"] = pd.to_datetime(a["week_start"]).dt.date
    a["drug_code"] = a["drug_code"].astype(str)
    col = "uncensored_units" if "uncensored_units" in a else "units"
    a = a.rename(columns={col: "y"})
    keep = ["drug_code", "week_start", "y"] + [c for c in ("units", "censored") if c in a
                                               and c != col]
    x = p.merge(a[keep], on=["drug_code", "week_start"], how="inner")
    x["ae"] = np.abs(x["mean"] - x["y"])
    rec = x["units"] if "units" in x else x["y"]
    clean = ~x["censored"].astype(bool) if "censored" in x else pd.Series(True, index=x.index)
    x["ae_clean"] = np.where(clean, np.abs(x["mean"] - rec), 0.0)
    x["y_clean"] = np.where(clean, rec, 0.0)
    return x


def _subsets(w: pd.DataFrame) -> dict[str, pd.Series]:
    c = w["censored"].astype(bool)
    return {"all": pd.Series(True, index=w.index), "clean": ~c, "censored": c}


def _tier_masks(tier: pd.Series) -> dict[str, pd.Series]:
    out = {t: tier == t for t in TIERS}
    out["pooled"] = tier.isin(GATE_TIERS)
    out["all"] = pd.Series(True, index=tier.index)
    return out


def summarise(w: pd.DataFrame, wk: pd.DataFrame) -> pd.DataFrame:
    """(subset, tier) rows of every metric. Weekly WAPE has no censored split."""
    rows = []
    for sub, sm in _subsets(w).items():
        for t, tm in _tier_masks(w["tier"]).items():
            x = w[sm & tm]
            r = {"subset": sub, "tier": t, "windows": len(x)}
            for k, (num, den) in RATIO.items():
                dsum = x[den].sum()
                r[k] = x[num].sum() / dsum if dsum else np.nan
            if sub == "all":
                y = wk[_tier_masks(wk["tier"])[t]]
                for k, (num, den) in WEEKLY.items():
                    r[k] = y[num].sum() / y[den].sum() if y[den].sum() else np.nan
            rows.append(r)
    return pd.DataFrame(rows).set_index(["subset", "tier"])


# ---- paired comparison --------------------------------------------------------


def compare_arms(a: Scores, b: Scores, metrics: Sequence[str] = METRICS,
                 tiers: Sequence[str] = ("pooled", "fast", "regular", "slow", "rare"),
                 subsets: Sequence[str] = ("all", "clean", "censored"),
                 n: int = bootstrap.N_DRAWS, seed: int = 0, ni_margin: float = 0.01,
                 drugs_only: bool = False) -> pd.DataFrame:
    """Paired differences a - b, with two-way bootstrap intervals (90%).

    Windows are matched on (origin, drug); tiers and censoring come from `b`
    (the reference). For each metric: value of a and b, diff, diff / b, the
    interval, and for lower-is-better scores the roadmap's verdicts:
    `non_inferior` (upper bound < ni_margin x b) and `superior` (upper < 0).
    Coverage verdicts are left to the caller (the target is 0.95, not 0).
    """
    keys = ["origin", "drug_code"]
    wa, wb = a.windows.set_index(keys), b.windows.set_index(keys)
    common = wa.index.intersection(wb.index)
    wa, wb = wa.loc[common], wb.loc[common]
    ka, kb = a.weekly.set_index(keys + ["h"]), b.weekly.set_index(keys + ["h"])
    ck = ka.index.intersection(kb.index)
    ka, kb = ka.loc[ck], kb.loc[ck]
    rows = []
    for metric in metrics:
        weekly = metric in WEEKLY
        if weekly and subsets == ("censored",):
            continue
        num, den = (WEEKLY if weekly else RATIO)[metric]
        xa, xb = (ka, kb) if weekly else (wa, wb)
        frame = pd.DataFrame({"origin": [i[0] for i in xb.index],
                              "drug_code": [i[1] for i in xb.index],
                              "tier": xb["tier"].to_numpy(),
                              "na": xa[num].to_numpy(float), "nb": xb[num].to_numpy(float),
                              "da": xa[den].to_numpy(float), "db": xb[den].to_numpy(float)})
        cens = (np.zeros(len(frame), bool) if weekly
                else xb["censored"].astype(bool).to_numpy())
        subs = {"all": np.ones(len(frame), bool), "clean": ~cens, "censored": cens}
        for sub in subsets:
            if weekly and sub != "all":
                continue
            for t in tiers:
                m = subs[sub] & _tier_masks(frame["tier"])[t].to_numpy()
                f = frame[m]
                if f["db"].sum() == 0:
                    continue
                iv = bootstrap.two_way(
                    f, ["na", "nb", "da", "db"],
                    lambda s: s["na"] / s["da"] - s["nb"] / s["db"],
                    n=n, seed=seed, drugs_only=drugs_only)
                va, vb = f["na"].sum() / f["da"].sum(), f["nb"].sum() / f["db"].sum()
                r = {"metric": metric, "subset": sub, "tier": t, "n": len(f), "a": va, "b": vb,
                     "diff": iv.point, "rel": iv.point / vb if vb else np.nan,
                     "lo": iv.lo, "hi": iv.hi,
                     "rel_lo": iv.lo / vb if vb else np.nan,
                     "rel_hi": iv.hi / vb if vb else np.nan}
                if metric.startswith(("pinball", "crps", "rwape", "wape")):
                    r["non_inferior"] = bool(iv.hi < ni_margin * vb)
                    r["superior"] = bool(iv.hi < 0)
                    r["worse"] = bool(iv.lo > 0)
                rows.append(r)
    return pd.DataFrame(rows)
