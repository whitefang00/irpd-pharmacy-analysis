"""Demand lost to stock-outs and shelf-outs, and how much of it switched to an alternative.

Everything here takes `(con, as_of)` and reads only rows dated up to `as_of`
(spells still running then are clipped there), so it is safe on a full
warehouse connection as well as on `warehouse.as_of` or `holdout.agent_view`.

**Expected demand while out.** For a spell of drug X starting on day s, the
baseline is X's last `BASE_DAYS` *clean* days before s (stock above zero on
the floor and in total, at the end of that day and the day before), reaching
back at most `LOOKBACK` days. X's rate is its units per unit of shop traffic
on those days, where traffic is the whole shop's sale lines that day.
Expected demand on a spell day is that rate times the day's traffic. Traffic
is observed on the day itself, so weekdays, Eid, closed days and slow months
come in with no model. The choices were tuned on the fake stock-out test
(`fake_spells` and `fake_score` below): traffic scaling had
lower bias and error than a weekday index or a flat mean on fast, regular
and slow drugs (the weekday index was no better than flat), and 56 baseline
days had the lowest error among settings with small bias. On that test the
total runs about right on fast and slow drugs and ~6% high on regular ones;
rare drugs are noise. Demand just before real stock-outs is no higher than
just after the restock (0.95-0.99x), so the baseline isn't inflated by the
spike that emptied the shelf.

**Lost** = expected - recorded over the spell, *signed*. On a stock-out
recorded sales are near zero, so it is the expected demand. On a shelf-out
(floor empty, store stocked) sales go on when someone fetches stock from
upstairs, so it is the shortfall below normal, and it can be negative when
sales held up. Sum it; don't floor it per spell: on fake spells, flooring
alone invents lost sales worth 16-29% of what sold. Only the first
`MAX_DAYS` of a spell count: after that the drug is more likely delisted or
unobtainable than missed day after day (`days_beyond` has the rest). Lost
units are valued at X's median selling price per base unit over the
`PRICE_DAYS` before the spell. `lost_lo` / `lost_hi` are a per-spell 80%
band from the fake test: what really sold on hidden in-stock periods ran
between `BAND` low and high times the estimate (wide for short, slow spells).
For a total, `total_band` gives the range from the fake test's aggregate
error instead.

**Switching.** Alternatives are drugs with the same ingredient set and form
family (`ingredient_map`), other than X. Strength is ignored: the map's
strengths are patchy, and a customer can take two 5mg for a 10mg. For each
spell, the alternatives' uplift is their recorded sale lines minus their own
expected lines (same estimator, own baselines), on the spell days each one
was on the shelf. It is counted in sale *lines* (roughly customers), because
brands of one ingredient sell in different units and pack sizes, so units
don't compare; lines do. `switched_raw` is a spell's uplift over X's lost
lines, clipped to 0-1, and it is noisy. The share applied is the
ingredient's pooled rate over its stock-outs (sum of uplift / sum of lost
lines), shrunk toward the overall rate with `SHRINK` pseudo lost lines and
clipped to 0-1. X's switched units and rupiah are that share of its lost
units and rupiah, in X's own units and price; the rest is truly lost. A spell
with no alternative on the shelf switches nothing, and an unmapped drug has no
alternatives, so its loss counts as truly lost (an upper bound: switching to
a different ingredient isn't seen). `placebo` reruns the measurement on
windows where nothing was out.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from irpd import config, holdout
from irpd.tables.sales import tiers_sql
from irpd.tables.stock import spells_as_of

BASE_DAYS = 56    # clean days in the baseline before a spell
MIN_BASE = 7      # fewer clean days than this: no estimate
LOOKBACK = 365    # calendar days the baseline may reach back
MAX_DAYS = 28     # spell days counted as lost
PRICE_DAYS = 182
SHRINK = 50       # pseudo lost lines pulling an ingredient's switch rate to the overall rate
# What really sold on fake spells over the estimate, 10th and 90th percentiles,
# by expected units: < 5, 5-20, 20-100, >= 100 (fake test up to the cutoff).
BAND_EDGES = (5, 20, 100)
BAND = ((0.0, 4.8), (0.0, 2.35), (0.0, 1.93), (0.45, 1.45))

FORM_FAMILY = {
    "tablet": "oral solid", "capsule": "oral solid", "lozenge": "oral solid",
    "liquid": "oral liquid", "syrup": "oral liquid", "suspension": "oral liquid",
    "emulsion": "oral liquid",
    "cream": "topical", "ointment": "topical", "gel": "topical", "lotion": "topical",
}
# A blank form is read from the name, else taken as an oral solid (most blanks
# are generics such as "Amlodipine 10mg Medicon").
NAME_FORMS = [("oral liquid", r"\b(?:syr|sirup|syrup|susp|elixir)\b"),
              ("drops", r"\bdrops?\b"),
              ("topical", r"\b(?:cr|cream|krim|salep|zalf|gel|lotion)\b"),
              ("suppository", r"\b(?:supp|sup)\b")]


def _d(day: date) -> str:
    return f"DATE '{day.isoformat()}'"


def _prefix(x: np.ndarray) -> np.ndarray:
    """Prefix sums along days with a leading zero: sum over [a, b) = P[:, b] - P[:, a]."""
    return np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(x, axis=1)], axis=1)


def _window(P: np.ndarray, j, s, e) -> np.ndarray:
    return P[j, e] - P[j, s]


@dataclass
class Panel:
    """Drugs x days arrays from FIRST_DAY to as_of, and the prefix sums the estimator reads."""

    drugs: list[str]
    days: list[date]
    units: np.ndarray
    lines: np.ndarray
    revenue: np.ndarray
    in_stock: np.ndarray   # total stock > 0 at the end of the day
    clean: np.ndarray      # floor and total > 0 at the end of the day and the day before
    traffic: np.ndarray    # per day: the shop's sale lines (0 = closed)

    def __post_init__(self):
        self.index = {d: i for i, d in enumerate(self.drugs)}
        self.P = {k: _prefix(getattr(self, k)) for k in ("units", "lines", "revenue")}
        self.P_clean = {k: _prefix(getattr(self, k) * self.clean)
                        for k in ("units", "lines", "revenue")}
        self.P_clean_days = _prefix(self.clean.astype(float))
        self.P_lines_clean = _prefix(self.lines * self.clean)
        self.set_weights(self.traffic.astype(float))

    def set_weights(self, w: np.ndarray) -> None:
        """How much demand each day carries: shop traffic, unless a test swaps it."""
        self.w = w
        self.P_w_clean = _prefix(self.clean * w)
        self.P_w = _prefix(w[None, :])[0]

    def day(self, d: date) -> int:
        return (d - config.FIRST_DAY).days


def panel(con: duckdb.DuckDBPyConnection, as_of: date) -> Panel:
    end = _d(as_of)
    days = [config.FIRST_DAY + timedelta(days=i)
            for i in range((as_of - config.FIRST_DAY).days + 1)]
    dem = con.execute(f"""
        SELECT drug_code, date, units, lines, revenue FROM daily_demand
        WHERE date <= {end}""").df()
    stock = con.execute(f"""
        SELECT drug_code, date,
               max(closing) FILTER (WHERE location = 'total') AS total,
               max(closing) FILTER (WHERE location = 'floor') AS floor
        FROM daily_stock WHERE date <= {end} AND location IN ('total', 'floor')
        GROUP BY ALL""").df()
    drugs = sorted(set(dem.drug_code) | set(stock.drug_code))
    di = pd.Index(drugs)
    day0 = np.datetime64(config.FIRST_DAY, "D")

    def grid(df, col, fill=0.0):
        out = np.full((len(drugs), len(days)), fill)
        t = (pd.to_datetime(df["date"]).to_numpy("datetime64[D]") - day0).astype(int)
        ok = t >= 0
        out[di.get_indexer(df["drug_code"])[ok], t[ok]] = df[col].to_numpy(float)[ok]
        return out

    units, lines, revenue = (grid(dem, c) for c in ("units", "lines", "revenue"))
    total, floor = grid(stock, "total", np.nan), grid(stock, "floor", np.nan)
    in_stock = total > 0
    on_shelf = in_stock & (floor > 0)
    before = np.concatenate([np.zeros((len(drugs), 1), bool), on_shelf[:, :-1]], axis=1)
    traffic = lines.sum(axis=0)
    return Panel(drugs, days, units, lines, revenue, in_stock,
                 on_shelf & before & (traffic > 0), traffic)


def rates(p: Panel, j: np.ndarray, s: np.ndarray,
          n_base: int = BASE_DAYS) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Clean baseline days used, and units / lines / revenue per unit of traffic, before day s.

    The baseline is drug j's last `n_base` clean days before s, at most
    LOOKBACK days back. NaN where it has fewer than MIN_BASE clean days.
    """
    a = np.array([np.searchsorted(p.P_clean_days[jj], p.P_clean_days[jj, ss] - n_base,
                                  side="left") for jj, ss in zip(j, s, strict=True)], int)
    a = np.maximum(a, np.asarray(s) - LOOKBACK).clip(0)
    n = _window(p.P_clean_days, j, a, s)
    w = _window(p.P_w_clean, j, a, s)
    ok = (w > 0) & (n >= MIN_BASE)
    return n, {k: np.divide(_window(P, j, a, s), w, out=np.full(len(j), np.nan), where=ok)
               for k, P in p.P_clean.items()}


def band(expected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The fake-test 80% band of what really sells, as multipliers on `expected`."""
    b = np.searchsorted(BAND_EDGES, expected, side="right")
    lo, hi = np.array(BAND).T
    return lo[b], hi[b]


def tiers(con: duckdb.DuckDBPyConnection, as_of: date) -> pd.Series:
    df = con.execute(
        f"SELECT drug_code, tier FROM ({tiers_sql(min(as_of, config.CUTOFF))})").df()
    return df.set_index("drug_code")["tier"]


def groups(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Mapped drugs: ingredients, form family and switching group (ingredients | family)."""
    m = con.execute("SELECT drug_code, drug_name, ingredients, form FROM ingredient_map "
                    "WHERE ingredients <> ''").df()
    fam = m.form.map(FORM_FAMILY).fillna(m.form)
    name = m.drug_name.str.lower()
    guess = pd.Series("oral solid", index=m.index)
    for f, pat in reversed(NAME_FORMS):
        guess = guess.mask(name.str.contains(pat, regex=True), f)
    m["family"] = fam.mask(fam.eq(""), guess)
    m["group"] = m.ingredients + " | " + m.family
    return m.set_index("drug_code")[["ingredients", "family", "group"]]


def measure(p: Panel, df: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """For windows [s, e) of drug row j: expected and recorded demand, and alternatives' uplift."""
    j, s, e = df.j.to_numpy(), df.s.to_numpy(), df.e.to_numpy()
    n, r = rates(p, j, s)
    wsum = p.P_w[e] - p.P_w[s]
    out = df.assign(base_days=n.astype(int), estimated=np.isfinite(r["units"]))
    for k in ("units", "lines", "revenue"):
        out[f"expected_{k}"] = np.nan_to_num(r[k] * wsum)
        out[f"recorded_{k}"] = _window(p.P[k], j, s, e)
    return pd.concat([out, alt_uplift(p, out, g)], axis=1)


def alternatives(p: Panel, df: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """(window row, alternative drug row) pairs: same group, not the drug itself.

    A `group` column on `df` overrides the drug's own group (the placebo uses it).
    """
    members = (g[g.index.isin(p.index)].reset_index()
               .groupby("group")["drug_code"].apply(list).to_dict())
    grp = df["group"] if "group" in df else df.drug_code.map(g["group"])
    rows = [(i, p.index[a]) for i, (code, gr) in enumerate(zip(df.drug_code, grp, strict=True))
            if isinstance(gr, str) for a in members.get(gr, []) if a != code]
    return pd.DataFrame(rows, columns=["row", "a"], dtype=int)


def alt_uplift(p: Panel, df: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """Per window: alternatives' expected and recorded lines on the days each was on the shelf."""
    pairs = alternatives(p, df, g)
    i, a = pairs.row.to_numpy(), pairs.a.to_numpy()
    s, e = df.s.to_numpy()[i], df.e.to_numpy()[i]
    _, r = rates(p, a, s)
    ok = np.isfinite(r["lines"])
    w_on = _window(p.P_w_clean, a, s, e)
    up = pd.DataFrame({
        "row": i[ok],
        "alt_expected_lines": (r["lines"] * w_on)[ok],
        "alt_recorded_lines": _window(p.P_lines_clean, a, s, e)[ok],
        "alt_on_shelf_days": _window(p.P_clean_days, a, s, e)[ok],
    }).groupby("row").sum().reindex(range(len(df)), fill_value=0)
    up.index = df.index
    up["alt_uplift_lines"] = up.alt_recorded_lines - up.alt_expected_lines
    up["has_alternative"] = up.alt_on_shelf_days > 0
    return up


def _spell_frame(con, as_of: date, p: Panel, max_days: int) -> pd.DataFrame:
    parts = [spells_as_of(con, as_of, table).df()[["drug_code", "start", "stop", "days"]]
             .assign(kind=kind)
             for kind, table in (("stockout", "stockouts"), ("shelfout", "shelfouts"))]
    df = pd.concat(parts, ignore_index=True)
    df = df[df.drug_code.isin(p.index)].copy()
    df["start"] = pd.to_datetime(df["start"]).dt.date
    df["stop"] = pd.to_datetime(df["stop"]).dt.date
    # The first day sold out partway through, so its sales are real: count from the next.
    df["counted_from"] = df.start + timedelta(days=1)
    df["counted_days"] = (df["days"] - 1).clip(0, max_days)
    df["days_beyond"] = df["days"] - 1 - df["counted_days"]
    df["j"] = df.drug_code.map(p.index)
    df["s"] = [p.day(d) for d in df.counted_from]
    df["e"] = df.s + df.counted_days
    return df.reset_index(drop=True)


def _prices(p: Panel, j: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Median price per base unit on sale days in the PRICE_DAYS before s.

    Falls back to any earlier sale day, then to any day in the panel (a drug
    with no sale before s is valued at its later price).
    """
    unit_price = np.divide(p.revenue, p.units, out=np.full(p.units.shape, np.nan),
                           where=p.units > 0)
    out = np.zeros(len(j))
    for i, (jj, ss) in enumerate(zip(j, s, strict=True)):
        for x in (unit_price[jj, max(ss - PRICE_DAYS, 0):ss], unit_price[jj, :ss],
                  unit_price[jj, :]):
            if np.isfinite(x).any():
                out[i] = np.nanmedian(x)
                break
    return out


def spells(con: duckdb.DuckDBPyConnection, as_of: date, max_days: int = MAX_DAYS,
           p: Panel | None = None) -> pd.DataFrame:
    """One row per stock-out and shelf-out spell up to `as_of`, with lost and switched demand.

    Spells with no usable baseline (`estimated` False, mostly new or rare
    drugs) get zero expected demand and so no lost demand.
    """
    p = p or panel(con, as_of)
    g = groups(con)
    df = measure(p, _spell_frame(con, as_of, p, max_days), g)
    df["tier"] = df.drug_code.map(tiers(con, as_of)).fillna("rare")
    df["ingredients"] = df.drug_code.map(g["ingredients"])
    df["lost_units"] = df.expected_units - df.recorded_units
    df["lost_lines"] = df.expected_lines - df.recorded_lines
    lo, hi = band(df.expected_units.to_numpy())
    df["lost_lo"] = df.expected_units * lo - df.recorded_units
    df["lost_hi"] = df.expected_units * hi - df.recorded_units
    df["price"] = _prices(p, df.j.to_numpy(), df.s.to_numpy())
    for c in ("lost", "lost_lo", "lost_hi"):
        df[f"{c}_revenue"] = df[c if c != "lost" else "lost_units"] * df.price
    df["switched_raw"] = (df.alt_uplift_lines / df.lost_lines.where(df.lost_lines > 0)) \
        .clip(0, 1).fillna(0)
    return apply_switching(df.drop(columns=["j", "s", "e"]), switch_rates(df))


def apply_switching(df: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """Split each spell's lost demand with per-ingredient switch rates (from `switch_rates`)."""
    df = df.assign(switch_rate=np.where(
        df.has_alternative, df.ingredients.map(rates["rate"]).fillna(rates.attrs["overall"]), 0))
    df["switched_units"] = df.lost_units * df.switch_rate
    df["switched_revenue"] = df.lost_revenue * df.switch_rate
    df["truly_lost_revenue"] = df.lost_revenue - df.switched_revenue
    return df


def switch_rates(df: pd.DataFrame, kind: str = "stockout") -> pd.DataFrame:
    """Per ingredient set: pooled alternatives' uplift / lost lines over its spells, shrunk.

    Uses stock-out spells that had an alternative in stock (on a shelf-out
    staff can still fetch the drug, so fewer customers need an alternative).
    `.attrs['overall']` is the same pooled ratio over all of them.
    """
    d = df[(df.kind == kind) & df.has_alternative & df.estimated]
    overall = float(np.clip(d.alt_uplift_lines.sum() / d.lost_lines.sum(), 0, 1)) \
        if d.lost_lines.sum() > 0 else 0.0
    by = d.groupby("ingredients").agg(spells=("lost_lines", "size"),
                                      lost_lines=("lost_lines", "sum"),
                                      uplift_lines=("alt_uplift_lines", "sum"))
    by["raw"] = by.uplift_lines / by.lost_lines
    by["rate"] = ((by.uplift_lines + SHRINK * overall) / (by.lost_lines.clip(lower=0) + SHRINK)
                  ).clip(0, 1)
    by = by.sort_values("lost_lines", ascending=False)
    by.attrs["overall"] = overall
    return by


def _first(p: Panel, j: int) -> int:
    """The first day of drug j's stock period: STOCK_START, or its first day in stock."""
    stocked = p.in_stock[j]
    return max(p.day(config.STOCK_START), int(np.argmax(stocked)) if stocked.any() else 0)


def _out_share(p: Panel, j: int, last: int) -> float:
    first = _first(p, j)
    return 1 - p.in_stock[j, first:last + 1].mean() if last >= first else 1.0


def _random_clean_window(p: Panel, j: int, length: int, rng, last: int,
                         tries: int = 20) -> int | None:
    """A start s with [s, s+length) all clean, in drug j's stock period up to `last`, or None."""
    first = _first(p, j)
    if last - length + 1 <= first:
        return None
    for _ in range(tries):
        s = int(rng.integers(first, last - length + 2))
        if p.P_clean_days[j, s + length] - p.P_clean_days[j, s] == length:
            return s
    return None


def _last_sale(p: Panel) -> np.ndarray:
    sold = p.units > 0
    return np.where(sold.any(axis=1), sold.shape[1] - 1 - np.argmax(sold[:, ::-1], axis=1), -1)


def fake_spells(con: duckdb.DuckDBPyConnection, as_of: date, per_drug: int = 20,
                max_out_share: float = 0.02, n_base: int = BASE_DAYS, seed: int = 0,
                p: Panel | None = None) -> pd.DataFrame:
    """Hide real in-stock periods as fake spells and estimate what sold in them.

    Drugs out of stock on at most `max_out_share` of days (from STOCK_START, or their
    first day in stock, to their last sale up to `as_of`) get up to `per_drug` fake spells: runs
    of clean days with lengths drawn from the real stock-out spells of the
    drug's tier (capped at MAX_DAYS). `expected` uses the same baseline as a
    real spell; `actual` is what sold. The lost demand of a fake spell is zero.
    """
    p = p or panel(con, as_of)
    rng = np.random.default_rng(seed)
    tier = tiers(con, as_of)
    real = spells_as_of(con, as_of, "stockouts").df()
    lengths = {t: (x.days - 1)[x.days > 1].clip(upper=MAX_DAYS).to_numpy()
               for t, x in real.assign(tier=real.drug_code.map(tier)).groupby("tier")}
    last = _last_sale(p)
    rows = []
    for j, code in enumerate(p.drugs):
        t = tier.get(code)
        if t not in lengths or _out_share(p, j, last[j]) > max_out_share:
            continue
        for _ in range(per_drug):
            length = int(rng.choice(lengths[t]))
            s = _random_clean_window(p, j, length, rng, last[j])
            if s is not None:
                rows.append((j, s, length, code, t))
    df = pd.DataFrame(rows, columns=["j", "s", "days", "drug_code", "tier"])
    j, s = df.j.to_numpy(int), df.s.to_numpy(int)
    e = s + df.days.to_numpy(int)
    n, r = rates(p, j, s, n_base)
    df["base_days"] = n.astype(int)
    df["expected"] = r["units"] * (p.P_w[e] - p.P_w[s])
    df["actual"] = _window(p.P["units"], j, s, e)
    df["start"] = [p.days[x] for x in s]
    return df[np.isfinite(df.expected)].drop(columns=["j", "s"]).reset_index(drop=True)


def fake_score(fake: pd.DataFrame, by: str = "tier") -> pd.DataFrame:
    """Bias, error and band coverage of the estimate on fake spells.

    `bias` is total expected over total sold, minus 1. `wape` is the summed
    absolute error over total sold. `floored_lost` is what flooring each
    spell's shortfall at zero would invent, as a share of what sold.
    """
    def one(x):
        e, a = x.expected, x.actual
        lo, hi = band(e.to_numpy())
        return pd.Series({
            "spells": len(x), "drugs": x.drug_code.nunique(),
            "bias": e.sum() / a.sum() - 1,
            "wape": (e - a).abs().sum() / a.sum(),
            "floored_lost": (e - a).clip(lower=0).sum() / a.sum(),
            "band_cover": ((a >= e * lo) & (a <= e * hi)).mean(),
        })
    return fake.groupby(by).apply(one, include_groups=False)


def total_band(fake: pd.DataFrame, n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Per tier, an 80% range for actual / expected summed over many spells.

    Bootstraps drugs (a drug's fake spells stay together). Multiply a total
    expected demand by it for a range on the total.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for t, x in fake.groupby("tier"):
        per = x.groupby("drug_code")[["expected", "actual"]].sum().to_numpy()
        idx = rng.integers(0, len(per), (n_boot, len(per)))
        ratio = per[idx, 1].sum(axis=1) / per[idx, 0].sum(axis=1)
        out[t] = {"ratio": per[:, 1].sum() / per[:, 0].sum(),
                  "lo": np.quantile(ratio, 0.1), "hi": np.quantile(ratio, 0.9)}
    return pd.DataFrame(out).T


def placebo(con: duckdb.DuckDBPyConnection, as_of: date, per_spell: int = 3,
            seed: int = 0, p: Panel | None = None,
            real: pd.DataFrame | None = None) -> pd.DataFrame:
    """Alternatives' uplift in windows where nothing was out, next to the real stock-outs.

    For each real stock-out with an alternative in stock, up to `per_spell`
    random clean windows of the same drug and length ('same drug'), and the
    real windows with a random other group's drugs standing in for the
    alternatives ('unrelated'). Each row: windows, the drug's expected and
    lost lines, the alternatives' expected and recorded lines, their uplift
    over the drug's expected lines (comparable across rows), and their
    recorded / expected.
    """
    p = p or panel(con, as_of)
    g = groups(con)
    rng = np.random.default_rng(seed)
    sp = _spell_frame(con, as_of, p, MAX_DAYS)
    sp = measure(p, sp[(sp.kind == "stockout") & (sp.counted_days > 0)], g)
    sp = sp[sp.has_alternative & sp.estimated]
    last = _last_sale(p)

    rows = []
    for r in sp.itertuples():
        for _ in range(per_spell):
            s = _random_clean_window(p, r.j, r.counted_days, rng, last[r.j])
            if s is not None:
                rows.append((r.drug_code, r.j, s, s + r.counted_days))
    same = measure(p, pd.DataFrame(rows, columns=["drug_code", "j", "s", "e"]), g)

    # Unrelated: the real windows, with a random other group as "alternatives".
    labels = g["group"].unique()
    own = sp.drug_code.map(g["group"]).to_numpy()
    other = rng.choice(labels, len(sp))
    while (clash := other == own).any():
        other[clash] = rng.choice(labels, clash.sum())
    unrelated = measure(p, sp[["drug_code", "j", "s", "e"]].assign(group=other), g)

    def row(x, lost):
        return {"windows": len(x), "drug_expected_lines": x.expected_lines.sum(),
                "drug_lost_lines": lost, "alt_expected_lines": x.alt_expected_lines.sum(),
                "alt_recorded_lines": x.alt_recorded_lines.sum(),
                "uplift_over_drug_expected": x.alt_uplift_lines.sum() / x.expected_lines.sum(),
                "alt_ratio": x.alt_recorded_lines.sum() / x.alt_expected_lines.sum()}

    out = {"stock-outs": row(sp, (sp.expected_lines - sp.recorded_lines).sum()),
           "placebo: same drug, in stock": row(same[same.has_alternative], 0.0),
           "placebo: unrelated drugs": row(unrelated[unrelated.has_alternative], 0.0)}
    return pd.DataFrame(out).T


def headline(sp: pd.DataFrame, since: date = config.STOCK_START,
             band_by_tier: pd.DataFrame | None = None) -> pd.DataFrame:
    """Lost, switched and truly lost rupiah by tier and kind, for spells starting from `since`.

    With `band_by_tier` (from `total_band`), adds an 80% range on lost revenue.
    """
    d = sp[sp.start >= since]
    t = d.groupby(["tier", "kind"]).agg(
        spells=("lost_units", "size"), counted_days=("counted_days", "sum"),
        days_beyond=("days_beyond", "sum"), expected_revenue=("expected_revenue", "sum"),
        recorded_revenue=("recorded_revenue", "sum"), lost_revenue=("lost_revenue", "sum"),
        switched_revenue=("switched_revenue", "sum"),
        truly_lost_revenue=("truly_lost_revenue", "sum"))
    if band_by_tier is not None:
        # Lost = expected x ratio - recorded, valued at each spell's price.
        val = d.assign(e=d.expected_units * d.price, r=d.recorded_units * d.price) \
            .groupby(["tier", "kind"])[["e", "r"]].sum()
        b = band_by_tier.reindex(val.index.get_level_values("tier")).to_numpy()
        t["lost_revenue_lo"] = val.e.to_numpy() * b[:, 1] - val.r.to_numpy()
        t["lost_revenue_hi"] = val.e.to_numpy() * b[:, 2] - val.r.to_numpy()
    total = t.sum().to_frame("all").T
    total.index = pd.MultiIndex.from_tuples([("all", "all")], names=t.index.names)
    t = pd.concat([t, total])
    t["switched_share"] = t.switched_revenue / t.lost_revenue
    return t


Demand = Callable[[duckdb.DuckDBPyConnection, list[str], list[date]], np.ndarray]


def uncensor(p: Panel, sp: pd.DataFrame, g: pd.DataFrame) -> np.ndarray:
    """Panel units with lost demand added back and switched demand taken off alternatives.

    Stock-outs only: on a shelf-out sales are only partly censored, and a
    spell's shortfall is too noisy to add back day by day without bias.
    X gets its whole lost demand on its counted out days, spread in
    proportion to each day's expected demand: had it been in stock, the
    customers who switched would have bought X. Each alternative gives up its
    share (by expected lines) of the switched lines, in its own units (its
    baseline units per line), spread over the spell days it was on the shelf,
    never taking its demand below zero.
    """
    out = p.units.copy()
    sp = sp[(sp.kind == "stockout") & sp.estimated & (sp.lost_units > 0)].reset_index(drop=True)
    if sp.empty:
        return out
    sp = sp.assign(j=sp.drug_code.map(p.index), s=[p.day(d) for d in sp.counted_from])
    sp["e"] = sp.s + sp.counted_days
    for r in sp.itertuples():
        days = slice(r.s, r.e)
        if p.w[days].sum() > 0:
            out[r.j, days] += r.lost_units * p.w[days] / p.w[days].sum()

    pairs = alternatives(p, sp, g)
    _, ar = rates(p, pairs.a.to_numpy(), sp.s.to_numpy()[pairs.row.to_numpy()])
    for ii, aa, rl, ru in zip(pairs.row, pairs.a, ar["lines"], ar["units"], strict=True):
        r = sp.iloc[ii]
        days = slice(r.s, r.e)
        w_in = p.w[days] * p.clean[aa, days]
        if not (rl > 0 and r.switch_rate > 0 and w_in.sum() > 0 and r.alt_expected_lines > 0):
            continue
        share = rl * w_in.sum() / r.alt_expected_lines
        take = r.switch_rate * r.lost_lines * share * ru / rl  # in the alternative's units
        out[aa, days] = np.maximum(out[aa, days] - take * w_in / w_in.sum(), 0)
    return out


def uncensored_demand(agent: Path = holdout.AGENT) -> Demand:
    """A `replay.load(demand=...)` hook: holdout demand with stock-out losses added back.

    Demand rates come from each spell's own past: the agent file up to the
    cutoff, then the sealed holdout up to the spell. Switch rates come from
    spells that started before the holdout. Scorer side only (it reads the
    sealed file).
    """
    def hook(sealed: duckdb.DuckDBPyConnection, drugs: list[str],
             days: list[date]) -> np.ndarray:
        with combined(sealed, agent) as con:
            return uncensored_units(con, drugs, days)
    return hook


def uncensored_units(con: duckdb.DuckDBPyConnection, drugs: list[str],
                     days: list[date]) -> np.ndarray:
    """Demand on `days` (drugs x days) with stock-out losses added back, from a
    connection whose history runs to the last of them."""
    p = panel(con, days[-1])
    sp = spells(con, days[-1], p=p)
    past = sp[sp.start < days[0]]  # switch rates from spells before the window
    sp = apply_switching(sp[sp.stop >= days[0]], switch_rates(past))
    units = uncensor(p, sp, groups(con))
    t0 = p.day(days[0])
    out = np.zeros((len(drugs), len(days)))
    for k, d in enumerate(drugs):
        if d in p.index:
            out[k] = units[p.index[d], t0:t0 + len(days)]
    return out


def combined(sealed: duckdb.DuckDBPyConnection, agent: Path) -> duckdb.DuckDBPyConnection:
    """An in-memory connection with the agent's past and the sealed holdout as one history."""
    con = duckdb.connect()
    con.execute(f"ATTACH '{agent}' AS ag (READ_ONLY)")
    for name in ("daily_demand", "daily_stock"):
        con.register("_held", sealed.execute(f"SELECT * FROM {name}").df())
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM ag.{name} UNION ALL BY NAME "
                    f"SELECT * REPLACE (date::DATE AS date) FROM _held")
        con.unregister("_held")
    for name in ("stock_moves", "ingredient_map"):
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM ag.{name}")
    return con
