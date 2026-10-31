"""Replay a reorder policy over the holdout and score it against the pharmacist.

The holdout is 1 Jul - 31 Dec 2025. Everything the policy may use comes from
the agent file (as of the cutoff); the real demand and the pharmacist's real
stock come from the sealed file, which only `load()` here opens.

**Scope.** Drugs that were fast or regular at the cutoff (`demand_tiers`) and
sold at least once after it. `candidates()` lists the first part, which is
all a policy-maker can know; the scorer drops drugs that never sold again.

**Policy.** A DataFrame, one row per drug:

- `s`, `S`: when the inventory position (floor + store + on order) ends a
  day at or below `s`, order up to `S`.
- `lead_days` (optional): days from order to delivery. Default: the observed
  delivery cadence of the drug's main supplier, clipped to 1-14 days (there
  are no purchase-order dates, so this is the best stand-in).
- `floor_min`, `floor_max` (optional, default 0 and `S`): each morning, if the
  shop floor holds `floor_min` or less and the store has stock, move enough
  down to bring the floor to `floor_max`.

**One simulated day**, per drug: deliveries due arrive (on the floor until
the store comes back into use, in the store after); the floor is topped up
from the store; stock-take adjustments booked that day are applied; demand
arrives and sales are capped by floor stock (the rest is lost); in the
evening an order is placed if the position is at or below `s`. The sim starts
from the real floor and store stock at the cutoff with nothing on order.

**Holdout events, taken as given.** Both are read from the sealed file:

- Stock-takes, including the August 2025 full count: each drug's net
  correction (floor + store) is applied on the day it was booked, to the
  floor first, never taking stock below zero. Counts mostly fix booking
  errors that the starting stock inherited, so both sides carry them.
  Where the count moved stock between floor and store, only the net counts.
- The store coming back into use: the first day after the cutoff with a
  delivery or transfer into the store (29 Jul 2025). Before it, the sim puts
  deliveries on the floor, as the pharmacist then did; from it, in the store.

**Demand.** By default the recorded holdout demand. It is censored: on days
the pharmacist was out, recorded sales drop to about 3% of normal, so the sim
sees almost no demand on those days. That hides the demand the pharmacist
missed, which favours the pharmacist (and leaves the policy's stock a little
higher than it would have been). Pass `demand=` to `load()` to swap in an
uncensored estimate.

**Scores.** Both sides are scored the same way from daily floor, store and
sales: stock-out days (total <= 0), shelf-out days (floor <= 0, store > 0),
units and revenue lost, average stock value, units sold. Days out count only
inside each drug's selling window, up to its last recorded sale in the
holdout, as in FINDINGS.md: after that the pharmacist had stopped stocking it,
and an empty shelf with no demand isn't a stock-out. Stock value counts every
day, so stock a policy still holds after a drug stops selling costs it. Lost
units on a day in a spell = the drug's demand rate at the spell's start, less
what sold that day (never below zero). The rate is the mean daily demand over the previous
`RATE_DAYS` days the pharmacist was in stock. Stock is valued at the drug's
median selling price per base unit over the half-year before the cutoff.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from irpd import config, holdout

FIRST = config.CUTOFF + timedelta(days=1)
LAST = config.LAST_DAY
SCOPE_TIERS = ("fast", "regular")
RATE_DAYS = 28
PRICE_DAYS = 182
LEAD_CLIP = (1, 14)
LEAD_FALLBACK = 7  # drugs with no supplier cadence
# Stock-out days in spells longer than this are reported apart: in the real
# ledger they look like a drug left unstocked (or unobtainable), which the sim,
# where every order is delivered on time, can't reproduce.
LONG_SPELL = 14

Demand = Callable[[duckdb.DuckDBPyConnection, list[str], list[date]], np.ndarray]


def _d(day: date) -> str:
    return f"DATE '{day.isoformat()}'"


def candidates(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Drugs a policy must cover: fast and regular at the cutoff."""
    tiers = ", ".join(f"'{t}'" for t in SCOPE_TIERS)
    return [r[0] for r in con.execute(
        f"SELECT drug_code FROM demand_tiers WHERE tier IN ({tiers}) ORDER BY 1").fetchall()]


def default_lead(con: duckdb.DuckDBPyConnection) -> pd.Series:
    """Lead time per drug: its main supplier's delivery cadence, clipped."""
    df = con.execute("""
        SELECT b.drug_code, c.median_gap_days AS lead
        FROM pharmacist_baseline b
        LEFT JOIN supplier_cadence c ON c.supplier = b.main_suppliers[1]""").df()
    return df.set_index("drug_code")["lead"].clip(*LEAD_CLIP).round()


def prices(con: duckdb.DuckDBPyConnection, day: date = config.CUTOFF) -> pd.Series:
    """Median selling price per base unit over the half-year before `day`."""
    since = _d(day - timedelta(days=PRICE_DAYS - 1))
    return con.execute(f"""
        SELECT drug_code,
               coalesce(median(revenue / units) FILTER (WHERE date >= {since}),
                        median(revenue / units)) AS price
        FROM daily_demand WHERE units > 0 AND date <= {_d(day)}
        GROUP BY 1""").df().set_index("drug_code")["price"]


def recorded_demand(sealed: duckdb.DuckDBPyConnection, drugs: list[str],
                    days: list[date]) -> np.ndarray:
    """The default demand: units recorded sold after the cutoff (censored)."""
    df = sealed.execute("SELECT drug_code, date, units FROM daily_demand").df()
    return _matrix(df, "units", drugs, days)


def _matrix(df: pd.DataFrame, col: str, drugs: list[str], days: list[date],
            fill: float = 0) -> np.ndarray:
    """A drugs x days array from long rows (drug_code, date, col)."""
    wide = (df.assign(date=pd.to_datetime(df["date"]).dt.date)
              .pivot_table(index="drug_code", columns="date", values=col, aggfunc="sum"))
    return wide.reindex(index=drugs, columns=days).fillna(fill).to_numpy(float)


@dataclass
class Holdout:
    """Everything the sim and the scorer need, as drugs x days arrays."""

    drugs: list[str]
    days: list[date]
    price: np.ndarray            # per drug
    lead: pd.Series              # default lead days, by drug_code
    start_floor: np.ndarray      # per drug, end of the cutoff day
    start_store: np.ndarray
    demand: np.ndarray           # drugs x days
    adjust: np.ndarray           # net stock-take correction booked each day
    store_opens: date
    rate: np.ndarray             # demand rate before each day (drugs x days)
    actual_floor: np.ndarray     # the pharmacist's real end-of-day stock
    actual_store: np.ndarray
    actual_sold: np.ndarray
    live: np.ndarray             # drugs x days: up to the drug's last sale after the cutoff


def load(agent: Path = holdout.AGENT, sealed: Path = holdout.SEALED,
         demand: Demand = recorded_demand) -> Holdout:
    """Read the scope, start stock, holdout events and both sides' inputs. Scorer only."""
    with holdout.agent_view(config.CUTOFF, agent) as con, holdout.open_sealed(sealed) as hc:
        return load_from(con, hc, FIRST, LAST, demand)


def load_from(con: duckdb.DuckDBPyConnection, hc: duckdb.DuckDBPyConnection,
              first: date, last: date, demand: Demand = recorded_demand) -> Holdout:
    """`load` for any window: `con` stands on the day before `first`; `hc` holds the
    window's rows of the sealed tables (`holdout.SEALED_TABLES`), first to last."""
    cutoff = first - timedelta(days=1)
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    sold_after = {r[0] for r in hc.execute(
        "SELECT DISTINCT drug_code FROM daily_demand WHERE units > 0").fetchall()}
    drugs = [d for d in candidates(con) if d in sold_after]
    lead = default_lead(con)
    price = prices(con, cutoff).reindex(drugs).fillna(0).to_numpy()
    start = con.execute(f"""
        SELECT drug_code, location, greatest(closing, 0) AS closing FROM daily_stock
        WHERE date = {_d(cutoff)} AND location IN ('floor', 'store')""").df()
    start = (start.pivot(index="drug_code", columns="location", values="closing")
                  .reindex(index=drugs, columns=["floor", "store"]).fillna(0))
    hist_days = [cutoff - timedelta(days=i) for i in range(RATE_DAYS, 0, -1)]
    hist = con.execute(f"""
        SELECT s.drug_code, s.date, coalesce(d.units, 0) AS units, s.closing > 0 AS in_stock
        FROM daily_stock s LEFT JOIN daily_demand d USING (drug_code, date)
        WHERE s.location = 'total' AND s.date > {_d(hist_days[0] - timedelta(days=1))}
        """).df()

    stock = hc.execute("SELECT drug_code, date, location, closing, sold "
                       "FROM daily_stock").df()
    takes = hc.execute("SELECT drug_code, date, adjustment FROM stock_takes").df()
    store_opens = hc.execute("""
        SELECT least((SELECT min(date) FROM purchases WHERE location = 'store'),
                     (SELECT min(date) FROM transfers WHERE to_location = 'store'))
        """).fetchone()[0] or last + timedelta(days=1)
    dem = demand(hc, drugs, days)
    last_sale = recorded_demand(hc, drugs, days) > 0

    floor = _matrix(stock[stock.location == "floor"], "closing", drugs, days)
    store = _matrix(stock[stock.location == "store"], "closing", drugs, days)
    sold = _matrix(stock[stock.location == "total"], "sold", drugs, days)
    # The rate uses the pharmacist's in-stock days only, before and after the cutoff.
    in_stock = np.concatenate([_matrix(hist, "in_stock", drugs, hist_days),
                               (floor + store) > 0], axis=1)
    units = np.concatenate([_matrix(hist, "units", drugs, hist_days), dem], axis=1)
    rate = _trailing_rate(units, in_stock)[:, len(hist_days):]
    return Holdout(drugs=drugs, days=days, price=price, lead=lead,
                   start_floor=start["floor"].to_numpy(float),
                   start_store=start["store"].to_numpy(float),
                   demand=dem, adjust=_matrix(takes, "adjustment", drugs, days),
                   store_opens=store_opens, rate=rate,
                   live=np.cumsum(last_sale[:, ::-1], axis=1)[:, ::-1] > 0,
                   actual_floor=floor, actual_store=store, actual_sold=sold)


def _trailing_rate(units: np.ndarray, in_stock: np.ndarray) -> np.ndarray:
    """Mean units over the previous RATE_DAYS in-stock days, for each day."""
    def window_sum(x):
        c = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(x, axis=1)], axis=1)
        out = np.zeros_like(x, float)
        for t in range(x.shape[1]):
            out[:, t] = c[:, t] - c[:, max(t - RATE_DAYS, 0)]
        return out

    n = window_sum(in_stock.astype(float))
    return np.divide(window_sum(units * in_stock), n, out=np.zeros_like(n), where=n > 0)


@dataclass
class Run:
    floor: np.ndarray  # end-of-day stock, drugs x days
    store: np.ndarray
    sold: np.ndarray
    unmet: np.ndarray  # demand the floor couldn't meet (direct, censored demand)
    ordered: np.ndarray  # units ordered each evening


def policy_arrays(policy: pd.DataFrame, h: Holdout) -> dict[str, np.ndarray]:
    p = policy.set_index("drug_code") if "drug_code" in policy else policy
    missing = [d for d in h.drugs if d not in p.index]
    if missing:
        raise ValueError(f"policy has no row for {len(missing)} drugs, e.g. {missing[:5]}")
    p = p.reindex(h.drugs)
    lead = p["lead_days"] if "lead_days" in p else pd.Series(np.nan, index=p.index)
    lead = lead.fillna(h.lead.reindex(h.drugs)).fillna(LEAD_FALLBACK)
    S = p["S"].to_numpy(float)
    return {
        "s": p["s"].to_numpy(float),
        "S": S,
        "lead": np.maximum(lead.to_numpy(), 1).round().astype(int),
        "floor_min": (p["floor_min"] if "floor_min" in p else pd.Series(0.0, index=p.index))
                     .fillna(0).to_numpy(float),
        "floor_max": (p["floor_max"].fillna(p["S"]) if "floor_max" in p else p["S"])
                     .to_numpy(float),
    }


def simulate(policy: pd.DataFrame, h: Holdout) -> Run:
    """Run a policy over the holdout, all drugs at once, one day at a time."""
    p = policy_arrays(policy, h)
    n, T = len(h.drugs), len(h.days)
    floor, store = h.start_floor.astype(float).copy(), h.start_store.astype(float).copy()
    pipeline = np.zeros((n, T + p["lead"].max() + 1))  # arrivals by day index
    out = {k: np.zeros((n, T)) for k in ("floor", "store", "sold", "unmet", "ordered")}
    rows = np.arange(n)

    for t, day in enumerate(h.days):
        arriving = pipeline[:, t]
        if day < h.store_opens:
            floor += arriving
        else:
            store += arriving
        move = np.where((floor <= p["floor_min"]) & (store > 0),
                        np.clip(p["floor_max"] - floor, 0, store), 0)
        floor, store = floor + move, store - move

        adj = h.adjust[:, t]
        from_floor = np.maximum(adj, -floor)  # gains, or losses the floor can cover
        floor = floor + from_floor
        store = np.maximum(store + (adj - from_floor), 0)

        sold = np.minimum(h.demand[:, t], np.maximum(floor, 0))
        floor -= sold
        position = floor + store + pipeline[:, t + 1:].sum(axis=1)
        order = np.where(position <= p["s"], np.maximum(p["S"] - position, 0), 0)
        pipeline[rows, t + p["lead"]] += order

        for k, v in (("floor", floor), ("store", store), ("sold", sold),
                     ("unmet", h.demand[:, t] - sold), ("ordered", order)):
            out[k][:, t] = v
    return Run(**out)


def _spell_rate(out: np.ndarray, rate: np.ndarray) -> np.ndarray:
    """For each out day, the rate on the first day of its spell (0 elsewhere)."""
    held = np.zeros_like(rate)
    for t in range(rate.shape[1]):
        started = out[:, t] & ~(out[:, t - 1] if t else np.zeros(len(out), bool))
        held[:, t] = np.where(started, rate[:, t], held[:, t - 1] if t else 0)
    return np.where(out, held, 0)


def _spell_length(out: np.ndarray) -> np.ndarray:
    """For each out day, the length of the spell it belongs to (0 elsewhere)."""
    fwd, back = np.zeros(out.shape, int), np.zeros(out.shape, int)
    for t in range(out.shape[1]):
        fwd[:, t] = np.where(out[:, t], (fwd[:, t - 1] if t else 0) + 1, 0)
    for t in reversed(range(out.shape[1])):
        nxt = back[:, t + 1] if t + 1 < out.shape[1] else 0
        back[:, t] = np.where(out[:, t], nxt + 1, 0)
    return np.where(out, fwd + back - 1, 0)


def metrics(floor: np.ndarray, store: np.ndarray, sold: np.ndarray,
            h: Holdout) -> pd.DataFrame:
    """Per-drug scores from daily end-of-day floor, store and units sold."""
    total = floor + store
    stockout = (total <= 0) & h.live
    shelfout = (floor <= 0) & (store > 0) & ~stockout & h.live
    lost_out = np.maximum(_spell_rate(stockout, h.rate) - sold, 0) * stockout
    lost_shelf = np.maximum(_spell_rate(shelfout, h.rate) - sold, 0) * shelfout
    price = h.price[:, None]
    return pd.DataFrame({
        "drug_code": h.drugs,
        "days": h.live.sum(axis=1),
        "stockout_days": stockout.sum(axis=1),
        "stockout_days_long": (_spell_length(stockout) > LONG_SPELL).sum(axis=1),
        "shelfout_days": shelfout.sum(axis=1),
        "lost_units_stockout": lost_out.sum(axis=1),
        "lost_units_shelfout": lost_shelf.sum(axis=1),
        "lost_revenue_stockout": (lost_out * price).sum(axis=1),
        "lost_revenue_shelfout": (lost_shelf * price).sum(axis=1),
        "avg_stock_value": (np.maximum(total, 0) * price).mean(axis=1),
        "units_sold": sold.sum(axis=1),
    })


def summarise(per_drug: pd.DataFrame) -> pd.Series:
    s = per_drug.drop(columns="drug_code").sum()
    s["stockout_share"] = s["stockout_days"] / s["days"]
    s["shelfout_share"] = s["shelfout_days"] / s["days"]
    s["lost_revenue"] = s["lost_revenue_stockout"] + s["lost_revenue_shelfout"]
    s["drugs"] = len(per_drug)
    return s


def actual(h: Holdout) -> pd.DataFrame:
    """The pharmacist's real holdout, per drug."""
    return metrics(h.actual_floor, h.actual_store, h.actual_sold, h)


def score(policies: dict[str, pd.DataFrame], h: Holdout) -> pd.DataFrame:
    """Summary rows for the pharmacist and each named policy."""
    rows = {"pharmacist (actual)": summarise(actual(h))}
    for name, policy in policies.items():
        run = simulate(policy, h)
        s = summarise(metrics(run.floor, run.store, run.sold, h))
        s["unmet_units_direct"] = run.unmet.sum()
        rows[name] = s
    return pd.DataFrame(rows).T


def scaled(policy: pd.DataFrame, k: float) -> pd.DataFrame:
    """The policy with its reorder point and order-up-to level times `k`."""
    out = policy.copy()
    out["s"] = policy["s"] * k
    out["S"] = policy["S"] * k
    return out


def stock_value(policy: pd.DataFrame, h: Holdout) -> float:
    run = simulate(policy, h)
    return float((np.maximum(run.floor + run.store, 0) * h.price[:, None]).mean(axis=1).sum())


def match_stock_value(policy: pd.DataFrame, h: Holdout, target: float | None = None,
                      tol: float = 0.005) -> tuple[pd.DataFrame, float]:
    """Scale `s` and `S` so the policy holds the pharmacist's average stock value.

    Returns the scaled policy and the factor. `target` defaults to the
    pharmacist's actual average stock value over the holdout.
    """
    if target is None:
        target = float(actual(h)["avg_stock_value"].sum())
    lo, hi = 0.0, 1.0
    while stock_value(scaled(policy, hi), h) < target:
        lo, hi = hi, hi * 2
        if hi > 64:
            raise ValueError("can't reach the target stock value")
    for _ in range(40):
        k = (lo + hi) / 2
        v = stock_value(scaled(policy, k), h)
        if abs(v / target - 1) < tol:
            break
        lo, hi = (k, hi) if v < target else (lo, k)
    return scaled(policy, k), k


def tradeoff(policy: pd.DataFrame, h: Holdout,
             factors: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)) -> pd.DataFrame:
    """Stock held vs stock-outs as the policy is scaled up and down."""
    rows = []
    for k in factors:
        run = simulate(scaled(policy, k), h)
        s = summarise(metrics(run.floor, run.store, run.sold, h))
        rows.append({"factor": k, **s[["avg_stock_value", "stockout_days", "shelfout_days",
                                       "lost_revenue"]].to_dict()})
    return pd.DataFrame(rows)
