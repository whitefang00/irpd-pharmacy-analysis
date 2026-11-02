"""Small, public demand histories for ML tests; no warehouse or private files needed."""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from irpd import forecast
from irpd.ml import history as hist
from irpd.ml import schema, split

PATTERNS = ("smooth", "trend", "eid", "intermittent", "lumpy", "late_start", "rare")
_START = date(2023, 1, 2)
_FEATURES = [*(f"d_lag_{k}" for k in range(1, 5)), "d_mean_4", "d_mean_13",
             "y_week_of_year", "c_eid"]
_DTYPES = {"drug_code": "str", "origin": "datetime64[s]", "target_week": "datetime64[s]",
           **{k: v for k, v in schema.BOOKKEEPING.items()
              if k not in ("drug_code", "origin", "target_week")}}


def _eid_week(week: date) -> bool:
    return any(e - timedelta(days=7) <= week + timedelta(days=6)
               and week <= e + timedelta(days=28) for e in forecast.EID)


def _mean(values: np.ndarray) -> float:
    observed = values[np.isfinite(values)]
    return float(observed.mean()) if len(observed) else np.nan


def synthetic_history(n_per_pattern: int = 4, seed: int = 0,
                      until: date = date(2025, 6, 29)) -> tuple[dict, pd.DataFrame]:
    """Return load_history-shaped observations and uncensored expected weekly demand.

    Sparse series are missing before their first positive sale, just as in the
    warehouse. Late starts begin in June/July 2024. Only a subset of dense
    series gets stock-out spells, keeping genuine intermittent zeros distinct.
    """
    if n_per_pattern < 1:
        raise ValueError("n_per_pattern must be positive")
    n_weeks = ((until - _START).days + 1) // 7
    if n_weeks < 1:
        raise ValueError("until must include at least one complete week from 2023-01-02")
    rng = np.random.default_rng(seed)
    weeks = np.array([_START + timedelta(weeks=i) for i in range(n_weeks)], dtype=object)
    eid = np.array([_eid_week(w) for w in weeks])
    codes, sales, outages, truth = [], [], [], []
    for pattern in PATTERNS:
        for j in range(n_per_pattern):
            code = f"SYN-{pattern}-{j}"
            level = rng.uniform(20, 60)
            mean = np.full(n_weeks, level)
            first = 0
            if pattern == "trend":
                mean *= 1.3 ** (np.arange(n_weeks) * 7 / 365.25)
            elif pattern == "eid":
                mean *= np.where(eid, 1.8, 1.0)
            elif pattern == "late_start":
                first = 74 + j % 8
                mean[:first] = 0
            if pattern in ("intermittent", "lumpy"):
                probability, maximum = (0.3, 5) if pattern == "intermittent" else (0.15, 40)
                mean[:] = probability * (maximum + 1) / 2
                y = (rng.random(n_weeks) < probability) * rng.integers(1, maximum + 1, n_weeks)
                y = y.astype(float)
            elif pattern == "rare":
                # Renewal arrivals, 9..17 weeks apart (roughly two to four months).
                arrivals = np.zeros(n_weeks)
                arrivals[:min(13, n_weeks)] = 1 / min(13, n_weeks)
                for t in range(n_weeks):
                    for gap in range(9, 18):
                        if t + gap < n_weeks:
                            arrivals[t + gap] += arrivals[t] / 9
                mean = 3 * arrivals  # exact renewal probability times mean sale size
                y = np.zeros(n_weeks)
                t = int(rng.integers(0, min(13, n_weeks)))
                while t < n_weeks:
                    y[t] = rng.integers(1, 6)
                    t += int(rng.integers(9, 18))
            else:
                y = rng.poisson(mean).astype(float)
            positive = np.flatnonzero(y > 0)
            y[:positive[0] if len(positive) else n_weeks] = np.nan
            out = np.zeros(n_weeks, dtype=bool)
            eligible = np.flatnonzero((weeks >= hist.first_stock_week())
                                      & (np.arange(n_weeks) >= first))
            if pattern in ("smooth", "trend", "eid", "late_start") and j % 2 == 0:
                # Separate blocks prevent adjacent spells combining into >6 weeks.
                for block in np.array_split(eligible, 3):
                    block = block[:-1]  # leave an uncensored gap between blocks
                    if not len(block):
                        continue
                    length = int(rng.integers(1, min(6, len(block)) + 1))
                    start = int(rng.integers(0, max(1, len(block) - length + 1)))
                    out[block[start:start + length]] = True
                multiplier = rng.uniform(0, 0.3, out.sum())
                # Bound against the expectation as well as the realised demand.
                y[out] = np.minimum(y[out] * multiplier, 0.3 * mean[out])
            codes.append(code)
            sales.append(y)
            outages.append(out)
            truth.extend(zip([code] * n_weeks, weeks, mean, [pattern] * n_weeks, strict=True))
    # load_history sorts drug codes lexically.
    order = np.argsort(codes)
    days = [_START + timedelta(days=i) for i in range(n_weeks * 7)]
    weekday = np.array([1.05, 1.0, 1.0, 1.05, 1.15, 0.9, 0.75])
    daily_eid = np.array([any(-7 <= (d - e).days <= 28 for e in forecast.EID) for d in days])
    lines = (800 * weekday[[d.weekday() for d in days]]
             * (1 + np.arange(len(days)) * 0.00015) * np.where(daily_eid, 1.8, 1.0)
             * rng.uniform(0.98, 1.02, len(days)))
    history = {"drugs": np.array(codes, dtype=object)[order], "weeks": weeks,
               "y": np.array(sales)[order], "out": np.array(outages)[order],
               "lines": pd.Series(lines, index=pd.Index(days, name="date"))}
    _extend(history, rng)
    return history, pd.DataFrame(truth, columns=["drug_code", "week_start", "true_mean", "pattern"])


def _price(i: int, day: date) -> float:
    factor = 1.06 if day >= date(2025, 1, 1) else 1.03 if day >= date(2024, 1, 1) else 1.0
    return float(round((1000 + 250 * i) * factor))


def _extend(history: dict, rng) -> None:
    """Add the keys `ml.history.load` adds: stock_known, revenue, shelfout_days, first_sale."""
    weeks, y = history["weeks"], history["y"]
    known = np.array([w >= hist.first_stock_week() for w in weeks], dtype=bool)
    history["stock_known"] = known
    history["out"] = history["out"] & known[None, :]
    prices = np.array([[_price(i, w) for w in weeks] for i in range(len(y))])
    history["revenue"] = y * prices
    shelf = np.where(known[None, :] & (rng.random(y.shape) < 0.05),
                     rng.integers(1, 7, y.shape), 0)
    history["shelfout_days"] = shelf.astype(np.int16)
    sold = np.nan_to_num(y) > 0
    history["first_sale"] = np.where(sold.any(axis=1), sold.argmax(axis=1), -1)


def synthetic_drugs(history) -> pd.DataFrame:
    """Ingredient-map rows with mixed-form siblings, singletons and unmapped drugs."""
    rows = []
    for i, code in enumerate(history["drugs"]):
        slot, group = i % 8, i // 8
        if slot < 3:
            ingredients = "paracetamol + caffeine" if group == 0 else f"ingredient-{group}"
        elif slot < 6:
            ingredients = f"singleton-{i}"
        else:
            ingredients = ""
        form = "syrup" if slot == 1 else "tablet"
        rows.append((code, f"Synthetic {code}", ingredients,
                     "120 mg/5 ml" if form == "syrup" else "500 mg", form, "drug",
                     bool(ingredients), "synthetic" if ingredients else ""))
    return pd.DataFrame(rows, columns=["drug_code", "drug_name", "ingredients", "strength",
                                       "form", "kind", "mapped", "source_model"])


def synthetic_prices(history) -> pd.DataFrame:
    """Rupiah unit prices with small, deterministic steps; dates stay within history."""
    rows = []
    end = history["weeks"][-1] + timedelta(days=6)
    for i, code in enumerate(history["drugs"]):
        for day in (_START, date(2024, 1, 1), date(2025, 1, 1)):
            if day <= end:
                rows.append((code, day, _price(i, day)))
    return pd.DataFrame(rows, columns=["drug_code", "valid_from", "price"])


def synthetic_panel(history, origins: list[date], seed: int = 0) -> pd.DataFrame:
    """Build a mask-mode panel. Features are deterministic; seed is API-compatible.

    Targets beyond history are missing. Eligibility, tiers, scales and demand
    features depend only on completed weeks at the origin. MASE uses adjacent
    differences on the masked weekly grid, exactly as forecast.backtest does.
    """
    if any(o.weekday() != 6 for o in origins):
        raise ValueError("origins must be Sundays")
    rows = []
    weeks, y, out = history["weeks"], history["y"], history["out"]
    week_index = {w: i for i, w in enumerate(weeks)}
    for origin in origins:
        n = sum(w + timedelta(days=6) <= origin for w in weeks)
        for i, code in enumerate(history["drugs"]):
            raw = y[i, :n]
            recent = raw[-52:]
            if not np.any(recent > 0):
                continue
            masked = np.where(out[i, :n], np.nan, raw)
            scale = schema.SCALE_FLOOR
            for window in schema.SCALE_WEEKS:
                candidate = _mean(masked[-window:])
                if candidate > 0:
                    scale = candidate
                    break
            share = np.count_nonzero(recent > 0) / len(recent)
            tier = next(t for threshold, t in [(0.9, "fast"), (0.5, "regular"),
                                               (0.15, "slow"), (0, "rare")] if share >= threshold)
            lags = [masked[-k] if n >= k else np.nan for k in range(1, 5)]
            means = [_mean(masked[-k:]) for k in (4, 13)]
            mase = _mean(np.abs(np.diff(masked)))
            scored = bool(np.nansum(raw[-schema.SCORED_WEEKS:]) > 0)
            for h in range(1, split.HORIZON + 1):
                week = schema.target_week(origin, h)
                target_i = week_index.get(week)
                available = target_i is not None
                known = week >= hist.first_stock_week()
                censored = bool(known and available and out[i, target_i])
                target = y[i, target_i] if available and not censored else np.nan
                eid = _eid_week(week)
                rows.append([code, origin, h, week, target, censored, known, available, scored,
                             tier, eid, scale, mase, *lags, *means,
                             week.isocalendar().week, float(eid)])
    panel = pd.DataFrame(rows, columns=[*schema.BOOKKEEPING, *_FEATURES])
    return panel.astype({**_DTYPES, **dict.fromkeys(_FEATURES, "float32")})


def synthetic_store(history) -> pd.DataFrame:
    """Daily store traffic, identical to the history's lines series."""
    return history["lines"].rename("lines").rename_axis("date").reset_index()
