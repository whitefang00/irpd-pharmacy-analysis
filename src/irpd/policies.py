"""Two reference policies for the replay, built only from data up to the cutoff.

- `mimic`: the pharmacist's own habits, read off `pharmacist_baseline` and
  the transfers before the cutoff. A sanity check: replayed, it should land
  near what the pharmacist actually did.
- `textbook`: a classic (s, S) from the mean and spread of daily demand,
  with a service level and a fixed order cycle.

Both return the policy DataFrame `irpd.replay.simulate` takes, for every
drug in `replay.candidates`. Pass a connection from `holdout.agent_view(day)`
and the same `day` (default the cutoff): the tiers and the pharmacist's
baseline are read as the connection shows them.
"""

from datetime import date, timedelta
from statistics import NormalDist

import duckdb
import numpy as np
import pandas as pd

from irpd import config, replay

HISTORY_DAYS = 182  # demand rate and spread: the half-year before the cutoff
TRANSFER_DAYS = 365  # floor habits: the year before the cutoff


def demand_stats(con: duckdb.DuckDBPyConnection, day: date = config.CUTOFF) -> pd.DataFrame:
    """Mean and sd of daily units per candidate drug, on days it was in stock."""
    since = f"DATE '{(day - timedelta(days=HISTORY_DAYS - 1)).isoformat()}'"
    df = con.execute(f"""
        SELECT s.drug_code, avg(coalesce(d.units, 0)) AS rate,
               coalesce(stddev_samp(coalesce(d.units, 0)), 0) AS sd
        FROM daily_stock s LEFT JOIN daily_demand d USING (drug_code, date)
        WHERE s.location = 'total' AND s.closing > 0
          AND s.date BETWEEN {since} AND DATE '{day.isoformat()}'
        GROUP BY 1""").df().set_index("drug_code")
    drugs = replay.candidates(con)
    return df.reindex(drugs).fillna(0).assign(lead=replay.default_lead(con).reindex(drugs)
                                              .fillna(replay.LEAD_FALLBACK))


def mimic(con: duckdb.DuckDBPyConnection, day: date = config.CUTOFF) -> pd.DataFrame:
    """Order when cover falls to what the pharmacist had left at arrival, in their usual qty.

    With lead time L and rate r, an order placed at position s arrives with
    about s - rL on hand, so s = r * (median cover at arrival + L) and
    S = s + median order qty. On the floor, transfers start at the floor
    level the pharmacist usually moved stock at, and fill to the level after.
    """
    st = demand_stats(con, day)
    base = con.execute("""
        SELECT drug_code, median_cover_days, median_order_qty FROM pharmacist_baseline
        """).df().set_index("drug_code").reindex(st.index)
    cover = base["median_cover_days"].fillna(base["median_cover_days"].median())
    qty = base["median_order_qty"].fillna(st["rate"] * 14)
    s = st["rate"] * (cover + st["lead"])

    since = f"DATE '{(day - timedelta(days=TRANSFER_DAYS - 1)).isoformat()}'"
    floor = con.execute(f"""
        SELECT t.drug_code, median(f.closing) AS floor_min,
               median(f.closing + t.qty) AS floor_max
        FROM transfers t JOIN daily_stock f
          ON f.drug_code = t.drug_code AND f.location = 'floor' AND f.date = t.date - 1
        WHERE t.to_location = 'floor'
          AND t.date BETWEEN {since} AND DATE '{day.isoformat()}'
        GROUP BY 1""").df().set_index("drug_code").reindex(st.index)
    return pd.DataFrame({
        "drug_code": st.index,
        "s": np.ceil(s).to_numpy(),
        "S": np.ceil(s + qty).to_numpy(),
        "lead_days": st["lead"].to_numpy(),
        "floor_min": floor["floor_min"].clip(lower=0).fillna(0).to_numpy(),
        "floor_max": floor["floor_max"].to_numpy(),  # NaN -> S in the sim
    })


def textbook(con: duckdb.DuckDBPyConnection, service: float = 0.95,
             cycle_days: int = 14, day: date = config.CUTOFF) -> pd.DataFrame:
    """(s, S) with safety stock for a service level over the lead time.

    s = rL + z * sd * sqrt(L), S = s + r * cycle_days. The floor is topped up
    when it can't cover a day of demand at the same service level, to that
    level plus a week of demand.
    """
    st = demand_stats(con, day)
    z = NormalDist().inv_cdf(service)
    s = st["rate"] * st["lead"] + z * st["sd"] * np.sqrt(st["lead"])
    floor_min = np.ceil(st["rate"] + z * st["sd"])
    return pd.DataFrame({
        "drug_code": st.index,
        "s": np.ceil(s).to_numpy(),
        "S": np.ceil(s + st["rate"] * cycle_days).to_numpy(),
        "lead_days": st["lead"].to_numpy(),
        "floor_min": floor_min.to_numpy(),
        "floor_max": (floor_min + np.ceil(st["rate"] * 7)).to_numpy(),
    })
