"""The (drug, origin, h) panel: rows, targets, censoring, bookkeeping and features.

`context(con, data_end, origins)` loads the weekly history once
(`ml.history.load`) and wraps it in a `schema.PanelContext`.
`build_panel(ctx, groups)` lays out the grid, fills every bookkeeping column
in `schema.BOOKKEEPING` and joins the feature groups' `build(ctx)` output.

Rows: for each origin, the drugs with a sale in the ACTIVE_WEEKS weeks
before it, times h = 1..8. Everything on a row is computed from the history
cut at its origin, or from `ctx.connection_at(origin)`, except the target
and its flags, which describe the target week:

- `available`: the target week is in the loaded history and ended by
  `ctx.data_end`. A history cut at an origin has none of its target weeks.
- `censor_known`: the target week starts on or after
  `history.first_stock_week()`.
- `censored`: the target week had a total stock-out day (only where known).
- `target`: units, NaN when censored, unavailable or before the first sale.
- `tier`: `forecast.tiers` on the connection as of the origin. A
  history-only context (`con` and `con_at` both None, as in unit tests) has
  no daily sales, so it falls back to the share of the last 52 weeks with a
  sale (>= 0.9 fast, >= 0.5 regular, >= 0.15 slow, else rare).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd

from irpd import forecast as fc
from irpd.ml import features, schema, split
from irpd.ml import history as hist

TIERS = ("fast", "regular", "slow", "rare")
_WEEKLY_TIER_CUTS = ((0.9, "fast"), (0.5, "regular"), (0.15, "slow"))
DATETIME = "datetime64[s]"


def context(con: duckdb.DuckDBPyConnection, data_end: date, origins: Iterable[date],
            censor_mode: str = "mask", group_def: str = "ingredient",
            con_at: Callable[[date], duckdb.DuckDBPyConnection] | None = None,
            forecast_only: bool = False, options: dict | None = None) -> schema.PanelContext:
    """A PanelContext with the history loaded once, up to `data_end`.

    On real data pass `con=split.selection_con()` and `con_at=split.selection_con`.
    `options` are experiment switches for feature builders (none: frozen behaviour).
    """
    origins = sorted(set(origins))
    return schema.PanelContext(con=con, data_end=data_end, origins=origins,
                               history=hist.load(con, data_end), censor_mode=censor_mode,
                               group_def=group_def, con_at=con_at,
                               forecast_only=forecast_only, options=dict(options or {}))


def has_sql(ctx: schema.PanelContext) -> bool:
    """Whether the context has a warehouse connection (not history-only)."""
    return ctx.con is not None or ctx.con_at is not None


def eid_week(week: date) -> bool:
    """The Monday-start week overlaps an Eid window (-7..+28 days, as in forecast.backtest)."""
    lo, hi = fc.EID_DAYS.start, fc.EID_DAYS.stop - 1
    return any(e + timedelta(days=lo) <= week + timedelta(days=6)
               and week <= e + timedelta(days=hi) for e in fc.EID)


def active_rows(ctx: schema.PanelContext, origin: date) -> np.ndarray:
    """Indices into `ctx.drugs` of the drugs that get rows at `origin`."""
    return np.flatnonzero(hist.active(ctx.history, origin, schema.ACTIVE_WEEKS))


def grid(ctx: schema.PanelContext, by_h: bool = False) -> pd.DataFrame:
    """The panel's (drug_code, origin[, h]) keys, in panel order (origin, drug, h).

    Feature builders use it to emit one row per panel row.
    """
    parts = []
    for origin in sorted(set(ctx.origins)):
        codes = ctx.drugs[active_rows(ctx, origin)]
        parts.append(pd.DataFrame({"drug_code": codes, "origin": origin}))
    keys = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        {"drug_code": [], "origin": []})
    keys = keys.astype({"drug_code": "str"})
    keys["origin"] = pd.to_datetime(keys["origin"]).astype(DATETIME)
    if not by_h:
        return keys
    hs = np.arange(1, split.HORIZON + 1, dtype=np.int8)
    out = keys.loc[keys.index.repeat(len(hs))].reset_index(drop=True)
    out["h"] = np.tile(hs, len(keys))
    return out


def frame(ctx: schema.PanelContext, origin: date, rows: np.ndarray,
          cols: dict[str, np.ndarray], by_h: bool = False) -> pd.DataFrame:
    """Feature columns for one origin's `rows` (drug indices), keyed for the join.

    With `by_h`, each column is drugs x HORIZON and is flattened drug-major.
    """
    codes = ctx.drugs[rows]
    if by_h:
        hs = np.arange(1, split.HORIZON + 1, dtype=np.int8)
        df = pd.DataFrame({"drug_code": np.repeat(codes, len(hs)),
                           "origin": np.datetime64(origin, "s"),
                           "h": np.tile(hs, len(codes))})
        for k, v in cols.items():
            df[k] = np.asarray(v).reshape(-1)
    else:
        df = pd.DataFrame({"drug_code": codes, "origin": np.datetime64(origin, "s")})
        for k, v in cols.items():
            df[k] = v
    return df


def finish(parts: list[pd.DataFrame], by_h: bool) -> pd.DataFrame:
    """Concatenate per-origin frames, cast floats to float32 and set the key index."""
    keys = ["drug_code", "origin", "h"] if by_h else ["drug_code", "origin"]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=keys)
    df = df.astype({"drug_code": "str"})
    df["origin"] = pd.to_datetime(df["origin"]).astype(DATETIME)
    floats = [c for c in df.columns if c not in keys and pd.api.types.is_float_dtype(df[c])]
    df[floats] = df[floats].astype(np.float32)
    return df.set_index(keys)


def _weekly_tiers(h: dict) -> np.ndarray:
    """History-only fallback: tier by the share of the last 52 weeks with a sale."""
    share = (np.nan_to_num(h["y"][:, -52:]) > 0).sum(axis=1) / 52
    out = np.full(len(share), "rare", dtype=object)
    for cut, tier in reversed(_WEEKLY_TIER_CUTS):
        out[share >= cut] = tier
    return out


def tiers_at(ctx: schema.PanelContext, origin: date) -> np.ndarray:
    """Tier per drug in `ctx.drugs` as of `origin`."""
    if not has_sql(ctx):
        return _weekly_tiers(ctx.history_at(origin))
    of = fc.tiers(ctx.connection_at(origin), origin)
    return np.array([of.get(c, "rare") for c in ctx.drugs], dtype=object)


def bookkeeping(ctx: schema.PanelContext) -> pd.DataFrame:
    """Every BOOKKEEPING column for the grid, in panel order."""
    h = ctx.history
    col = {w: j for j, w in enumerate(h["weeks"])}
    first_stock = hist.first_stock_week()
    parts = []
    for origin in sorted(set(ctx.origins)):
        rows = active_rows(ctx, origin)
        k = len(rows)
        n = ctx.weeks_to(origin)
        weeks = [schema.target_week(origin, hh) for hh in range(1, split.HORIZON + 1)]
        avail = np.array([w in col and w + timedelta(days=6) <= ctx.data_end
                          for w in weeks])
        known = np.array([w >= first_stock for w in weeks])
        t = np.array([col.get(w, 0) for w in weeks])
        y = np.where(avail[None, :], h["y"][rows][:, t], np.nan)
        out = h["out"][rows][:, t] & (avail & known)[None, :]
        target = np.where(out, np.nan, y)
        scored = np.nansum(h["y"][rows, max(0, n - schema.SCORED_WEEKS):n], axis=1) > 0
        tier = tiers_at(ctx, origin)[rows]
        parts.append(pd.DataFrame({
            "drug_code": np.repeat(ctx.drugs[rows], split.HORIZON),
            "origin": np.datetime64(origin, "s"),
            "h": np.tile(np.arange(1, split.HORIZON + 1, dtype=np.int8), k),
            "target_week": np.tile(np.array(weeks, dtype="datetime64[s]"), k),
            "target": target.reshape(-1).astype(np.float32),
            "censored": out.reshape(-1),
            "censor_known": np.tile(known, k),
            "available": np.tile(avail, k),
            "scored": np.repeat(scored, split.HORIZON),
            "tier": np.repeat(tier, split.HORIZON),
            "eid": np.tile([eid_week(w) for w in weeks], k),
            "target_scale": np.repeat(hist.target_scale(h, origin)[rows], split.HORIZON),
            "mase_scale": np.repeat(hist.mase_scale(h, origin)[rows], split.HORIZON),
        }))
    if parts:
        df = pd.concat(parts, ignore_index=True)
    else:
        df = pd.DataFrame({c: pd.Series(dtype="float32") for c in schema.BOOKKEEPING})
    return df.astype({"drug_code": "str", "origin": DATETIME, "target_week": DATETIME,
                      "h": "int8", "target": "float32", "censored": "bool",
                      "censor_known": "bool", "available": "bool", "scored": "bool",
                      "tier": pd.CategoricalDtype(TIERS), "eid": "bool",
                      "target_scale": "float32", "mase_scale": "float32"})


def _check_group(name: str, feat: pd.DataFrame) -> None:
    prefix = features.GROUPS[name][0]
    if set(feat.index.names) not in ({"drug_code", "origin"}, {"drug_code", "origin", "h"}):
        raise ValueError(f"{name}: index must be (drug_code, origin[, h]), "
                         f"not {feat.index.names}")
    bad = [c for c in feat.columns if not c.startswith(prefix)]
    if bad:
        raise ValueError(f"{name}: columns without the {prefix} prefix: {bad}")
    if feat.index.duplicated().any():
        raise ValueError(f"{name}: duplicate keys")


def join(panel: pd.DataFrame, name: str, feat: pd.DataFrame) -> pd.DataFrame:
    """Left-join one group's features onto the panel by its key columns."""
    _check_group(name, feat)
    keys = list(feat.index.names)
    feat = feat.reorder_levels(keys) if len(keys) > 1 else feat
    lookup = pd.MultiIndex.from_arrays(
        [panel[k].astype(feat.index.get_level_values(k).dtype) for k in keys], names=keys)
    taken = feat.reindex(lookup)
    taken.index = panel.index
    return pd.concat([panel, taken], axis=1)


def build_panel(ctx: schema.PanelContext, groups: Iterable[str] = ()) -> pd.DataFrame:
    """The validated panel for `ctx.origins` with the given feature groups joined."""
    panel = bookkeeping(ctx)
    for name in groups:
        feat = features.builder(name)(ctx)
        panel = join(panel, name, feat)
    floats = [c for c in panel.columns
              if c not in schema.BOOKKEEPING and pd.api.types.is_float_dtype(panel[c])]
    panel[floats] = panel[floats].astype(np.float32)
    schema.validate_panel(panel)
    return panel
