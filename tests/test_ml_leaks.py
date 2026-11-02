"""Leak tests for the panel and every feature group (plan 5.5).

1. Truncation: an origin's features are the same whether built from data cut
   at the origin or from all the data.
2. Per horizon: changing anything after the origin (sales, lines, revenue,
   shelf-out days, stock-out flags; on a warehouse, every row after it),
   including just one target week, moves no feature on the origin's rows.
3. Training rows: `trainable(panel, O)` rows' target weeks end by O, and
   their targets are what data cut at O says.
4. Nothing reads past `ctx.data_end`, and builders reach the warehouse only
   through `ctx.connection_at`.

Each runs on every registered feature group whose module exists, so groups
added later are covered by the same harness. Groups that need SQL (module
flag `NEEDS_SQL`, assumed True when absent) are skipped on the synthetic
history and run on the synthetic warehouse (`ml_warehouse`) and, when
`split.SELECTION_DB` exists, on the real selection data.
"""

from __future__ import annotations

import importlib
import importlib.util
from datetime import date, timedelta
from functools import cache
from pathlib import Path

import duckdb
import ml_warehouse
import numpy as np
import pandas as pd
import pytest
from ml_fixture import synthetic_history

from irpd import warehouse
from irpd.ml import features, panel, schema, split
from irpd.ml import history as hist

PRESENT = [g for g in features.GROUPS
           if importlib.util.find_spec(f"irpd.ml.features.{g}") is not None]
MISSING = [g for g in features.GROUPS if g not in PRESENT]


def needs_sql(group: str) -> bool:
    return bool(getattr(importlib.import_module(f"irpd.ml.features.{group}"), "NEEDS_SQL", True))


HISTORY_GROUPS = [g for g in PRESENT if not needs_sql(g)]
# (group, censor mode): the censoring mode only changes the demand group.
CASES = [(g, m) for g in PRESENT for m in (("mask", "raw", "impute") if g == "demand"
                                           else ("mask",))]
HISTORY_CASES = [(g, m) for g, m in CASES if g in HISTORY_GROUPS and m != "impute"]

SYN_END = date(2025, 6, 29)
SYN_ORIGINS = [date(2023, 6, 4), date(2024, 3, 3), date(2024, 3, 31)]
WH_END = ml_warehouse.END
WH_ORIGINS = [date(2023, 6, 4), date(2024, 3, 3)]
REAL_ORIGINS = [date(2023, 11, 26), date(2024, 3, 3)]


def test_every_group_is_tested_or_missing():
    assert set(PRESENT) | set(MISSING) == set(features.GROUPS)
    assert {"demand", "calendar", "store", "price", "drug"} <= set(PRESENT)


# ---- helpers --------------------------------------------------------------


def at(feat: pd.DataFrame, origin: date) -> pd.DataFrame:
    """One origin's rows of a builder's output, comparable across builds."""
    named = any(n is not None for n in feat.index.names)
    df = feat.reset_index(drop=not named)
    df = df[df["origin"] == pd.Timestamp(origin)]
    df = df.sort_values([c for c in schema.KEYS if c in df.columns]).reset_index(drop=True)
    cats = [c for c in df.columns if isinstance(df[c].dtype, pd.CategoricalDtype)]
    return df.astype(dict.fromkeys(cats, "object"))


def same(a: pd.DataFrame, b: pd.DataFrame) -> None:
    assert len(a), "nothing to compare"
    pd.testing.assert_frame_equal(a, b, check_exact=True)


KNOWN_AT_ORIGIN = ["drug_code", "origin", "h", "target_week", "censor_known", "scored",
                   "tier", "eid", "target_scale", "mase_scale"]


def bookkeeping_at(ctx, origin):
    return at(panel.bookkeeping(ctx)[KNOWN_AT_ORIGIN], origin)


# ---- synthetic history: groups that need no SQL ---------------------------


@pytest.fixture(scope="module")
def syn():
    history, _ = synthetic_history()
    return history


def syn_ctx(history, origins, data_end=SYN_END, mode="mask"):
    return schema.PanelContext(con=None, data_end=data_end, origins=list(origins),
                               history=history, censor_mode=mode)


def perturbed(history, origin, week=None, seed=7):
    """History with every week after `origin` (or only target week `week`) changed."""
    n = hist.weeks_to(history, origin)
    cols = slice(n, None) if week is None else slice(n + week - 1, n + week)
    rng = np.random.default_rng(seed)
    out = {**history}
    for k in ("y", "revenue"):
        out[k] = history[k].copy()
        out[k][:, cols] = rng.uniform(0, 5000, out[k][:, cols].shape)
    out["out"] = history["out"].copy()
    out["out"][:, cols] = ~out["out"][:, cols]
    out["shelfout_days"] = history["shelfout_days"].copy()
    out["shelfout_days"][:, cols] = 7 - out["shelfout_days"][:, cols]
    lines = history["lines"].copy()
    if week is None:
        lines[lines.index > origin] *= 3
    else:
        start = schema.target_week(origin, week)
        inside = (lines.index >= start) & (lines.index <= start + timedelta(days=6))
        lines[inside] *= 3
    out["lines"] = lines
    sold = np.nan_to_num(out["y"]) > 0
    out["first_sale"] = np.where(sold.any(axis=1), sold.argmax(axis=1), -1)
    return out


@pytest.mark.parametrize("group, mode", HISTORY_CASES)
def test_truncation_history(syn, group, mode):
    build = features.builder(group)
    full = build(syn_ctx(syn, SYN_ORIGINS, mode=mode))
    for origin in SYN_ORIGINS:
        cut = syn_ctx(hist.cut(syn, origin), [origin], origin + timedelta(days=7), mode)
        same(at(full, origin), at(build(cut), origin))


@pytest.mark.parametrize("group, mode", HISTORY_CASES)
def test_future_changes_move_nothing_history(syn, group, mode):
    build = features.builder(group)
    for origin in SYN_ORIGINS:
        base = at(build(syn_ctx(syn, [origin], mode=mode)), origin)
        for week in (None, *range(1, split.HORIZON + 1)):
            changed = perturbed(syn, origin, week)
            same(base, at(build(syn_ctx(changed, [origin], mode=mode)), origin))


def test_bookkeeping_truncation_and_future_history(syn):
    full = syn_ctx(syn, SYN_ORIGINS)
    for origin in SYN_ORIGINS:
        expect = bookkeeping_at(full, origin)
        cut = syn_ctx(hist.cut(syn, origin), [origin], origin + timedelta(days=7))
        same(expect, bookkeeping_at(cut, origin))
        same(expect, bookkeeping_at(syn_ctx(perturbed(syn, origin), [origin]), origin))


def test_trainable_rows_are_known_at_their_origin(syn):
    origins = split.training_origins(date(2024, 10, 13))
    full = panel.build_panel(syn_ctx(syn, origins), HISTORY_GROUPS)
    for origin in split.selection_origins():
        rows = schema.trainable(full, origin)
        assert len(rows)
        assert (rows.target_week + pd.Timedelta(days=6) <= pd.Timestamp(origin)).all()
        assert (rows.origin < pd.Timestamp(origin)).all()
        # Data cut at the origin gives the same targets for those rows.
        cut_ctx = syn_ctx(hist.cut(syn, origin), [o for o in origins if o < origin],
                          origin + timedelta(days=7))
        cut = schema.trainable(panel.bookkeeping(cut_ctx), origin)
        cols = ["drug_code", "origin", "h", "target", "censored"]
        same(at_all(rows[cols]), at_all(cut[cols]))


def at_all(df):
    return df.sort_values(schema.KEYS).reset_index(drop=True)


# ---- synthetic warehouse: every group -------------------------------------


@pytest.fixture(scope="module")
def wh(tmp_path_factory):
    return ml_warehouse.build(tmp_path_factory.mktemp("mlwh"))


@pytest.fixture(scope="module")
def wh_perturbed(tmp_path_factory):
    return {o: ml_warehouse.build(tmp_path_factory.mktemp("mlwh_p"), perturb_after=o)
            for o in WH_ORIGINS}


@cache
def _as_of(path: Path, day: date) -> duckdb.DuckDBPyConnection:
    return warehouse.as_of(day, path)


def wh_ctx(path, origins, mode="mask"):
    """All data, read through connections as of each day."""
    con = duckdb.connect(str(path), read_only=True)
    return panel.context(con, WH_END, origins, mode, con_at=lambda d: _as_of(path, d))


def wh_cut_ctx(path, origin, mode="mask"):
    """Data cut at `origin`: every connection stops there."""
    return panel.context(_as_of(path, origin), origin + timedelta(days=7), [origin], mode,
                         con_at=lambda d: _as_of(path, min(d, origin)))


@pytest.mark.parametrize("group, mode", CASES)
def test_truncation_warehouse(wh, group, mode):
    build = features.builder(group)
    full = build(wh_ctx(wh, WH_ORIGINS, mode))
    for origin in WH_ORIGINS:
        same(at(full, origin), at(build(wh_cut_ctx(wh, origin, mode)), origin))


@pytest.mark.parametrize("group, mode", CASES)
def test_future_changes_move_nothing_warehouse(wh, wh_perturbed, group, mode):
    build = features.builder(group)
    for origin in WH_ORIGINS:
        base = at(build(wh_ctx(wh, [origin], mode)), origin)
        same(base, at(build(wh_ctx(wh_perturbed[origin], [origin], mode)), origin))


def test_perturbation_really_changes_later_data(wh, wh_perturbed):
    origin = WH_ORIGINS[-1]
    q = "SELECT sum(units) FROM weekly_demand WHERE week_start {} DATE '2024-03-04'"
    a, b = (duckdb.connect(str(p), read_only=True) for p in (wh, wh_perturbed[origin]))
    assert a.execute(q.format("<")).fetchone() == b.execute(q.format("<")).fetchone()
    assert a.execute(q.format(">=")).fetchone() != b.execute(q.format(">=")).fetchone()


def test_bookkeeping_warehouse(wh, wh_perturbed):
    full = wh_ctx(wh, WH_ORIGINS)
    for origin in WH_ORIGINS:
        expect = bookkeeping_at(full, origin)
        same(expect, bookkeeping_at(wh_cut_ctx(wh, origin), origin))
        same(expect, bookkeeping_at(wh_ctx(wh_perturbed[origin], [origin]), origin))


def test_price_blip_is_only_seen_as_of_the_origin(wh):
    """The blip on the origin is the price then; hindsight would drop it."""
    origin = ml_warehouse.BLIP_DAY
    feat = at(features.builder("price")(wh_ctx(wh, [origin])), origin).set_index("drug_code")
    assert feat.loc["W6", "p_price"] == pytest.approx(12000 * 1.2)


class _Forbidden:
    def __getattr__(self, name):
        raise AssertionError(f"a builder used ctx.con.{name}; use ctx.connection_at")


@pytest.mark.parametrize("mode", ["mask", "impute"])
def test_nothing_reads_past_data_end(wh, mode):
    data_end = date(2024, 4, 28)
    origins = [date(2023, 6, 4), date(2024, 3, 3)]
    asked: list[date] = []

    def spy(day):
        asked.append(day)
        return _as_of(wh, day)

    ctx = panel.context(_as_of(wh, data_end), data_end, origins, mode, con_at=spy)
    # Loading through a full connection reads nothing past data_end either.
    loaded = hist.load(duckdb.connect(str(wh), read_only=True), data_end)
    for k in ("y", "out", "revenue", "shelfout_days"):
        np.testing.assert_array_equal(ctx.history[k], loaded[k])
    pd.testing.assert_series_equal(ctx.history["lines"], loaded["lines"])
    ctx.con = _Forbidden()
    panel.build_panel(ctx, PRESENT)
    assert asked and max(asked) <= data_end
    with pytest.raises(ValueError, match="after data_end"):
        ctx.connection_at(data_end + timedelta(days=1))


# ---- real selection data --------------------------------------------------


@pytest.fixture(scope="module")
def selection():
    if not split.SELECTION_DB.exists():
        pytest.skip("selection.duckdb not present")
    return split.selection_con()


@pytest.mark.parametrize("group, mode", CASES)
def test_truncation_real_selection(selection, group, mode):
    build = features.builder(group)
    full = build(panel.context(selection, split.SELECTION_END, REAL_ORIGINS, mode,
                               con_at=split.selection_con))
    for origin in REAL_ORIGINS:
        cut = panel.context(split.selection_con(origin), origin + timedelta(days=7), [origin],
                            mode, con_at=lambda d, o=origin: split.selection_con(min(d, o)))
        same(at(full, origin), at(build(cut), origin))


def test_bookkeeping_real_selection(selection):
    full = panel.context(selection, split.SELECTION_END, REAL_ORIGINS,
                         con_at=split.selection_con)
    for origin in REAL_ORIGINS:
        cut = panel.context(split.selection_con(origin), origin + timedelta(days=7), [origin],
                            con_at=lambda d, o=origin: split.selection_con(min(d, o)))
        same(bookkeeping_at(full, origin), bookkeeping_at(cut, origin))


# ---- the harness catches leaks ----------------------------------------------


def _leaky_history(ctx):
    """Reads the target week from the uncut history: test 1 and 2 must catch it."""
    parts = []
    for origin in ctx.origins:
        rows = panel.active_rows(ctx, origin)
        n = ctx.weeks_to(origin)
        nxt = ctx.history["y"][rows, n] if n < ctx.history["y"].shape[1] else np.nan
        parts.append(panel.frame(ctx, origin, rows, {"d_leak": np.broadcast_to(nxt, len(rows))}))
    return panel.finish(parts, by_h=False)


def _leaky_sql(ctx):
    """Reads prices through ctx.con at the end of the data: test 1 must catch it."""
    parts = []
    for origin in ctx.origins:
        rows = panel.active_rows(ctx, origin)
        p = dict(ctx.con.execute("SELECT drug_code, max(price) FROM prices GROUP BY 1").fetchall())
        parts.append(panel.frame(ctx, origin, rows,
                                 {"p_leak": [p.get(c, np.nan) for c in ctx.drugs[rows]]}))
    return panel.finish(parts, by_h=False)


def test_harness_catches_a_history_leak(syn):
    origin = SYN_ORIGINS[1]
    full = at(_leaky_history(syn_ctx(syn, [origin])), origin)
    cut = syn_ctx(hist.cut(syn, origin), [origin], origin + timedelta(days=7))
    with pytest.raises(AssertionError):
        same(full, at(_leaky_history(cut), origin))
    with pytest.raises(AssertionError):
        same(full, at(_leaky_history(syn_ctx(perturbed(syn, origin, week=1), [origin])), origin))


def test_harness_catches_an_sql_leak(wh):
    origin = WH_ORIGINS[0]
    full = at(_leaky_sql(wh_ctx(wh, [origin])), origin)
    with pytest.raises(AssertionError):
        same(full, at(_leaky_sql(wh_cut_ctx(wh, origin)), origin))
