"""The panel grid and bookkeeping, on the synthetic history and the synthetic warehouse."""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import ml_warehouse
import numpy as np
import pandas as pd
import pytest
from ml_fixture import synthetic_history, synthetic_panel

from irpd import forecast, warehouse
from irpd.ml import features, panel, schema, split
from irpd.ml import history as hist

END = date(2025, 6, 29)
ORIGINS = split.selection_origins()
HISTORY_GROUPS = ["demand", "calendar", "store", "season"]


@pytest.fixture(scope="module")
def syn():
    return synthetic_history()[0]


@pytest.fixture(scope="module")
def built(syn):
    ctx = schema.PanelContext(con=None, data_end=END, origins=ORIGINS, history=syn)
    return panel.build_panel(ctx, HISTORY_GROUPS)


def test_matches_the_reference_fixture_panel(syn, built):
    ref = synthetic_panel(syn, ORIGINS)
    cols = [c for c in ref.columns if c not in ("tier", "c_eid")]
    ours = built[cols].sort_values(schema.KEYS).reset_index(drop=True)
    ref = ref[cols].sort_values(schema.KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(ours, ref, check_dtype=False, rtol=1e-6)
    assert (built["c_eid"] == built["eid"].astype("float32")).all()


def test_weekly_tier_fallback_matches_the_fixture(syn, built):
    ref = synthetic_panel(syn, ORIGINS).sort_values(schema.KEYS).reset_index(drop=True)
    ours = built.sort_values(schema.KEYS).reset_index(drop=True)
    assert (ours.tier.astype(str) == ref.tier.astype(str)).all()


def test_contract(built):
    schema.validate_panel(built)
    assert not {"h", "tier"} & set(schema.feature_columns(built, features.PREFIXES))
    feats = schema.feature_columns(built, features.PREFIXES)
    assert all(str(built[c].dtype) == "float32" for c in feats)
    assert built.groupby(["drug_code", "origin"]).size().eq(split.HORIZON).all()
    assert list(built.tier.cat.categories) == list(panel.TIERS)


def test_unavailable_and_censoring(syn):
    late = [date(2025, 6, 1)]
    p = panel.build_panel(schema.PanelContext(con=None, data_end=END, origins=late,
                                              history=syn))
    end_of_target = p.target_week + pd.Timedelta(days=6)
    assert (p.available == (end_of_target <= pd.Timestamp(END))).all()
    assert p.loc[~p.available, "target"].isna().all()
    assert not p.loc[~p.available, "censored"].any()
    early = panel.build_panel(schema.PanelContext(
        con=None, data_end=END, origins=[date(2023, 8, 27)], history=syn))
    assert not early.loc[early.target_week.dt.date < hist.first_stock_week(),
                         "censor_known"].any()
    assert early.loc[early.target_week.dt.date >= hist.first_stock_week(), "censor_known"].all()


def test_rows_are_the_active_drugs(syn, built):
    for origin in ORIGINS:
        sub = built[built.origin == pd.Timestamp(origin)]
        expect = syn["drugs"][hist.active(syn, origin, schema.ACTIVE_WEEKS)]
        assert set(sub.drug_code) == set(expect)


def test_eid_flag():
    assert panel.eid_week(date(2024, 4, 8))          # Eid week
    assert panel.eid_week(date(2024, 4, 1))          # within 7 days before
    assert panel.eid_week(date(2024, 5, 6))          # +26..+32 days
    assert not panel.eid_week(date(2024, 5, 13))     # +33 onwards
    assert not panel.eid_week(date(2024, 3, 25))     # ends Mar 31, before Apr 3


def test_join_rejects_bad_builder_output(syn):
    ctx = schema.PanelContext(con=None, data_end=END, origins=ORIGINS[:1], history=syn)
    base = panel.bookkeeping(ctx)
    good = features.builder("store")(ctx)
    with pytest.raises(ValueError, match="prefix"):
        panel.join(base, "store", good.rename(columns={"s_trend": "trend"}))
    with pytest.raises(ValueError, match="duplicate"):
        panel.join(base, "store", pd.concat([good, good]))
    with pytest.raises(ValueError, match="index"):
        panel.join(base, "store", good.reset_index())


def test_join_leaves_missing_rows_nan(syn):
    ctx = schema.PanelContext(con=None, data_end=END, origins=ORIGINS[:2], history=syn)
    base = panel.bookkeeping(ctx)
    feat = features.builder("store")(ctx)
    p = panel.join(base, "store", feat.iloc[1:])
    first = feat.index[0]
    rows = (p.drug_code == first[0]) & (p.origin == first[1])
    assert p.loc[rows, "s_lines_4"].isna().all()
    assert p.loc[~rows, "s_lines_4"].notna().all()


@pytest.fixture(scope="module")
def wh(tmp_path_factory):
    return ml_warehouse.build(tmp_path_factory.mktemp("mlwh"))


def test_warehouse_panel_with_every_group(wh):
    origins = [date(2023, 6, 4), date(2024, 3, 3)]
    con = duckdb.connect(str(wh), read_only=True)
    ctx = panel.context(con, ml_warehouse.END, origins,
                        con_at=lambda d: warehouse.as_of(d, wh))
    present = [g for g in features.GROUPS if g in ("demand", "calendar", "store", "price",
                                                   "drug")]
    p = panel.build_panel(ctx, present)
    for origin in origins:
        tiers = forecast.tiers(warehouse.as_of(origin, wh), origin)
        sub = p[(p.origin == pd.Timestamp(origin)) & (p.h == 1)]
        assert dict(zip(sub.drug_code, sub.tier.astype(str), strict=True)) == {
            c: tiers[c] for c in sub.drug_code}
    # W1 ran out in the week of 4 Mar 2024 (spell 3-5 Mar): censored, NaN target.
    row = p[(p.drug_code == "W1") & (p.target_week == pd.Timestamp(2024, 3, 4))]
    assert row.censored.all() and row.target.isna().all()
    # The late starter has no rows before it sold, and its scales come from history.
    assert "W5" not in set(p[p.origin == pd.Timestamp(2023, 6, 4)].drug_code)
    sub = p[p.origin == pd.Timestamp(2024, 3, 3)].groupby("drug_code").first()
    idx = {c: i for i, c in enumerate(ctx.drugs)}
    ts = hist.target_scale(ctx.history, date(2024, 3, 3))
    assert np.allclose(sub.target_scale, [ts[idx[c]] for c in sub.index])


def test_context_rejects_bad_settings(wh):
    con = duckdb.connect(str(wh), read_only=True)
    with pytest.raises(ValueError, match="censor_mode"):
        panel.context(con, ml_warehouse.END, [date(2024, 3, 3)], censor_mode="nope")
    with pytest.raises(ValueError, match="target week"):
        panel.context(con, ml_warehouse.END, [ml_warehouse.END])
    ctx = panel.context(con, ml_warehouse.END, [date(2024, 3, 3), date(2024, 3, 3)])
    assert ctx.origins == [date(2024, 3, 3)]
    assert ctx.connection_at(date(2024, 3, 3)) is con
    with pytest.raises(ValueError):
        ctx.connection_at(ml_warehouse.END + timedelta(days=1))
