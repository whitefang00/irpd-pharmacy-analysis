"""Feature groups B owns: demand, calendar, store, price, drug."""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import ml_warehouse
import numpy as np
import pandas as pd
import pytest
from ml_fixture import synthetic_history

from irpd import forecast, warehouse
from irpd.ml import features, panel, schema
from irpd.ml import history as hist
from irpd.ml.features import calendar, demand, season, store

END = date(2025, 6, 29)
ORIGIN = date(2024, 3, 31)


@pytest.fixture(scope="module")
def syn():
    return synthetic_history()[0]


def ctx_of(history, origins=(ORIGIN,), mode="mask"):
    return schema.PanelContext(con=None, data_end=END, origins=list(origins),
                               history=history, censor_mode=mode)


def rows_at(feat, origin=ORIGIN, h=None):
    df = feat.reset_index()
    df = df[df.origin == pd.Timestamp(origin)]
    if h is not None:
        df = df[df.h == h]
    return df.set_index("drug_code")


def test_every_group_has_its_prefix(syn):
    ctx = ctx_of(syn)
    for name in ("demand", "calendar", "store"):
        feat = features.builder(name)(ctx)
        assert all(c.startswith(features.GROUPS[name][0]) for c in feat.columns)


def test_censor_modes_on_lags(syn):
    i = next(i for i in range(len(syn["drugs"])) if syn["out"][i].any())
    j = np.flatnonzero(syn["out"][i])[0]           # first stock-out week of drug i
    origin = syn["weeks"][j] + timedelta(days=6)    # so it is lag 1
    code = syn["drugs"][i]
    got = {m: rows_at(demand.build(ctx_of(syn, [origin], m)), origin, 1).loc[code]
           for m in ("mask", "raw")}
    assert np.isnan(got["mask"].d_lag_1)
    assert got["raw"].d_lag_1 == np.float32(syn["y"][i, j])
    assert got["mask"].d_lag_1_out == got["raw"].d_lag_1_out == 1
    assert got["mask"].d_out_4 >= 1
    s = np.where(syn["out"][i, :j + 1], np.nan, syn["y"][i, :j + 1])
    assert got["mask"].d_mean_4 == pytest.approx(np.nanmean(s[-4:]), rel=1e-6)
    assert got["raw"].d_mean_4 == pytest.approx(np.mean(syn["y"][i, j - 3:j + 1]), rel=1e-6)
    with pytest.raises(ValueError, match="warehouse"):
        demand.build(ctx_of(syn, [origin], "impute"))


def test_demand_values(syn):
    feat = demand.build(ctx_of(syn))
    n = hist.weeks_to(syn, ORIGIN)
    idx = {c: i for i, c in enumerate(syn["drugs"])}
    for h in (1, 8):
        sub = rows_at(feat, h=h)
        for code, r in sub.iterrows():
            i = idx[code]
            y = syn["y"][i, :n]
            s = np.where(syn["out"][i, :n], np.nan, y)
            np.testing.assert_allclose(r.d_mean_52, np.nanmean(s[-52:]), rtol=1e-6)
            win = s[-13:]
            np.testing.assert_allclose(r.d_sale_share_13,
                                       (win > 0).sum() / np.isfinite(win).sum(), rtol=1e-6)
            sold = np.flatnonzero(np.nan_to_num(y) > 0)
            assert r.d_since_sale == n - 1 - sold[-1]
            assert r.d_since_first == n - 1 - sold[0]
            assert r.d_shelf_13 == syn["shelfout_days"][i, n - 13:n].sum()


def test_out_counts_are_nan_before_the_ledger(syn):
    early = date(2023, 10, 22)  # 3 weeks after the ledger starts
    sub = rows_at(demand.build(ctx_of(syn, [early])), early, 1)
    assert sub.d_out_13.isna().all() and sub.d_shelf_13.isna().all()
    assert sub.d_lag_4_out.isna().all() and sub.d_lag_3_out.notna().all()


def test_calendar_values(syn):
    feat = calendar.build(ctx_of(syn, [date(2024, 3, 3)]))
    sub = rows_at(feat, date(2024, 3, 3)).reset_index().groupby("h").first()
    weeks = [schema.target_week(date(2024, 3, 3), h) for h in range(1, 9)]
    lines = syn["lines"][syn["lines"].index <= date(2024, 3, 3)]
    expect = forecast.eid_index(weeks, forecast.eid_factors(lines))
    np.testing.assert_allclose(sub.c_eid_uplift, expect, rtol=1e-6)
    assert sub.c_eid_uplift.max() > 1.1                      # the Eid week is in range
    assert sub.c_eid_days.iloc[0] == (date(2024, 4, 10) - date(2024, 3, 7)).days
    assert calendar.ramadan_share(date(2024, 3, 25)) == 1.0  # Ramadan 2024 ran to 9 Apr
    assert calendar.ramadan_share(date(2024, 4, 8)) == pytest.approx(2 / 7)
    assert calendar.ramadan_share(date(2024, 6, 3)) == 0.0


def test_eid_uplift_uses_only_past_eids(syn):
    """Before any Eid was seen, there is no uplift to learn."""
    sub = rows_at(calendar.build(ctx_of(syn, [date(2023, 3, 26)])), date(2023, 3, 26))
    assert (sub.c_eid_uplift == 1).all()


def test_store_values(syn):
    sub = rows_at(store.build(ctx_of(syn)))
    lines = syn["lines"]
    last4 = lines[(lines.index > ORIGIN - timedelta(weeks=4)) & (lines.index <= ORIGIN)]
    assert sub.s_lines_4.iloc[0] == pytest.approx(last4.sum() / 4, rel=1e-6)
    assert sub.s_trend.iloc[0] == pytest.approx(
        sub.s_lines_4.iloc[0] / sub.s_lines_13.iloc[0] - 1, rel=1e-4)
    assert sub.s_lines_4.nunique() == 1


# ---- warehouse groups -------------------------------------------------------


@pytest.fixture(scope="module")
def wh(tmp_path_factory):
    return ml_warehouse.build(tmp_path_factory.mktemp("mlwh"))


def wh_ctx(path, origins, mode="mask"):
    con = duckdb.connect(str(path), read_only=True)
    return panel.context(con, ml_warehouse.END, origins, mode,
                         con_at=lambda d: warehouse.as_of(d, path))


def test_price_per_base_unit_and_changes(wh):
    origin = date(2024, 4, 28)
    sub = rows_at(features.builder("price")(wh_ctx(wh, [origin])), origin)
    assert sub.loc["W2", "p_price"] == 33000 / 10                 # strips of 10 tablets
    assert sub.loc["W2", "p_change_13"] == pytest.approx(33000 / 31500 - 1, rel=1e-6)
    assert sub.loc["W2", "p_change_52"] == pytest.approx(33000 / 30000 - 1, rel=1e-6)
    assert sub.loc["W1", "p_price"] == 750
    assert np.isnan(sub.loc["W5", "p_change_52"])                 # no price a year back


def test_drug_category(wh):
    origin = date(2024, 3, 3)
    feat = features.builder("drug")(wh_ctx(wh, [origin]))
    sub = rows_at(feat, origin)
    assert isinstance(feat.x_category.dtype, pd.CategoricalDtype)
    assert sub.loc["W2", "x_category"] == "Patent"
    assert sub.loc["W6", "x_category"] == "unknown"


def test_impute_fills_stockout_weeks_from_the_baseline(wh):
    ctx = wh_ctx(wh, [date(2024, 3, 3)], "impute")
    est = demand.imputed_units(ctx)
    h = ctx.history
    seen = ~h["out"] & ~np.isnan(h["y"])
    np.testing.assert_allclose(est[seen], h["y"][seen])            # only out days change
    i = list(h["drugs"]).index("W5")                               # out 18 Jan - 15 Feb
    j = list(h["weeks"]).index(date(2024, 1, 29))
    assert h["out"][i, j] and h["y"][i, j] == 0
    assert est[i, j] > 10                                          # ~3 customers a day
    j_late = list(h["weeks"]).index(date(2024, 2, 12))             # spell days 26-28
    assert est[i, j_late] > 0
    feat = demand.build(ctx)
    row = rows_at(feat, date(2024, 3, 3), 1).loc["W5"]
    assert row.d_lag_1_out == 0


def test_impute_series_is_used_in_lags(wh):
    origin = date(2024, 2, 4)  # lag 1 = week of 29 Jan, inside W5's stock-out
    got = {m: rows_at(demand.build(wh_ctx(wh, [origin], m)), origin, 1).loc["W5"]
           for m in ("mask", "raw", "impute")}
    assert np.isnan(got["mask"].d_lag_1)
    assert got["raw"].d_lag_1 == 0
    assert got["impute"].d_lag_1 > 10


def test_season_values(syn):
    feat = season.build(ctx_of(syn))
    n = hist.weeks_to(syn, ORIGIN)
    idx = {c: i for i, c in enumerate(syn["drugs"])}
    weeks = [schema.target_week(ORIGIN, h) for h in range(1, 9)]
    for h in (1, 8):
        for code, r in rows_at(feat, h=h).iterrows():
            i = idx[code]
            s = np.where(syn["out"][i, :n], np.nan, syn["y"][i, :n])
            t = n + h - 1 - 52
            np.testing.assert_allclose(r.y_ly, s[t], rtol=1e-6)
            np.testing.assert_allclose(r.y_ly_mean3, np.nanmean(s[t - 1:t + 2]), rtol=1e-6)
            assert r.y_week_of_year == weeks[h - 1].isocalendar().week


def test_ewm_level_matches_ses():
    s = np.array([[np.nan, 10.0, 20.0, np.nan, 10.0]])
    # starts at the first observed week, skips the gap
    want = 10 + 0.15 * (20 - 10)
    want = want + 0.15 * (10 - want)
    assert demand.ewm_level(s, 0.15)[0] == pytest.approx(want)
    assert np.isnan(demand.ewm_level(np.full((1, 3), np.nan), 0.15)[0])
