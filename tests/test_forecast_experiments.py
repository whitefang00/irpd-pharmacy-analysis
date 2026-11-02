"""Experiments E3 (Eid repair), E4 (censoring-aware targets) and E5 (TSB, segment labels).

Each switch defaults to the shipped behaviour, so the frozen system stays
reproducible; these tests cover the switched-on paths.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import duckdb
import ml_warehouse
import numpy as np
import pandas as pd
import pytest
from ml_fixture import synthetic_history

from irpd import forecast as fc
from irpd import warehouse
from irpd.ml import censor, panel, schema, segments, split
from irpd.ml import history as hist
from irpd.ml.features import calendar as cal

# ---- E3: Eid dates, cap, day counts ------------------------------------------


def test_eid_list_covers_2026_and_2027_and_keeps_the_frozen_list():
    years = {e.year for e in fc.EID}
    assert {2023, 2024, 2025, 2026, 2027} <= years
    assert fc.EID == sorted(fc.EID)
    assert fc.EID_LEGACY == [date(2023, 4, 22), date(2024, 4, 10), date(2025, 3, 31)]
    # The frozen feature still reads the frozen list.
    assert cal.eid_days(date(2025, 12, 1)) == cal.eid_days(date(2025, 12, 1), fc.EID_LEGACY)


def test_eid_days_stays_in_the_training_range_after_sep_2025():
    """Roadmap E3 check (c): with the full list, late-2025 and 2026 weeks look like seen ones."""
    def weeks(a, b):
        return [a + timedelta(weeks=k) for k in range((b - a).days // 7 + 1)]

    seen = [cal.eid_days(w, fc.EID) for w in weeks(date(2023, 2, 27), date(2025, 9, 22))]
    later = [cal.eid_days(w, fc.EID) for w in weeks(date(2025, 9, 29), date(2026, 12, 28))]
    assert min(seen) <= min(later) and max(later) <= max(seen)
    legacy = [cal.eid_days(w) for w in weeks(date(2025, 9, 29), date(2025, 12, 29))]
    assert min(legacy) < min(seen)  # the defect the full list fixes


def test_eid_days_cap():
    eids = fc.EID
    assert cal.capped_eid_days(date(2025, 3, 24), eids, 45) == 4
    assert cal.capped_eid_days(date(2025, 2, 17), eids, 45) == 39  # Thu 20 Feb
    assert np.isnan(cal.capped_eid_days(date(2025, 2, 10), eids, 45))  # Thu 13 Feb: 46 days
    far = cal.capped_eid_days(date(2025, 1, 6), eids, 45)
    assert np.isnan(far)
    assert cal.capped_eid_days(date(2025, 1, 6), eids, None) == cal.eid_days(date(2025, 1, 6), eids)
    for w in (date(2024, 1, 1) + timedelta(weeks=k) for k in range(104)):
        v = cal.capped_eid_days(w, eids, 45)
        d = cal.eid_days(w, eids)
        assert (np.isnan(v) and abs(d) > 45) or (v == d and abs(d) <= 45)


def test_day_counts_for_known_weeks():
    # 2025: Eid Mon 31 Mar. The week before is all pre-Eid shopping days.
    assert cal.day_counts(date(2025, 3, 24)) == {
        "c_eid_pre7": 7, "c_eid_closed": 0, "c_eid_post7": 0, "c_eid_post14": 0}
    assert cal.day_counts(date(2025, 3, 31)) == {
        "c_eid_pre7": 0, "c_eid_closed": 1, "c_eid_post7": 6, "c_eid_post14": 0}
    assert cal.day_counts(date(2025, 4, 7)) == {
        "c_eid_pre7": 0, "c_eid_closed": 0, "c_eid_post7": 1, "c_eid_post14": 6}
    # 2024: Eid Wed 10 Apr. The week of 8 Apr splits it: Mon-Tue before, Eid, Thu-Sun after.
    assert cal.day_counts(date(2024, 4, 8)) == {
        "c_eid_pre7": 2, "c_eid_closed": 1, "c_eid_post7": 4, "c_eid_post14": 0}
    # 2026 (expected Fri 20 Mar).
    assert cal.day_counts(date(2026, 3, 16)) == {
        "c_eid_pre7": 4, "c_eid_closed": 1, "c_eid_post7": 2, "c_eid_post14": 0}
    # Every week: each day lands in at most one band, and days -7..+14 all count.
    for w in (date(2023, 1, 2) + timedelta(weeks=k) for k in range(220)):
        c = cal.day_counts(w)
        days = [(w + timedelta(days=i) - e).days for i in range(7) for e in fc.EID]
        assert sum(c.values()) == sum(-7 <= k <= 14 for k in days) <= 7
    assert cal.day_counts(date(2025, 1, 6)) == dict.fromkeys(cal.DAY_BANDS, 0)


@pytest.fixture(scope="module")
def syn():
    history, _ = synthetic_history()
    return history


def _ctx(history, options=None):
    return schema.PanelContext(con=None, data_end=date(2025, 6, 29),
                               origins=[date(2024, 1, 7), date(2024, 3, 3), date(2025, 3, 2)],
                               history=history, options=options or {})


def test_calendar_default_is_the_frozen_feature_set(syn):
    feat = cal.build(_ctx(syn))
    assert list(feat.columns) == ["c_eid_days", "c_eid", "c_ramadan", "c_eid_uplift"]
    weeks = feat.reset_index()
    expect = [cal.eid_days(o.date() + timedelta(days=1 + 7 * (h - 1)), fc.EID_LEGACY)
              for o, h in zip(weeks.origin, weeks.h, strict=True)]
    np.testing.assert_array_equal(weeks.c_eid_days.to_numpy(), np.array(expect, np.float32))


def test_calendar_switches(syn):
    opts = {"eid_list": "full", "eid_days_cap": 45, "eid_day_counts": True}
    feat = cal.build(_ctx(syn, opts)).reset_index()
    assert {"c_eid_pre7", "c_eid_closed", "c_eid_post7", "c_eid_post14"} <= set(feat.columns)
    jan = feat[(feat.origin == pd.Timestamp(2024, 1, 7)) & (feat.h <= 7)]
    assert jan.c_eid_days.isna().all()  # Jan-Feb 2024 is 48+ days from Eid: the dip it fitted
    mar = feat[(feat.origin == pd.Timestamp(2025, 3, 2)) & (feat.h == 4)].iloc[0]
    assert (mar.c_eid_days, mar.c_eid_pre7, mar.c_eid_closed) == (4, 7, 0)
    with pytest.raises(ValueError):
        cal.build(_ctx(syn, {"eid_list": "nope"}))


def test_units_profile_needs_store_units_and_differs_from_lines():
    lines = pd.Series(1.0, index=[date(2024, 1, 1) + timedelta(days=i) for i in range(200)])
    units = lines.copy()
    eid = date(2024, 4, 10)
    lines[eid + timedelta(days=3)] = 3.0  # lines spike after Eid
    units[eid - timedelta(days=2)] = 3.0  # units spike before it
    h = {"lines": lines, "store_units": units}
    f_lines, f_units = fc.eid_profile(h, "lines"), fc.eid_profile(h, "units")
    assert f_lines[3] == pytest.approx(1 + fc.EID_SHRINK * 2) and f_lines[-2] == 1
    assert f_units[-2] == pytest.approx(3.0) and f_units[3] == 1
    with pytest.raises(ValueError):
        fc.eid_profile(h, "both")


# ---- E5: TSB and labels ---------------------------------------------------------


def test_tsb_on_a_toy_series():
    nan = np.nan
    y = np.array([[nan, 4, 0, 6, 0, 0, 0],      # sells, then fades
                  [2, 2, 2, 2, 2, 2, 2],         # steady
                  [nan] * 7,                     # never sold
                  [5, 0, 0, nan, 0, 0, 0]])      # a censored week is skipped
    out = np.zeros(y.shape, bool)
    out[3, 3] = True
    h = {"y": np.where(out, 9, y), "out": out}
    r = fc.tsb(h, horizon=3, alpha_p=0.15, alpha_d=0.1)
    # Drug 0: p = 1, then 0.85, then 0.15 + 0.85 * 0.85, then x 0.85 three times.
    p = 1.0
    p = 0.85 * p
    p = 0.15 + 0.85 * p
    p = p * 0.85 ** 3
    size = 0.1 * 6 + 0.9 * 4
    assert r["rate"][0] == pytest.approx(p * size)
    assert r["rate"][1] == pytest.approx(2.0)
    assert r["rate"][2] == 0
    assert r["rate"][3] == pytest.approx(5 * 0.85 ** 5)  # 5 observed zeros, not 6
    assert r["mean"].shape == (4, 3) and (r["mean"] == r["rate"][:, None]).all()
    assert (r["var"] >= r["mean"]).all()
    np.testing.assert_allclose(r["var"], r["mean"] * r["disp"][:, None])


def test_segment_labels_at_origin():
    n = 60
    y = np.zeros((5, n))
    y[0] = 3                                  # steady
    y[1, :40] = 2                             # fading: nothing in the last 20 weeks
    y[2, ::10] = 1                            # sporadic: 1 week in 10 sells
    y[3, :50] = np.nan; y[3, 50:] = 1         # new
    y[4] = 3                                  # censored: often out lately
    out = np.zeros(y.shape, bool)
    out[4, -26::3] = True                     # 9 of the last 26 weeks
    h = {"drugs": np.array(list("abcde"), dtype=object), "y": y, "out": out,
         "stockout": out.copy(), "stock_known": np.ones(n, bool)}
    lab = segments.labels(h, np.array(["fast", "regular", "slow", "slow", "fast"],
                                       dtype=object)).set_index("drug_code")
    assert lab.fading.tolist() == [False, True, False, False, False]
    assert lab.sporadic.tolist() == [False, False, True, False, False]
    assert lab.new.tolist() == [False, False, False, True, False]
    assert lab.censored.tolist() == [False, False, False, False, True]
    assert lab.loc["b", "weeks_since_sale"] == 20
    assert lab.loc["e", "stockout_share_26"] == pytest.approx(9 / 26)
    # A rare drug is sporadic whatever its weeks say.
    lab2 = segments.labels(h, np.array(["rare"] * 5, dtype=object))
    assert lab2.sporadic.all()


# ---- E4: censoring-aware targets are leak-free -----------------------------------

ORIGIN = date(2024, 3, 31)  # after W1's Feb-Mar outage and W5's Jan-Feb one


@pytest.fixture(scope="module")
def wh(tmp_path_factory):
    return ml_warehouse.build(tmp_path_factory.mktemp("mlwh_censor"))


@pytest.fixture(scope="module")
def wh_perturbed(tmp_path_factory):
    return ml_warehouse.build(tmp_path_factory.mktemp("mlwh_censor_p"), perturb_after=ORIGIN)


@lru_cache(maxsize=4)  # each connection holds threads; one per origin ran other tests out
def _as_of(path: Path, day: date) -> duckdb.DuckDBPyConnection:
    con = warehouse.as_of(day, path)
    con.execute("SET threads = 1")
    return con


def _training_rows(ctx, target="impute", shelf="flag"):
    book = panel.bookkeeping(ctx)
    rows = schema.trainable(censor.censor_aware(book, ctx, target, shelf), ORIGIN)
    keep = ["drug_code", "origin", "h", "target", "censored", censor.SHELF_FLAG]
    return rows[[c for c in keep if c in rows]].sort_values(schema.KEYS).reset_index(drop=True)


def _full(path):
    con = duckdb.connect(str(path), read_only=True)
    origins = split.training_origins(ORIGIN)
    return panel.context(con, ml_warehouse.END, origins, con_at=lambda d: _as_of(path, d))


def _cut(path):
    origins = split.training_origins(ORIGIN)
    return panel.context(_as_of(path, ORIGIN), ORIGIN + timedelta(days=7), origins,
                         con_at=lambda d: _as_of(path, min(d, ORIGIN)))


def test_censor_aware_targets_put_stockout_weeks_back(wh):
    ctx = _full(wh)
    book = panel.bookkeeping(ctx)
    aware = censor.censor_aware(book, ctx)
    was = book.censored & book.available
    assert was.any()
    back = aware.loc[was, "target"].notna()
    assert back.mean() > 0.5
    # Uncensored rows are untouched; imputed weeks are at least what was recorded.
    same = ~was
    pd.testing.assert_series_equal(aware.loc[same, "target"], book.loc[same, "target"])
    h = ctx.history
    j = book.drug_code.map({c: i for i, c in enumerate(h["drugs"])})
    t = book.target_week.map({pd.Timestamp(w): i for i, w in enumerate(h["weeks"])})
    sel = was & aware.target.notna()
    rec = h["y"][j[sel].to_numpy(int), t[sel].to_numpy(int)]
    assert (aware.loc[sel, "target"].to_numpy() >= np.nan_to_num(rec) - 1e-3).mean() > 0.9
    # With both switches off, nothing changes.
    off = censor.censor_aware(book, ctx, "mask", "mask")
    pd.testing.assert_series_equal(off.target, book.target)
    pred = censor.for_prediction(aware)
    assert (pred[censor.SHELF_FLAG] == 0).all()


def test_censor_aware_targets_truncation(wh):
    """A training row's target is the same from data cut at the origin as from all data."""
    full, cut = _training_rows(_full(wh)), _training_rows(_cut(wh))
    assert len(full) and full.censored.any() and full.target[full.censored].notna().any()
    pd.testing.assert_frame_equal(full, cut, check_exact=False, rtol=1e-6)


def test_censor_aware_targets_ignore_later_changes(wh, wh_perturbed):
    pd.testing.assert_frame_equal(_training_rows(_full(wh)), _training_rows(_full(wh_perturbed)),
                                  check_exact=False, rtol=1e-6)


def test_history_cut_keeps_new_series_aligned(wh):
    ctx = _full(wh)
    c = hist.cut(ctx.history, ORIGIN)
    n = c["y"].shape[1]
    assert c["stockout"].shape[1] == n
    assert (c["stockout"] <= c["out"]).all()
    assert c["store_units"].index.max() <= ORIGIN
