from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from ml_fixture import synthetic_drugs, synthetic_history, synthetic_panel

from irpd.ml import history as hist
from irpd.ml import schema, split
from irpd.ml.features import groups

DATA_END = date(2025, 6, 29)
ORIGINS = [date(2024, 3, 3), date(2024, 8, 18)]
DYN_LOO = [c for c in groups.DYN_COLUMNS if not c.startswith("g_dyn_share_")]


@pytest.fixture(scope="module")
def data():
    history, _ = synthetic_history()
    return history, synthetic_drugs(history)


def _ctx(history, origins=ORIGINS, **kw):
    return schema.PanelContext(con=None, data_end=DATA_END, origins=list(origins),
                               history=history, **kw)


def _copy(history):
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in history.items()}


def test_helpers():
    assert groups.ingredient_key("Paracetamol + caffeine") == "caffeine + paracetamol"
    assert groups.ingredient_key(" caffeine +paracetamol ") == "caffeine + paracetamol"
    assert groups.ingredient_key("") == "" and groups.ingredient_key(None) == ""
    assert [groups.route(f) for f in ("tablet", "Syrup", "cream", "inhaler", "injection",
                                      "other", "", None)] == [
        "oral", "oral", "topical", "inhaled", "injected", "other", "other", "other"]
    assert groups.strength_mg("500 mg") == 500
    assert groups.strength_mg("0,5mg") == 0.5
    assert groups.strength_mg("120 mg/5 ml") == 120
    assert groups.strength_mg("1 g") == 1000
    assert np.isnan(groups.strength_mg("12pcs")) and np.isnan(groups.strength_mg(""))


def test_group_definitions_split_as_expected(data):
    history, drugs = data
    drugs = drugs.copy()
    codes = drugs.drug_code.to_numpy()
    drugs.loc[2, "form"] = "cream"  # group 0: tablet, syrup, cream
    ids = {d: groups.group_ids(drugs, d) for d in schema.GROUP_DEFS}
    first3 = codes[:3]
    assert ids["ingredient"][first3].nunique() == 1
    assert ids["ingredient"][first3].iloc[0] == "caffeine + paracetamol"
    route = ids["ingredient_route"][first3]
    assert route.iloc[0] == route.iloc[1] != route.iloc[2]   # tablet & syrup are oral
    assert ids["ingredient_form"][first3].nunique() == 3
    for d in schema.GROUP_DEFS:
        unmapped = drugs.ingredients.eq("").to_numpy()
        assert ids[d][codes[unmapped]].isna().all()
        assert ids[d][codes[~unmapped]].notna().all()
    # Every sibling group in the fixture: 3 under ingredient, 2 tablets + 1 syrup by form.
    sizes = ids["ingredient"].value_counts()
    assert (sizes[sizes > 1] == 3).all() and (sizes > 1).sum() == 4
    by_form = groups.group_ids(data[1], "ingredient_form").value_counts()
    assert sorted(by_form[by_form > 1]) == [2, 2, 2, 2]

    with pytest.raises(ValueError):
        groups.group_ids(drugs, "nonsense")


def test_build_from_shapes_and_dtypes(data):
    history, drugs = data
    f = groups.build_from(_ctx(history), drugs)
    assert list(f.columns) == groups.COLUMNS
    assert f.index.names == ["drug_code", "origin"]
    # One row per drug active at each origin (a sale in the 52 weeks before it).
    active = sum(int(hist.active(history, o, schema.ACTIVE_WEEKS).sum()) for o in ORIGINS)
    assert len(f) == active < len(history["drugs"]) * len(ORIGINS)
    for c in f.columns:
        assert str(f[c].dtype) in ("float32", "category", "bool", "int8"), c
    # Joins onto a panel that still validates.
    panel = synthetic_panel(history, ORIGINS)
    joined = panel.join(f, on=["drug_code", "origin"])
    assert len(joined) == len(panel)
    assert joined["g_static_size"].notna().any()
    schema.validate_panel(joined)


def test_singletons_and_unmapped_have_nan_dynamic(data):
    history, drugs = data
    f = groups.build_from(_ctx(history), drugs).reset_index()
    alone = f.drug_code.isin(drugs.drug_code[drugs.ingredients.str.startswith("singleton")])
    unmapped = f.drug_code.isin(drugs.drug_code[drugs.ingredients.eq("")])
    for mask in (alone, unmapped):
        assert f.loc[mask, groups.DYN_COLUMNS].isna().all().all()
        assert f.loc[mask, "g_static_no_siblings"].all()
    assert (f.loc[alone, "g_static_size"] == 1).all()
    assert f.loc[unmapped, "g_static_size"].isna().all()
    assert f.loc[unmapped, "g_static_group"].isna().all()
    sib = ~f.g_static_no_siblings
    assert sib.any()
    # Single weeks can be masked; the 13-week means always have something.
    assert f.loc[sib, ["g_dyn_idx_mean_13", "g_dyn_rev_mean_13"]].notna().all().all()
    assert (f.loc[sib, "g_static_size"] >= 2).all()


def test_leave_one_out_is_exact(data):
    history, drugs = data
    base = groups.build_from(_ctx(history), drugs)
    # A drug in the paracetamol group, which has siblings at both origins.
    code = drugs.drug_code.iloc[0]
    sibs = list(drugs.drug_code.iloc[1:3])
    i = list(history["drugs"]).index(code)
    changed = _copy(history)
    changed["y"][i] = changed["y"][i] * 3 + 7
    changed["revenue"][i] = changed["revenue"][i] * 5 + 11
    changed["out"][i] = ~changed["out"][i] & changed["stock_known"]
    after = groups.build_from(_ctx(changed), drugs)
    pd.testing.assert_frame_equal(base.loc[code, DYN_LOO], after.loc[code, DYN_LOO])
    assert not np.allclose(base.loc[code, "g_dyn_share_13"], after.loc[code, "g_dyn_share_13"])
    for s in sibs:
        assert not base.loc[s, DYN_LOO].equals(after.loc[s, DYN_LOO])
    # Drugs outside the group don't move at all.
    other = [c for c in history["drugs"] if c not in [code, *sibs]]
    pd.testing.assert_frame_equal(base.loc[other], after.loc[other])


def test_history_after_origin_is_ignored(data):
    history, drugs = data
    for group_def in schema.GROUP_DEFS:
        for mode in ("mask", "raw"):
            base = groups.build_from(_ctx(history, censor_mode=mode, group_def=group_def), drugs)
            for origin in ORIGINS:
                n = sum(w + timedelta(days=6) <= origin for w in history["weeks"])
                changed = _copy(history)
                rng = np.random.default_rng(1)
                changed["y"][:, n:] = rng.integers(0, 50, changed["y"][:, n:].shape)
                changed["revenue"][:, n:] = changed["y"][:, n:] * 1234.0
                changed["out"][:, n:] = rng.random(changed["out"][:, n:].shape) < 0.5
                after = groups.build_from(
                    _ctx(changed, origins=[origin], censor_mode=mode, group_def=group_def),
                    drugs)
                pd.testing.assert_frame_equal(base.xs(pd.Timestamp(origin), level="origin"),
                                              after.xs(pd.Timestamp(origin), level="origin"))


def test_censor_mode_masks_sibling_weeks(data):
    history, drugs = data
    origins = split.selection_origins()
    mask = groups.build_from(_ctx(history, origins=origins, censor_mode="mask"), drugs)
    raw = groups.build_from(_ctx(history, origins=origins, censor_mode="raw"), drugs)
    assert not mask[DYN_LOO].equals(raw[DYN_LOO])
    pd.testing.assert_frame_equal(mask[groups.STATIC_COLUMNS], raw[groups.STATIC_COLUMNS])


def test_in_stock_counts_siblings(data):
    history, drugs = data
    origin = date(2024, 8, 18)
    f = groups.build_from(_ctx(history, origins=[origin]), drugs)
    code, sibs = drugs.drug_code.iloc[0], list(drugs.drug_code.iloc[1:3])
    n = sum(w + timedelta(days=6) <= origin for w in history["weeks"])
    idx = [list(history["drugs"]).index(s) for s in sibs]
    changed = _copy(history)
    changed["out"][idx, n - 1] = False
    assert groups.build_from(_ctx(changed, origins=[origin]), drugs).loc[
        code, "g_dyn_in_stock"].iloc[0] == 2
    changed["out"][idx[0], n - 1] = True
    assert groups.build_from(_ctx(changed, origins=[origin]), drugs).loc[
        code, "g_dyn_in_stock"].iloc[0] == 1
    # Before the stock ledger starts, stock-outs can't be seen.
    early = groups.build_from(_ctx(history, origins=[date(2023, 6, 4)]), drugs)
    assert early["g_dyn_in_stock"].isna().all()
    assert f["g_dyn_in_stock"].notna().any()


def test_demand_index_matches_hand_calculation(data):
    history, drugs = data
    origin = date(2024, 8, 18)
    f = groups.build_from(_ctx(history, origins=[origin], censor_mode="raw"), drugs)
    from irpd.ml import history as hist

    code, sibs = drugs.drug_code.iloc[0], list(drugs.drug_code.iloc[1:3])
    idx = [list(history["drugs"]).index(s) for s in sibs]
    c = hist.cut(history, origin)
    scale = hist.target_scale(history, origin)[idx].astype(float)
    weekly = np.nanmean(c["y"][idx][:, -13:] / scale[:, None], axis=0)
    row = f.loc[code].iloc[0]
    assert row["g_dyn_idx_lag_1"] == pytest.approx(weekly[-1], rel=1e-6)
    assert row["g_dyn_idx_mean_13"] == pytest.approx(np.nanmean(weekly), rel=1e-6)
    rev = np.nanmean(c["revenue"][idx][:, -2:], axis=0)
    assert row["g_dyn_rev_lag_2"] == pytest.approx(rev[0], rel=1e-6)
    own = np.nansum(c["revenue"][list(history["drugs"]).index(code), -13:])
    total = own + np.nansum(c["revenue"][idx][:, -13:])
    assert row["g_dyn_share_13"] == pytest.approx(own / total, rel=1e-6)
