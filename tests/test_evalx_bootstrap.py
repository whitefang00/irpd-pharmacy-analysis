import numpy as np
import pandas as pd
import pytest

from irpd.evalx import bootstrap


def _frame(n_o=10, n_d=50, seed=0):
    rng = np.random.default_rng(seed)
    o, d = np.meshgrid(np.arange(n_o), np.arange(n_d), indexing="ij")
    return pd.DataFrame({"origin": o.ravel(), "drug_code": [f"D{x}" for x in d.ravel()],
                         "a": rng.gamma(2, 1, o.size), "y": np.ones(o.size)})


def test_origin_blocks_move_together():
    w = bootstrap.origin_weights(5, 200, np.random.default_rng(0), block=2)
    assert (w[:, 0] == w[:, 1]).all() and (w[:, 2] == w[:, 3]).all()
    assert (w.sum(1) == 6).all() or (w.sum(1) <= 6).all()  # 3 blocks of 2, 2, 1


def test_constant_difference_has_a_degenerate_interval():
    f = _frame()
    f["b"] = f["a"] - 0.5
    iv = bootstrap.paired_diff(f, "a", "b", den="y")
    assert iv.point == pytest.approx(0.5)
    assert iv.lo == pytest.approx(0.5) and iv.hi == pytest.approx(0.5)


def test_identical_arms_give_zero():
    f = _frame()
    iv = bootstrap.paired_diff(f, "a", "a", den="y")
    assert iv.point == 0 and iv.lo == 0 and iv.hi == 0 and not iv.excludes_zero()


def test_two_way_is_wider_than_drug_only_when_origins_matter():
    f = _frame(n_o=12, n_d=200)
    shock = np.random.default_rng(1).normal(0, 1, 12)
    f["b"] = f["a"] + shock[f["origin"]]  # a difference that moves by origin
    two = bootstrap.paired_diff(f, "b", "a", den="y")
    one = bootstrap.paired_diff(f, "b", "a", den="y", drugs_only=True)
    assert (two.hi - two.lo) > 3 * (one.hi - one.lo)


def test_mean_of_draws_is_near_the_point():
    f = _frame()
    iv = bootstrap.two_way(f, ["a", "y"], lambda s: s["a"] / s["y"])
    assert iv.draws.mean() == pytest.approx(iv.point, rel=0.02)
    assert iv.lo < iv.point < iv.hi
