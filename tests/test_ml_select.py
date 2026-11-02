import numpy as np
import pandas as pd
import pytest

from irpd.ml import select


def _table(rows):
    return pd.DataFrame(rows, columns=["tier", "method", "wape", "bias", "complexity"])


def _pick(table, tier):
    c = select.choose(table)
    return c.set_index("tier").loc[tier]


def test_lowest_wape_in_band():
    t = _table([("fast", "a", 0.30, 1.05, 0),     # out of band
                ("fast", "b", 0.33, 1.02, 3),
                ("fast", "c", 0.34, 0.98, 3)])
    p = _pick(t, "fast")
    assert p["candidate"] == "b" and p["rule"] == "band" and p["in_band"] == 2


def test_simpler_wins_within_half_a_point():
    t = _table([("regular", "ml", 0.330, 1.00, 5),
                ("regular", "base", 0.334, 1.01, 1),   # 0.4 points worse, simpler
                ("regular", "ml_small", 0.331, 1.00, 3)])
    p = _pick(t, "regular")
    assert p["candidate"] == "base" and p["best"] == "ml"


def test_simpler_loses_past_half_a_point():
    t = _table([("regular", "ml", 0.330, 1.00, 5),
                ("regular", "base", 0.336, 1.01, 1)])
    assert _pick(t, "regular")["candidate"] == "ml"


def test_tie_break_takes_lowest_wape_among_simplest():
    t = _table([("fast", "x", 0.300, 1.0, 4), ("fast", "y", 0.303, 1.0, 1),
                ("fast", "z", 0.302, 1.0, 1)])
    assert _pick(t, "fast")["candidate"] == "z"


def test_fallback_closest_bias_within_one_point():
    t = _table([("fast", "a", 0.300, 1.10, 1),
                ("fast", "b", 0.309, 1.05, 1),    # within 1 point, bias closer
                ("fast", "c", 0.320, 1.04, 1)])   # bias closer still, but 2 points worse
    p = _pick(t, "fast")
    assert p["candidate"] == "b" and p["rule"] == "fallback" and p["in_band"] == 0


def test_slow_and_rare_use_the_wide_band():
    rows = [("slow", "a", 0.50, 1.05, 1), ("slow", "b", 0.55, 1.00, 1),
            ("fast", "a", 0.50, 1.05, 1), ("fast", "b", 0.55, 1.00, 1)]
    c = select.choose(_table(rows)).set_index("tier")
    assert c.loc["slow", "candidate"] == "a" and c.loc["slow", "rule"] == "band"
    assert c.loc["fast", "candidate"] == "b"


def test_band_edges_are_inclusive():
    t = _table([("fast", "a", 0.30, 1.03, 2), ("fast", "b", 0.31, 1.0, 2)])
    assert _pick(t, "fast")["candidate"] == "a"


def test_choose_validates():
    with pytest.raises(ValueError, match="missing"):
        select.choose(pd.DataFrame({"tier": ["fast"], "method": ["a"], "wape": [0.1],
                                    "bias": [1.0]}))
    with pytest.raises(ValueError, match="one row"):
        select.choose(_table([("fast", "a", 0.3, 1, 1), ("fast", "a", 0.3, 1, 1)]))


def test_objective_known_value():
    def rows(tier, actual, mean):
        return [{"tier": tier, "actual": a, "mean": m, "config_hash": "c", "seed": -1}
                for a, m in zip(actual, mean, strict=True)]

    pred = pd.DataFrame(rows("fast", [10, 10], [12, 12])          # bias 1.2: penalty 0.17
                        + rows("regular", [10, 10], [10, 10.4])   # bias 1.02: no penalty
                        + rows("slow", [5, 5], [0, 5])
                        + rows("rare", [1, np.nan], [50, 50]))    # ignored
    wape = (2 + 2 + 0 + 0.4 + 5 + 0) / 50
    assert select.objective(pred) == pytest.approx(wape + 0.17)


def test_objective_needs_one_config():
    pred = pd.DataFrame({"tier": ["fast"] * 2, "actual": [1.0, 1.0], "mean": [1.0, 1.0],
                         "config_hash": ["a", "b"], "seed": [-1, -1]})
    with pytest.raises(ValueError, match="one config_hash"):
        select.objective(pred)
