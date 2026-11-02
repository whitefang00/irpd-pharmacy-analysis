import ast
import json
import subprocess
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pytest

from irpd import config, warehouse
from irpd.ml import split
from irpd.ml.config import FrozenSystem, MLConfig

FIXTURES = Path(__file__).parent / "fixtures"
ROUTING = {"fast": "ml", "regular": "ml", "slow": "ses+eid", "rare": "snaive"}


def test_origin_sets():
    allo = split.backtest_origins()
    sel, purged, ev = split.selection_origins(), split.purged_origins(), split.eval_origins()
    assert len(allo) == 18
    assert (len(sel), len(purged), len(ev)) == (11, 2, 5)
    assert sel[0] == date(2024, 1, 7) and sel[-1] == date(2024, 10, 13)
    assert purged == [date(2024, 11, 10), date(2024, 12, 8)]
    assert ev[0] == date(2025, 1, 5) and ev[-1] == date(2025, 4, 27)
    assert sel + purged + ev == allo
    assert all(o.weekday() == 6 for o in allo)


def test_selection_targets_stay_in_2024():
    assert max(split.target_end(o) for o in split.selection_origins()) == split.SELECTION_END
    assert all(split.target_end(o) > split.SELECTION_END for o in split.purged_origins())


def test_eval_every_two_weeks():
    ev = split.eval_origins(step_weeks=2)
    assert len(ev) == 9
    assert all(split.target_end(o) <= config.CUTOFF for o in ev)
    assert split.target_end(ev[-1] + timedelta(weeks=2)) > config.CUTOFF
    with pytest.raises(ValueError):
        split.eval_origins(step_weeks=0)


def test_screen_origins():
    assert split.screen_origins() == [date(2024, 3, 3), date(2024, 5, 26),
                                      date(2024, 8, 18), date(2024, 10, 13)]


def test_training_origins_are_weekly_sundays_with_a_target_week():
    tr = split.training_origins(split.SELECTION_END)
    assert tr[0] == split.FIRST_TRAINING_ORIGIN and tr[-1] == date(2024, 12, 1)
    assert all((b - a).days == 7 for a, b in zip(tr, tr[1:], strict=False))
    assert all(o.weekday() == 6 for o in tr)


def test_check_scored_refuses_late_targets():
    split.check_scored(split.selection_origins(), split.SELECTION_END)
    with pytest.raises(ValueError, match="after 2024-12-08"):
        split.check_scored(split.purged_origins(), split.SELECTION_END)


@pytest.fixture(scope="module")
def sealed(tmp_path_factory):
    """A warehouse whose rows are all in 2025, and its selection file."""
    d = tmp_path_factory.mktemp("sel")
    src = warehouse.build(FIXTURES / "leaks", d / "leaks.duckdb")
    return src, split.seal_selection(source=src, out=d / "selection.duckdb")


def test_selection_file_holds_nothing_after_selection_end(sealed):
    src, sel = sealed
    with duckdb.connect(str(src), read_only=True) as con:
        assert con.execute("SELECT count(*) FROM sales_lines").fetchone()[0] > 0
    con = split.selection_con(path=sel)
    # Even behind the views, the attached file has nothing later.
    assert con.execute("SELECT count(*) FROM wh.sales_lines").fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM sales_lines").fetchone()[0] == 0


def test_selection_con_is_locked(sealed):
    src, sel = sealed
    con = split.selection_con(path=sel)
    with pytest.raises(duckdb.Error):
        con.execute(f"ATTACH '{src}' AS other (READ_ONLY)")
    with pytest.raises(duckdb.Error):
        con.execute(f"SELECT * FROM read_csv('{FIXTURES / 'leaks/raw_dataset/stock_raw.csv'}')")
    with pytest.raises(duckdb.Error):
        con.execute("SET enable_external_access = true")


def test_selection_con_refuses_later_days(sealed):
    with pytest.raises(ValueError):
        split.selection_con(split.SELECTION_END + timedelta(days=1), path=sealed[1])


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    return tmp_path


def test_eval_guard_needs_a_committed_frozen_system(repo):
    frozen = repo / "frozen.json"
    with pytest.raises(PermissionError, match="no frozen config"):
        split.check_frozen(frozen)
    system = FrozenSystem(MLConfig(model="lgbm", loss="tweedie", tweedie_power=1.5), ROUTING)
    frozen.write_text(system.to_json())
    with pytest.raises(PermissionError, match="not committed"):
        split.check_frozen(frozen)
    _git(repo, "add", "frozen.json")
    _git(repo, "commit", "-qm", "freeze")
    f = split.check_frozen(frozen)
    assert (f.config_hash, f.system_hash) == (system.config.hash(), system.hash())
    assert f.commit and f.blob
    # Any edit after the commit is refused, staged or not.
    frozen.write_text(frozen.read_text() + " ")
    with pytest.raises(PermissionError, match="uncommitted"):
        split.check_frozen(frozen)
    _git(repo, "add", "frozen.json")
    with pytest.raises(PermissionError, match="uncommitted"):
        split.check_frozen(frozen)


def test_tampered_frozen_file_is_refused(repo):
    frozen = repo / "frozen.json"
    body = json.loads(FrozenSystem(MLConfig(model="lgbm"), ROUTING).to_json())
    body["config"]["loss"] = "poisson"
    frozen.write_text(json.dumps(body))
    _git(repo, "add", "frozen.json")
    _git(repo, "commit", "-qm", "tamper")
    with pytest.raises(PermissionError, match="config_hash"):
        split.check_frozen(frozen)


def test_log_run_appends(tmp_path):
    log = tmp_path / "runs.jsonl"
    split.log_run("eval", None, {"wape": 0.3}, log)
    split.log_run("eval", None, {"wape": 0.31}, log)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["rows"]["wape"] for r in rows] == [0.3, 0.31]


def test_ml_code_never_opens_the_sealed_holdout():
    root = Path(split.__file__).parent
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Attribute):
                names.append(node.attr)
            elif isinstance(node, ast.Name):
                names.append(node.id)
            elif isinstance(node, ast.ImportFrom):
                names += [a.name for a in node.names]
            bad = {"open_sealed", "SEALED", "load_from", "recorded_demand"} & set(names)
            assert not bad, f"{path.name} uses {bad}"
