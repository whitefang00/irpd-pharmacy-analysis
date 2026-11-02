"""Forecast origins, the selection / evaluation split, and the guards around it.

An origin is a Sunday. A model standing on an origin knows the weeks that
ended by it and forecasts the 8 Monday-start weeks after it.

The backtest grid is the existing one (`forecast.backtest_origins`): every
4 weeks from 7 Jan 2024. It is split by where each origin's targets end:

- selection: targets end by SELECTION_END (8 Dec 2024). Every choice
  (features, loss, hyperparameters, EDA decisions) is made on these only;
- purged: 2024 origins whose targets run into 2025. Scored, used for nothing;
- evaluation: 2025 origins, opened once, after the configuration is frozen.

**Guards.**

- `seal_selection()` writes `selection.duckdb` beside the warehouse: every
  table as it stood on SELECTION_END, written out. `selection_con()` opens
  that file with external access off and locked, so no query (not even one
  on `wh.*` behind the views, and no ATTACH or file read) can reach later
  data. This is the same design as `holdout.agent_view`.
- `eval_con()` opens the agent file (data to the cutoff) only once a frozen
  config is committed, and returns where the freeze is recorded.
- Nothing in `irpd.ml` may open the sealed holdout (a test checks this).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb

from irpd import config, holdout, warehouse
from irpd import forecast as fc

HORIZON = 8
SELECTION_END = date(2024, 12, 8)
EVAL_START = date(2025, 1, 1)
FIRST_ORIGIN = date(2024, 1, 7)
# Training rows start at the first Sunday with 8 weeks of sales behind it (sales
# start Mon 2 Jan 2023). Earlier rows would have almost no lag features.
FIRST_TRAINING_ORIGIN = date(2023, 2, 26)
SCREEN_PICK = (2, 5, 8, 10)  # indices into selection_origins(): 3 Mar, 26 May, 18 Aug, 13 Oct
SELECTION_DB = config.WAREHOUSE.with_name("selection.duckdb")
RUN_LOG_NAME = "runs.jsonl"


def target_end(origin: date, horizon: int = HORIZON) -> date:
    """The last day of the last target week (a Sunday)."""
    return origin + timedelta(weeks=horizon)


def backtest_origins() -> list[date]:
    """The 18 origins of the existing backtest, every 4 weeks from 7 Jan 2024."""
    return fc.backtest_origins(FIRST_ORIGIN, HORIZON)


def selection_origins() -> list[date]:
    """Origins whose targets all end by SELECTION_END (11: 7 Jan to 13 Oct 2024)."""
    return [o for o in backtest_origins() if target_end(o) <= SELECTION_END]


def purged_origins() -> list[date]:
    """2024 origins whose targets cross into 2025 (10 Nov and 8 Dec 2024)."""
    return [o for o in backtest_origins()
            if target_end(o) > SELECTION_END and o < EVAL_START]


def eval_origins(step_weeks: int = 4) -> list[date]:
    """2025 origins whose targets end by the cutoff: 5 at the default step, 9 at 2."""
    if step_weeks < 1:
        raise ValueError("step_weeks must be at least 1")
    first = next(o for o in backtest_origins() if o >= EVAL_START)
    out, o = [], first
    while target_end(o) <= config.CUTOFF:
        out.append(o)
        o += timedelta(weeks=step_weeks)
    return out


def screen_origins() -> list[date]:
    """4 selection origins spread over 2024, for cheap screening runs."""
    sel = selection_origins()
    return [sel[i] for i in SCREEN_PICK]


def training_origins(until: date) -> list[date]:
    """Every Sunday from FIRST_TRAINING_ORIGIN to `until`, at least one week before it.

    Panel rows are built at these origins. Rows whose target week ends after
    the data the caller holds get a NaN target (`schema.PanelContext` marks
    them `available=False`); `schema.trainable` and `schema.train_valid`
    pick the rows a model for a given origin may use.
    """
    out, o = [], FIRST_TRAINING_ORIGIN
    while o + timedelta(weeks=1) <= until:
        out.append(o)
        o += timedelta(weeks=1)
    return out


def check_scored(origins: list[date], data_end: date, horizon: int = HORIZON) -> None:
    """Refuse scored origins whose targets end after the data the caller holds."""
    late = [o for o in origins if target_end(o, horizon) > data_end]
    if late:
        raise ValueError(f"{len(late)} origins have targets after {data_end} "
                         f"(first: {late[0]}); use the connection for that period")


# ---- connections --------------------------------------------------------


def _lock(con: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    con.execute("SET enable_external_access = false")
    con.execute("SET lock_configuration = true")
    return con


def seal_selection(source: Path | None = None, out: Path | None = None) -> Path:
    """Write every table as of SELECTION_END to its own file (run once, by the lead)."""
    source, out = source or holdout.AGENT, out or SELECTION_DB
    out.unlink(missing_ok=True)
    with warehouse.as_of(SELECTION_END, source) as con:
        con.execute(f"ATTACH '{out}' AS sel")
        views = [r[0] for r in con.execute(
            "SELECT view_name FROM duckdb_views() WHERE NOT internal ORDER BY 1").fetchall()]
        for name in views:
            con.execute(f"CREATE TABLE sel.{name} AS SELECT * FROM {name}")
        con.execute("DETACH sel")
    return out


def selection_con(day: date = SELECTION_END, path: Path | None = None
                  ) -> duckdb.DuckDBPyConnection:
    """The selection file as of `day` (<= SELECTION_END), locked against other files.

    Every choice-making step uses this. Pass an earlier `day` to rebuild
    tables that are reconstructed in hindsight (prices, spells) as they stood
    on an origin.
    """
    if day > SELECTION_END:
        raise ValueError(f"selection data ends on {SELECTION_END}")
    path = path or SELECTION_DB
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `irpd ml seal` once")
    return _lock(warehouse.as_of(day, path))


@dataclass(frozen=True)
class Freeze:
    """Where the frozen config is recorded: its hash, commit and git blob."""

    config_hash: str
    system_hash: str
    commit: str
    blob: str


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(config.ROOT), *args],
                          capture_output=True, text=True)


def check_frozen(frozen: Path) -> Freeze:
    """Refuse unless `frozen` is committed at HEAD byte for byte and its hashes check out."""
    if not frozen.exists():
        raise PermissionError(f"no frozen config at {frozen}; freeze before evaluating")
    rel = frozen.resolve().relative_to(config.ROOT.resolve()).as_posix()
    blob = _git("rev-parse", f"HEAD:{rel}")
    if blob.returncode != 0:
        raise PermissionError(f"{rel} is not committed at HEAD")
    shown = subprocess.run(["git", "-C", str(config.ROOT), "show", f"HEAD:{rel}"],
                           capture_output=True)
    if shown.stdout != frozen.read_bytes():
        raise PermissionError(f"{rel} has uncommitted changes")
    from irpd.ml.config import FrozenSystem  # local: config imports schema, which imports this

    system = FrozenSystem.from_json(frozen.read_text())
    commit = _git("rev-parse", "--short", "HEAD").stdout.strip()
    return Freeze(system.config.hash(), system.hash(), commit, blob.stdout.strip()[:12])


def eval_con(frozen: Path) -> tuple[duckdb.DuckDBPyConnection, Freeze]:
    """The agent's data up to the cutoff, only once a frozen config is committed."""
    freeze = check_frozen(frozen)
    return holdout.agent_view(config.CUTOFF), freeze


def log_run(kind: str, freeze: Freeze | None, rows: dict, log: Path | None = None) -> None:
    """Append one run (what, which frozen config, its results) to the ML run log."""
    log = log or config.ML_DIR / RUN_LOG_NAME
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(json.dumps({
            "ts": datetime.now(UTC).isoformat(timespec="seconds"), "kind": kind,
            "freeze": freeze.__dict__ if freeze else None,
            "commit": _git("rev-parse", "--short", "HEAD").stdout.strip(),
            "rows": rows,
        }, default=str) + "\n")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
