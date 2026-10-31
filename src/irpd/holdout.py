"""The sealed holdout: the second half of 2025, readable only by the scorer.

`seal()` splits a built warehouse into two files beside it:

- `agent.duckdb`: every table as `warehouse.as_of(CUTOFF)` shows it, written
  out. It holds no row from after the cutoff, so nothing that opens it can
  reach the holdout, not even by querying `wh.*` behind the views.
- `holdout.duckdb`: what happened after the cutoff (demand, daily stock,
  deliveries, transfers, stock-takes). Only `open_sealed()` opens it, and
  only scorer code calls that: `irpd.replay` and the API's results routes.

`agent_view()` is the connection agent tools should use: the agent file as of
a day, with external access switched off and locked, so SQL can't ATTACH or
read the holdout file (or any other file) either.
"""

from datetime import date
from pathlib import Path

import duckdb

from irpd import config, warehouse

AGENT = config.WAREHOUSE.with_name("agent.duckdb")
SEALED = config.WAREHOUSE.with_name("holdout.duckdb")

# Holdout tables and the date column that puts a row after the cutoff.
SEALED_TABLES = {
    "daily_demand": "date",
    "daily_stock": "date",
    "purchases": "date",
    "transfers": "date",
    "stock_takes": "date",
}


def seal(wh: Path = config.WAREHOUSE, agent: Path = AGENT,
         sealed: Path = SEALED) -> tuple[Path, Path]:
    """Write the agent file and the sealed holdout file from a full warehouse."""
    cutoff = f"DATE '{config.CUTOFF.isoformat()}'"
    for path in (agent, sealed):
        path.unlink(missing_ok=True)
    with warehouse.as_of(config.CUTOFF, wh) as con:
        con.execute(f"ATTACH '{agent}' AS agent")
        views = [r[0] for r in con.execute(
            "SELECT view_name FROM duckdb_views() WHERE NOT internal ORDER BY 1").fetchall()]
        for name in views:
            con.execute(f"CREATE TABLE agent.{name} AS SELECT * FROM {name}")
        con.execute("DETACH agent")

        con.execute(f"ATTACH '{sealed}' AS sealed")
        for name, col in SEALED_TABLES.items():
            con.execute(f"CREATE TABLE sealed.{name} AS "
                        f"SELECT * FROM wh.{name} WHERE {col} > {cutoff}")
        con.execute("DETACH sealed")
    return agent, sealed


def agent_view(day: date = config.CUTOFF, agent: Path = AGENT) -> duckdb.DuckDBPyConnection:
    """The agent's connection: the agent file as of `day`, unable to open other files."""
    con = warehouse.as_of(day, agent)
    con.execute("SET enable_external_access = false")
    con.execute("SET lock_configuration = true")
    return con


def open_sealed(sealed: Path = SEALED) -> duckdb.DuckDBPyConnection:
    """The holdout, read-only. For the scorer in `irpd.replay` only."""
    return duckdb.connect(str(sealed), read_only=True)
