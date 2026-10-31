"""Build the DuckDB warehouse, and open it as it looked on a given day.

`build()` loads the base tables, then runs every module in `irpd.tables` in
dependency order. `as_of()` returns a connection where every table is a view
that stops at that day, which is how the agent is kept from seeing the future.
"""

import importlib
import pkgutil
from datetime import date
from pathlib import Path
from types import ModuleType

import duckdb

from irpd import config, tables
from irpd.load import DRUGS_SQL, load

# Base tables from `irpd.load`, with their date columns.
BASE_TABLES = {"sales_lines": "date", "stock_moves": "date", "drugs": None}


def table_modules() -> list[ModuleType]:
    """All modules in irpd.tables, ordered so each runs after what it depends on."""
    mods = [importlib.import_module(f"irpd.tables.{m.name}")
            for m in pkgutil.iter_modules(tables.__path__)]
    made_by = {t: m for m in mods for t in m.TABLES}
    ordered: list[ModuleType] = []
    visiting: set[str] = set()

    def visit(m: ModuleType) -> None:
        if m in ordered:
            return
        if m.__name__ in visiting:
            raise ValueError(f"dependency cycle at {m.__name__}")
        visiting.add(m.__name__)
        for dep in m.DEPENDS:
            if dep in BASE_TABLES:
                continue
            if dep not in made_by:
                raise ValueError(f"{m.__name__} depends on unknown table {dep!r}")
            visit(made_by[dep])
        visiting.discard(m.__name__)
        ordered.append(m)

    for m in sorted(mods, key=lambda m: m.__name__):
        visit(m)
    return ordered


def date_columns() -> dict[str, str | None]:
    cols = dict(BASE_TABLES)
    for m in table_modules():
        cols.update(m.TABLES)
    return cols


def build(data: Path = config.DATA, path: Path = config.WAREHOUSE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".building")
    tmp.unlink(missing_ok=True)
    with duckdb.connect(str(tmp)) as con:
        load(con, data)
        for m in table_modules():
            m.build(con)
    tmp.replace(path)  # never leave a half-built warehouse behind
    return path


def rebuilt_as_of(day: date) -> dict[str, str | None]:
    """Tables whose rows carry facts from after their own date, rebuilt as of `day`.

    A module may define `as_of_views(day)`, returning table -> a SELECT over
    the other (already filtered) views, or None to hide the table on that day.
    """
    out: dict[str, str | None] = {"drugs": DRUGS_SQL}
    for m in table_modules():
        out.update(getattr(m, "as_of_views", lambda _: {})(day))
    return out


def as_of(day: date, path: Path = config.WAREHOUSE) -> duckdb.DuckDBPyConnection:
    """An in-memory connection where every table only shows what was known on `day`.

    Most tables are filtered on their date column. Those listed by
    `rebuilt_as_of` are recomputed from the filtered views instead, so a spell
    never shows its later end, a price its later valid_to, and so on.
    """
    con = duckdb.connect()
    con.execute(f"ATTACH '{path}' AS wh (READ_ONLY)")
    present = {r[0] for r in con.execute(
        "SELECT table_name FROM duckdb_tables() WHERE database_name = 'wh'").fetchall()}
    rebuilt = rebuilt_as_of(day)
    for name, col in date_columns().items():
        if name not in present:
            continue
        if name in rebuilt:
            if rebuilt[name] is not None:
                con.execute(f"CREATE VIEW {name} AS {rebuilt[name]}")
            continue
        where = f"WHERE {col} <= DATE '{day.isoformat()}'" if col else ""
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM wh.{name} {where}")
    return con
