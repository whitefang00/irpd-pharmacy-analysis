"""Active ingredients per drug, from the LLM mapping (`irpd map-ingredients`).

- `ingredient_map`: one row per drug, loaded from
  `data/ingredients/ingredient_map.csv`. `ingredients` is a " + "-joined list
  of English INN names, empty when no model was confident. `source_model` is
  the model whose answer the row uses.

The map has no date: it describes the products, not anything that happened,
so `as_of` shows it whole. If the CSV hasn't been made yet, the table is
created empty so the rest of the build still works.
"""

import duckdb

from irpd import config

TABLES = {"ingredient_map": None}
DEPENDS: list[str] = []

MAP = config.ROOT / "data/ingredients/ingredient_map.csv"

COLUMNS = {
    "drug_code": "VARCHAR", "drug_name": "VARCHAR", "ingredients": "VARCHAR",
    "strength": "VARCHAR", "form": "VARCHAR", "kind": "VARCHAR",
    "mapped": "BOOLEAN", "source_model": "VARCHAR",
}


def build(con: duckdb.DuckDBPyConnection) -> None:
    cols = ", ".join(f"{c} {t}" for c, t in COLUMNS.items())
    con.execute(f"CREATE OR REPLACE TABLE ingredient_map ({cols})")
    if MAP.exists():
        # By name, so a CSV from before a column was added still loads.
        con.execute(f"""
            INSERT INTO ingredient_map BY NAME
            SELECT * FROM read_csv('{MAP}', header = true, all_varchar = true)""")
        # Empty text fields come back from the CSV as NULL; keep them as ''.
        text = [c for c, t in COLUMNS.items() if t == "VARCHAR"]
        con.execute("UPDATE ingredient_map SET "
                    + ", ".join(f"{c} = coalesce({c}, '')" for c in text))
