"""Derived tables, one module per area.

Each module in this package defines:

- `TABLES`: dict of table name -> its date column (None if it has no date).
  The date column is what the `as_of` views filter on.
- `DEPENDS`: list of table names it reads, other than the base tables.
- `build(con)`: creates its tables from tables that already exist.

`irpd.warehouse.build` finds the modules on its own and runs them in
dependency order, so adding a module never means editing a shared file.
"""
