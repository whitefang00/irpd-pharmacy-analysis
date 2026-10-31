"""Paths and project-wide constants."""

import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# The IRPD files (not committed). Override with IRPD_DATA to point elsewhere,
# e.g. at the small test fixtures.
DATA = Path(os.environ.get("IRPD_DATA", ROOT / "data/raw/irpd"))
WAREHOUSE = Path(os.environ.get("IRPD_DB", ROOT / "data/warehouse/irpd.duckdb"))

FIRST_DAY = date(2023, 1, 2)
LAST_DAY = date(2025, 12, 31)

# The agent never sees data after this day (the replay holdout starts the next day).
CUTOFF = date(2025, 6, 30)

# Opening balances were wrong for some drugs until the stock counts of
# July–September 2023, so stock analysis starts here.
STOCK_START = date(2023, 10, 1)
