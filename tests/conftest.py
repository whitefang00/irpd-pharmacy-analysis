from pathlib import Path

import pytest

from irpd import config, warehouse

FIXTURES = Path(__file__).parent / "fixtures/irpd"


@pytest.fixture(scope="session")
def fixture_db(tmp_path_factory) -> Path:
    """A warehouse built from the small hand-made fixture files."""
    return warehouse.build(FIXTURES, tmp_path_factory.mktemp("wh") / "fixture.duckdb")


@pytest.fixture(scope="session")
def real_db(tmp_path_factory) -> Path:
    """A warehouse built from the real IRPD files; skipped if they aren't present."""
    if not (config.DATA / "processed_dataset/sales_processed.csv").exists():
        pytest.skip("real IRPD data not present")
    return warehouse.build(config.DATA, tmp_path_factory.mktemp("wh") / "real.duckdb")
