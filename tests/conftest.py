from pathlib import Path

import pytest

from irpd import config, warehouse

FIXTURES = Path(__file__).parent / "fixtures/irpd"


@pytest.fixture(autouse=True)
def no_saved_ml_forecasts(tmp_path, monkeypatch):
    """Tests never read the real saved ML forecasts; ones that need them write their own."""
    from irpd.ml import rolling, serve

    monkeypatch.setattr(serve, "FORECAST_FILE", tmp_path / "ml_forecast.csv")
    monkeypatch.setattr(rolling, "FILE", tmp_path / "ml_rolling.csv")


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
