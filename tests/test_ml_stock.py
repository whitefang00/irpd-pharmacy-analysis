from datetime import date

import duckdb
import numpy as np
import pytest

from irpd import warehouse
from irpd.ml import history as hist
from irpd.ml import schema
from irpd.ml.features import stock

DATA_END = date(2024, 1, 14)


@pytest.fixture
def ctx(fixture_db):
    con = duckdb.connect(str(fixture_db), read_only=True)
    # DX has sales history in no table, so it has no ledger rows either.
    h = hist.load(con, DATA_END, ["D1", "D2", "DX"])
    yield schema.PanelContext(con=con, data_end=DATA_END,
                              origins=[date(2023, 12, 31), date(2024, 1, 7)], history=h,
                              con_at=lambda d: warehouse.as_of(d, fixture_db))
    con.close()


def test_stock_on_hand_and_cover(ctx):
    f = stock.build(ctx)
    assert list(f.columns) == stock.COLUMNS
    assert all(str(t) == "float32" for t in f.dtypes)
    o = np.datetime64("2024-01-07")
    # D1: 30 in, 20 sold by 4 Jan; D2 ends at 2 after the count on 5 Jan.
    assert f.loc[("D1", o), "k_on_hand"] == 10
    # 20 sold over the 7 ledger days 1-7 Jan: 10 / (20 / 7) days.
    assert f.loc[("D1", o), "k_days_cover"] == pytest.approx(3.5, rel=1e-6)
    assert f.loc[("D2", o), "k_on_hand"] == 2
    # 10 sold over its 6 ledger days, 2-7 Jan: 2 / (10 / 6) days.
    assert f.loc[("D2", o), "k_days_cover"] == pytest.approx(1.2, rel=1e-6)
    # Rows exist only for drugs with panel rows: DX never sold, and nothing had
    # sold by 31 Dec 2023, so neither gets a row at all.
    assert "DX" not in f.index.get_level_values("drug_code")
    assert np.datetime64("2023-12-31") not in f.index.get_level_values("origin")


def test_stock_reads_each_origin_as_of_that_day(ctx, fixture_db):
    days = []

    def con_at(d):
        days.append(d)
        return warehouse.as_of(d, fixture_db)

    ctx.con_at = con_at
    stock.build(ctx)
    assert days == ctx.origins
