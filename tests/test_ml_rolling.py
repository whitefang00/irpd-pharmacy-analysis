"""The rolling H2 forecasts: which weeks get one, and reading them back."""

from datetime import date

from irpd.ml import rolling


def test_origins_are_sundays_whose_4_weeks_end_by_the_last_day():
    os_ = rolling.origins()
    assert os_[0] == date(2025, 6, 29) and os_[-1] == date(2025, 11, 30)
    assert all(o.weekday() == 6 for o in os_) and len(os_) == 23
    assert (os_[-1] - os_[0]).days == 7 * 22


def test_for_drug_reads_one_drugs_weeks(tmp_path):
    path = tmp_path / "r.csv"
    assert rolling.for_drug("A", path) is None
    path.write_text("origin,week_start,drug_code,current,ml\n"
                    "2025-06-29,2025-06-30,A,10.5,12\n"
                    "2025-07-06,2025-07-07,A,11,\n"
                    "2025-06-29,2025-06-30,B,1,1\n")
    assert rolling.for_drug("A", path) == [
        {"week_start": "2025-06-30", "current": 10.5, "ml": 12.0},
        {"week_start": "2025-07-07", "current": 11.0, "ml": None}]
    assert rolling.for_drug("Z", path) == []
