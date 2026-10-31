import duckdb

from irpd.tables import ingredients as t


def test_missing_map_gives_an_empty_table(tmp_path, monkeypatch):
    monkeypatch.setattr(t, "MAP", tmp_path / "missing.csv")
    con = duckdb.connect()
    t.build(con)
    assert con.execute("SELECT count(*) FROM ingredient_map").fetchone()[0] == 0
    assert [r[0] for r in con.execute("DESCRIBE ingredient_map").fetchall()] == list(t.COLUMNS)


def test_map_loads_with_empty_text_as_blank(tmp_path, monkeypatch):
    path = tmp_path / "map.csv"
    path.write_text("drug_code,drug_name,ingredients,strength,form,kind,mapped,source_model\n"
                    "A,Zetamol 500,paracetamol,500 mg,tablet,drug,True,gpt-6-luna\n"
                    "B,Mystery,,,,unknown,True,gpt-6-luna\n")
    monkeypatch.setattr(t, "MAP", path)
    con = duckdb.connect()
    t.build(con)
    rows = con.execute("SELECT drug_code, ingredients, mapped FROM ingredient_map "
                       "ORDER BY drug_code").fetchall()
    assert rows == [("A", "paracetamol", True), ("B", "", True)]
