import json
from types import SimpleNamespace

import pandas as pd
import pytest

from irpd import ingredients, llm


class FakeClient:
    """Answers like the Responses API, from a canned list of products."""

    def __init__(self, answers, output_tokens=500):
        self.answers, self.output_tokens, self.calls = answers, output_tokens, 0
        self.responses = self

    def create(self, **kwargs):
        self.calls += 1
        items = json.loads(kwargs["input"])
        products = [{"id": it["id"], **self.answers.get(it["name"], {
            "ingredients": [], "strength": "", "form": "", "kind": "unknown", "confident": False})}
            for it in items]
        usage = SimpleNamespace(input_tokens=1_000, output_tokens=self.output_tokens,
                                input_tokens_details=SimpleNamespace(cached_tokens=0))
        return SimpleNamespace(output_text=json.dumps({"products": products}), usage=usage)


def drug(name, ingredients_, confident=True):
    return {name: {"ingredients": ingredients_, "strength": "", "form": "tablet",
                   "kind": "drug", "confident": confident}}


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "LEDGER", tmp_path / "spend.jsonl")
    monkeypatch.setattr(ingredients, "CACHE", tmp_path / "map.jsonl")
    monkeypatch.setattr(ingredients, "OUT", tmp_path / "map.csv")
    return tmp_path


def test_mapping_is_cached_and_logged(paths, fixture_db):
    client = FakeClient({**drug("Amoxicillin 500mg", ["amoxicillin"]),
                         **drug("Flazen 50", ["diclofenac"])})
    table = ingredients.run(client=client, db=fixture_db)
    assert client.calls == 1
    assert dict(zip(table["drug_name"], table["ingredients"], strict=True)) == {
        "Amoxicillin 500mg": "amoxicillin", "Flazen 50": "diclofenac"}
    assert llm.spent() == pytest.approx(llm.cost(llm.BULK_MODEL, 1_000, 0, 500))
    ingredients.run(client=client, db=fixture_db)  # everything cached: no new call
    assert client.calls == 1


def test_unconfident_answers_count_as_unknown(paths, fixture_db):
    client = FakeClient(drug("Flazen 50", ["ibuprofen"], confident=False))
    table = ingredients.run(client=client, db=fixture_db)
    assert table.set_index("drug_name").loc["Flazen 50", "ingredients"] == ""


def test_cap_refuses_a_call_that_could_pass_it(paths):
    with pytest.raises(llm.BudgetExceeded):
        llm.call("test", "x", "y", client=FakeClient({}), cap=0.001)
    assert llm.spent() == 0.0


def test_score_covers_partial_keys_and_is_exact_on_generics(tmp_path):
    key = pd.DataFrame({
        "DrugCode": ["A", "B", "C", "D"],
        "ingredients": ["diclofenac", "paracetamol", "cyanocobalamin", "omeprazole"],
        "source": ["generic_name", "generic_name", "brand_list", "generic_name"],
        "confidence": ["high"] * 4})
    key.to_csv(tmp_path / "key.csv", index=False)
    table = pd.DataFrame({
        "drug_code": ["A", "B", "C", "D"],
        "ingredients": ["diclofenac", "ibuprofen", "cyanocobalamin + pyridoxine + thiamine", ""],
        "mapped": [True] * 4})
    s = ingredients.score(table, tmp_path / "key.csv")
    assert s["covered"] == 2 / 3   # A and C cover the key (C's extras are fine), B doesn't
    assert s["exact"] == 0.5       # generic names only: A right, B wrong
    assert s["abstained"] == 0.25  # D said nothing


def answer(code, model, ingredients_, confident=True):
    return {"drug_code": code, "prompt": "v2", "model": model, "ingredients": ingredients_,
            "strength": "", "form": "tablet", "kind": "drug", "confident": confident}


def test_table_prefers_a_confident_sol_answer_over_luna(paths):
    sol, luna = llm.AGENT_MODEL, llm.BULK_MODEL
    rows = [answer("A", luna, ["ibuprofen"]), answer("A", sol, ["diclofenac"]),
            answer("B", luna, ["paracetamol"]), answer("B", sol, [], confident=False),
            answer("C", luna, [], confident=False),
            answer("D", luna, ["omeprazole"], confident=False)]
    ingredients.CACHE.write_text("".join(json.dumps(r) + "\n" for r in rows))
    drugs = pd.DataFrame({"drug_code": list("ABCDE"), "drug_name": list("ABCDE")})
    t = ingredients.ingredient_table(drugs, version="v2").set_index("drug_code")
    assert t.loc["A", ["ingredients", "source_model"]].tolist() == ["diclofenac", sol]
    assert t.loc["B", ["ingredients", "source_model"]].tolist() == ["paracetamol", luna]
    assert t.loc["C", "ingredients"] == "" and t.loc["C", "mapped"]
    assert t.loc["D", "ingredients"] == ""  # unconfident answers never count
    assert t.loc["E", ["mapped", "source_model"]].tolist() == [False, ""]
    only_luna = ingredients.ingredient_table(drugs, version="v2", models=[luna])
    assert only_luna.set_index("drug_code").loc["A", "ingredients"] == "ibuprofen"


def test_cli_passes_model_and_prompt(monkeypatch):
    from irpd import cli
    seen = {}

    def fake_run(**kwargs):
        seen.update(kwargs)
        return pd.DataFrame({"drug_code": [], "ingredients": [], "mapped": []})

    monkeypatch.setattr(ingredients, "run", fake_run)
    monkeypatch.setattr(ingredients, "score", lambda t: dict.fromkeys(
        ["scored", "covered", "covered_n", "exact", "exact_n", "abstained"], 0))
    monkeypatch.setattr(llm, "spent", lambda: 0.0)
    cli.main(["map-ingredients", "--model", "sol", "--prompt", "v1", "--limit", "5"])
    assert seen == {"limit": 5, "model": "sol", "version": "v1"}
    cli.main(["map-ingredients"])
    assert seen == {"limit": None, "model": "luna", "version": "v2"}


def test_run_refuses_an_unknown_model(paths, fixture_db):
    with pytest.raises(ValueError):
        ingredients.run(model="gpt-nope", client=FakeClient({}), db=fixture_db)
