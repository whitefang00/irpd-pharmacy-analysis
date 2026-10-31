from irpd.agent import registry


def test_register_call_and_errors(monkeypatch):
    monkeypatch.setattr(registry, "TOOLS", {})
    schema = {"type": "object", "additionalProperties": False,
              "required": ["x"], "properties": {"x": {"type": "integer"}}}

    @registry.tool("double", "Double x.", schema)
    def double(con, x):
        return {"value": 2 * x}

    assert registry.openai_tools()[0]["name"] == "double"
    assert registry.call("double", '{"x": 4}', None) == {"value": 8}
    assert "error" in registry.call("double", {"y": 1}, None)
    assert "error" in registry.call("nope", {}, None)
