"""`/contest`: the frozen contest runs read from their files, on a tiny hand-made run dir."""

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from irpd.api import app as api_app
from irpd.api import sealed

REFS = {"pharmacist (ledger)": (40, 20, 30e6, None), "mimic": (15, 9, 11e6, 0.77),
        "textbook": (11, 12, 4.4e6, 1.9), "plain tools": (5, 60, 4.9e6, 1.7)}


def write_run(out: Path, name: str, model: str, agent: tuple) -> None:
    d = out / name
    d.mkdir(parents=True)
    knobs = {"default": {"service_level": 0.95, "cycle_days": 14, "floor_min_days": 14,
                         "floor_max_days": 30},
             "segments": [{"name": "Rising", "trend": "rising", "long_stockout": None,
                           "service_level": 0.98, "cycle_days": 21}],
             "stop_after_idle_days": None, "overrides": []}
    (d / "knobs.json").write_text(json.dumps({
        "run_id": name, "model": model, "as_of": "2025-06-30", "frozen_at": "2026-10-30",
        "usd": 0.04, "knobs": knobs, "rationale": "Cover rising drugs."}))
    pd.DataFrame({"drug_code": ["A", "B", "C"], "rule": ["Rising", "default", "default"]}
                 ).to_csv(d / "rules.csv", index=False)
    rows = []
    for demand in ("uncensored", "recorded"):
        for p, (out_d, shelf, lost, scale) in {**REFS, "agent": agent}.items():
            rows.append({"policy": p, "stockout_days": out_d, "shelfout_days": shelf,
                         "lost_revenue": lost, "avg_stock_value": 312e6,
                         "stockout_days_long": 0, "drugs": 3, "scale": scale,
                         "demand": demand})
    pd.DataFrame(rows).set_index("policy").to_csv(d / "holdout_score.csv")
    for demand in ("uncensored", "recorded"):
        curve = pd.DataFrame({"factor": [0.5, 1.0], "avg_stock_value": [1e8, 2e8],
                              "stockout_days": [90, 30], "shelfout_days": [3, 4],
                              "lost_revenue": [9e6, 3e6]})
        pd.concat({"agent": curve, "mimic": curve * 2}).to_csv(d / f"tradeoff_{demand}.csv")
    table = [{"policy": "agent", "scale": 1.6, "stockout_days": 4, "shelfout_days": 3,
              "lost_revenue_rp_m": 2.36, "avg_stock_rp_m": 365.0}]
    (d / "in_sample.json").write_text(json.dumps({"windows": [
        {"window": "2025-01-01 to 2025-06-30", "table": table}]}))


def test_contest_reads_the_frozen_runs(tmp_path):
    write_run(tmp_path, "final-luna", "luna", (8, 36, 5.8e6, 1.5))
    write_run(tmp_path, "final-sol", "sol", (7, 3, 4.2e6, 1.6))
    write_run(tmp_path, "draft-unscored", "x", (1, 1, 1, 1))
    (tmp_path / "draft-unscored" / "holdout_score.csv").unlink()
    (tmp_path / "holdout_runs.jsonl").write_text("{}\n{}\n")

    r = sealed.contest_report(tmp_path)
    assert r["drugs"] == 3 and r["holdout_runs"] == 2
    assert [x["model"] for x in r["runs"]] == ["sol", "luna"]  # best first
    t = r["table"]["uncensored"]
    assert [x["policy"] for x in t] == [*REFS, "agent (sol)", "agent (luna)"]
    assert t[-2]["shelfout_days"] == 3 and t[0]["scale"] is None
    assert set(r["table"]) == {"uncensored", "recorded"}

    sol = r["runs"][0]
    assert sol["rules"][0] == {
        "name": "Rising", "drugs": 1,
        "when": "Drugs with demand rising (last 8 weeks over 1.25x the half-year rate)",
        "sets": "98% service level; orders cover the lead time plus 21 days"}
    assert sol["rules"][-1]["name"] == "default" and sol["rules"][-1]["drugs"] == 2
    assert "floor" in sol["rules"][-1]["sets"]
    assert [p["factor"] for p in sol["tradeoff"]["uncensored"]] == [0.5, 1.0]
    assert r["mimic_tradeoff"]["uncensored"][0]["stockout_days"] == 180
    assert sol["in_sample"][0]["rows"][0]["lost_revenue"] == 2.36e6


def test_contest_route_is_null_without_runs(tmp_path):
    def client(d):
        return TestClient(api_app.create_app(tmp_path / "a", tmp_path / "s", contest_dir=d))

    assert client(tmp_path / "missing").get("/contest").json() is None
    write_run(tmp_path / "c", "final-sol", "sol", (7, 3, 4.2e6, 1.6))
    assert client(tmp_path / "c").get("/contest").json()["runs"][0]["model"] == "sol"
