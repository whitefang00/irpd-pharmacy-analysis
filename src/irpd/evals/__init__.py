"""Eval questions for the pharmacy analyst, their gold answers, and a scorer.

`gold_answers()` computes every gold answer from the warehouse as of
`config.CUTOFF`. Questions that need the ingredient map are skipped unless one
is passed in. "Can't tell" questions have no gold value (None).
"""

from pathlib import Path
from typing import Any

import pandas as pd

from irpd import config, warehouse
from irpd.evals.questions import QUESTIONS, Question, prepare

__all__ = ["QUESTIONS", "Question", "gold_answers"]


def gold_answers(db: Path = config.WAREHOUSE,
                 ingredient_map: pd.DataFrame | None = None) -> dict[str, Any]:
    """Gold answer per question id. Skipped questions are left out."""
    out: dict[str, Any] = {}
    with warehouse.as_of(config.CUTOFF, db) as con:
        prepare(con)
        if ingredient_map is not None:
            m = ingredient_map[["drug_code", "ingredients"]].fillna("")
            con.register("ingredient_map", m)
        for q in QUESTIONS:
            if q.needs == "ingredient_map" and ingredient_map is None:
                continue
            out[q.id] = q.gold(con) if q.gold else None
    return out
