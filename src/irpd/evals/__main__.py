"""Print every gold answer: `python -m irpd.evals [ingredient_map.csv]`."""

import sys

import pandas as pd

from irpd.evals import QUESTIONS, gold_answers

imap = pd.read_csv(sys.argv[1], dtype=str) if len(sys.argv) > 1 else None
gold = gold_answers(ingredient_map=imap)
for q in QUESTIONS:
    value = gold.get(q.id, "(skipped: needs " + str(q.needs) + ")")
    print(f"{q.id:<28} {q.kind:<9} {value}")
