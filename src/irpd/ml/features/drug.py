"""Drug attributes (x_), per (drug, origin).

- `x_category`: the source category from the `drugs` table as of the origin
  (Generik, Patent, OTC, ...; '-' and missing become 'unknown'), as a
  categorical. Form, kind and strength come with the ingredient groups (g_).
"""

from __future__ import annotations

import pandas as pd

from irpd.ml import panel as pn
from irpd.ml import schema

NEEDS_SQL = True
UNKNOWN = "unknown"


def build(ctx: schema.PanelContext) -> pd.DataFrame:
    parts = []
    for origin in sorted(set(ctx.origins)):
        rows = pn.active_rows(ctx, origin)
        cat = dict(ctx.connection_at(origin).execute(
            "SELECT drug_code, category FROM drugs").fetchall())
        values = [cat.get(c) for c in ctx.drugs[rows]]
        values = [UNKNOWN if v in (None, "", "-") else v for v in values]
        parts.append(pn.frame(ctx, origin, rows, {"x_category": values}))
    df = pn.finish(parts, by_h=False)
    df["x_category"] = df["x_category"].astype(
        pd.CategoricalDtype(sorted(df["x_category"].unique())))
    return df
