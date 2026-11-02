"""Price features (p_), per (drug, origin), from the prices table as of the origin.

`prices` is rebuilt in hindsight (a blip is only spotted once the price
after it is seen), so each origin reads `ctx.connection_at(origin)`. Prices
are per sales unit; each drug uses its main unit (most sale lines to date),
converted to rupiah per stock base unit with `pack_sizes`.

- `p_price`: price per base unit on the origin.
- `p_change_{13,52}`: `p_price` over the price 13 / 52 weeks earlier, minus 1.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from irpd.ml import panel as pn
from irpd.ml import schema

NEEDS_SQL = True
CHANGES = (13, 52)


def prices_at(con, origin: date) -> pd.DataFrame:
    """Per drug: base-unit price on the origin and CHANGES weeks before, main unit only."""
    d = f"DATE '{origin.isoformat()}'"
    pr = con.execute(f"""
        SELECT p.drug_code, p.unit, p.price::DOUBLE / k.base_per_unit AS price,
               p.valid_from, p.lines
        FROM prices p LEFT JOIN pack_sizes k
          ON k.drug_code = p.drug_code AND k.sales_unit = p.unit
        WHERE p.valid_from <= {d}""").df()
    pr["valid_from"] = pd.to_datetime(pr.valid_from).dt.date
    lines = pr.groupby(["drug_code", "unit"], as_index=False)["lines"].sum()
    main = (lines.sort_values(["drug_code", "lines", "unit"], ascending=[True, False, True])
            .drop_duplicates("drug_code"))
    pr = pr.merge(main[["drug_code", "unit"]], on=["drug_code", "unit"])
    pr = pr.sort_values(["drug_code", "valid_from"])
    out = pd.DataFrame(index=pd.Index(main.drug_code, name="drug_code"))
    for w in (0, *CHANGES):
        day = origin - timedelta(weeks=w)
        last = pr[pr.valid_from <= day].groupby("drug_code")["price"].last()
        out[w] = last.reindex(out.index)
    return out


def build(ctx: schema.PanelContext) -> pd.DataFrame:
    parts = []
    for origin in sorted(set(ctx.origins)):
        rows = pn.active_rows(ctx, origin)
        p = prices_at(ctx.connection_at(origin), origin).reindex(ctx.drugs[rows])
        now = p[0].to_numpy(float)
        cols = {"p_price": now}
        for w in CHANGES:
            then = p[w].to_numpy(float)
            cols[f"p_change_{w}"] = np.divide(now, then, out=np.full(len(now), np.nan),
                                              where=then > 0) - 1
        parts.append(pn.frame(ctx, origin, rows, cols))
    return pn.finish(parts, by_h=False)
