"""E5 diagnostic (exploratory): fading-drug bias split by stock status of the 4-week window."""
import pandas as pd, numpy as np
import common as C
from e5 import fading_frame
from irpd.evalx.scorer import weekly_from_daily
wk = weekly_from_daily(C.daily("selection")); wk["week_start"] = pd.to_datetime(wk["week_start"]).dt.date
lb = C.labels()[["origin", "drug_code", "fading", "sporadic"]]
rows = []
for base in ("ml_control", "cur_control"):
    for variant, df in (("base", None), ("fading->tsb", fading_frame(base))):
        p = C.preds(base, C.origins_for("sel"), df)
        x = p[p.h <= 4].merge(wk, on=["drug_code", "week_start"])
        g = x.groupby(["origin", "drug_code"]).agg(f=("mean", "sum"), y=("uncensored_units", "sum"),
            out=("censored", "sum"), n=("h", "size")).reset_index()
        g = g[(g.n == 4)].merge(lb, on=["origin", "drug_code"])
        g = g[g.fading]
        for nm, m in (("no stock-out week", g.out == 0), ("1-3 stock-out weeks", g.out.between(1, 3)), ("all 4 weeks out", g.out == 4)):
            s = g[m]
            rows.append({"base": base, "variant": variant, "window": nm, "windows": len(s),
                         "forecast": s.f.sum(), "truth": s.y.sum(), "bias": s.f.sum() / s.y.sum()})
d = pd.DataFrame(rows); d.to_csv(C.OUT / "e5_diag_fading_by_stock.csv", index=False); print(d.round(3).to_string())
