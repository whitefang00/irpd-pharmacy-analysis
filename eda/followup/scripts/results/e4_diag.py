"""E4 diagnostic (exploratory): why the censored segment's bias is ~2.
Splits censored-segment 4-week windows by how many of the 4 target weeks hold a total
stock-out day, and shows the share of lost units the day-level estimator covers."""
import pandas as pd, numpy as np
import common as C
from irpd.evalx.scorer import weekly_from_daily
wk = weekly_from_daily(C.daily("selection")); wk["week_start"] = pd.to_datetime(wk["week_start"]).dt.date
rows = []
for arm in ("ml_control", "ml_censor"):
    p = C.preds(arm, C.origins_for("sel"))
    p = p[p.h <= 4].merge(wk, on=["drug_code", "week_start"])
    g = p.groupby(["origin", "drug_code", "tier"]).agg(f4=("mean", "sum"), y4=("uncensored_units", "sum"),
        u4=("units", "sum"), cw=("censored", "sum"), n=("h", "size")).reset_index()
    g = g[g.n == 4].merge(C.labels()[["origin", "drug_code", "censored"]], on=["origin", "drug_code"])
    for lab in (True, False):
        for cw in range(5):
            x = g[(g.censored == lab) & (g.cw == cw)]
            if len(x):
                rows.append({"arm": arm, "censored_label": lab, "stockout_weeks_in_4": cw, "windows": len(x),
                             "bias_gross": x.f4.sum() / x.y4.sum(), "share_of_f4": x.f4.sum() / g[g.censored == lab].f4.sum()})
d = pd.DataFrame(rows); d.to_csv(C.OUT / "e4_diag_by_stockout_weeks.csv", index=False)
print(d.round(3).to_string())
