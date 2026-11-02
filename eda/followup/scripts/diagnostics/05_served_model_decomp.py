"""Load the saved served boosters (read-only) and show which inputs move the path over h.

Builds the served panel rows (origin 29 Jun 2025, agent view, as serve.train does) and,
for comparison, rows at origin 2 Mar 2025 (Eid 31 Mar ahead). The served model was
trained on data to 29 Jun 2025, so the March rows are in-sample: used only to show the
mechanism, not accuracy. Writes only to data/followup/diagnostics/.
"""
import os
import time
from datetime import date

import lightgbm as lgb
import numpy as np
import pandas as pd

from irpd import holdout
from irpd.ml import panel, schema
from irpd.ml.serve import frozen_config

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402

DATA = fp.DATA
OUT = fp.out("diagnostics")
MD = f"{DATA}/warehouse/ml_forecast_models"
cfg = frozen_config()
boosters = [lgb.Booster(model_file=f"{MD}/global_seed{s}.txt") for s in (0, 1, 2)]
feats = boosters[0].feature_name()
print(len(feats), "features")

import json
note = json.load(open(f"{DATA}/warehouse/ml_forecast.json"))
assert note["features"] == feats, "feature order differs"


def rows_at(origin: date, day: date):
    t = time.time()
    with holdout.agent_view(day) as con:
        ctx = panel.context(con, day, {origin}, censor_mode=cfg.censor_mode,
                            con_at=holdout.agent_view, forecast_only=True)
        p = panel.build_panel(ctx, cfg.groups)
    print(origin, p.shape, f"{time.time()-t:.0f}s")
    return p


def design(p):
    x = pd.DataFrame(index=p.index)
    for c in feats:
        s = p[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            x[c] = s
        elif pd.api.types.is_bool_dtype(s):
            x[c] = s.astype(np.float32)
        else:
            x[c] = s.astype(np.float32)
    return x


def predict(x):
    return np.mean([b.predict(x) for b in boosters], axis=0)


# Categorical levels: LightGBM text models store pandas_categorical; reuse it.
pc = boosters[0].pandas_categorical


def fix_cats(x):
    cats = [c for c in feats if isinstance(x[c].dtype, pd.CategoricalDtype)]
    for c, levels in zip(cats, pc or []):
        x[c] = pd.Categorical(x[c].astype(object).where(x[c].astype(object).isin(levels)),
                              categories=levels)
    return x


res = {}
for name, origin, day in [("served", date(2025, 6, 29), date(2025, 6, 30)),
                          ("eid_mar", date(2025, 3, 2), date(2025, 3, 2))]:
    p = rows_at(origin, day)
    p = p.sort_values(["drug_code", "h"]).reset_index(drop=True)
    x = fix_cats(design(p))
    base = predict(x)
    if name == "served":
        f = pd.read_csv(f"{DATA}/warehouse/ml_forecast.csv", dtype={"drug_code": str})
        f = f.sort_values(["drug_code", "h"]).reset_index(drop=True)
        mm = p[["drug_code", "h"]].astype({"h": int}).merge(f, on=["drug_code", "h"], how="left")
        print("reproduces served CSV: max abs diff",
              float(np.nanmax(np.abs(mm["mean"].to_numpy() - base))))
    hvary = ["h", "c_eid_days", "c_eid", "c_ramadan", "c_eid_uplift"]
    # check nothing else varies with h within a drug
    other = [c for c in feats if c not in hvary]
    nun = p.groupby("drug_code")[[c for c in other if c in p.columns and c != "tier"]].nunique(dropna=False).max().max()
    print(name, "max distinct values over h of non-calendar features:", nun)
    # counterfactuals
    x_h1cal = x.copy()
    first = x.groupby(p.drug_code.values).transform("first")
    for c in ["c_eid_days", "c_eid", "c_ramadan", "c_eid_uplift"]:
        x_h1cal[c] = first[c].values
    only_h = predict(x_h1cal)
    x_h1 = x.copy()
    x_h1["h"] = 1.0
    only_cal = predict(x_h1)
    d = pd.DataFrame({"drug_code": p.drug_code, "h": p.h.astype(int), "tier": p.tier,
                      "full": base, "only_h": only_h, "only_cal": only_cal})

    def cv(col):
        g = d.groupby("drug_code")[col]
        return (g.std() / g.mean()).median()
    res[name] = {"median_path_cv_full": cv("full"), "median_path_cv_h_only": cv("only_h"),
                 "median_path_cv_calendar_only": cv("only_cal"),
                 "store_path_full": d.groupby("h").full.sum().round(0).tolist(),
                 "store_path_h_only": d.groupby("h").only_h.sum().round(0).tolist(),
                 "store_path_cal_only": d.groupby("h").only_cal.sum().round(0).tolist()}
    # SHAP contributions (raw log-scale, seed 0) for h-varying features
    contrib = boosters[0].predict(x, pred_contrib=True)
    cdf = pd.DataFrame(contrib[:, :-1], columns=feats)
    cdf["drug_code"] = p.drug_code.values
    # spread over h of each feature's contribution, averaged over drugs
    spread = cdf.groupby("drug_code")[hvary].agg(lambda s: s.max() - s.min()).mean()
    level = cdf[feats].abs().mean().sort_values(ascending=False)
    res[name]["mean_abs_shap_top"] = level.head(8).round(4).to_dict()
    res[name]["mean_over_drugs_range_over_h_of_shap(log)"] = spread.round(4).to_dict()
    res[name]["calendar_values_by_h"] = p[p.drug_code == p.drug_code.iloc[0]][hvary].astype(float).round(3).to_dict("list")
    d.to_parquet(f"{OUT}/decomp_{name}.parquet")

with open(f"{OUT}/served_model_decomp.json", "w") as fh:
    json.dump(res, fh, indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
