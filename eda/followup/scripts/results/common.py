"""Shared loaders for the results scripts (E2-E6).

Run any script from the repo root with:
    uv run python eda/followup/scripts/results/<script>.py
(figs.py needs `uv run --with matplotlib`). Inputs: the forecast arms in
data/followup/preds/ and the daily truth from harness/build_data.py.

Workarounds (harness code is not edited):
- labels.parquet names the 182-day price `price_avg182`; the scorer looks for
  `price`/`avg_price`/... and would silently use 0. We rename it to `price`.
- Every arm is scored on the same (origin, drug) set: rows present in
  ml_control AND cur_control (the ML's active-52-week drugs), so summaries of
  different arms are comparable, not only their paired differences.
"""
from __future__ import annotations

import pickle
import sys
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402

fp.use_cache("results")
P = fp.PREDS
H = fp.out("harness")
OUT = fp.out("results")
FIG = fp.fig("results")

SEL_FIRST, SEL_LAST = date(2024, 1, 7), date(2024, 10, 13)
FIT_BEFORE = date(2024, 7, 1)
EID = [date(2023, 4, 22), date(2024, 4, 10), date(2025, 3, 31)]
ARMS = ["cur_control", "cur_eid", "ml_control", "ml_eid", "ml_censor", "ml_eid_censor",
        "ml_eid_offset", "tsb"]


def grid() -> list[date]:
    return [d.date() for d in pd.date_range("2024-01-07", "2025-12-21", freq="14D")]


def sel_origins() -> list[date]:
    return [o for o in grid() if SEL_FIRST <= o <= SEL_LAST]


def diag_origins() -> list[date]:
    """2025 grid origins whose lead + cycle window (<= 28 days) ends by 31 Dec 2025."""
    return [o for o in grid() if o.year == 2025 and o <= date(2025, 11, 30)]


@lru_cache(None)
def labels() -> pd.DataFrame:
    lb = pd.read_parquet(P / "labels.parquet")
    lb["origin"] = pd.to_datetime(lb["origin"]).dt.date
    lb["drug_code"] = lb["drug_code"].astype(str)
    return lb.rename(columns={"price_avg182": "price"})


@lru_cache(None)
def actuals() -> pd.DataFrame:
    a = pd.read_parquet(P / "actuals.parquet")
    a["week_start"] = pd.to_datetime(a["week_start"]).dt.date
    a["drug_code"] = a["drug_code"].astype(str)
    return a


@lru_cache(None)
def daily(kind: str) -> pd.DataFrame:
    return pd.read_parquet(H / f"daily_truth_{kind}.parquet")


@lru_cache(None)
def raw(arm: str) -> pd.DataFrame:
    p = pd.read_parquet(P / f"{arm}.parquet")
    p["origin"] = pd.to_datetime(p["origin"]).dt.date
    p["week_start"] = pd.to_datetime(p["week_start"]).dt.date
    p["drug_code"] = p["drug_code"].astype(str)
    p["tier"] = p["tier"].astype(str)
    return p


@lru_cache(None)
def common_keys() -> pd.MultiIndex:
    a = raw("ml_control")[["origin", "drug_code"]].drop_duplicates()
    b = raw("cur_control")[["origin", "drug_code"]].drop_duplicates()
    k = a.merge(b, on=["origin", "drug_code"])
    return pd.MultiIndex.from_frame(k)


def preds(arm: str, origins: list[date], df: pd.DataFrame | None = None) -> pd.DataFrame:
    """An arm on the common (origin, drug) set and the given origins."""
    p = raw(arm) if df is None else df
    p = p[p["origin"].isin(set(origins))]
    idx = pd.MultiIndex.from_frame(p[["origin", "drug_code"]])
    return p[idx.isin(common_keys())][["origin", "drug_code", "h", "week_start", "mean",
                                       "var", "tier"]].reset_index(drop=True)


def truth_for(period: str) -> pd.DataFrame:
    return daily("selection" if period == "sel" else "full")


def origins_for(period: str) -> list[date]:
    return {"sel": sel_origins(), "diag": diag_origins(),
            "fit": [o for o in sel_origins() if o < FIT_BEFORE],
            "test": [o for o in sel_origins() if o >= FIT_BEFORE]}[period]


def score(arm: str, period: str, rho=None, df=None, tag: str | None = None):
    """score_arm on a period, cached on disk by (tag or arm, period, rho)."""
    from irpd.evalx import score_arm
    key = f"{tag or arm}_{period}_{'none' if not rho else '_'.join(f'{k}{v:.4f}' for k, v in sorted(rho.items()))}"
    f = OUT / "scores" / f"{key}.pkl"
    f.parent.mkdir(exist_ok=True)
    if f.exists():
        return pickle.loads(f.read_bytes())
    tp = "sel" if period in ("sel", "fit", "test") else "diag"
    s = score_arm(preds(arm, origins_for(period), df), actuals(), labels(),
                  daily=truth_for(tp), rho=rho)
    f.write_bytes(pickle.dumps(s))
    return s


def fmt(x, n=4):
    return "" if pd.isna(x) else f"{x:.{n}f}"


def md_table(df: pd.DataFrame, floatfmt: int = 4) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(fmt(v, floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
