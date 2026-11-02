"""A rolling H2 2025 forecast, for the holdout view of the drug page only.

Each Sunday from the cutoff on, both forecasters forecast the next 4 weeks
using only the data up to that day: the current method straight from
`forecast.forecast`, the ML model retrained on everything up to that Sunday
(the frozen system, as `serve.train` fits it). The last origin is the last
one whose 4 weeks end by `config.LAST_DAY`.

This reads the full warehouse, past the cutoff, so it must never feed the
agent. Only `irpd.api.sealed` reads its output (`FILE`), and only when the
page asks for the holdout.

    uv run --extra ml irpd ml rolling
"""

from __future__ import annotations

import json
import time
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd

from irpd import config, warehouse
from irpd import forecast as fc

FILE = config.WAREHOUSE.with_name("ml_rolling.csv")
WEEKS = 4


def origins(first: date = config.CUTOFF, last: date = config.LAST_DAY) -> list[date]:
    """Sundays from the one by `first` while 4 whole weeks after them end by `last`."""
    from irpd.ml import schema

    out, o = [], schema.policy_origin(first)
    while o + timedelta(weeks=WEEKS) <= last:
        out.append(o)
        o += timedelta(weeks=1)
    return out


def build(path: Path | None = None, wh: Path = config.WAREHOUSE, log=print) -> pd.DataFrame:
    """Forecast every origin with both forecasters and write `path` (CSV).

    One row per origin and drug: origin, week_start (the first forecast
    week), drug_code, current, ml (4-week totals; ml empty where the model
    has no row for the drug, as for drugs with no sale in the past year).
    """
    from irpd.ml import panel, schema, split
    from irpd.ml.models.lgbm import make_gbm
    from irpd.ml.serve import CONFIG_HASH, frozen_config

    path = path or FILE
    cfg = frozen_config()
    os_ = origins()

    def con_at(day: date):
        return warehouse.as_of(day, wh)

    end = schema.policy_origin(config.LAST_DAY)  # the last whole week's Sunday
    t = time.time()
    with con_at(end) as con:
        ctx = panel.context(con, end, set(split.training_origins(os_[-1])) | set(os_),
                            censor_mode=cfg.censor_mode, con_at=con_at)
        p = panel.build_panel(ctx, cfg.groups)
    log(f"panel {p.shape} in {time.time() - t:.0f}s")

    parts = []
    for i, o in enumerate(os_, 1):
        t = time.time()
        day = o + timedelta(days=1)  # the Monday the forecast is read, as in the app
        model = make_gbm(cfg).fit(schema.trainable(p, o), o)
        rows = p[(pd.to_datetime(p["origin"]) == pd.Timestamp(o)) & (p["h"] <= WEEKS)]
        ml = (pd.DataFrame({"drug_code": rows["drug_code"].astype(str).to_numpy(),
                            "ml": model.predict(rows)})
              .groupby("drug_code").ml.sum())
        with con_at(day) as c:
            cur = fc.forecast(c, None, day, WEEKS)
        cur = cur[cur.tier != "none"].groupby("drug_code")["mean"].sum().rename("current")
        part = pd.concat([cur, ml], axis=1).reset_index(names="drug_code")
        part.insert(0, "week_start", day)
        part.insert(0, "origin", o)
        parts.append(part)
        log(f"{i}/{len(os_)} {o}: {len(part)} drugs, {time.time() - t:.0f}s")

    out = pd.concat(parts, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format="%.3f")
    note = {"config_hash": CONFIG_HASH, "weeks": WEEKS,
            "origins": [os_[0].isoformat(), os_[-1].isoformat()],
            "made_at": datetime.now(UTC).isoformat(timespec="seconds")}
    path.with_suffix(".json").write_text(json.dumps(note, indent=1) + "\n")
    return out


def for_drug(code: str, path: Path | None = None) -> list[dict] | None:
    """One drug's rolling forecasts, oldest first, or None if none are saved."""
    path = path or FILE
    if not path.exists():
        return None
    f = _load(path, path.stat().st_mtime_ns)
    rows = f[f.drug_code == code]
    return [{"week_start": r.week_start, "current": round(float(r.current), 2),
             "ml": None if pd.isna(r.ml) else round(float(r.ml), 2)}
            for r in rows.itertuples()]


@lru_cache(maxsize=2)
def _load(path: Path, mtime: int) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"drug_code": str, "week_start": str}).fillna(
        {"current": 0.0})
