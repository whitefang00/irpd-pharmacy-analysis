"""The ML forecaster for the app: trained once on the app's day, forecasts saved.

The app stands on one day (`config.CUTOFF`), so the model doesn't run live.
`train` fits the frozen system (LightGBM, Tweedie 1.3, mask censoring, A2
features, default settings, seeds 0-2) on the agent's data up to that day,
forecasts weeks 1-8 for every drug with a sale in the last 52 weeks, and
writes them to `FORECAST_FILE` (CSV, so reading needs no ML libraries) with
a JSON note of what made them. The boosters are saved next to it as LightGBM
text files, for the record; the app reads only the CSV.

`forecast.forecast(..., forecaster="ml")` reads the file through `load`.
Drugs the file lacks fall back to the current forecaster there, and weeks
past 8 repeat week 8, as in the decision replay (`ml.decide`).

    uv run --extra ml irpd ml serve
"""

from __future__ import annotations

import json
import time
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd

from irpd import config

# The system frozen at gate 3 (lab/ml-frozen.json, config 36ad7c615b96) and
# confirmed by the full-budget run. Kept here so the app doesn't depend on lab/.
CONFIG = {
    "model": "lgbm", "target": "raw", "loss": "tweedie", "tweedie_power": 1.3,
    "censor_mode": "mask", "groups": ["demand", "calendar", "store", "price", "drug"],
    "group_def": "ingredient", "drop_features": [], "per_tier": False,
    "retrain_every": 1, "recency_halflife_weeks": None, "params": {}, "seeds": [0, 1, 2],
}
CONFIG_HASH = "36ad7c615b96"

FORECAST_FILE = config.WAREHOUSE.with_name("ml_forecast.csv")


def frozen_config():
    from irpd.ml.config import MLConfig

    cfg = MLConfig.from_dict(CONFIG)
    if cfg.hash() != CONFIG_HASH:
        raise ValueError(f"serving config hashes to {cfg.hash()}, not {CONFIG_HASH}")
    return cfg


def _note_path(path: Path) -> Path:
    return path.with_suffix(".json")


def train(day: date = config.CUTOFF, path: Path | None = None,
          con_at=None) -> pd.DataFrame:
    """Fit the frozen system on data up to `day`; write and return its weekly forecasts.

    `con_at(day)` opens the data as of a day (default: the agent's view).
    Returns drug_code, h (1..8), week_start, mean.
    """
    from irpd import holdout
    from irpd.ml import panel, schema, split
    from irpd.ml.models.lgbm import make_gbm

    con_at = con_at or holdout.agent_view
    path = path or FORECAST_FILE
    cfg = frozen_config()
    origin = schema.policy_origin(day)
    t = time.time()
    with con_at(day) as con:
        ctx = panel.context(con, day, set(split.training_origins(day)) | {origin},
                            censor_mode=cfg.censor_mode, con_at=con_at, forecast_only=True)
        p = panel.build_panel(ctx, cfg.groups)
    model = make_gbm(cfg).fit(schema.trainable(p, origin), origin)
    rows = p[pd.to_datetime(p["origin"]) == pd.Timestamp(origin)]
    out = pd.DataFrame({"drug_code": rows["drug_code"].astype(str).to_numpy(),
                        "h": rows["h"].astype(int).to_numpy(),
                        "mean": model.predict(rows)})
    out["week_start"] = [origin + pd.Timedelta(weeks=h - 1, days=1) for h in out["h"]]
    out["week_start"] = pd.to_datetime(out["week_start"]).dt.date
    out = out.sort_values(["drug_code", "h"])[["drug_code", "h", "week_start", "mean"]]

    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format="%.4f")
    models = path.with_name(path.stem + "_models")
    models.mkdir(exist_ok=True)
    for (key, seed), booster in model.boosters.items():
        booster.save_model(str(models / f"{key or 'global'}_seed{seed}.txt"))
    note = {"day": day.isoformat(), "origin": origin.isoformat(), "config_hash": CONFIG_HASH,
            "config": CONFIG, "drugs": int(out.drug_code.nunique()),
            "train_rows": int(model.fit_info["n_train"]),
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "seconds": round(time.time() - t), "features": model.features}
    _note_path(path).write_text(json.dumps(note, indent=1) + "\n")
    return out


def load(day: date, path: Path | None = None) -> pd.DataFrame | None:
    """The saved forecasts for `day` (drug_code x week_start means).

    None if none are saved, or they were made for another day. Raises if they
    were made by another config.
    """
    path = path or FORECAST_FILE
    if not path.exists():
        return None
    return _load(day, path, path.stat().st_mtime_ns)


@lru_cache(maxsize=4)
def _load(day: date, path: Path, mtime: int) -> pd.DataFrame | None:
    note = json.loads(_note_path(path).read_text())
    if note["config_hash"] != CONFIG_HASH:
        raise ValueError(f"{path.name} was made by config {note['config_hash']}, "
                         f"not {CONFIG_HASH}; rerun `irpd ml serve`")
    if note["day"] != day.isoformat():
        return None
    f = pd.read_csv(path, dtype={"drug_code": str})
    f["week_start"] = pd.to_datetime(f["week_start"]).dt.date
    return f.pivot(index="drug_code", columns="week_start", values="mean")
