"""Shared paths for the follow-up scripts.

Inputs come from the repo's own data/ folder (the same files irpd.config uses).
Outputs (tables, intermediate parquet, caches) go to data/followup/<topic>/,
which git ignores. Figures go to eda/followup/fig/<topic>/.

Each script puts this folder on sys.path and imports it:

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import followup_paths as fp
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
WAREHOUSE = DATA / "warehouse"
ML = DATA / "ml"
FOLLOWUP = DATA / "followup"
# Weekly forecasts of every arm on the shared origin grid (written by arms/run_arms.py).
PREDS = FOLLOWUP / "preds"
FIG_ROOT = ROOT / "eda" / "followup" / "fig"

os.environ.setdefault("MPLBACKEND", "Agg")


def out(topic: str) -> Path:
    """data/followup/<topic>/, created if missing."""
    p = FOLLOWUP / topic
    p.mkdir(parents=True, exist_ok=True)
    return p


def fig(topic: str) -> Path:
    """eda/followup/fig/<topic>/, created if missing."""
    p = FIG_ROOT / topic
    p.mkdir(parents=True, exist_ok=True)
    return p


def use_cache(topic: str) -> Path:
    """Point irpd.evalx's pickle cache at data/followup/cache/<topic>/ unless
    IRPD_EVALX_CACHE is already set. The cache only saves time on reruns."""
    p = FOLLOWUP / "cache" / topic
    os.environ.setdefault("IRPD_EVALX_CACHE", str(p))
    return Path(os.environ["IRPD_EVALX_CACHE"])
