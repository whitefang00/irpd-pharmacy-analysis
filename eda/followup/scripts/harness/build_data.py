"""Harness inputs: daily truth (selection-cut and full) and current-forecaster preds.

Writes to data/followup/harness/:
- daily_truth_selection.parquet: 1 Jan - 8 Dec 2024, as of 8 Dec 2024 (selection data only).
- daily_truth_full.parquet: 1 Jan 2024 - 31 Dec 2025, as of 31 Dec 2025 (diagnostic for 2025).
- cur_harness.parquet + cur_harness_labels.parquet: current forecaster at the 21
  selection G origins (7 Jan - 13 Oct 2024), in the same format as the forecast arms (arms/run_arms.py). For power calibration.

Run from the repo root:  uv run python eda/followup/scripts/harness/build_data.py
"""
import sys
import time
from datetime import date
from pathlib import Path
from irpd.evalx import data, rolling

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup_paths as fp  # noqa: E402
fp.use_cache("harness")

OUT = fp.out("harness")
t = time.time()
if not (OUT / "cur_harness.parquet").exists():
    p, lb = data.current_preds(rolling.grid(last=date(2024, 10, 13)))
    p.to_parquet(OUT / "cur_harness.parquet"); lb.to_parquet(OUT / "cur_harness_labels.parquet")
    print("preds", p.shape, lb.shape, f"{time.time()-t:.0f}s", flush=True)
if not (OUT / "daily_truth_selection.parquet").exists():
    d = data.daily_truth(date(2024, 1, 1), date(2024, 12, 8))
    d.to_parquet(OUT / "daily_truth_selection.parquet"); print("truth sel", d.shape, f"{time.time()-t:.0f}s", flush=True)
if not (OUT / "daily_truth_full.parquet").exists():
    d = data.daily_truth(date(2024, 1, 1), date(2025, 12, 31))
    d.to_parquet(OUT / "daily_truth_full.parquet"); print("truth full", d.shape, f"{time.time()-t:.0f}s", flush=True)
