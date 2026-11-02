"""EXPLORATORY: compare rolling-replay variants at EQUAL average stock value (frontier), H2 2024.
Each variant is simulated at scales k in K; lost revenue is interpolated (log-log) at fixed stock
values. Reuses power_rolling.py's simulator; needs the same inputs.
Run from the repo root:  uv run python eda/followup/scripts/architecture/frontier_rolling.py"""
import sys, runpy
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.argv = ["x"]
# import simulator pieces without running power_rolling's main block
src = (HERE / "power_rolling.py").read_text().split("base_name, base = run")[0]
g = {"__file__": str(HERE / "power_rolling.py")}; exec(compile(src, "power_rolling_head", "exec"), g)
from irpd import contest, replay
K = [0.7, 0.85, 1.0, 1.2, 1.45, 1.75, 2.1]
def curve(make_basis):
    p = g["policies_over_time"](lambda j: contest.to_policy({}, make_basis(g["bases"][j])))
    out = []
    for k in K:
        s = replay.summarise(g["simulate_tv"](p, k))
        out.append((s.avg_stock_value, s.lost_revenue, s.lost_revenue_stockout))
    return np.array(out)
targets = np.array([2.6e8, 2.9e8, 3.3e8, 3.8e8, 4.4e8])
def at(c, col):
    return np.exp(np.interp(np.log(targets), np.log(c[:, 0]), np.log(c[:, col])))
variants = {"base": lambda b: b,
            "variance x1.5": lambda b: g["perturbed"](b, None, 1.5),
            "all means x0.9": lambda b: g["perturbed"](b, np.full(len(b.mean), 0.9))}
for s_ in (0.1, 0.25):
    for seed in range(5):
        f = g["drug_noise"](s_, 500 + seed)
        variants[f"noise s={s_} seed{seed}"] = (lambda b, f=f: g["perturbed"](b, f(b)))
rows = []
for name, mk in variants.items():
    c = curve(mk)
    lr, lso = at(c, 1), at(c, 2)
    for t, a, b_ in zip(targets, lr, lso):
        rows.append({"variant": name, "stock_value_M": t / 1e6, "lost_rev_M": a / 1e6, "lost_stockout_M": b_ / 1e6})
    print(name, np.round(lr / 1e6, 3), flush=True)
df = pd.DataFrame(rows)
base = df[df.variant == "base"].set_index("stock_value_M").lost_rev_M
df["ratio_vs_base"] = df.lost_rev_M / df.stock_value_M.map(base)
df.to_csv(g["OUT"] / "frontier_rolling_H2-2024.csv", index=False)
with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
    print(df.pivot_table(index="variant", columns="stock_value_M", values="ratio_vs_base").to_string())
