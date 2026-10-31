"""Lost demand, the fake stock-out test, switching and the headline numbers.

Run from the repo root on a built warehouse:
    uv run --no-sync python scripts/lost_demand.py [--grid]

`--grid` reruns the method grid the estimator was tuned on (weights x baseline days).
"""

import sys
import time

import duckdb
import numpy as np
import pandas as pd

from irpd import config, lost

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
T0 = time.time()


def section(title):
    print(f"\n=== {title} ({time.time() - T0:.0f}s)")


def rp(x):
    return f"Rp {x / 1e6:,.1f}M"


con = duckdb.connect(str(config.WAREHOUSE), read_only=True)
p = lost.panel(con, config.CUTOFF)

if "--grid" in sys.argv:
    section("Method grid on the fake test (bias / wape by tier)")
    base = p.w.copy()
    open_ = (p.traffic > 0).astype(float)
    wd = np.array([d.weekday() for d in p.days])
    by_wd = np.array([p.traffic[(wd == w) & (open_ > 0)].mean() for w in range(7)])
    for name, w in [("traffic", base), ("weekday", open_ * by_wd[wd] / by_wd.mean()),
                    ("flat", open_)]:
        p.set_weights(w)
        for n in (14, 28, 56, 91):
            sc = lost.fake_score(lost.fake_spells(con, config.CUTOFF, n_base=n, p=p))
            print(f"{name:8} n={n:3}  " + "  ".join(
                f"{t}: {r.bias:+.3f}/{r.wape:.2f}" for t, r in sc.iterrows()))
    p.set_weights(base)

section("Fake stock-out test (drugs out <= 2% of days; 20 fake spells each)")
fake = lost.fake_spells(con, config.CUTOFF, p=p)
with pd.option_context("display.float_format", "{:.3f}".format):
    print(lost.fake_score(fake).to_string())
    print("\nBy expected size (band buckets):")
    print(lost.fake_score(fake.assign(size=pd.cut(fake.expected, [0, 5, 20, 100, np.inf],
                                                   right=False)), by="size").to_string())
    tb = lost.total_band(fake)
    print("\nActual / expected summed over spells, 80% bootstrap range by tier:")
    print(tb.to_string())

section("Spells up to the cutoff")
sp = lost.spells(con, config.CUTOFF, p=p)
print(sp.groupby(["kind", "tier"]).agg(spells=("estimated", "size"),
                                       estimated=("estimated", "mean"),
                                       with_alt=("has_alternative", "mean")).round(2))
so = sp[(sp.kind == "stockout") & sp.estimated & (sp.counted_days >= 1)]
print(f"\nStock-out days: sold {so.recorded_units.sum() / so.expected_units.sum():.1%} "
      "of expected units")
first = sp[(sp.kind == "shelfout") & sp.estimated]
print(f"Shelf-out spells: sold {first.recorded_units.sum() / first.expected_units.sum():.2f}x "
      "of expected units")

section("Switching: real stock-outs vs placebos")
with pd.option_context("display.float_format", "{:,.3f}".format):
    print(lost.placebo(con, config.CUTOFF, p=p).to_string())
r = lost.switch_rates(sp)
print(f"\nSwitched share of lost lines, stock-outs with an alternative in stock: "
      f"{r.attrs['overall']:.1%}")
real = sp[(sp.kind == "stockout") & sp.has_alternative & sp.estimated]
rng = np.random.default_rng(0)
idx = rng.integers(0, len(real), (2000, len(real)))
boot = real.alt_uplift_lines.to_numpy()[idx].sum(1) / real.lost_lines.to_numpy()[idx].sum(1)
print(f"  95% CI (bootstrap over spells): {np.quantile(boot, .025):.1%} - "
      f"{np.quantile(boot, .975):.1%}")
pos = real.lost_lines.clip(lower=0)
print("Per-spell uplift capped at the spell's lost lines, as a share of lost "
      f"lines: {(real.switched_raw * pos).sum() / pos.sum():.1%}")
sh = sp[(sp.kind == "shelfout") & sp.has_alternative & sp.estimated]
print("Shelf-outs: alternatives' uplift / lost lines "
      f"{sh.alt_uplift_lines.sum() / sh.lost_lines.sum():.1%}")
with pd.option_context("display.float_format", "{:.2f}".format):
    print("\nPer ingredient (stock-outs with an alternative; >= 30 lost lines):")
    print(r[r.lost_lines >= 30].round(2).to_string())
    for ing in ("amlodipine", "paracetamol"):
        if ing in r.index:
            print(f"{ing}: {r.loc[ing].to_dict()}")

section("Headline: spells from 1 Oct 2023 to the cutoff")
h = lost.headline(sp, band_by_tier=tb[["ratio", "lo", "hi"]])
rev = con.execute(f"""SELECT sum(revenue) FROM daily_demand
    WHERE date BETWEEN DATE '{config.STOCK_START}' AND DATE '{config.CUTOFF}'""").fetchone()[0]
with pd.option_context("display.float_format", "{:,.0f}".format):
    cols = ["spells", "counted_days", "days_beyond", "lost_revenue", "lost_revenue_lo",
            "lost_revenue_hi", "switched_revenue", "truly_lost_revenue"]
    print((h[cols] / [1, 1, 1, 1e6, 1e6, 1e6, 1e6, 1e6]).round(1).to_string())
    print(h["switched_share"].round(2).to_string())
a = h.loc[("all", "all")]
print(f"\nRevenue over the period: {rp(rev)}. Lost {rp(a.lost_revenue)} "
      f"({a.lost_revenue / rev:.2%}; range {rp(a.lost_revenue_lo)} - {rp(a.lost_revenue_hi)}), "
      f"switched {a.switched_share:.0%}, truly lost {rp(a.truly_lost_revenue)} "
      f"({a.truly_lost_revenue / rev:.2%}).")
nocap = lost.spells(con, config.CUTOFF, max_days=10_000, p=p)
nocap = nocap[nocap.start >= config.STOCK_START]
print(f"Counting every spell day (no {lost.MAX_DAYS}-day cap): lost "
      f"{rp(nocap.lost_revenue.sum())}, truly lost {rp(nocap.truly_lost_revenue.sum())}.")
mapped = sp[sp.start >= config.STOCK_START]
print(f"Truly lost on unmapped drugs (no alternatives known): "
      f"{rp(mapped[mapped.ingredients.isna()].truly_lost_revenue.sum())}")

section("Top 10 drugs by truly lost revenue")
names = dict(con.execute("SELECT drug_code, drug_name FROM drugs").fetchall())
top = (mapped.groupby("drug_code")
       .agg(tier=("tier", "first"), ingredients=("ingredients", "first"),
            spells=("kind", "size"), days=("counted_days", "sum"),
            lost=("lost_revenue", "sum"), switched=("switched_revenue", "sum"),
            truly_lost=("truly_lost_revenue", "sum"))
       .sort_values("truly_lost", ascending=False).head(10))
top.insert(0, "name", top.index.map(names))
with pd.option_context("display.float_format", "{:,.0f}".format):
    print(top.to_string())
