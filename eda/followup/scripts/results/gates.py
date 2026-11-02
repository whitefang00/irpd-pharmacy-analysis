"""Collect each experiment's gate values into data/followup/results/gates.json (for the summary chart)."""
import json

import pandas as pd

import common as C

O = C.OUT
e2 = pd.read_csv(O / "e2_scores.csv"); e2c = pd.read_csv(O / "e2_compare.csv")
e3 = json.loads((O / "e3_verdict.json").read_text()); e3a = pd.read_csv(O / "e3_eid_week_ape.csv", index_col=0)
e3c = pd.read_csv(O / "e3_compare.csv")
e4 = json.loads((O / "e4_verdict.json").read_text())
e5 = json.loads((O / "e5_verdict.json").read_text())
e6 = json.loads((O / "e6_verdict.json").read_text())
G = []


def add(name, value, txt, lo, hi, axis):
    G.append({"name": name, "value": float(value), "value_txt": txt, "pass": [lo, hi],
              "ok": bool(lo <= value <= hi), "axis": axis})


add("E0 rolling replay reproduces the earlier rolling run (Rp M)", 4.3332, "4.333 vs 4.33 +/-0.5%", 4.308, 4.352, [4.2, 4.45])
add("E1 14-day re-plan: net lost revenue ratio, H2 2024", 0.901, "0.90 (need <= 0.85)", 0, 0.85, [0.6, 1.1])
for arm, lab in (("ml_control", "ML"), ("cur_control", "current")):
    t = e2[(e2.arm == arm) & (e2.period == "test") & (e2.variant == "rho") & e2.tier.isin(["fast", "regular", "slow"])]
    worst = t.loc[(t[["cov_l", "cov_lc"]] - 0.95).abs().max(axis=1).idxmax()]
    v = worst["cov_lc"] if abs(worst["cov_lc"] - .95) >= abs(worst["cov_l"] - .95) else worst["cov_l"]
    add(f"E2 {lab}+rho: worst tier p95 coverage, Jul-Oct 2024 ({worst['tier']})", v, f"{v:.3f}", 0.935, 0.965, [0.9, 1.0])
    c = e2c[(e2c.arm == arm) & (e2c.period == "test") & (e2c.variant == "rho") & (e2c.tier == "pooled")].iloc[0]
    add(f"E2 {lab}+rho: pooled pinball, upper 90% bound (%)", c.rel_hi * 100, f"{c.rel_hi*100:+.1f}%", -100, 1, [-5, 6])
for arm, ref, lab in (("cur_eid", "cur_control", "current+Eid"), ("ml_eid", "ml_control", "ML+Eid")):
    p25 = int(e3[arm]["placed_2025"].split("/")[0])
    add(f"E3 {lab}: Eid 2025 peak placed right (of 4)", p25, e3[arm]["placed_2025"], 3, 4, [0, 4.2])
    d = (e3a.loc[arm, "2024"] - e3a.loc[ref, "2024"]) * 100
    add(f"E3 {lab}: Eid-week store error 2024 vs control (pts)", d, f"{e3a.loc[arm,'2024']:.3f} vs {e3a.loc[ref,'2024']:.3f}", -100, -1e-9, [-6, 6])
    c = e3c[(e3c.period == "sel") & (e3c.arm == arm) & (e3c.metric == "pinball_lc") & (e3c.tier == "pooled")].iloc[0]
    add(f"E3 {lab}: pooled pinball, upper 90% bound (%)", c.rel_hi * 100, f"{c.rel_hi*100:+.2f}%", -100, 1, [-3, 3])
v = e4["ml_censor"]
add("E4 ML+censor: censored-segment 4-week bias", v["censored_bias"], f"{v['censored_bias']:.2f}", 0.95, 1.05, [0.8, 2.5])
add("E4 ML+censor: pooled pinball, upper 90% bound (%)", v["pinball_rel_hi"] * 100, f"{v['pinball_rel_hi']*100:+.2f}%", -100, 1, [-3, 3])
for k, lab in (("ml_control:fading", "fading->TSB (ML)"), ("ml_control:sporadic", "sporadic min-stock (ML)"),
               ("cur_control:sporadic", "sporadic min-stock (current)")):
    add(f"E5 {lab}: stock freed on affected drugs (%)", e5[k]["stock_freed_mean"] * 100,
        f"{e5[k]['stock_freed_mean']*100:.1f}%", 5, 100, [-5, 35])
    add(f"E5 {lab}: extra stock-out days per drug", e5[k]["extra_stockout_days_per_drug"],
        f"{e5[k]['extra_stockout_days_per_drug']:.1f}", -100, 2, [0, 10])
v = e6["ml_vs_cur"]
add("E6 ML vs current: pooled pinball, upper 90% bound (%)", v["pooled_rel_hi"] * 100, f"{v['pooled_rel_hi']*100:+.1f}%", -100, -1e-9, [-8, 3])
add("E6 ML vs current: fast-tier pinball, upper bound (%)", 4.09, "+4.1%", -100, 1, [-8, 6])
add("E6 ML: regular-tier 4-week bias", v["bias"]["regular"]["bias4"], f"{v['bias']['regular']['bias4']:.3f}", 0.97, 1.03, [0.9, 1.15])
add("E6 ML vs current: replay net stock-out ratio, H2 2024", v["replay_net_out_ratio"]["H2-2024"],
    f"{v['replay_net_out_ratio']['H2-2024']:.3f}", 0, 1.075, [0.8, 1.2])
e = e6["ens_vs_ml"]
add("E6 ensemble vs ML: pooled pinball, upper bound (%)", e["pooled_rel_hi"] * 100, f"{e['pooled_rel_hi']*100:+.2f}%", -100, -1e-9, [-3, 3])
(O / "gates.json").write_text(json.dumps(G, indent=1))
print(pd.DataFrame(G)[["name", "value_txt", "ok"]].to_string())
