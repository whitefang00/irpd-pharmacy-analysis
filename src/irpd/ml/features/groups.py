"""Ingredient-group features (g_, ablation step A3).

Drugs are grouped by the `ingredient_map` table, under `ctx.group_def`:

- `ingredient`: the same set of active ingredients (order and case ignored);
- `ingredient_route`: plus the route, read from the form (ROUTES);
- `ingredient_form`: plus the form itself.

Drugs with no ingredients (unmapped, or missing from the map) have no group.

A drug's **siblings** at an origin are the other members of its group with a
sale in the ACTIVE_WEEKS weeks before it, the same rule that puts a drug in
the panel. Using only the history cut at the origin keeps every feature,
static ones included, free of later data.

`g_static_*` (what the drug is): group id, form, kind, strength in mg, group
size (the drug plus its siblings) and a no-siblings flag.

`g_dyn_*` (what the siblings sold), leave-one-out, from `ctx.history_at`:

- `g_dyn_idx_lag_1..4`, `g_dyn_idx_mean_4/13/26`: the siblings' demand index.
  Each sibling's weekly units are divided by its own `history.target_scale`
  at the origin, then averaged across siblings each week. Means average the
  weekly values, skipping missing weeks.
- `g_dyn_rev_lag_1..2`, `g_dyn_rev_mean_4/13/26`: rupiah revenue per sibling
  (the mean across siblings each week), so groups of different sizes compare.
- `g_dyn_in_stock`: siblings not out of stock in the last week (NaN where the
  ledger can't show stock-outs yet).
- `g_dyn_share_13`: the drug's own share of group revenue (itself plus
  siblings) over 13 weeks. Not leave-one-out, by definition.

Censored sibling weeks follow `ctx.censor_mode`: `raw` keeps them, `mask`
makes them missing so they are skipped. `impute` is treated as `mask` here:
there is no as-of-origin lost-demand estimate for siblings.

Drugs without siblings get NaN dynamic features.
"""

from __future__ import annotations

import re
import warnings
from datetime import date

import numpy as np
import pandas as pd

from irpd.ml import history as hist
from irpd.ml import schema

NEEDS_SQL = True  # reads ingredient_map

ROUTES = {
    "oral": ("tablet", "capsule", "syrup", "suspension", "drops", "powder", "liquid",
             "lozenge", "emulsion"),
    "topical": ("cream", "ointment", "gel", "lotion"),
    "inhaled": ("inhaler",),
    "injected": ("injection",),
}
_ROUTE = {f: r for r, forms in ROUTES.items() for f in forms}
# 26-week windows added after the EDA: siblings co-move over 13+ weeks, not week to week.
IDX_LAGS, IDX_MEANS = (1, 2, 3, 4), (4, 13, 26)
REV_LAGS, REV_MEANS = (1, 2), (4, 13, 26)
SHARE_WEEKS = 13
_MG = {"mg": 1.0, "g": 1000.0, "gr": 1000.0, "mcg": 1e-3, "µg": 1e-3, "ug": 1e-3}
_STRENGTH = re.compile(r"(\d+(?:[.,]\d+)?)\s*(mg|mcg|µg|ug|gr|g)\b", re.IGNORECASE)

STATIC_COLUMNS = ["g_static_group", "g_static_form", "g_static_kind", "g_static_strength_mg",
                  "g_static_size", "g_static_no_siblings"]
DYN_COLUMNS = [*(f"g_dyn_idx_lag_{k}" for k in IDX_LAGS),
               *(f"g_dyn_idx_mean_{k}" for k in IDX_MEANS),
               *(f"g_dyn_rev_lag_{k}" for k in REV_LAGS),
               *(f"g_dyn_rev_mean_{k}" for k in REV_MEANS),
               "g_dyn_in_stock", f"g_dyn_share_{SHARE_WEEKS}"]
COLUMNS = STATIC_COLUMNS + DYN_COLUMNS


def ingredient_key(ingredients: str | None) -> str:
    """The ingredient set as a canonical string ('' when there are none)."""
    if not isinstance(ingredients, str):
        return ""
    parts = {p.strip().lower() for p in ingredients.split("+")}
    return " + ".join(sorted(p for p in parts if p))


def route(form: str | None) -> str:
    """oral, topical, inhaled, injected or other."""
    return _ROUTE.get((form or "").strip().lower(), "other")


def strength_mg(strength: str | None) -> float:
    """The first mass in a strength label, in mg ('0,5 mg' -> 0.5); NaN if none."""
    if not isinstance(strength, str):
        return np.nan
    m = _STRENGTH.search(strength)
    if not m:
        return np.nan
    return float(m.group(1).replace(",", ".")) * _MG[m.group(2).lower()]


def group_ids(drugs: pd.DataFrame, group_def: str) -> pd.Series:
    """Group id per drug_code under `group_def`; NaN for drugs without ingredients."""
    if group_def not in schema.GROUP_DEFS:
        raise ValueError(f"group_def must be one of {schema.GROUP_DEFS}")
    key = drugs["ingredients"].map(ingredient_key)
    form = drugs["form"].fillna("").str.strip().str.lower()
    if group_def == "ingredient_route":
        key = key + " | " + form.map(route)
    elif group_def == "ingredient_form":
        key = key + " | " + form.mask(form.eq(""), "unknown")
    gid = key.where(drugs["ingredients"].map(ingredient_key).ne(""))
    return pd.Series(gid.to_numpy(), index=drugs["drug_code"].to_numpy(), name="group")


def read_drugs(ctx: schema.PanelContext) -> pd.DataFrame:
    """The `ingredient_map` table. It has no date, so any origin's connection gives it whole."""
    con = ctx.connection_at(min(ctx.origins) if ctx.origins else ctx.data_end)
    return con.execute("SELECT drug_code, ingredients, strength, form, kind "
                       "FROM ingredient_map").df()


def build(ctx: schema.PanelContext) -> pd.DataFrame:
    """Group features indexed by (drug_code, origin), read from the warehouse."""
    return build_from(ctx, read_drugs(ctx))


def _nanmean(x: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
        return np.nanmean(x, axis=axis)


def _category(values, categories) -> pd.Categorical:
    return pd.Categorical(values, categories=categories)


def _active(ctx: schema.PanelContext, origin) -> np.ndarray:
    return np.flatnonzero(hist.active(ctx.history, origin, schema.ACTIVE_WEEKS))


def build_from(ctx: schema.PanelContext, drugs: pd.DataFrame) -> pd.DataFrame:
    """Group features from an ingredient table (the columns of `ingredient_map`)."""
    codes = ctx.drugs
    drugs = drugs.drop_duplicates("drug_code").set_index("drug_code", drop=False)
    known = drugs.reindex(codes)
    all_gid = group_ids(drugs, ctx.group_def)
    gid = all_gid.reindex(codes)
    # Category levels come from the whole map (it has no dates), not from the drugs
    # seen so far, so a group's code never depends on how far the data runs.
    groups = sorted(all_gid.dropna().unique())
    g = gid.map({k: i for i, k in enumerate(groups)}).fillna(-1).to_numpy(int)
    members = [np.flatnonzero(g == i) for i in range(len(groups))]

    def label(col: str) -> pd.Series:
        s = known[col].fillna("").astype(str).str.strip().str.lower()
        return s.mask(s.eq(""))

    form, kind = label("form"), label("kind")

    def levels(col: str) -> list[str]:
        s = drugs[col].fillna("").astype(str).str.strip().str.lower()
        return sorted(s[s.ne("")].unique())

    forms, kinds = levels("form"), levels("kind")
    strength = known["strength"].map(strength_mg).to_numpy(float)

    frames = []
    for origin in ctx.origins:
        stat, dyn = _origin(ctx, origin, g, members)
        df = pd.DataFrame({
            "drug_code": codes,
            "origin": pd.Timestamp(origin),
            "g_static_group": _category(gid.to_numpy(), groups),
            "g_static_form": _category(form.to_numpy(), forms),
            "g_static_kind": _category(kind.to_numpy(), kinds),
            "g_static_strength_mg": strength,
            **stat, **dyn,
        })
        # Only drugs that have panel rows at this origin: a drug that first sells
        # later must not appear at all.
        frames.append(df.iloc[_active(ctx, origin)])
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["drug_code", "origin", *COLUMNS])
    out["origin"] = out["origin"].astype("datetime64[s]")
    for c in COLUMNS:
        if c == "g_static_no_siblings":
            out[c] = out[c].astype(bool)
        elif c in ("g_static_group", "g_static_form", "g_static_kind"):
            out[c] = out[c].astype(pd.CategoricalDtype(
                {"g_static_group": groups, "g_static_form": forms,
                 "g_static_kind": kinds}[c]))
        else:
            out[c] = out[c].astype("float32")
    return out.set_index(["drug_code", "origin"])[COLUMNS]


def _origin(ctx: schema.PanelContext, origin: date, g: np.ndarray,
            members: list[np.ndarray]) -> tuple[dict, dict]:
    """Static size columns and leave-one-out dynamic columns for one origin."""
    c = ctx.history_at(origin)
    n_drugs = len(g)
    active = hist.active(ctx.history, origin, schema.ACTIVE_WEEKS)
    scale = hist.target_scale(ctx.history, origin).astype(float)
    y, rev = c["y"].astype(float), c["revenue"].astype(float)
    if ctx.censor_mode != "raw":
        y, rev = np.where(c["out"], np.nan, y), np.where(c["out"], np.nan, rev)
    weeks = max(max(IDX_MEANS), max(REV_MEANS), SHARE_WEEKS)
    idx = y[:, -weeks:] / scale[:, None]
    rev = rev[:, -weeks:]
    last_out = c["out"][:, -1] if c["out"].shape[1] else np.zeros(n_drugs, bool)
    stock_known = bool(c["stock_known"][-1]) if len(c["stock_known"]) else False

    size = np.full(n_drugs, np.nan)
    no_sib = np.ones(n_drugs, bool)
    dyn = {k: np.full(n_drugs, np.nan) for k in DYN_COLUMNS}
    for m in members:
        act = m[active[m]]
        for d in m:
            sib = act[act != d]
            size[d] = 1 + len(sib)
            if not len(sib):
                continue
            no_sib[d] = False
            si = _nanmean(idx[sib], axis=0)   # weekly sibling index
            sr = _nanmean(rev[sib], axis=0)   # weekly rupiah per sibling
            for k in IDX_LAGS:
                dyn[f"g_dyn_idx_lag_{k}"][d] = si[-k] if len(si) >= k else np.nan
            for k in IDX_MEANS:
                dyn[f"g_dyn_idx_mean_{k}"][d] = _nanmean(si[-k:], 0) if len(si) else np.nan
            for k in REV_LAGS:
                dyn[f"g_dyn_rev_lag_{k}"][d] = sr[-k] if len(sr) >= k else np.nan
            for k in REV_MEANS:
                dyn[f"g_dyn_rev_mean_{k}"][d] = _nanmean(sr[-k:], 0) if len(sr) else np.nan
            if stock_known:
                dyn["g_dyn_in_stock"][d] = float((~last_out[sib]).sum())
            own = np.nansum(rev[d, -SHARE_WEEKS:])
            total = own + np.nansum(rev[sib, -SHARE_WEEKS:])
            dyn[f"g_dyn_share_{SHARE_WEEKS}"][d] = own / total if total > 0 else np.nan
    return {"g_static_size": size, "g_static_no_siblings": no_sib}, dyn
