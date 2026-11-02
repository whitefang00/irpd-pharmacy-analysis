"""Screen losses and tune LightGBM on the selection origins (plan 7.1 to 7.3).

Three stages, each a subcommand of `uv run python -m irpd.ml.tune`:

- `screen`: every (target, loss) setting in `SCREEN_SETTINGS` times the three
  censoring modes, at default parameters and seed 0, on
  `split.screen_origins()`. Writes `tune/screen.csv`: every row, losers
  included, with per-tier WAPE, bias, MASE and cover90, the pooled objective
  (`select.objective`) and the fit time.
- `search`: the best two setups from the screen (`pick_two`) each get an
  Optuna TPE search (seed 0) on all 11 `split.selection_origins()` with one
  seed. Every trial goes to `tune/search_<setup>.csv`; the studies live in
  `tune/optuna.db`.
- `summary`: each setup's best trial rerun with seeds 0, 1 and 2, compared
  per tier with the gate-1 baseline (paired bootstrap over drugs, origin
  wins), then `select.choose` per tier over the ML setups and every
  statistical method. Writes `tune/summary.md` and `tune/best_configs.json`.

**Data.** Only `split.selection_con()` (data to 8 Dec 2024). Panels are built
once per censoring mode with the A2 groups and cached as parquet, keyed by
(censor mode, groups, `schema.PANEL_VERSION`). Predictions go through
`backtest.run` with its per-origin cache keyed by config hash, so any stage
rerun resumes where it stopped.

**Huber.** LightGBM's huber delta (`alpha`) defaults to 1 unit, which on raw
weekly units acts like L1 and fits slowly. The screen runs it at two fixed
deltas, 5 and 20 units (`HUBER_DELTAS`), as two rows of the same loss family.

**Picking two setups.** A setup is one screen row (target, loss, power or
delta, censoring mode). The first is the best pooled objective. The second
is the best row from a different loss family, where a family is (target,
loss): two Tweedie powers, two huber deltas or two censoring modes of the
same loss are one family. That way the search covers two genuinely
different setups.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from irpd import config
from irpd import forecast as fc
from irpd.ml import backtest, panel, schema, select, split, stats
from irpd.ml.config import TIERS, MLConfig
from irpd.ml.features import groups_for_step
from irpd.ml.models.baseline import BaselineForecaster
from irpd.ml.models.baseline import config_hash as baseline_hash
from irpd.ml.models.lgbm import make_gbm

GROUPS = groups_for_step(2)
CENSOR_MODES = schema.CENSOR_MODES
BASELINE = {"fast": "mean8", "regular": "ses", "slow": "ses", "rare": "mean8"}
TWEEDIE_POWERS = (1.1, 1.3, 1.5, 1.7, 1.9)
HUBER_DELTAS = (5.0, 20.0)
ML_COMPLEXITY = 10  # above every statistical method in select.BASELINE_COMPLEXITY
SUMMARY_SEEDS = (0, 1, 2)
TRIALS = 60
PRUNE_STARTUP = 10  # trials that always run to the end
PRUNE_WARMUP = 3    # origins every trial runs before it can be pruned
METRICS = ("wape", "bias", "mase", "cover90")


@dataclass(frozen=True)
class Setting:
    """One (target, loss) screen setting, before the censoring mode is chosen."""

    target: str
    loss: str
    power: float | None = None
    params: tuple[tuple[str, float], ...] = ()

    @property
    def name(self) -> str:
        tag = f"{self.power:g}" if self.power is not None else ""
        extra = "".join(f"-{k}{v:g}" for k, v in self.params)
        return f"{self.target}-{self.loss}{tag}{extra}"


SCREEN_SETTINGS: tuple[Setting, ...] = (
    Setting("raw", "l2"), Setting("raw", "poisson"),
    *(Setting("raw", "tweedie", p) for p in TWEEDIE_POWERS),
    Setting("raw", "log1p_l2"),
    *(Setting("raw", "huber", params=(("alpha", d),)) for d in HUBER_DELTAS),
    Setting("scaled", "l2"), Setting("scaled", "poisson"),
)


def tune_dir() -> Path:
    return config.ML_DIR / "tune"


def make_config(s: Setting, censor_mode: str, params: dict | None = None,
                seeds: tuple[int, ...] = (0,)) -> MLConfig:
    return MLConfig(model="lgbm", target=s.target, loss=s.loss, tweedie_power=s.power,
                    censor_mode=censor_mode, groups=GROUPS,
                    params={**dict(s.params), **(params or {})}, seeds=seeds)


def setup_name(cfg: MLConfig) -> str:
    """A readable name for a config's setup: target, loss, power/delta, censoring."""
    power = f"{cfg.tweedie_power:g}" if cfg.loss == "tweedie" else ""
    delta = f"-alpha{cfg.params['alpha']:g}" if cfg.loss == "huber" and "alpha" in cfg.params \
        else ""
    return f"{cfg.target}-{cfg.loss}{power}{delta}-{cfg.censor_mode}"


def family(cfg: MLConfig) -> tuple[str, str]:
    return cfg.target, cfg.loss


def screen_configs() -> list[MLConfig]:
    """Every screen config: each setting under each censoring mode, defaults, seed 0."""
    return [make_config(s, m) for s in SCREEN_SETTINGS for m in CENSOR_MODES]


# ---- panels and runs -------------------------------------------------------


def panel_path(censor_mode: str, groups: tuple[str, ...] = GROUPS) -> Path:
    return (tune_dir() / "panels"
            / f"panel_{censor_mode}_{'-'.join(groups)}_v{schema.PANEL_VERSION}.parquet")


def _restore(p: pd.DataFrame) -> pd.DataFrame:
    """Undo what a parquet round trip may change in the bookkeeping dtypes."""
    p = p.astype({"drug_code": "str", "origin": panel.DATETIME,
                  "target_week": panel.DATETIME})
    p["tier"] = p["tier"].astype(pd.CategoricalDtype(panel.TIERS))
    schema.validate_panel(p)
    return p


def load_panel(censor_mode: str) -> tuple[pd.DataFrame, schema.PanelContext]:
    """The A2 panel for one censoring mode on selection data, cached as parquet."""
    ctx = panel.context(split.selection_con(), split.SELECTION_END,
                        split.training_origins(split.SELECTION_END), censor_mode=censor_mode,
                        con_at=split.selection_con)
    path = panel_path(censor_mode)
    if path.exists():
        return _restore(pd.read_parquet(path)), ctx
    p = panel.build_panel(ctx, GROUPS)
    path.parent.mkdir(parents=True, exist_ok=True)
    p.to_parquet(path, index=False)
    return p, ctx


def pred_dir() -> Path:
    return tune_dir() / "predictions"


def run_config(cfg: MLConfig, origins: list[date], p: pd.DataFrame,
               ctx: schema.PanelContext, cache_dir: Path | None = None
               ) -> tuple[pd.DataFrame, float]:
    """Predictions for `cfg` on `origins`, and the seconds spent on the origins not cached."""
    t = time.time()
    pred = backtest.run(lambda: make_gbm(cfg), cfg.hash(), origins, p, ctx,
                        cache_dir=cache_dir or pred_dir())
    pred["method"] = setup_name(cfg)
    return pred, time.time() - t


def run_baseline(routing: dict[str, str], origins: list[date], p: pd.DataFrame,
                 ctx: schema.PanelContext, cache_dir: Path | None = None) -> pd.DataFrame:
    pred = backtest.run(lambda: BaselineForecaster(ctx, routing), baseline_hash(routing),
                        origins, p, ctx, cache_dir=cache_dir or pred_dir())
    pred["method"] = "baseline" if routing == BASELINE else next(iter(set(routing.values())))
    return pred


def final(pred: pd.DataFrame, seed: int = schema.ENSEMBLE_SEED) -> pd.DataFrame:
    """One config's rows for one seed (the ensemble by default)."""
    return pred[pred["seed"] == seed]


def tier_scores(pred: pd.DataFrame, seed: int = schema.ENSEMBLE_SEED) -> pd.DataFrame:
    """Per-tier WAPE, bias, MASE, cover90 and n on the selection origins."""
    s = backtest.score(pred, "selection", seed=seed, by=("tier",))
    s["tier"] = s["tier"].astype(str)
    return s[s["tier"].isin(TIERS)].set_index("tier")


def wide(scores: pd.DataFrame) -> dict[str, float]:
    """Per-tier scores as flat columns: fast_wape, fast_bias, ..."""
    return {f"{t}_{m}": float(scores.loc[t, m]) if t in scores.index else np.nan
            for t in TIERS for m in METRICS}


def objective(pred: pd.DataFrame, seed: int = schema.ENSEMBLE_SEED) -> float:
    """`select.objective` on one config's selection-origin rows for one seed."""
    sub = final(pred, seed)
    sub = sub[backtest.period_of(sub["origin"]) == "selection"]
    return select.objective(sub)


# ---- stage 1: screen -------------------------------------------------------


def _timings() -> dict:
    path = tune_dir() / "timings.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _save_timing(key: str, seconds: float) -> None:
    t = _timings()
    if seconds > 1 or key not in t:  # a cache hit takes well under a second; keep the real fit time
        t[key] = round(seconds, 1)
        path = tune_dir() / "timings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(t, indent=1, sort_keys=True) + "\n")


def screen(configs: list[MLConfig] | None = None, origins: list[date] | None = None,
           panels=load_panel, out: Path | None = None) -> pd.DataFrame:
    """Stage 1: score every screen config and the baseline; write the full table."""
    configs = configs if configs is not None else screen_configs()
    origins = origins or split.screen_origins()
    rows = []
    for mode in CENSOR_MODES:
        todo = [c for c in configs if c.censor_mode == mode]
        if not todo:
            continue
        p, ctx = panels(mode)
        if mode == "mask":
            base = run_baseline(BASELINE, origins, p, ctx)
            rows.append({"setup": "baseline", "family": "baseline", "censor_mode": "",
                         "config_hash": "baseline", **wide(tier_scores(base)),
                         "objective": objective(base), "fit_s": np.nan})
        for cfg in todo:
            pred, secs = run_config(cfg, origins, p, ctx)
            _save_timing(cfg.hash(), secs)
            rows.append({"setup": setup_name(cfg), "family": "-".join(family(cfg)),
                         "censor_mode": mode, "config_hash": cfg.hash(),
                         **wide(tier_scores(pred)), "objective": objective(pred),
                         "fit_s": _timings().get(cfg.hash(), secs)})
            print(f"{setup_name(cfg):32s} objective {rows[-1]['objective']:.4f} "
                  f"({secs:.0f}s)", flush=True)
        del p
    table = pd.DataFrame(rows).sort_values("objective").reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    out = out or tune_dir() / "screen.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, float_format="%.5f")
    return table


def pick_two(table: pd.DataFrame) -> list[str]:
    """The config hashes of the two setups to search (module docstring).

    `table` has `config_hash`, `family` and `objective`; the baseline row is
    ignored. The first is the best objective; the second the best of another
    family.
    """
    ml = table[table["config_hash"] != "baseline"].sort_values("objective", kind="stable")
    if ml.empty:
        raise ValueError("no ML rows in the screen table")
    first = ml.iloc[0]
    other = ml[ml["family"] != first["family"]]
    if other.empty:
        raise ValueError("every screen row is the same loss family")
    return [first["config_hash"], other.iloc[0]["config_hash"]]


# ---- stage 2: search -------------------------------------------------------


def suggest(trial, base: MLConfig) -> MLConfig:
    """The trial's config: `base` with the plan 7.2 search space applied."""
    params = dict(base.params)
    params.update(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 20, 1000, log=True),
        feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 30.0, log=True))
    d = {**base.to_dict(), "params": params}
    if base.loss == "tweedie":
        d["tweedie_power"] = trial.suggest_float("tweedie_variance_power", 1.05, 1.95)
    return MLConfig.from_dict(d)


def study_storage() -> str:
    return f"sqlite:///{tune_dir() / 'optuna.db'}"


def open_study(name: str, storage: str | None = None, seed: int = 0, prune: bool = True,
               pruner=None):
    """Create or load a study.

    Trials left running by a killed process are failed and their parameters
    re-queued, so the rerun hits the prediction cache. A resumed study's
    sampler is seeded with `seed` plus the trials already there, so it
    doesn't replay the first run's random start-up points. With `prune`, a
    median pruner stops a trial whose objective on the origins so far is
    worse than the median of earlier trials at the same point (`PRUNE_*`).
    """
    import optuna

    if pruner is None:
        pruner = (optuna.pruners.MedianPruner(n_startup_trials=PRUNE_STARTUP,
                                              n_warmup_steps=PRUNE_WARMUP)
                  if prune else optuna.pruners.NopPruner())
    study = optuna.create_study(study_name=name, storage=storage or study_storage(),
                                direction="minimize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=seed), pruner=pruner)
    if study.trials:
        study.sampler = optuna.samplers.TPESampler(seed=seed + len(study.trials))
    for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        study.enqueue_trial(t.params)
        study.tell(t.number, state=optuna.trial.TrialState.FAIL)
    return study


def search(base: MLConfig, origins: list[date], p: pd.DataFrame, ctx: schema.PanelContext,
           n_trials: int = TRIALS, storage: str | None = None, out: Path | None = None,
           cache_dir: Path | None = None, prune: bool = True, study_name: str | None = None,
           suggest_fn=suggest, pruner=None, enqueue: dict | None = None, n_jobs: int = 1):
    """Stage 2 for one setup: TPE until `n_trials` trials have finished; write every trial.

    A trial runs the origins in date order and reports the objective on the
    origins done so far after each one, so the pruner can stop it early.
    Finished means complete or pruned; only complete trials have the
    objective over all origins.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    name = study_name or setup_name(base)
    study = open_study(name, storage, prune=prune, pruner=pruner)
    if enqueue and not study.trials:
        study.enqueue_trial(enqueue)
    origins = sorted(origins)

    def run_trial(trial) -> float:
        cfg = suggest_fn(trial, base)
        trial.set_user_attr("config_hash", cfg.hash())
        parts, secs = [], 0.0
        for step, origin in enumerate(origins):
            pred, s = run_config(cfg, [origin], p, ctx, cache_dir)
            parts.append(pred)
            secs += s
            trial.set_user_attr("fit_s", round(secs, 1))
            trial.set_user_attr("origins_run", step + 1)
            sofar = pd.concat(parts, ignore_index=True)
            trial.report(objective(sofar), step)
            if step + 1 < len(origins) and trial.should_prune():
                print(f"{name} trial {trial.number}: pruned after {step + 1} origins "
                      f"({secs:.0f}s)", flush=True)
                raise optuna.TrialPruned()
        for k, v in wide(tier_scores(sofar)).items():
            trial.set_user_attr(k, v)
        value = objective(sofar)
        print(f"{name} trial {trial.number}: {value:.4f} ({secs:.0f}s)", flush=True)
        return value

    finished = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    done = sum(t.state in finished for t in study.trials)
    if done < n_trials:
        # n_jobs > 1 runs trials as threads in this process (LightGBM releases the
        # GIL while training). Set OMP_NUM_THREADS so they share the cores.
        study.optimize(run_trial, n_trials=n_trials - done, n_jobs=n_jobs)
    table = trials_table(study)
    out = out or tune_dir() / f"search_{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, float_format="%.5f")
    return study


def trials_table(study) -> pd.DataFrame:
    """Every finished trial: state, objective (NaN if pruned), the objective on the
    origins it ran, params, per-tier scores (complete trials), hash, fit time."""
    import optuna

    rows = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            value = t.value
        elif t.state == optuna.trial.TrialState.PRUNED:
            value = np.nan
        else:
            continue
        last = t.intermediate_values.get(max(t.intermediate_values, default=-1), np.nan)
        rows.append({"trial": t.number, "state": t.state.name.lower(), "objective": value,
                     "objective_so_far": last, **t.params, **t.user_attrs})
    return pd.DataFrame(rows).sort_values(["objective", "objective_so_far"]).reset_index(
        drop=True)


def best_config(study, base: MLConfig, seeds: tuple[int, ...] = SUMMARY_SEEDS) -> MLConfig:
    """The best trial's config, with `seeds`."""
    import optuna

    trial = optuna.trial.FixedTrial(study.best_params)
    cfg = suggest(trial, base)
    return MLConfig.from_dict({**cfg.to_dict(), "seeds": list(seeds)})


def chosen_setups(table: pd.DataFrame | None = None) -> list[MLConfig]:
    """The two screen configs `pick_two` picks from `tune/screen.csv`."""
    table = table if table is not None else pd.read_csv(tune_dir() / "screen.csv")
    by_hash = {c.hash(): c for c in screen_configs()}
    return [by_hash[h] for h in pick_two(table)]


# ---- stage 3: summary ------------------------------------------------------


def compare(ml: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Per tier: ML vs baseline WAPE difference with a 90% interval, and origin wins."""
    a, b = final(ml), final(base)
    boot = stats.paired_bootstrap(a, b, by="tier")
    _, wins = stats.origin_wins(a, b, by="tier")
    return boot.merge(wins[["tier", "origins", "a_wins", "b_wins"]], on="tier")


def choose_routing(ml_preds: dict[str, pd.DataFrame], stat_preds: dict[str, pd.DataFrame]
                   ) -> pd.DataFrame:
    """`select.choose` per tier over the ML setups and the statistical methods."""
    rows = []
    for name, pred in ml_preds.items():
        s = tier_scores(pred).reset_index()
        s["method"], s["complexity"] = name, ML_COMPLEXITY
        rows.append(s)
    for name, pred in stat_preds.items():
        s = tier_scores(pred).reset_index()
        s["method"], s["complexity"] = name, select.BASELINE_COMPLEXITY[name]
        rows.append(s)
    table = pd.concat(rows, ignore_index=True)
    return select.choose(table)


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(
            f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def finals(setups: list[tuple[str, MLConfig]], origins: list[date] | None = None,
           panels=load_panel) -> dict[str, pd.DataFrame]:
    """Each labelled config's predictions on `origins` (all 11 selection ones by default)."""
    origins = origins or split.selection_origins()
    out, by_mode = {}, {}
    for label, cfg in setups:
        by_mode.setdefault(cfg.censor_mode, []).append((label, cfg))
    for mode, items in by_mode.items():
        p, ctx = panels(mode)
        for label, cfg in items:
            pred, secs = run_config(cfg, origins, p, ctx)
            _save_timing(cfg.hash(), secs)
            pred["method"] = label
            out[label] = pred
        del p
    return {label: out[label] for label, _ in setups}


def summary(setups: list[tuple[str, MLConfig]], notes: list[str] | None = None,
            origins: list[date] | None = None, panels=load_panel,
            out_dir: Path | None = None) -> str:
    """Stage 3: labelled configs (3 seeds) vs the baseline, and the per-tier routing."""
    origins = origins or split.selection_origins()
    out_dir = out_dir or tune_dir()
    ml_preds = finals(setups, origins, panels)
    timings = _timings()
    p, ctx = panels("mask")
    base = run_baseline(BASELINE, origins, p, ctx)
    stat_preds = {m: run_baseline(dict.fromkeys(TIERS, m), origins, p, ctx)
                  for m in fc.CANDIDATES if m not in select.NOT_CANDIDATES}
    del p

    order = {t: i for i, t in enumerate(TIERS)}
    base_scores = tier_scores(base)
    parts, seed_rows = [], []
    for label, cfg in setups:
        pred = ml_preds[label]
        sc = tier_scores(pred)
        cmp_ = compare(pred, base).set_index("tier")
        for t in sc.index:
            parts.append({"setup": label, "tier": t,
                          **{m: sc.loc[t, m] for m in METRICS},
                          "base_wape": base_scores.loc[t, "wape"],
                          "base_bias": base_scores.loc[t, "bias"],
                          "base_mase": base_scores.loc[t, "mase"],
                          "base_cover90": base_scores.loc[t, "cover90"],
                          "diff": cmp_.loc[t, "diff"], "lo": cmp_.loc[t, "lo"],
                          "hi": cmp_.loc[t, "hi"],
                          "ml_wins": f"{cmp_.loc[t, 'a_wins']}/{cmp_.loc[t, 'origins']}"})
        for sd in (*cfg.seeds, schema.ENSEMBLE_SEED):
            sc_s = tier_scores(pred, sd)
            seed_rows.append({"setup": label,
                              "seed": "ensemble" if sd == schema.ENSEMBLE_SEED else sd,
                              "objective": objective(pred, sd),
                              **{f"{t}_wape": sc_s["wape"].get(t, np.nan) for t in TIERS}})
    per_tier = pd.DataFrame(parts).sort_values(["setup", "tier"], key=lambda s: s.map(order)
                                               if s.name == "tier" else s)
    seeds = pd.DataFrame(seed_rows)
    routing = choose_routing(ml_preds, stat_preds)
    routing = routing.sort_values("tier", key=lambda s: s.map(order))
    ml_tiers = [f"{t} ({c})" for t, c in zip(routing["tier"], routing["candidate"], strict=True)
                if c in ml_preds]
    lines = [
        "# LightGBM tuning summary",
        "",
        "Selection data only (to 8 Dec 2024), all 11 selection origins, A2 features, "
        "seeds 0, 1 and 2 averaged. The baseline is the gate-1 re-selection "
        f"({', '.join(f'{t} {m}' for t, m in BASELINE.items())}).",
        "",
        *(notes or []),
        "",
        f"Baseline pooled objective: {objective(base):.4f}.",
        "",
        "## By tier, ensemble vs baseline",
        "",
        "`diff` is ML WAPE minus baseline WAPE; negative means ML is better. `lo` and `hi` "
        "bound a 90% paired bootstrap over drugs: variation across this mix of drugs only, "
        "not across calendar periods. `ml_wins` counts origins where ML had the lower WAPE.",
        "",
        _md_table(per_tier),
        "",
        "## Pooled objective and WAPE by seed",
        "",
        _md_table(seeds),
        "",
        "## Routing by the plan 7.3 rule",
        "",
        "`select.choose` per tier over the ML configs above (complexity "
        f"{ML_COMPLEXITY}, above every statistical method) and every statistical method "
        "except zero.",
        "",
        _md_table(routing),
        "",
        ("ML is chosen for: " + ", ".join(ml_tiers) + ".") if ml_tiers
        else "ML is chosen for no tier.",
        "",
        "## Fit time (11 origins, 3 seeds)",
        "",
        *(f"- {label}: {timings.get(cfg.hash(), float('nan')):.0f} s" for label, cfg in setups),
        "",
    ]
    text = "\n".join(lines)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(text)
    (out_dir / "best_configs.json").write_text(json.dumps(
        {label: {"config": c.to_dict(), "config_hash": c.hash()} for label, c in setups},
        indent=1, sort_keys=True) + "\n")
    per_tier.to_csv(out_dir / "summary_by_tier.csv", index=False, float_format="%.5f")
    seeds.to_csv(out_dir / "summary_by_seed.csv", index=False, float_format="%.5f")
    routing.to_csv(out_dir / "routing.csv", index=False, float_format="%.5f")
    return text


# ---- the focused search (budget decision) ------------------------------------
#
# The full plan (60 trials per setup on 11 origins) would take about 9 hours
# at the real ~10 s per fit per origin. The lead cut it to one focused search
# on the screen winner: a narrow space, learning rate fixed at 0.1 for speed,
# 20 TPE trials on the 4 screen origins, defaults enqueued as trial 0, and a
# median pruner. The winner's params then run at learning rate 0.05 on all 11
# origins with 3 seeds against the defaults; within KEEP_DEFAULTS the defaults
# stay (simpler).

FOCUSED_TRIALS = 20
FOCUSED_LR = 0.1
KEEP_DEFAULTS = 0.005
DEFAULT_SPACE = {"num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.8}


def suggest_focused(trial, base: MLConfig) -> MLConfig:
    params = dict(base.params)
    params.update(
        learning_rate=FOCUSED_LR,
        num_leaves=trial.suggest_int("num_leaves", 31, 255, log=True),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 50, 500, log=True),
        feature_fraction=trial.suggest_float("feature_fraction", 0.5, 0.9))
    d = {**base.to_dict(), "params": params}
    if base.loss == "tweedie":
        d["tweedie_power"] = trial.suggest_float("tweedie_variance_power", 1.05, 1.4)
    return MLConfig.from_dict(d)


def focused_defaults(base: MLConfig) -> dict:
    d = dict(DEFAULT_SPACE)
    if base.loss == "tweedie":
        d["tweedie_variance_power"] = base.tweedie_power
    return d


def focused_config(base: MLConfig, best_params: dict, seeds=SUMMARY_SEEDS,
                   learning_rate: float = 0.05) -> MLConfig:
    """The focused search's best params at the final learning rate, with `seeds`."""
    cfg = suggest_focused(__import__("optuna").trial.FixedTrial(best_params), base)
    params = {**dict(cfg.params), "learning_rate": learning_rate}
    return MLConfig.from_dict({**cfg.to_dict(), "params": params, "seeds": list(seeds)})


def focused_search(base: MLConfig, origins: list[date], p: pd.DataFrame,
                   ctx: schema.PanelContext, n_trials: int = FOCUSED_TRIALS,
                   storage: str | None = None, cache_dir: Path | None = None):
    import optuna

    name = "focused_" + setup_name(base)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    return search(base, origins, p, ctx, n_trials=n_trials, storage=storage,
                  out=tune_dir() / f"search_{name}.csv", cache_dir=cache_dir,
                  study_name=name, suggest_fn=suggest_focused, pruner=pruner,
                  enqueue=focused_defaults(base))


# ---- CLI -------------------------------------------------------------------


def best_l2(table: pd.DataFrame) -> MLConfig:
    """The best raw-units L2 row of the screen: the simple runner-up."""
    by_hash = {c.hash(): c for c in screen_configs()}
    l2 = table[table["family"] == "raw-l2"].sort_values("objective")
    return by_hash[l2.iloc[0]["config_hash"]]


def winner(table: pd.DataFrame) -> MLConfig:
    by_hash = {c.hash(): c for c in screen_configs()}
    return by_hash[pick_two(table)[0]]


def with_seeds(cfg: MLConfig, seeds=SUMMARY_SEEDS) -> MLConfig:
    return MLConfig.from_dict({**cfg.to_dict(), "seeds": list(seeds)})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m irpd.ml.tune")
    sub = ap.add_subparsers(dest="stage", required=True)
    sub.add_parser("screen", help="stage 1: every loss x censoring mode on the screen origins")
    s = sub.add_parser("search", help="stage 2 (full plan): Optuna on the best two setups")
    s.add_argument("--trials", type=int, default=TRIALS)
    s.add_argument("--setup", type=int, choices=(1, 2), help="only the first or second setup")
    s.add_argument("--no-prune", action="store_true", help="run every trial on every origin")
    f = sub.add_parser("focused", help="stage 2 (budget): focused search on the screen winner")
    f.add_argument("--trials", type=int, default=FOCUSED_TRIALS)
    sub.add_parser("final", help="3-seed runs of the screen winner and best L2 (warms cache)")
    sub.add_parser("summary", help="stage 3: defaults vs focused best vs L2, 3 seeds")
    args = ap.parse_args(argv)
    if args.stage == "screen":
        table = screen()
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            print(table[["rank", "setup", "objective", "fast_wape", "fast_bias",
                         "regular_wape", "regular_bias", "slow_wape", "fit_s"]].to_string(
                index=False, float_format=lambda v: f"{v:.3f}"))
        print("search will use:", [setup_name(c) for c in chosen_setups(table)])
        return
    if args.stage == "search":
        setups = chosen_setups()
        if args.setup:
            setups = [setups[args.setup - 1]]
        for base in setups:
            p, ctx = load_panel(base.censor_mode)
            study = search(base, split.selection_origins(), p, ctx, n_trials=args.trials,
                           prune=not args.no_prune)
            print(f"{setup_name(base)}: best {study.best_value:.4f} {study.best_params}")
            del p
        return
    table = pd.read_csv(tune_dir() / "screen.csv")
    win = winner(table)
    if args.stage == "focused":
        p, ctx = load_panel(win.censor_mode)
        study = focused_search(win, split.screen_origins(), p, ctx, n_trials=args.trials)
        print(f"focused {setup_name(win)}: best {study.best_value:.4f} {study.best_params}")
    elif args.stage == "final":
        finals([("defaults", with_seeds(win)), ("l2", with_seeds(best_l2(table)))])
    else:
        import optuna

        study = optuna.load_study(study_name="focused_" + setup_name(win),
                                  storage=study_storage())
        tuned = focused_config(win, study.best_params)
        states = pd.Series([t.state.name.lower() for t in study.trials]).value_counts()
        setups = [(f"{setup_name(win)} defaults", with_seeds(win)),
                  (f"{setup_name(win)} focused", tuned),
                  (f"{setup_name(best_l2(table))} defaults", with_seeds(best_l2(table)))]
        preds = finals(setups)
        obj = {label: objective(preds[label]) for label, _ in setups}
        d, t = obj[setups[0][0]], obj[setups[1][0]]
        keep = setups[1][0] if t < d - KEEP_DEFAULTS else setups[0][0]
        notes = [
            "**Budget decision (disclosed).** Fits cost about 10 s per origin, not the "
            "pilot's 3-4 s, so the planned search (60 Optuna trials for each of two setups "
            "on all 11 origins, about 9 hours) was cut by the lead. Instead: one focused "
            f"search on the screen winner ({setup_name(win)}), {len(study.trials)} TPE "
            f"trials ({', '.join(f'{n} {k}' for k, n in states.items())}), seed 0, on "
            "the 4 screen origins, learning rate fixed at 0.1, median pruning, defaults "
            "enqueued as trial 0. Space: num_leaves [31, 255], min_data_in_leaf "
            "[50, 500], feature_fraction [0.5, 0.9], tweedie power [1.05, 1.4]; "
            "everything else at defaults. The second setup is the best raw L2 config "
            "at default params, a simple runner-up, not a searched one.",
            "",
            f"Focused best (search objective {study.best_value:.4f} on 4 origins at "
            f"lr 0.1): {study.best_params}. Rerun at learning rate 0.05.",
            "",
            f"Pooled objective on 11 origins, 3-seed ensemble: defaults {d:.4f}, "
            f"focused {t:.4f}. Rule: keep the focused params only if better by more than "
            f"{KEEP_DEFAULTS}. Kept: **{keep}**.",
        ]
        print(summary(setups, notes))


if __name__ == "__main__":
    main()
