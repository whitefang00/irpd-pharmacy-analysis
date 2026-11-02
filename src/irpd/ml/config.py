"""Experiment settings, and stable hashes to name them by.

- `MLConfig`: everything that changes one model's predictions. Its hash keys
  cached predictions and tuning trials (with the origin and
  `schema.PANEL_VERSION`).
- `FrozenSystem`: the whole forecasting system the evaluation runs: the
  model config, which tiers use it or a baseline, and the uncertainty
  method. Its hash names the frozen evaluation.

Settings must be plain JSON (no NaN or infinity). They are deep-copied and
made read-only on construction, so a hash can't drift after it is taken.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any

from irpd.ml.schema import CENSOR_MODES, GROUP_DEFS

MODELS = ("baseline", "lgbm", "xgb", "prophet")
TARGETS = ("raw", "scaled")
LOSSES = ("l2", "poisson", "tweedie", "log1p_l2", "huber")
UNCERTAINTY = ("baseline_nb", "u1_nb", "conformal")
TIERS = ("fast", "regular", "slow", "rare")


def _freeze(x: Any) -> Any:
    """A read-only deep copy of plain JSON data; refuses anything else."""
    if isinstance(x, dict | MappingProxyType):
        if not all(isinstance(k, str) for k in x):
            raise TypeError("setting keys must be strings")
        return MappingProxyType({k: _freeze(x[k]) for k in sorted(x)})
    if isinstance(x, list | tuple):
        return tuple(_freeze(v) for v in x)
    if isinstance(x, bool) or x is None or isinstance(x, str | int):
        return x
    if isinstance(x, float):
        if not math.isfinite(x):
            raise ValueError("settings can't hold NaN or infinity")
        return x
    raise TypeError(f"settings must be plain JSON, not {type(x).__name__}")


def _thaw(x: Any) -> Any:
    if isinstance(x, MappingProxyType):
        return {k: _thaw(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return [_thaw(v) for v in x]
    return x


def _digest(d: dict[str, Any]) -> str:
    body = json.dumps(d, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class MLConfig:
    model: str
    target: str = "raw"
    loss: str = "l2"
    tweedie_power: float | None = None
    censor_mode: str = "mask"
    groups: tuple[str, ...] = ("demand", "calendar")
    group_def: str = "ingredient"
    drop_features: tuple[str, ...] = ()       # single columns left out within the groups
    per_tier: bool = False                    # one model per tier instead of one global model
    retrain_every: int = 1                    # refit every n scored origins
    recency_halflife_weeks: float | None = None  # sample weights by target-week age
    params: Any = field(default_factory=dict)  # library hyperparameters (read-only after init)
    seeds: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self) -> None:
        from irpd.ml.features import GROUPS

        checks = [("model", self.model, MODELS), ("target", self.target, TARGETS),
                  ("loss", self.loss, LOSSES), ("censor_mode", self.censor_mode, CENSOR_MODES),
                  ("group_def", self.group_def, GROUP_DEFS)]
        for name, value, allowed in checks:
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, not {value!r}")
        unknown = [g for g in self.groups if g not in GROUPS]
        if unknown:
            raise ValueError(f"unknown feature groups {unknown}; known: {list(GROUPS)}")
        if (self.loss == "tweedie") != (self.tweedie_power is not None):
            raise ValueError("tweedie_power is set if and only if loss is 'tweedie'")
        if self.tweedie_power is not None and not 1 < self.tweedie_power < 2:
            raise ValueError("tweedie_power must be between 1 and 2")
        if self.retrain_every < 1:
            raise ValueError("retrain_every must be at least 1")
        if not self.seeds:
            raise ValueError("at least one seed")
        set_ = object.__setattr__
        set_(self, "groups", tuple(g for g in GROUPS if g in set(self.groups)))
        set_(self, "drop_features", tuple(sorted(set(self.drop_features))))
        set_(self, "seeds", tuple(int(s) for s in self.seeds))
        set_(self, "params", _freeze(dict(self.params)))
        if self.tweedie_power is not None:
            set_(self, "tweedie_power", float(self.tweedie_power))
        if self.recency_halflife_weeks is not None:
            set_(self, "recency_halflife_weeks", float(self.recency_halflife_weeks))

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _thaw(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MLConfig:
        """Rebuild from `to_dict`; keys left out take the constructor defaults."""
        known = {f.name for f in fields(cls)}
        extra = set(d) - known
        if extra:
            raise ValueError(f"unknown config keys {sorted(extra)}")
        return cls(**{k: tuple(v) if isinstance(v, list) and k != "params" else v
                      for k, v in d.items()})

    def hash(self) -> str:
        """First 12 hex characters of the sha256 of the canonical JSON."""
        return _digest(self.to_dict())

    def prefixes(self) -> tuple[str, ...]:
        from irpd.ml.features import GROUPS

        return tuple(GROUPS[g][0] for g in self.groups)

    def __hash__(self) -> int:
        return hash(self.hash())


@dataclass(frozen=True)
class FrozenSystem:
    """The full forecasting system an evaluation runs."""

    config: MLConfig
    routing: Any                 # tier -> "ml" or a baseline method name
    uncertainty: str = "baseline_nb"
    uncertainty_params: Any = field(default_factory=dict)
    note: str = ""

    def __post_init__(self) -> None:
        if set(self.routing) != set(TIERS):
            raise ValueError(f"routing must name every tier: {TIERS}")
        if self.uncertainty not in UNCERTAINTY:
            raise ValueError(f"uncertainty must be one of {UNCERTAINTY}")
        object.__setattr__(self, "routing", _freeze(dict(self.routing)))
        object.__setattr__(self, "uncertainty_params", _freeze(dict(self.uncertainty_params)))

    def to_dict(self) -> dict[str, Any]:
        return {"config": self.config.to_dict(), "config_hash": self.config.hash(),
                "routing": _thaw(self.routing), "uncertainty": self.uncertainty,
                "uncertainty_params": _thaw(self.uncertainty_params), "note": self.note}

    def hash(self) -> str:
        d = self.to_dict()
        d.pop("note")
        return _digest(d)

    def to_json(self) -> str:
        return json.dumps({**self.to_dict(), "system_hash": self.hash()}, indent=1,
                          sort_keys=True, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, text: str) -> FrozenSystem:
        """Parse and check a frozen file: both recorded hashes must match its contents."""
        body = json.loads(text)
        system = cls(MLConfig.from_dict(body["config"]), body["routing"], body["uncertainty"],
                     body.get("uncertainty_params", {}), body.get("note", ""))
        if system.config.hash() != body.get("config_hash"):
            raise PermissionError("config_hash doesn't match its config")
        if system.hash() != body.get("system_hash"):
            raise PermissionError("system_hash doesn't match the frozen system")
        return system
