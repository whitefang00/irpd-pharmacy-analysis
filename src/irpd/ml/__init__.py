"""Machine-learning demand forecasters, tested against the statistical ones in `irpd.forecast`.

The contracts every part builds on:

- `split`: the forecast origins, the selection / evaluation split and the
  guards that keep 2025 data out of every choice.
- `schema`: the panel (one row per drug, origin and week ahead), the
  context feature builders read from, and the predictions table.
- `config`: one experiment's full settings, with a stable hash.
- `models.base`: the `Forecaster` interface every model implements.
- `features`: the registry of feature groups and their column prefixes.
"""
