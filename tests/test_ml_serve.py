"""The ML forecaster the app serves: the frozen system, nothing else."""

import json

from irpd import config
from irpd.ml import serve


def test_serving_config_is_the_frozen_one():
    assert serve.frozen_config().hash() == serve.CONFIG_HASH
    frozen = config.ROOT / "lab/ml-frozen.json"
    if frozen.exists():  # lab/ is private, so only on the dev checkout
        assert json.loads(frozen.read_text())["config_hash"] == serve.CONFIG_HASH
