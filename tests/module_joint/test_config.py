import torch

from data import schemas
from module_joint.config import (
    ALPHA,
    HORIZON,
    LOAD,
    LOOKBACK,
    PRICE,
    QUANTILE_COLS,
    QUANTILES,
    JointConfig,
    select_device,
    set_seed,
)


def test_constants():
    assert HORIZON == 24 and LOOKBACK == 168
    assert QUANTILES == (0.1, 0.5, 0.9)
    assert QUANTILE_COLS == ("q10", "q50", "q90")
    assert ALPHA == 0.20


def test_target_names_match_schemas():
    assert LOAD == schemas.LOAD_COL
    assert PRICE == schemas.PRICE_COL


def test_config_defaults():
    c = JointConfig()
    assert c.lookback == 168 and c.horizon == 24
    assert c.seed == 42 and c.hidden > 0
    assert c.backbone == "tide"
    assert c.task_weights == {"load": 1.0, "price": 1.0}


def test_set_seed_reproducible():
    set_seed(7)
    a = torch.randn(5)
    set_seed(7)
    b = torch.randn(5)
    assert torch.allclose(a, b)


def test_select_device_force_cpu():
    assert select_device(force_cpu=True).type == "cpu"
