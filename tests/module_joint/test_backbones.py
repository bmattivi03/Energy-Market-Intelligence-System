import torch

from module_joint.config import JointConfig
from module_joint.backbones import (
    DLinearBackbone,
    NHiTSxBackbone,
    TiDEBackbone,
    build_backbone,
)

LOOKBACK = 168
NFEAT = 50


def _check(bb):
    out_dim = bb.out_dim
    for B in (1, 4):
        z = bb(torch.randn(B, LOOKBACK, NFEAT))
        assert z.shape == (B, out_dim)
        assert torch.all(torch.isfinite(z))


def test_tide_backbone():
    _check(TiDEBackbone(NFEAT, LOOKBACK, hidden=64, layers=2, dropout=0.1))


def test_dlinear_backbone():
    _check(DLinearBackbone(NFEAT, LOOKBACK, hidden=64))


def test_nhitsx_backbone():
    _check(NHiTSxBackbone(NFEAT, LOOKBACK, hidden=64))


def test_build_backbone_dispatch():
    cfg = JointConfig(hidden=64)
    for name in ("tide", "dlinear", "nhitsx"):
        bb = build_backbone(name, NFEAT, LOOKBACK, cfg)
        assert bb.out_dim == 64
