"""The physical-constraint aux loss must fire iff physical_constraints is set,
and must compose with the VIB (mod_e) core. Regression guard for the fork merge."""
import sys
import pathlib

import torch

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_GLOCAL = _ROOT / "GlocalIB"
if str(_GLOCAL) not in sys.path:
    sys.path.insert(0, str(_GLOCAL))

from pypots.nn.modules.loss import MAE, MSE  # noqa: E402

from otherModel.SAITS.core import _SAITS  # noqa: E402
from otherModel.SAITS.physical_loss import PhysicalConstraintSpec  # noqa: E402


def _make_core(physical_constraints):
    return _SAITS(
        loss_type="13",
        loss_weight=[1.0, 0.0, 0.1],
        align_type="contras_1",
        n_layers=1,
        n_steps=8,
        n_features=3,
        d_model=16,
        n_heads=2,
        d_k=8,
        d_v=8,
        d_ffn=16,
        dropout=0.0,
        attn_dropout=0.0,
        diagonal_attention_mask=True,
        ORT_weight=1.0,
        MIT_weight=1.0,
        training_loss=MAE(),
        validation_metric=MSE(),
        mod_e=1,  # VIB ON — must compose with the physical hook
        physical_constraints=physical_constraints,
    )


def _batch():
    torch.manual_seed(0)
    X = torch.randn(2, 8, 3)
    mask = (torch.rand(2, 8, 3) > 0.3).float()
    ind = (torch.rand(2, 8, 3) > 0.7).float()
    return {
        "X": X * mask,
        "missing_mask": mask,
        "X_ori": X,
        "indicating_mask": ind,
    }


def test_physical_loss_absent_when_spec_none():
    core = _make_core(None).train()
    out = core(_batch(), calc_criterion=True)
    assert "Physical_loss" not in out or out.get("Physical_loss") is None


def test_physical_loss_fires_when_spec_set():
    spec = PhysicalConstraintSpec(
        nonneg_indices=(0, 1),
        nonneg_zero_in_scaled=(0.0, 0.0),
        weight=0.5,
    )
    core = _make_core(spec).train()
    out = core(_batch(), calc_criterion=True)
    assert "Physical_loss" in out
    assert torch.is_tensor(out["Physical_loss"])
    assert out["loss"].requires_grad
