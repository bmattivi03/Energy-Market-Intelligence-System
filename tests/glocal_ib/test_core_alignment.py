"""Verify that the SAITS_MY contrastive-target bug fix uses the real X_ori mask.

The original code in ``GlocalIB/otherModel/SAITS/core.py`` passed
``torch.ones_like(missing_mask)`` to the encoder when computing the X_ori
embedding for the contrastive loss. Since ``X_ori`` already had its NaNs
replaced with 0s in the data loader, treating those zeros as observed made
the contrastive alignment target partly noise.

The fix introduces a ``use_real_xori_mask`` flag (default True) and, when
True, computes ``xori_mask = missing_mask + indicating_mask`` - the genuine
observation mask of the ground-truth tensor.

These tests exercise the encoder call by capturing the mask argument via a
``unittest.mock.patch.object``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# vendored repo importable via tests/glocal_ib/conftest.py
# Append (not insert-at-0) so GlocalIB's ``utils``/``data`` packages do not
# shadow the project's same-named packages; ``otherModel`` is GlocalIB-only.
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT / "GlocalIB") not in sys.path:
    sys.path.append(str(_PROJECT_ROOT / "GlocalIB"))

torch = pytest.importorskip("torch")


def _build_core(use_real_xori_mask: bool = True):
    from otherModel.SAITS.core import _SAITS
    from pypots.nn.modules.loss import MAE, MSE

    return _SAITS(
        loss_type="13",
        loss_weight=[1.0, 0.0, 0.1],
        align_type="contras_1",
        n_layers=1,
        n_steps=4,
        n_features=2,
        d_model=8,
        n_heads=2,
        d_k=4,
        d_v=4,
        d_ffn=8,
        dropout=0.0,
        attn_dropout=0.0,
        diagonal_attention_mask=False,
        ORT_weight=1.0,
        MIT_weight=1.0,
        training_loss=MAE(),
        validation_metric=MSE(),
        use_real_xori_mask=use_real_xori_mask,
    )


def _make_inputs(B: int = 2, T: int = 4, N: int = 2):
    X_ori = torch.randn(B, T, N)
    # Simulate two missing positions per batch
    missing_mask = torch.ones(B, T, N)
    missing_mask[0, 0, 0] = 0
    missing_mask[1, 2, 1] = 0
    indicating_mask = torch.zeros(B, T, N)
    indicating_mask[0, 1, 1] = 1  # artificially masked but observed in X_ori
    indicating_mask[1, 3, 0] = 1
    X = X_ori * missing_mask  # zero-fill the missing positions
    return {
        "X": X,
        "X_ori": X_ori,
        "missing_mask": missing_mask,
        "indicating_mask": indicating_mask,
    }


def test_default_uses_real_xori_mask() -> None:
    """When use_real_xori_mask=True, encoder receives missing_mask + indicating_mask."""
    core = _build_core(use_real_xori_mask=True)
    core.train()
    inputs = _make_inputs()

    captured_masks: list[torch.Tensor] = []

    def hook(_module, args):
        # encoder.forward(X, missing_mask, diagonal_attention_mask)
        captured_masks.append(args[1].detach().clone())

    handle = core.encoder.register_forward_pre_hook(hook)
    try:
        core(inputs)
    finally:
        handle.remove()

    # First call: encoder on X with missing_mask. Second call: encoder on X_ori with the new mask.
    assert len(captured_masks) == 2
    expected_xori_mask = (inputs["missing_mask"] + inputs["indicating_mask"]).clamp(max=1.0)
    assert torch.allclose(captured_masks[1], expected_xori_mask)
    # And it must NOT be all ones - indicating_mask + missing_mask still has gaps
    # at positions where both are zero.
    assert not torch.allclose(captured_masks[1], torch.ones_like(captured_masks[1]))


def test_legacy_behavior_when_flag_disabled() -> None:
    """Setting use_real_xori_mask=False reproduces the original (buggy) all-ones behavior."""
    core = _build_core(use_real_xori_mask=False)
    core.train()
    inputs = _make_inputs()

    captured_masks: list[torch.Tensor] = []

    def hook(_module, args):
        # encoder.forward(X, missing_mask, diagonal_attention_mask)
        captured_masks.append(args[1].detach().clone())

    handle = core.encoder.register_forward_pre_hook(hook)
    try:
        core(inputs)
    finally:
        handle.remove()

    assert len(captured_masks) == 2
    # Legacy: the X_ori encoder pass receives all-ones
    assert torch.allclose(captured_masks[1], torch.ones_like(captured_masks[1]))


def test_no_indicating_mask_falls_back_to_ones() -> None:
    """If indicating_mask isn't provided, use_real_xori_mask gracefully falls back."""
    core = _build_core(use_real_xori_mask=True)
    core.train()
    inputs = _make_inputs()
    inputs.pop("indicating_mask")

    captured_masks: list[torch.Tensor] = []

    def hook(_module, args):
        # encoder.forward(X, missing_mask, diagonal_attention_mask)
        captured_masks.append(args[1].detach().clone())

    handle = core.encoder.register_forward_pre_hook(hook)
    try:
        core(inputs)
    finally:
        handle.remove()
    assert torch.allclose(captured_masks[1], torch.ones_like(captured_masks[1]))


def test_physical_loss_added_when_spec_provided() -> None:
    """When physical_constraints is set, the loss includes a Physical_loss term."""
    from otherModel.SAITS.physical_loss import PhysicalConstraintSpec

    spec = PhysicalConstraintSpec(
        nonneg_indices=(0,),
        nonneg_zero_in_scaled=(0.0,),
        weight=1.0,  # large weight so any imputation puts a non-zero penalty
    )
    core = _build_core(use_real_xori_mask=True)
    core.physical_constraints = spec
    core.train()
    inputs = _make_inputs()
    results = core(inputs, calc_criterion=True)
    assert "Physical_loss" in results
    assert results["Physical_loss"].requires_grad


def test_physical_loss_omitted_when_no_spec() -> None:
    core = _build_core(use_real_xori_mask=True)
    assert core.physical_constraints is None
    core.train()
    inputs = _make_inputs()
    results = core(inputs, calc_criterion=True)
    assert "Physical_loss" not in results
