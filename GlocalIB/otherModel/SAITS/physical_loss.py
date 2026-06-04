"""Auxiliary physical-bound loss for SAITS-style imputers.

Adds a soft-hinge penalty on imputed values that violate physical bounds:

* **Non-negativity** for generation columns (e.g. ``gen_wind_onshore`` cannot
  be < 0).
* **Flow sign coherence** for cross-border flow pairs (``A_to_B`` and
  ``B_to_A`` cannot both be > 0 simultaneously).

Time-position-aware constraints (e.g. ``gen_solar = 0`` at night) are NOT
included here because they would require per-batch timestamp metadata which
SAITS does not currently propagate. Those constraints are enforced post-hoc
via ``src/preprocessing/physical_constraints.py``.

The penalty is computed in the *scaled* space (matching the rest of SAITS
training) - callers must convert their physical constraint specs to scaled
units, or rely on the fact that ``relu(-x_scaled)`` is monotonic in
``relu(-x_unscaled)`` for non-negativity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class PhysicalConstraintSpec:
    """Per-feature physical-bound spec referenced by feature *index*."""

    # Indices of features that must be ≥ 0 (in scaled space - see note below).
    nonneg_indices: tuple[int, ...] = ()
    # Per-feature scaled value of "0" - the lower bound after StandardScaler.
    # If a column was scaled with mean μ and std σ, then the scaled lower bound
    # is -μ/σ. Callers compute and pass these.
    nonneg_zero_in_scaled: tuple[float, ...] = ()
    # Index pairs (a, b) where ``a`` and ``b`` cannot both be > 0.
    flow_pairs: tuple[tuple[int, int], ...] = ()
    flow_zero_in_scaled: tuple[tuple[float, float], ...] = ()
    weight: float = 0.05


def physical_constraint_loss(
    imputed_data: torch.Tensor,
    spec: PhysicalConstraintSpec,
) -> torch.Tensor:
    """Compute the soft-hinge physical-constraint penalty.

    Parameters
    ----------
    imputed_data : Tensor of shape [B, T, N] in scaled space.
    spec : :class:`PhysicalConstraintSpec` with the indices and scaled zeros.

    Returns
    -------
    Scalar tensor (already multiplied by ``spec.weight``). If the spec is
    empty, returns 0 on the same device.
    """
    if (not spec.nonneg_indices) and (not spec.flow_pairs):
        return imputed_data.new_zeros(())

    losses = []

    for idx, scaled_zero in zip(spec.nonneg_indices, spec.nonneg_zero_in_scaled):
        # Penalty proportional to amount below scaled_zero
        below = torch.relu(scaled_zero - imputed_data[..., idx])
        losses.append(below.mean())

    for (a, b), (za, zb) in zip(spec.flow_pairs, spec.flow_zero_in_scaled):
        # min(positive_a, positive_b) > 0 ⇒ violation. Penalty is the min.
        pos_a = torch.relu(imputed_data[..., a] - za)
        pos_b = torch.relu(imputed_data[..., b] - zb)
        losses.append(torch.minimum(pos_a, pos_b).mean())

    if not losses:
        return imputed_data.new_zeros(())

    return spec.weight * torch.stack(losses).sum()


def build_spec_from_scaler(
    column_names: list[str],
    scaler_mean: torch.Tensor | "np.ndarray",  # type: ignore[name-defined]
    scaler_scale: torch.Tensor | "np.ndarray",  # type: ignore[name-defined]
    nonneg_columns: tuple[str, ...],
    flow_pairs: tuple[tuple[str, str], ...],
    weight: float = 0.05,
) -> PhysicalConstraintSpec:
    """Convenience builder mapping *named* constraints to scaled indices.

    ``scaler_mean`` and ``scaler_scale`` come from the
    ``preprocessing.impute.NaNScaler`` fitted on training data. The scaled
    zero of column ``c`` is ``-mean[c] / scale[c]``.
    """
    import numpy as np

    name_to_idx = {n: i for i, n in enumerate(column_names)}
    nonneg_idx, nonneg_zero = [], []
    for c in nonneg_columns:
        if c in name_to_idx:
            i = name_to_idx[c]
            nonneg_idx.append(i)
            nonneg_zero.append(float(-scaler_mean[i] / scaler_scale[i]))
    pairs_idx, pairs_zero = [], []
    for a, b in flow_pairs:
        if a in name_to_idx and b in name_to_idx:
            ai, bi = name_to_idx[a], name_to_idx[b]
            pairs_idx.append((ai, bi))
            pairs_zero.append((
                float(-scaler_mean[ai] / scaler_scale[ai]),
                float(-scaler_mean[bi] / scaler_scale[bi]),
            ))
    return PhysicalConstraintSpec(
        nonneg_indices=tuple(nonneg_idx),
        nonneg_zero_in_scaled=tuple(nonneg_zero),
        flow_pairs=tuple(pairs_idx),
        flow_zero_in_scaled=tuple(pairs_zero),
        weight=weight,
    )
