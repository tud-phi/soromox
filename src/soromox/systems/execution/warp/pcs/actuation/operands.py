"""Runtime contracts for PlanarPCS/PCS Warp threadlike actuation."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
from jax import Array

from soromox.systems.execution.warp.actuation.threadlike import (
    LinearThreadlikeActuationData,
)
from soromox.systems.execution.warp.pcs.operands import PCSOperandSource


class PCSThreadlikeOperands(eqx.Module):
    """Runtime operands for PlanarPCS/PCS linear threadlike actuation.

    Array fields are JAX-owned model/routing data. External Warp-native callers
    may create zero-copy Warp views once and retain them together with their
    caller-owned output buffers.
    """

    is_planar: bool = eqx.field(static=True)
    num_segments: int = eqx.field(static=True)
    num_dofs: int = eqx.field(static=True)
    num_quadrature: int = eqx.field(static=True)
    routing: LinearThreadlikeActuationData
    active_strain_indices: Array
    active_strain_scales: Array
    reference_strain: Array
    local_points: Array
    quadrature_weights: Array
    segment_lengths: Array
    segment_starts: Array
    global_eps: float

    @classmethod
    def from_model(cls, model: PCSOperandSource) -> PCSThreadlikeOperands:
        """Build the ordered body/routing operand bundle for one PCS model."""

        routing = LinearThreadlikeActuationData.from_model(model)
        return cls(
            is_planar=model.is_planar,
            num_segments=model.num_segments,
            num_dofs=model.num_dofs,
            num_quadrature=model.num_gauss_points,
            routing=routing,
            active_strain_indices=model.active_strain_indices,
            active_strain_scales=model.active_strain_scales,
            reference_strain=model.xi_ref,
            local_points=model.dynamics_local_points[:, : model.num_gauss_points],
            quadrature_weights=model.integration_weights[1:-1],
            segment_lengths=model.L,
            segment_starts=model.L_cum[:-1],
            global_eps=model.global_eps,
        )


@dataclass(frozen=True)
class PCSThreadlikeShapes:
    """Allocation contract for batched PCS threadlike matrix/force launches."""

    batch_size: int
    num_dofs: int
    num_paths: int

    @classmethod
    def from_operands(
        cls, operands: PCSThreadlikeOperands, *, batch_size: int
    ) -> PCSThreadlikeShapes:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        return cls(batch_size, operands.num_dofs, operands.routing.num_paths)

    def matrix_output(self) -> tuple[int, int, int]:
        """Return the caller-owned batched matrix output shape."""

        return self.batch_size, self.num_dofs, self.num_paths

    def force_output(self) -> tuple[int, int]:
        """Return the caller-owned batched generalized-force output shape."""

        return self.batch_size, self.num_dofs


__all__ = ["PCSThreadlikeOperands", "PCSThreadlikeShapes"]
