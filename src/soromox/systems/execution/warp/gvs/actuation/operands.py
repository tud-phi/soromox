"""Runtime contracts for GVS Warp threadlike actuation."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
from jax import Array

from soromox.systems.execution.warp.actuation.threadlike import (
    LinearThreadlikeActuationData,
)
from soromox.systems.execution.warp.gvs.operands import GVSOperandSource


class GVSThreadlikeOperands(eqx.Module):
    """Runtime operands for shape-generic GVS linear threadlike actuation."""

    num_segments: int = eqx.field(static=True)
    num_dofs: int = eqx.field(static=True)
    max_dof: int = eqx.field(static=True)
    num_quadrature: int = eqx.field(static=True)
    scale_rotational_basis_by_length: bool = eqx.field(static=True)
    routing: LinearThreadlikeActuationData
    link_local_to_global: Array
    link_basis: Array
    reference_strain: Array
    integration_points: Array
    integration_weights: Array
    segment_lengths: Array
    segment_starts: Array
    global_eps: float

    @classmethod
    def from_model(cls, model: GVSOperandSource) -> GVSThreadlikeOperands:
        """Build GVS body/routing operands while retaining runtime dimensions."""

        routing = LinearThreadlikeActuationData.from_model(model)
        return cls(
            num_segments=model.num_segments,
            num_dofs=model.num_dofs,
            max_dof=model.max_dof,
            num_quadrature=model.max_num_integration_points - 2,
            scale_rotational_basis_by_length=model.scale_rotational_basis_by_length,
            routing=routing,
            link_local_to_global=model.link_local_to_global,
            link_basis=model.B_Xs[:, 1 : model.max_num_integration_points - 1],
            reference_strain=model.xi_ref_Xs[
                :, 1 : model.max_num_integration_points - 1
            ],
            integration_points=model.integration_points[
                :, 1 : model.max_num_integration_points - 1
            ],
            integration_weights=model.inner_integration_weights,
            segment_lengths=model.segment_lengths,
            segment_starts=model.segment_end_positions[:-1],
            global_eps=model.global_eps,
        )


@dataclass(frozen=True)
class GVSThreadlikeShapes:
    """Allocation contract for batched GVS threadlike matrix/force launches."""

    batch_size: int
    num_dofs: int
    num_paths: int

    @classmethod
    def from_operands(
        cls, operands: GVSThreadlikeOperands, *, batch_size: int
    ) -> GVSThreadlikeShapes:
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


__all__ = ["GVSThreadlikeOperands", "GVSThreadlikeShapes"]
