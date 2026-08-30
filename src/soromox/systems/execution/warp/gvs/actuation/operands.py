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
    """Runtime operands for shape-generic GVS linear threadlike actuation.

    Attributes:
        num_segments: Number of serial joint-link segments.
        num_dofs: Number of active generalized coordinates.
        max_dof: Padded link-local coordinate width.
        num_quadrature: Interior quadrature points per segment.
        routing: Ordered linear-routing arrays and coordinate scales.
        link_local_to_global: Padded link-local to active-coordinate map.
        link_global_to_local: Active-coordinate to link-local map.
        link_basis: Length-scaled link basis at interior quadrature points.
        reference_strain: Reference link strains at quadrature points.
        integration_weights: Physical quadrature weights for every segment.
        physical_points: Global backbone coordinates of quadrature points.
        segment_lengths: Physical length of every segment.
        segment_starts: Cumulative segment-start coordinates.
        global_eps: Safe-normalization threshold shared with the JAX model.
    """

    num_segments: int = eqx.field(static=True)
    num_dofs: int = eqx.field(static=True)
    max_dof: int = eqx.field(static=True)
    num_quadrature: int = eqx.field(static=True)
    routing: LinearThreadlikeActuationData
    link_local_to_global: Array
    link_global_to_local: Array
    link_basis: Array
    reference_strain: Array
    integration_weights: Array
    physical_points: Array
    segment_lengths: Array
    segment_starts: Array
    global_eps: float

    @classmethod
    def from_model(cls, model: GVSOperandSource) -> GVSThreadlikeOperands:
        """Build threadlike operands from a GVS-compatible model.

        Args:
            model: Model providing the GVS body, coordinate-map, quadrature,
                and linear-threadlike routing data.

        Returns:
            Operand bundle with ordered routing arrays, physical quadrature
            data, and any required length scaling applied to the link basis.
        """

        routing = LinearThreadlikeActuationData.from_model(model)
        link_basis = model.B_Xs[:, 1 : model.max_num_integration_points - 1]
        if model.scale_rotational_basis_by_length:
            link_basis = link_basis.at[:, :, :3, :].divide(
                model.segment_lengths[:, None, None, None]
            )
        integration_points = model.integration_points[
            :, 1 : model.max_num_integration_points - 1
        ]
        segment_starts = model.segment_end_positions[:-1]
        return cls(
            num_segments=model.num_segments,
            num_dofs=model.num_velocities,
            max_dof=model.max_dof,
            num_quadrature=model.max_num_integration_points - 2,
            routing=routing,
            link_local_to_global=model.link_local_to_global,
            link_global_to_local=model.link_global_to_local,
            link_basis=link_basis,
            reference_strain=model.xi_ref_Xs[
                :, 1 : model.max_num_integration_points - 1
            ],
            integration_weights=model.inner_integration_weights,
            physical_points=(
                segment_starts[:, None]
                + integration_points * model.segment_lengths[:, None]
            ),
            segment_lengths=model.segment_lengths,
            segment_starts=segment_starts,
            global_eps=model.global_eps,
        )


@dataclass(frozen=True)
class GVSThreadlikeShapes:
    """Describe caller-owned arrays for GVS threadlike launches.

    Attributes:
        batch_size: Number of independent environments.
        num_segments: Number of serial joint-link segments.
        num_dofs: Number of active generalized coordinates.
        num_paths: Number of ordered threadlike paths.
        num_quadrature: Interior quadrature points per segment.
    """

    batch_size: int
    num_segments: int
    num_dofs: int
    num_paths: int
    num_quadrature: int

    @classmethod
    def from_operands(
        cls, operands: GVSThreadlikeOperands, *, batch_size: int
    ) -> GVSThreadlikeShapes:
        """Construct output shapes from public runtime operands.

        Args:
            operands: Runtime dimensions prepared by
                :meth:`GVSThreadlikeOperands.from_model`.
            batch_size: Positive number of independent environments.

        Returns:
            Immutable GVS threadlike workspace and output-shape contract.

        Raises:
            TypeError: If ``batch_size`` is not an integer or is a boolean.
            ValueError: If ``batch_size`` is not positive.
        """

        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        return cls(
            batch_size,
            operands.num_segments,
            operands.num_dofs,
            operands.routing.num_paths,
            operands.num_quadrature,
        )

    def strain_workspace(self) -> tuple[int, int, int, int]:
        """Return the caller-owned quadrature-strain workspace shape.

        Returns:
            Shape ``(batch_size, num_segments, num_quadrature, 6)``.
        """

        return self.batch_size, self.num_segments, self.num_quadrature, 6

    def matrix_output(self) -> tuple[int, int, int]:
        """Return the caller-owned batched matrix output shape.

        Returns:
            Shape ``(batch_size, num_dofs, num_paths)``.
        """

        return self.batch_size, self.num_dofs, self.num_paths

    def force_output(self) -> tuple[int, int]:
        """Return the caller-owned batched generalized-force output shape.

        Returns:
            Shape ``(batch_size, num_dofs)``.
        """

        return self.batch_size, self.num_dofs

    def matrix_outputs(self) -> dict[str, tuple[int, ...]]:
        """Return named output shapes for the matrix callable.

        Returns:
            Mapping for the caller-owned strain workspace and matrix output.
        """

        return {
            "strain_workspace": self.strain_workspace(),
            "output": self.matrix_output(),
        }

    def force_outputs(self) -> dict[str, tuple[int, ...]]:
        """Return named output shapes for the fused-force callable.

        Returns:
            Mapping for the caller-owned strain workspace and force output.
        """

        return {
            "strain_workspace": self.strain_workspace(),
            "output": self.force_output(),
        }


__all__ = ["GVSThreadlikeOperands", "GVSThreadlikeShapes"]
