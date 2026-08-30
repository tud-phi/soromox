"""Explicit runtime data contract for PlanarPCS and PCS Warp execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import equinox as eqx
from jax import Array

from soromox.execution.config import PCSBackendParams


class PCSOperandSource(Protocol):
    """Structural source of precomputed PCS Warp operands.

    Both PlanarPCS and PCS satisfy this contract. Keeping the contract in the
    execution package avoids importing either concrete class and lets their
    constant-strain pipelines share orchestration without circular imports.
    """

    is_planar: bool
    floating_base: bool
    num_segments: int
    num_internal_dofs: int
    num_velocities: int
    num_dofs: int
    num_gauss_points: int
    backend_params: PCSBackendParams
    active_strain_indices: Array
    active_strain_scales: Array
    xi_ref: Array
    dynamics_local_points: Array
    integration_weights: Array
    active_dof_ends: Array
    inertia_upper_rows: Array
    inertia_upper_columns: Array
    weighted_mass_diagonals: Array
    gravity_base: Array
    g: Array
    global_eps: float
    tangent_eps: float
    L: Array
    L_cum: Array
    fixed_base_pose: Array | None
    g0: Array


class PCSOperands(eqx.Module):
    """Minimal constant-strain data passed to a PCS Warp executor.

    Static fields select planar versus spatial kernels and determine launch
    shapes. Array fields are runtime model data shared with JAX, allowing one
    executor interface to cover PlanarPCS and PCS without embedding a model
    object in the Warp layer.

    External Warp-native integrations may convert the array fields to zero-copy
    Warp views with :func:`warp.from_jax` during solver construction and retain
    them for steady-state stepping. The object itself performs no allocation or
    device transfer.

    Attributes:
        is_planar: Whether the operands describe PlanarPCS (SE(2)) rather than
            spatial PCS (SE(3)).
        num_segments: Number of serial constant-strain segments.
        num_dofs: Number of active generalized coordinates.
        num_gauss_points: Quadrature points per segment.
        block_dim: Cooperative lanes per persistent chain block.
        active_strain_indices: Active-coordinate index of every full strain
            component, with negative entries for inactive components.
        active_strain_scales: Length-normalization scale of each strain row.
        reference_strain: Reference strain of every segment.
        local_points: Segment-local operator and quadrature coordinates.
        active_dof_ends: Cumulative active coordinate count after each segment.
        inertia_upper_rows: Row indices for packed upper inertia assembly.
        inertia_upper_columns: Columns matching ``inertia_upper_rows``.
        weighted_mass_diagonals: Quadrature-weighted diagonal spatial inertias.
        gravity_base: Gravity acceleration expressed in the base frame.
        global_eps: Small-angle threshold used by planar exponential operators.
        tangent_eps: Small-angle derivative threshold used by planar operators.
    """

    is_planar: bool = eqx.field(static=True)
    num_segments: int = eqx.field(static=True)
    num_dofs: int = eqx.field(static=True)
    num_gauss_points: int = eqx.field(static=True)
    block_dim: int = eqx.field(static=True)
    active_strain_indices: Array
    active_strain_scales: Array
    reference_strain: Array
    local_points: Array
    active_dof_ends: Array
    inertia_upper_rows: Array
    inertia_upper_columns: Array
    weighted_mass_diagonals: Array
    gravity_base: Array
    global_eps: float
    tangent_eps: float

    @classmethod
    def from_model(cls, model: PCSOperandSource) -> PCSOperands:
        """Build an operand view over a precomputed PCS-compatible model.

        Args:
            model: Planar or spatial PCS model satisfying
                :class:`PCSOperandSource`.

        Returns:
            Operand bundle referencing the model's existing arrays. No
            numerical preprocessing or array copy is performed here.
        """

        if model.floating_base:
            raise ValueError("Fixed PCS Warp operands require a fixed-base model.")
        return cls(
            is_planar=model.is_planar,
            num_segments=model.num_segments,
            num_dofs=model.num_internal_dofs,
            num_gauss_points=model.num_gauss_points,
            block_dim=model.backend_params.warp_block_dim,
            active_strain_indices=model.active_strain_indices,
            active_strain_scales=model.active_strain_scales,
            reference_strain=model.xi_ref,
            local_points=model.dynamics_local_points,
            active_dof_ends=model.active_dof_ends,
            inertia_upper_rows=model.inertia_upper_rows,
            inertia_upper_columns=model.inertia_upper_columns,
            weighted_mass_diagonals=model.weighted_mass_diagonals,
            gravity_base=model.gravity_base,
            global_eps=model.global_eps,
            tangent_eps=model.tangent_eps,
        )


class PCSFloatingOperands(eqx.Module):
    """Runtime data for augmented floating PlanarPCS and PCS dynamics.

    The local constant-strain arrays are identical to the fixed-base operand
    contract. Separate static dimensions and world gravity allow the floating
    entry kernels to prepend root columns without adding base operands or
    branches to the fixed kernel signature.

    Attributes:
        is_planar: Whether to use the SE(2) rather than SE(3) kernels.
        num_segments: Number of serial constant-strain segments.
        num_internal_dofs: Number of material generalized coordinates.
        num_velocities: Total augmented generalized-velocity dimension.
        num_gauss_points: Quadrature points per segment.
        block_dim: Cooperative lanes per persistent chain block.
        active_strain_indices: Internal-coordinate index of every strain row.
        active_strain_scales: Scale applied to every active strain row.
        reference_strain: Reference strain of every segment.
        local_points: Segment-local operator and quadrature coordinates.
        active_dof_ends: Cumulative internal-coordinate count by segment.
        inertia_upper_rows: Row indices for augmented upper-inertia assembly.
        inertia_upper_columns: Columns matching ``inertia_upper_rows``.
        weighted_mass_diagonals: Quadrature-weighted local inertia diagonals.
        gravity_world: Spatial gravity acceleration in the inertial frame.
        global_eps: Small-angle threshold for planar exponential operators.
        tangent_eps: Small-angle derivative threshold for planar operators.
    """

    is_planar: bool = eqx.field(static=True)
    num_segments: int = eqx.field(static=True)
    num_internal_dofs: int = eqx.field(static=True)
    num_velocities: int = eqx.field(static=True)
    num_gauss_points: int = eqx.field(static=True)
    block_dim: int = eqx.field(static=True)
    active_strain_indices: Array
    active_strain_scales: Array
    reference_strain: Array
    local_points: Array
    active_dof_ends: Array
    inertia_upper_rows: Array
    inertia_upper_columns: Array
    weighted_mass_diagonals: Array
    gravity_world: Array
    global_eps: float
    tangent_eps: float

    @classmethod
    def from_model(cls, model: PCSOperandSource) -> PCSFloatingOperands:
        """Build floating operands over a precomputed PCS-compatible model.

        Args:
            model: Floating PlanarPCS or PCS model satisfying
                :class:`PCSOperandSource`.

        Returns:
            Operand bundle referencing the model's existing arrays without
            numerical preprocessing or copies.

        Raises:
            ValueError: If ``model`` is fixed-base.
        """
        if not model.floating_base:
            raise ValueError("Floating PCS Warp operands require a floating model.")
        return cls(
            is_planar=model.is_planar,
            num_segments=model.num_segments,
            num_internal_dofs=model.num_internal_dofs,
            num_velocities=model.num_velocities,
            num_gauss_points=model.num_gauss_points,
            block_dim=model.backend_params.warp_block_dim,
            active_strain_indices=model.active_strain_indices,
            active_strain_scales=model.active_strain_scales,
            reference_strain=model.xi_ref,
            local_points=model.dynamics_local_points,
            active_dof_ends=model.active_dof_ends,
            inertia_upper_rows=model.inertia_upper_rows,
            inertia_upper_columns=model.inertia_upper_columns,
            weighted_mass_diagonals=model.weighted_mass_diagonals,
            gravity_world=model.g,
            global_eps=model.global_eps,
            tangent_eps=model.tangent_eps,
        )


@dataclass(frozen=True)
class PCSPipelineShapes:
    """Describe preallocated arrays for a planar or spatial PCS Warp pipeline.

    Attributes:
        batch_size: Number of independent environments.
        num_segments: Number of serial PCS segments.
        num_dofs: Number of active generalized coordinates.
        num_gauss_points: Quadrature points per segment.
        spatial_dim: Three for PlanarPCS and six for spatial PCS.
    """

    batch_size: int
    num_segments: int
    num_dofs: int
    num_gauss_points: int
    spatial_dim: int

    @classmethod
    def from_operands(
        cls, operands: PCSOperands, *, batch_size: int
    ) -> PCSPipelineShapes:
        """Construct pipeline shapes from public runtime operands.

        Args:
            operands: Runtime dimensions and model data prepared by
                :meth:`PCSOperands.from_model`.
            batch_size: Positive number of independent environments.

        Returns:
            Immutable planar or spatial PCS allocation contract.

        Raises:
            TypeError: If ``batch_size`` is not an integer or is a boolean.
            ValueError: If ``batch_size`` is not positive.
        """

        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        return cls(
            batch_size=batch_size,
            num_segments=operands.num_segments,
            num_dofs=operands.num_dofs,
            num_gauss_points=operands.num_gauss_points,
            spatial_dim=3 if operands.is_planar else 6,
        )

    def operator_outputs(self) -> dict[str, tuple[int, ...]]:
        """Return output shapes for the dimension-specific operator launch.

        Returns:
            Mapping from local-operator output names to Warp array shapes.
        """

        rows = (
            self.batch_size
            * self.num_segments
            * (self.num_gauss_points + 1)
            * self.spatial_dim
        )
        return {
            "adjoint_inverse": (rows, self.spatial_dim),
            "transported_tangent": (rows, self.spatial_dim),
            "local_velocity": (rows, 1),
            "transported_tangent_dot_velocity": (rows, 1),
        }

    def chain_outputs(self) -> dict[str, tuple[int, ...]]:
        """Return workspace and result shapes for the persistent chain.

        Returns:
            Mapping from dimension-specific chain output names to Warp array
            shapes. Ping-pong state arrays precede final dynamics outputs.
        """

        rows = self.batch_size * self.spatial_dim
        outputs = {
            "jacobian_first": (rows, self.num_dofs),
            "derivative_first": (rows, 1),
            "gravity_first": (rows, 1),
            "jacobian_second": (rows, self.num_dofs),
            "derivative_second": (rows, 1),
            "gravity_second": (rows, 1),
        }
        if self.spatial_dim == 6:
            outputs = {
                "jacobian_first": (rows, self.num_dofs),
                "derivative_first": (rows, 1),
                "velocity_first": (rows, 1),
                "gravity_first": (rows, 1),
                "jacobian_second": (rows, self.num_dofs),
                "derivative_second": (rows, 1),
                "velocity_second": (rows, 1),
                "gravity_second": (rows, 1),
            }
        outputs.update(
            {
                "inertia": (self.batch_size, self.num_dofs, self.num_dofs),
                "coriolis_qd": (self.batch_size, self.num_dofs),
                "gravity_force": (self.batch_size, self.num_dofs),
            }
        )
        return outputs


@dataclass(frozen=True)
class PCSFloatingPipelineShapes(PCSPipelineShapes):
    """Preallocated array shapes for an augmented floating PCS pipeline."""

    @classmethod
    def from_operands(
        cls, operands: PCSFloatingOperands, *, batch_size: int
    ) -> PCSFloatingPipelineShapes:
        """Construct augmented pipeline shapes from floating operands.

        Args:
            operands: Runtime dimensions and data prepared by
                :meth:`PCSFloatingOperands.from_model`.
            batch_size: Positive number of independent environments.

        Returns:
            Immutable allocation contract using the total velocity dimension.

        Raises:
            TypeError: If ``batch_size`` is not an integer or is a boolean.
            ValueError: If ``batch_size`` is not positive.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        return cls(
            batch_size=batch_size,
            num_segments=operands.num_segments,
            num_dofs=operands.num_velocities,
            num_gauss_points=operands.num_gauss_points,
            spatial_dim=3 if operands.is_planar else 6,
        )


class PCSKinematicsOperands(eqx.Module):
    """Runtime model data required by public PCS kinematics launchers.

    Attributes:
        is_planar: Select PlanarPCS or spatial PCS storage and kernels.
        num_segments: Number of constant-strain segments.
        num_dofs: Number of active generalized strains.
        block_dim: Preferred Warp block size.
        active_strain_indices: Active-coordinate index for each full strain row.
        active_strain_scales: Scale matching every active strain row.
        reference_strain: Model reference strain.
        segment_lengths: Physical segment lengths.
        segment_starts: Cumulative segment-start coordinates.
        base_pose: Planar base pose or spatial base transform.
        global_eps: Exponential-map threshold.
        tangent_eps: Tangent-map threshold.
    """

    is_planar: bool = eqx.field(static=True)
    num_segments: int = eqx.field(static=True)
    num_dofs: int = eqx.field(static=True)
    block_dim: int = eqx.field(static=True)
    active_strain_indices: Array
    active_strain_scales: Array
    reference_strain: Array
    segment_lengths: Array
    segment_starts: Array
    base_pose: Array
    global_eps: float
    tangent_eps: float

    @classmethod
    def from_model(cls, model: PCSOperandSource) -> PCSKinematicsOperands:
        """Create an allocation-free view of PlanarPCS or PCS model arrays.

        Args:
            model: Object satisfying :class:`PCSOperandSource`.

        Returns:
            Operand bundle referencing the model's existing JAX arrays.
        """

        base_pose = model.fixed_base_pose if model.is_planar else model.g0
        if base_pose is None:
            raise ValueError("Fixed PCS Warp operands require fixed_base_pose.")
        return cls(
            is_planar=model.is_planar,
            num_segments=model.num_segments,
            num_dofs=model.num_internal_dofs,
            block_dim=model.backend_params.warp_block_dim,
            active_strain_indices=model.active_strain_indices,
            active_strain_scales=model.active_strain_scales,
            reference_strain=model.xi_ref,
            segment_lengths=model.L,
            segment_starts=model.L_cum,
            base_pose=base_pose,
            global_eps=model.global_eps,
            tangent_eps=model.tangent_eps,
        )


class PCSFloatingKinematicsOperands(eqx.Module):
    """Floating PCS kinematics specialization over fixed relative operands.

    The relative recurrence retains its internal coordinate width and identity
    mounting. Runtime base coordinates and total Jacobian widths are handled by
    a separate composition kernel.
    """

    is_planar: bool = eqx.field(static=True)
    num_base_coordinates: int = eqx.field(static=True)
    num_base_velocities: int = eqx.field(static=True)
    num_velocities: int = eqx.field(static=True)
    relative: PCSKinematicsOperands

    @classmethod
    def from_model(cls, model: PCSOperandSource) -> PCSFloatingKinematicsOperands:
        """Build floating kinematics operands without copying model arrays.

        Args:
            model: Floating PlanarPCS or PCS model.

        Returns:
            Floating specialization containing identity-mounted relative
            operands.

        Raises:
            ValueError: If the supplied model is fixed-base.
        """
        if not model.floating_base:
            raise ValueError(
                "Floating PCS kinematics operands require a floating model."
            )
        relative_model = model._fixed_base_robot_at_pose()
        return cls(
            is_planar=model.is_planar,
            num_base_coordinates=3 if model.is_planar else 7,
            num_base_velocities=3 if model.is_planar else 6,
            num_velocities=model.num_velocities,
            relative=PCSKinematicsOperands.from_model(relative_model),
        )


@dataclass(frozen=True)
class PCSKinematicsShapes:
    """Allocation contract for fused PCS pose/Jacobian execution.

    Attributes:
        batch_size: Number of independent environments.
        num_samples: Number of backbone samples per environment.
        num_segments: Number of constant-strain segments.
        num_dofs: Number of active generalized strains.
        spatial_dim: Three for PlanarPCS and six for spatial PCS.
        is_planar: Select planar or spatial output storage.
    """

    batch_size: int
    num_samples: int
    num_segments: int
    num_dofs: int
    spatial_dim: int
    is_planar: bool

    @classmethod
    def from_operands(
        cls,
        operands: PCSKinematicsOperands,
        *,
        batch_size: int,
        num_samples: int,
    ) -> PCSKinematicsShapes:
        """Construct kinematics shapes from runtime operands.

        Args:
            operands: Public PCS kinematics operand bundle.
            batch_size: Positive number of independent environments.
            num_samples: Positive number of samples per environment.

        Returns:
            Immutable kinematics workspace and output dimensions.

        Raises:
            TypeError: If either requested dimension is not an integer.
            ValueError: If either requested dimension is not positive.
        """

        for name, value in (("batch_size", batch_size), ("num_samples", num_samples)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 1:
                raise ValueError(f"{name} must be positive.")
        return cls(
            batch_size=batch_size,
            num_samples=num_samples,
            num_segments=operands.num_segments,
            num_dofs=operands.num_dofs,
            spatial_dim=3 if operands.is_planar else 6,
            is_planar=operands.is_planar,
        )

    def workspace(self) -> dict[str, tuple[int, ...]]:
        """Return caller-owned fused segment-boundary workspace shapes.

        Returns:
            Mapping from fused launcher workspace names to array shapes.
        """

        pose_shape = (3,) if self.is_planar else (4, 4)
        return {
            "segment_strain": (
                self.batch_size,
                self.num_segments,
                self.spatial_dim,
            ),
            "segment_pose": (
                self.batch_size,
                self.num_segments + 1,
                *pose_shape,
            ),
            "segment_jacobian": (
                self.batch_size,
                self.num_segments + 1,
                self.spatial_dim,
                self.num_dofs,
            ),
        }

    def pose_workspace(self) -> dict[str, tuple[int, ...]]:
        """Return the reduced pose-only workspace shapes.

        Returns:
            Mapping containing only segment strains and boundary poses.
        """

        workspace = self.workspace()
        return {
            "segment_strain": workspace["segment_strain"],
            "segment_pose": workspace["segment_pose"],
        }

    def jacobian_workspace(self) -> dict[str, tuple[int, ...]]:
        """Return the Jacobian-only workspace shapes.

        Returns:
            Mapping containing the fused recurrence workspaces required to
            rotate body Jacobians into the inertial frame.
        """

        return self.workspace()

    def pose_output(self) -> tuple[int, ...]:
        """Return the native PlanarPCS or PCS pose output shape.

        Returns:
            Shape ``(E, N, 3)`` for PlanarPCS or ``(E, N, 4, 4)`` for PCS.
        """

        tail = (3,) if self.is_planar else (4, 4)
        return self.batch_size, self.num_samples, *tail

    def jacobian_output(self) -> tuple[int, ...]:
        """Return the inertial-frame Jacobian output shape.

        Returns:
            Shape ``(E, N, spatial_dim, num_dofs)``.
        """

        return (
            self.batch_size,
            self.num_samples,
            self.spatial_dim,
            self.num_dofs,
        )


__all__ = [
    "PCSFloatingKinematicsOperands",
    "PCSKinematicsOperands",
    "PCSKinematicsShapes",
    "PCSOperandSource",
    "PCSFloatingOperands",
    "PCSFloatingPipelineShapes",
    "PCSOperands",
    "PCSPipelineShapes",
]
