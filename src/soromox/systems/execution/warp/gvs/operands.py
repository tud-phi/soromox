"""Explicit runtime data contract for the GVS Warp executor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import equinox as eqx
import jax.numpy as jnp
from jax import Array


class GVSOperandSource(Protocol):
    """Structural source of precomputed GVS Warp operands.

    The protocol lists only data used by the executor. It deliberately avoids
    importing the concrete :class:`GVS` model and therefore keeps the execution
    package acyclic.
    """

    num_segments: int
    floating_base: bool
    num_internal_dofs: int
    num_velocities: int
    num_dofs: int
    max_dof: int
    max_num_integration_points: int
    B_Xs: Array
    xi_ref_Xs: Array
    inner_integration_weights: Array
    B_joint: Array
    xi_ref_joint: Array
    joint_local_to_global: Array
    joint_global_to_local: Array
    link_local_to_global: Array
    link_global_to_local: Array
    active_dofs_per_segment: Array
    scaled_B_Z1_values: Array
    scaled_B_Z2_values: Array
    link_basis_rows: Array
    xi_ref_Z1: Array
    xi_ref_Z2: Array
    segment_lengths: Array
    cell_widths: Array
    inertia_upper_rows: Array
    inertia_upper_columns: Array
    inner_weighted_mass_diagonals: Array
    gravity_base: Array
    g: Array
    g0: Array
    integration_points: Array
    segment_end_positions: Array
    basis_type_index: Array
    basis_active_params: Array
    basis_order_params: Array
    scale_rotational_basis_by_length: bool
    Z1: float
    Z2: float
    global_eps: float
    tangent_eps: float


class GVSOperands(eqx.Module):
    """Minimal shape-generic data passed to the GVS Warp executor.

    Static fields determine launch and output shapes and therefore participate
    in JAX/Warp compilation caching. Array fields remain runtime operands, so
    changing physical parameters does not generate new Warp source. The bundle
    contains no concrete model reference.

    External Warp-native integrations may use this object as the authoritative
    bridge from a Soromox :class:`GVS` model to the public GVS launch functions.
    Its arrays are JAX arrays owned by the model; obtain zero-copy Warp views
    with :func:`warp.from_jax` once during solver construction, then retain the
    views and caller-owned output buffers for steady-state stepping.

    Attributes:
        num_segments: Number of serial joint-link segments.
        num_dofs: Number of active generalized coordinates.
        max_dof: Padded local coordinate width shared by joints and links.
        num_cells: Integration cells per padded segment.
        num_quadrature: Interior quadrature points per padded segment.
        block_dim: Cooperative lanes used by the persistent chain launch.
        joint_basis: Flattened joint basis matrices.
        joint_reference: Reference joint strains.
        joint_local_to_global: Padded joint-local to active-coordinate map.
        joint_global_to_local: Active-coordinate to joint-local map.
        link_local_to_global: Padded link-local to active-coordinate map.
        link_global_to_local: Active-coordinate to link-local map.
        active_dofs_per_segment: Cumulative active coordinate count after each
            segment.
        link_basis_z1_values: Sparse link basis values at first Magnus nodes.
        link_basis_z2_values: Sparse link basis values at second Magnus nodes.
        link_basis_rows: Spatial row occupied by every sparse basis column.
        link_reference_z1: Reference strains at first Magnus nodes.
        link_reference_z2: Reference strains at second Magnus nodes.
        segment_lengths: Physical length of every segment.
        cell_widths: Width of every padded integration cell.
        inertia_upper_rows: Row indices for packed upper-triangular inertia
            assembly.
        inertia_upper_columns: Column indices matching ``inertia_upper_rows``.
        weighted_mass_diagonals: Quadrature-weighted diagonal spatial inertia at
            every interior point.
        gravity_base: Spatial gravity acceleration expressed in the base frame.
    """

    num_segments: int = eqx.field(static=True)
    num_dofs: int = eqx.field(static=True)
    max_dof: int = eqx.field(static=True)
    num_cells: int = eqx.field(static=True)
    num_quadrature: int = eqx.field(static=True)
    block_dim: int = eqx.field(static=True)
    joint_basis: Array
    joint_reference: Array
    joint_local_to_global: Array
    joint_global_to_local: Array
    link_local_to_global: Array
    link_global_to_local: Array
    active_dofs_per_segment: Array
    link_basis_z1_values: Array
    link_basis_z2_values: Array
    link_basis_rows: Array
    link_reference_z1: Array
    link_reference_z2: Array
    segment_lengths: Array
    cell_widths: Array
    inertia_upper_rows: Array
    inertia_upper_columns: Array
    weighted_mass_diagonals: Array
    gravity_base: Array

    @classmethod
    def from_model(cls, model: GVSOperandSource, *, block_dim: int) -> GVSOperands:
        """Build an operand view over a precomputed GVS-compatible model.

        Args:
            model: Object satisfying :class:`GVSOperandSource`.
            block_dim: Cooperative threads per persistent chain block, or one
                lane for Warp CPU execution.

        Returns:
            Operand bundle referencing the model's existing JAX arrays. No
            numerical preprocessing or array copy is performed here.
        """

        if model.floating_base:
            raise ValueError("Fixed GVS Warp operands require a fixed-base model.")
        return cls(
            num_segments=model.num_segments,
            num_dofs=model.num_dofs,
            max_dof=model.max_dof,
            num_cells=model.max_num_integration_points - 1,
            num_quadrature=model.max_num_integration_points - 2,
            block_dim=block_dim,
            joint_basis=model.B_joint,
            joint_reference=model.xi_ref_joint,
            joint_local_to_global=model.joint_local_to_global,
            joint_global_to_local=model.joint_global_to_local,
            link_local_to_global=model.link_local_to_global,
            link_global_to_local=model.link_global_to_local,
            active_dofs_per_segment=model.active_dofs_per_segment,
            link_basis_z1_values=model.scaled_B_Z1_values,
            link_basis_z2_values=model.scaled_B_Z2_values,
            link_basis_rows=model.link_basis_rows,
            link_reference_z1=model.xi_ref_Z1,
            link_reference_z2=model.xi_ref_Z2,
            segment_lengths=model.segment_lengths,
            cell_widths=model.cell_widths,
            inertia_upper_rows=model.inertia_upper_rows,
            inertia_upper_columns=model.inertia_upper_columns,
            weighted_mass_diagonals=model.inner_weighted_mass_diagonals,
            gravity_base=model.gravity_base,
        )


class GVSFloatingOperands(eqx.Module):
    """Runtime data for a statically augmented floating GVS Warp pipeline.

    Local joint and link operands are shared with :class:`GVSOperands`.
    Separate static internal and total dimensions keep runtime base state out
    of the fixed persistent kernel and allocate augmented workspaces only for
    floating models.
    """

    num_segments: int = eqx.field(static=True)
    num_internal_dofs: int = eqx.field(static=True)
    num_velocities: int = eqx.field(static=True)
    max_dof: int = eqx.field(static=True)
    num_cells: int = eqx.field(static=True)
    num_quadrature: int = eqx.field(static=True)
    block_dim: int = eqx.field(static=True)
    joint_basis: Array
    joint_reference: Array
    joint_local_to_global: Array
    joint_global_to_local: Array
    link_local_to_global: Array
    link_global_to_local: Array
    active_dofs_per_segment: Array
    link_basis_z1_values: Array
    link_basis_z2_values: Array
    link_basis_rows: Array
    link_reference_z1: Array
    link_reference_z2: Array
    segment_lengths: Array
    cell_widths: Array
    inertia_upper_rows: Array
    inertia_upper_columns: Array
    weighted_mass_diagonals: Array
    gravity_world: Array

    @classmethod
    def from_model(
        cls, model: GVSOperandSource, *, block_dim: int
    ) -> GVSFloatingOperands:
        """Build floating operands over a precomputed GVS model.

        Args:
            model: Floating GVS model satisfying :class:`GVSOperandSource`.
            block_dim: Cooperative threads per chain block, or one on CPU.

        Returns:
            Operand bundle referencing existing model arrays without copies.

        Raises:
            ValueError: If ``model`` is fixed-base.
        """
        if not model.floating_base:
            raise ValueError("Floating GVS Warp operands require a floating model.")
        return cls(
            num_segments=model.num_segments,
            num_internal_dofs=model.num_internal_dofs,
            num_velocities=model.num_velocities,
            max_dof=model.max_dof,
            num_cells=model.max_num_integration_points - 1,
            num_quadrature=model.max_num_integration_points - 2,
            block_dim=block_dim,
            joint_basis=model.B_joint,
            joint_reference=model.xi_ref_joint,
            joint_local_to_global=model.joint_local_to_global,
            joint_global_to_local=model.joint_global_to_local,
            link_local_to_global=model.link_local_to_global,
            link_global_to_local=model.link_global_to_local,
            active_dofs_per_segment=model.active_dofs_per_segment,
            link_basis_z1_values=model.scaled_B_Z1_values,
            link_basis_z2_values=model.scaled_B_Z2_values,
            link_basis_rows=model.link_basis_rows,
            link_reference_z1=model.xi_ref_Z1,
            link_reference_z2=model.xi_ref_Z2,
            segment_lengths=model.segment_lengths,
            cell_widths=model.cell_widths,
            inertia_upper_rows=model.inertia_upper_rows,
            inertia_upper_columns=model.inertia_upper_columns,
            weighted_mass_diagonals=model.inner_weighted_mass_diagonals,
            gravity_world=model.g,
        )


@dataclass(frozen=True)
class GVSPipelineShapes:
    """Describe every array shape in the public GVS Warp pipeline.

    External integrators can construct this object once for their fixed batch
    size, allocate Warp arrays for each returned output mapping, and reuse those
    buffers for every step and CUDA graph replay. The names match the arguments
    of :func:`launch_joint_terms`, :func:`launch_cell_terms`, and
    :func:`launch_persistent_chain`.

    Attributes:
        batch_size: Number of independent environments.
        num_segments: Number of serial joint-link segments.
        num_dofs: Number of active generalized coordinates.
        max_dof: Padded local coordinate width.
        num_cells: Integration cells per segment.
    """

    batch_size: int
    num_segments: int
    num_dofs: int
    max_dof: int
    num_cells: int

    @classmethod
    def from_operands(
        cls, operands: GVSOperands, *, batch_size: int
    ) -> GVSPipelineShapes:
        """Construct pipeline shapes from public runtime operands.

        Args:
            operands: Runtime dimensions and model data prepared by
                :meth:`GVSOperands.from_model`.
            batch_size: Positive number of independent environments.

        Returns:
            Immutable collection of pipeline dimensions and output mappings.

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
            max_dof=operands.max_dof,
            num_cells=operands.num_cells,
        )

    @property
    def local_state(self) -> tuple[int, int]:
        """Return the shape of padded link-state workspace arrays.

        Returns:
            Shape ``(batch_size * num_segments, max_dof)`` used by ``q_link``
            and ``qd_link``.
        """

        return self.batch_size * self.num_segments, self.max_dof

    def joint_outputs(self) -> dict[str, tuple[int, ...]]:
        """Return output shapes for a joint-terms launch.

        Returns:
            Mapping from public joint output argument names to array shapes.
        """

        rows = self.batch_size * self.num_segments * 6
        return {
            "adjoint": (rows, 6),
            "adjoint_dot": (rows, 6),
            "tangent_local": (rows, self.max_dof),
            "tangent_dot_qd": (rows, 1),
            "joint_velocity": (rows, 1),
        }

    def cell_outputs(self) -> dict[str, tuple[int, ...]]:
        """Return output shapes for a link-cell terms launch.

        Returns:
            Mapping from public cell output argument names to array shapes.
        """

        rows = self.batch_size * self.num_segments * self.num_cells * 6
        return {
            "adjoint": (rows, 6),
            "tangent_local": (rows, self.max_dof),
            "link_velocity": (rows, 1),
            "step_velocity": (rows, 1),
            "tangent_velocity_dot": (rows, 1),
        }

    def chain_outputs(self) -> dict[str, tuple[int, ...]]:
        """Return scratch and final output shapes for a persistent-chain launch.

        Returns:
            Mapping from public persistent-chain output argument names to array
            shapes. The first eight arrays are caller-owned ping-pong workspace;
            the last three are the public dynamics results.
        """

        rows = self.batch_size * 6
        return {
            "jacobian_first": (rows, self.num_dofs),
            "jacobian_dot_qd_first": (rows, 1),
            "velocity_first": (rows, 1),
            "gravity_first": (rows, 1),
            "jacobian_second": (rows, self.num_dofs),
            "jacobian_dot_qd_second": (rows, 1),
            "velocity_second": (rows, 1),
            "gravity_second": (rows, 1),
            "inertia": (self.batch_size, self.num_dofs, self.num_dofs),
            "coriolis_qd": (self.batch_size, self.num_dofs),
            "gravity_force": (self.batch_size, self.num_dofs),
        }


@dataclass(frozen=True)
class GVSFloatingPipelineShapes(GVSPipelineShapes):
    """Allocation contract for augmented floating GVS workspaces."""

    @classmethod
    def from_operands(
        cls, operands: GVSFloatingOperands, *, batch_size: int
    ) -> GVSFloatingPipelineShapes:
        """Construct augmented shapes using the total velocity dimension.

        Args:
            operands: Floating runtime operand bundle.
            batch_size: Positive number of independent environments.

        Returns:
            Immutable floating GVS allocation contract.

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
            max_dof=operands.max_dof,
            num_cells=operands.num_cells,
        )


class GVSKinematicsOperands(eqx.Module):
    """Runtime model data required by public GVS kinematics launchers.

    Static dimensions determine output and workspace shapes. Array fields are
    runtime operands so physical parameters and configurations remain reusable
    with compiled Warp kernels and captured CUDA graphs.

    Attributes:
        num_segments: Number of serial joint-link segments.
        num_dofs: Number of active generalized coordinates.
        max_dof: Padded local coordinate width.
        num_cells: Padded integration cells per segment.
        block_dim: Preferred cooperative block size.
        scale_rotational_basis_by_length: Whether angular basis rows are scaled.
        base_transform: Absolute robot base transform.
        joint_basis: Padded joint basis matrices.
        joint_reference: Reference joint strains.
        joint_local_to_global: Joint-local to active-coordinate map.
        joint_global_to_local: Active-coordinate to joint-local map.
        link_local_to_global: Link-local to active-coordinate map.
        link_global_to_local: Active-coordinate to link-local map.
        link_basis_z1_values: Sparse full-cell basis at first Magnus nodes.
        link_basis_z2_values: Sparse full-cell basis at second Magnus nodes.
        link_basis_rows: Spatial row occupied by every local basis column.
        link_reference_z1: Reference strain at first Magnus nodes.
        link_reference_z2: Reference strain at second Magnus nodes.
        segment_lengths: Physical link lengths.
        segment_starts: Cumulative link-start coordinates.
        integration_points: Normalized cell-boundary coordinates.
        cell_widths: Normalized full-cell widths.
        basis_type_index: Basis-family index for every segment.
        basis_active_params: Per-row basis activation parameters.
        basis_order_params: Per-row basis-order parameters.
        magnus_points: Normalized two-point Magnus quadrature coordinates.
        global_eps: Exponential-map threshold.
        tangent_eps: Tangent-map threshold.
    """

    num_segments: int = eqx.field(static=True)
    num_dofs: int = eqx.field(static=True)
    max_dof: int = eqx.field(static=True)
    num_cells: int = eqx.field(static=True)
    block_dim: int = eqx.field(static=True)
    scale_rotational_basis_by_length: bool = eqx.field(static=True)
    base_transform: Array
    joint_basis: Array
    joint_reference: Array
    joint_local_to_global: Array
    joint_global_to_local: Array
    link_local_to_global: Array
    link_global_to_local: Array
    link_basis_z1_values: Array
    link_basis_z2_values: Array
    link_basis_rows: Array
    link_reference_z1: Array
    link_reference_z2: Array
    segment_lengths: Array
    segment_starts: Array
    integration_points: Array
    cell_widths: Array
    basis_type_index: Array
    basis_active_params: Array
    basis_order_params: Array
    magnus_points: Array
    global_eps: float
    tangent_eps: float

    @classmethod
    def from_model(
        cls, model: GVSOperandSource, *, block_dim: int
    ) -> GVSKinematicsOperands:
        """Create an allocation-free view of a GVS model's kinematics arrays.

        Args:
            model: Object satisfying :class:`GVSOperandSource`.
            block_dim: Cooperative lanes per environment, or one on Warp CPU.

        Returns:
            Operand bundle referencing the model's existing JAX arrays.
        """

        return cls(
            num_segments=model.num_segments,
            num_dofs=model.num_dofs,
            max_dof=model.max_dof,
            num_cells=model.max_num_integration_points - 1,
            block_dim=block_dim,
            scale_rotational_basis_by_length=model.scale_rotational_basis_by_length,
            base_transform=model.g0,
            joint_basis=model.B_joint,
            joint_reference=model.xi_ref_joint,
            joint_local_to_global=model.joint_local_to_global,
            joint_global_to_local=model.joint_global_to_local,
            link_local_to_global=model.link_local_to_global,
            link_global_to_local=model.link_global_to_local,
            link_basis_z1_values=model.scaled_B_Z1_values,
            link_basis_z2_values=model.scaled_B_Z2_values,
            link_basis_rows=model.link_basis_rows,
            link_reference_z1=model.xi_ref_Z1,
            link_reference_z2=model.xi_ref_Z2,
            segment_lengths=model.segment_lengths,
            segment_starts=model.segment_end_positions,
            integration_points=model.integration_points,
            cell_widths=model.cell_widths,
            basis_type_index=model.basis_type_index,
            basis_active_params=model.basis_active_params,
            basis_order_params=model.basis_order_params,
            magnus_points=jnp.asarray([model.Z1, model.Z2]),
            global_eps=model.global_eps,
            tangent_eps=model.tangent_eps,
        )


class GVSFloatingKinematicsOperands(eqx.Module):
    """Floating GVS kinematics specialization over relative operands."""

    num_base_coordinates: int = eqx.field(static=True)
    num_base_velocities: int = eqx.field(static=True)
    num_velocities: int = eqx.field(static=True)
    relative: GVSKinematicsOperands

    @classmethod
    def from_model(
        cls, model: GVSOperandSource, *, block_dim: int
    ) -> GVSFloatingKinematicsOperands:
        """Build identity-mounted relative operands for a floating GVS model.

        Args:
            model: Floating GVS model.
            block_dim: Cooperative lanes per environment, or one on Warp CPU.

        Returns:
            Floating specialization referencing the model's existing arrays.

        Raises:
            ValueError: If the supplied model is fixed-base.
        """
        if not model.floating_base:
            raise ValueError("Floating GVS kinematics operands require a floating model.")
        relative_model = model._fixed_evaluation_view()
        return cls(
            num_base_coordinates=7,
            num_base_velocities=6,
            num_velocities=model.num_velocities,
            relative=GVSKinematicsOperands.from_model(
                relative_model, block_dim=block_dim
            ),
        )


@dataclass(frozen=True)
class GVSKinematicsShapes:
    """Allocation contract for fused GVS pose/Jacobian execution.

    Attributes:
        batch_size: Number of independent environments.
        num_samples: Number of backbone samples per environment.
        num_segments: Number of serial joint-link segments.
        num_cells: Padded integration cells per segment.
        num_dofs: Number of active generalized coordinates.
    """

    batch_size: int
    num_samples: int
    num_segments: int
    num_cells: int
    num_dofs: int

    @classmethod
    def from_operands(
        cls,
        operands: GVSKinematicsOperands,
        *,
        batch_size: int,
        num_samples: int,
    ) -> GVSKinematicsShapes:
        """Construct kinematics shapes from runtime operands.

        Args:
            operands: Public GVS kinematics operand bundle.
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
            num_cells=operands.num_cells,
            num_dofs=operands.num_dofs,
        )

    def workspace(self) -> dict[str, tuple[int, ...]]:
        """Return caller-owned fused cell-boundary workspace shapes.

        Returns:
            Mapping from fused launcher workspace names to array shapes.
        """

        node_count = self.batch_size * self.num_segments * (self.num_cells + 1)
        return {
            "node_pose": (node_count * 4, 4),
            "node_jacobian": (node_count * 6, self.num_dofs),
        }

    def pose_workspace(self) -> dict[str, tuple[int, ...]]:
        """Return the reduced pose-only workspace shapes.

        Returns:
            Mapping containing only the flattened node-pose workspace.
        """

        return {"node_pose": self.workspace()["node_pose"]}

    def pose_output(self) -> tuple[int, ...]:
        """Return the native GVS pose output shape.

        Returns:
            Shape ``(batch_size, num_samples, 4, 4)``.
        """

        return self.batch_size, self.num_samples, 4, 4

    def jacobian_output(self) -> tuple[int, ...]:
        """Return the native GVS inertial-Jacobian output shape.

        Returns:
            Shape ``(batch_size, num_samples, 6, num_dofs)``.
        """

        return self.batch_size, self.num_samples, 6, self.num_dofs


__all__ = [
    "GVSFloatingKinematicsOperands",
    "GVSKinematicsOperands",
    "GVSKinematicsShapes",
    "GVSOperandSource",
    "GVSFloatingOperands",
    "GVSFloatingPipelineShapes",
    "GVSOperands",
    "GVSPipelineShapes",
]
