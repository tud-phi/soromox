__all__ = [
    "BaseSystemParams",
    "BaseSoftRobotParams",
    "BaseContinuumSoftRobotParams",
    "BaseArticulatedSoftRobotParams",
    "BaseTendonRoutingParams",
    "LinearTendonRoutingParams",
    "PassiveTendonParams",
    "PCSStructure",
    "PlanarPCSStructure",
    "PCSParams",
    "PlanarPCSParams",
    "TendonActuatedPCSParams",
    "TendonActuatedPlanarPCSParams",
    "PneumaticActuatedPlanarPCSParams",
    "ISupportParams",
    "PendulumParams",
    "TendonActuatedPendulumParams",
    "ArticulatedSoftRobotParams",
    "GVSStructure",
    "GVSLinkParams",
    "GVSParams",
    "TendonActuatedGVSParams",
    "PlanarHSAStructure",
    "PlanarHSAParams",
]

from collections.abc import Sequence
from dataclasses import fields
from typing import Any

import equinox as eqx
import jax
from jax import Array
from jax import numpy as jnp


class BaseSystemParams(eqx.Module):
    """Base class for dynamic system parameters stored as JAX PyTrees."""

    def replace(self, **updates: Any) -> "BaseSystemParams":
        """Return a copy with selected fields replaced."""
        valid_names = {field.name for field in fields(self)}
        unknown_names = set(updates) - valid_names
        if unknown_names:
            unknown = ", ".join(sorted(unknown_names))
            raise KeyError(f"Unknown parameter field(s): {unknown}")

        updated = self
        for name, value in updates.items():
            updated = eqx.tree_at(
                lambda params, field_name=name: getattr(params, field_name),
                updated,
                value,
            )
        updated.validate()
        return updated

    def validate(self) -> None:
        """Validate parameter consistency."""
        return None


class BaseSoftRobotParams(BaseSystemParams):
    """Common dynamic parameters for soft robot systems.

    ``gravity`` is a JAX array in the dimensional convention of the concrete
    model, for example 2D planar gravity or 3D spatial gravity.
    """

    gravity: Array


class BaseContinuumSoftRobotParams(BaseSoftRobotParams):
    """Shared dynamic parameters for continuum soft robots.

    Field names denote one segment's physical quantity; the leading axis stores
    the segment batch. ``reference_strain`` contains the flattened per-segment
    reference strain used by PCS-style continuum models.
    """

    length: Array
    density: Array
    reference_strain: Array


class BaseArticulatedSoftRobotParams(BaseSoftRobotParams):
    """Shared dynamic parameters for articulated systems.

    The leading axis indexes joints/links. Stiffness, damping, and reference
    coordinates are dynamic arrays so optimization can update them without
    changing the system structure. ``joint_rest_configuration`` is the joint
    coordinate at which the elastic joint force is zero.
    """

    mass: Array
    joint_stiffness: Array
    joint_damping: Array
    joint_rest_configuration: Array


class BaseTendonRoutingParams(BaseSystemParams):
    """Base class for batched tendon routing parameter sets.

    A tendon routing params object represents all tendons of one routing family.
    The leading axis of every array indexes tendon number; changing this axis
    changes the actuator layout and requires model reconstruction.
    """

    @property
    def num_tendons(self) -> int:
        raise NotImplementedError


class LinearTendonRoutingParams(BaseTendonRoutingParams):
    """Batched linear routing parameters for active or passive tendons.

    Each field has shape ``(num_tendons,)``. Tendon ``k`` has cross-section
    offsets ``y_intercept[k] + y_slope[k] * s`` and
    ``z_intercept[k] + z_slope[k] * s`` and attaches to
    ``attachment_segment_index[k]``. Systems use ``jax.vmap`` over the leading
    tendon axis, so every tendon can have distinct routing parameters.
    """

    y_intercept: Array
    y_slope: Array
    z_intercept: Array
    z_slope: Array
    attachment_segment_index: Array

    @classmethod
    def empty(cls) -> "LinearTendonRoutingParams":
        return cls(
            y_intercept=jnp.array([], dtype=jnp.float64),
            y_slope=jnp.array([], dtype=jnp.float64),
            z_intercept=jnp.array([], dtype=jnp.float64),
            z_slope=jnp.array([], dtype=jnp.float64),
            attachment_segment_index=jnp.array([], dtype=jnp.int32),
        )

    @property
    def num_tendons(self) -> int:
        return int(self.y_intercept.shape[0])

    def validate(self) -> None:
        if len(self.y_intercept.shape) != 1:
            raise ValueError(
                "LinearTendonRoutingParams fields must be one-dimensional with "
                "shape (num_tendons,)."
            )
        n_tendons = self.y_intercept.shape[0]
        expected_shape = (n_tendons,)
        for name in (
            "y_slope",
            "z_intercept",
            "z_slope",
            "attachment_segment_index",
        ):
            value = getattr(self, name)
            if value.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {value.shape}."
                )

    def routing_for_tendon(self, index: Array) -> "LinearTendonRoutingParams":
        """Return the scalar-field PyTree for one tendon index."""
        return jax.tree.map(lambda value: value[index], self)


class PassiveTendonParams(BaseSystemParams):
    """Batched passive tendon impedance and rest-length parameters.

    Each field has shape ``(num_passive_tendons,)``. Passive tendon ``k`` uses
    ``stiffness[k]``, ``damping[k]``, and ``rest_length_offset[k]``. The length
    must match the paired passive routing params.
    """

    stiffness: Array
    damping: Array
    rest_length_offset: Array

    @classmethod
    def empty(cls) -> "PassiveTendonParams":
        return cls(
            stiffness=jnp.array([], dtype=jnp.float64),
            damping=jnp.array([], dtype=jnp.float64),
            rest_length_offset=jnp.array([], dtype=jnp.float64),
        )

    @property
    def num_tendons(self) -> int:
        return int(self.stiffness.shape[0])

    def validate(self) -> None:
        if len(self.stiffness.shape) != 1:
            raise ValueError(
                "PassiveTendonParams fields must be one-dimensional with shape "
                "(num_passive_tendons,)."
            )
        n_tendons = self.stiffness.shape[0]
        expected_shape = (n_tendons,)
        for name in ("damping", "rest_length_offset"):
            value = getattr(self, name)
            if value.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {value.shape}."
                )


class PCSStructure(eqx.Module):
    """Static PCS layout that determines JAX compilation structure.

    This stores quadrature count, strain activation, and basis scaling choices.
    Changing any of these choices changes array shapes or static control flow and
    therefore requires constructing a new ``PCS`` instance.
    """

    num_gauss_points: int = eqx.field(static=True, default=5)
    strain_selector: Array | None = None
    scale_rotational_basis_by_length: bool = eqx.field(static=True, default=False)


class PlanarPCSStructure(eqx.Module):
    """Static planar PCS layout.

    The fields mirror ``PCSStructure`` but apply to planar strain coordinates.
    Dynamic physical values live in ``PlanarPCSParams``.
    """

    num_gauss_points: int = eqx.field(static=True, default=5)
    strain_selector: Array | None = None
    scale_rotational_basis_by_length: bool = eqx.field(static=True, default=False)


class PCSParams(BaseContinuumSoftRobotParams):
    """Dynamic parameters for the spatial PCS model.

    ``length``, ``radius``, material parameters, and density use a leading
    segment axis. ``damping_matrix`` is the full flattened strain damping matrix.
    ``base_pose`` is the SE(3) base pose vector used to initialize the base
    transform.
    """

    radius: Array
    young_modulus: Array
    shear_modulus: Array
    damping_matrix: Array
    base_pose: Array


class PlanarPCSParams(BaseContinuumSoftRobotParams):
    """Dynamic parameters for the planar PCS model.

    The leading axis of per-segment fields indexes planar constant-strain
    segments. ``base_angle`` stores the planar base orientation.
    """

    radius: Array
    young_modulus: Array
    shear_modulus: Array
    damping_matrix: Array
    base_angle: Array


class TendonActuatedPCSParams(BaseSystemParams):
    """Dynamic parameters for spatial tendon-actuated PCS.

    ``body`` contains the underlying PCS dynamic parameters. Active and passive
    tendon routing fields are batched by tendon; passive impedance fields are
    batched by passive tendon and must have the same leading length as
    ``passive_tendon_routing``.
    """

    body: PCSParams
    active_tendon_routing: LinearTendonRoutingParams
    passive_tendon_routing: LinearTendonRoutingParams = eqx.field(
        default_factory=LinearTendonRoutingParams.empty
    )
    passive_tendon: PassiveTendonParams = eqx.field(
        default_factory=PassiveTendonParams.empty
    )

    def validate(self) -> None:
        self.body.validate()
        self.active_tendon_routing.validate()
        self.passive_tendon_routing.validate()
        self.passive_tendon.validate()
        if (
            self.passive_tendon.num_tendons
            != self.passive_tendon_routing.num_tendons
        ):
            raise ValueError(
                "passive_tendon must have one impedance entry per "
                "passive_tendon_routing entry."
            )


class TendonActuatedPlanarPCSParams(PlanarPCSParams):
    """Dynamic parameters for parallel-routed planar tendon PCS.

    ``tendon_distance`` has shape ``(num_segments, num_tendons_per_segment)``.
    Each row defines the offsets of the parallel tendons routed through that
    segment.
    """

    tendon_distance: Array


class PneumaticActuatedPlanarPCSParams(PlanarPCSParams):
    """Dynamic parameters for pneumatic planar PCS.

    Chamber geometry arrays describe the per-segment chamber layout and can be
    updated without changing the strain layout as long as their shapes are
    unchanged.
    """

    chamber_inner_radius: Array
    chamber_outer_radius: Array
    chamber_angle: Array
    chamber_distance: Array


class ISupportParams(PCSParams):
    """Dynamic parameters for the I-SUPPORT spatial pneumatic PCS model.

    The PCS body parameters are inherited. Chamber fields describe per-segment
    pneumatic chamber geometry and angular offsets.
    """

    chamber_inner_radius: Array
    chamber_outer_radius: Array
    chamber_distance: Array
    chamber_angle_offset: Array


class PendulumParams(BaseArticulatedSoftRobotParams):
    """Dynamic parameters for planar pendulum chains.

    The leading axis indexes links/joints. ``moment_inertia`` and
    ``center_of_mass_length`` are per link, while stiffness and damping are
    generalized-coordinate matrices inherited from the articulated base class.
    """

    moment_inertia: Array
    length: Array
    center_of_mass_length: Array
    radius: Array


class TendonActuatedPendulumParams(BaseSystemParams):
    """Dynamic parameters for tendon-actuated pendulum chains.

    ``body`` stores pendulum dynamics. Active/passive routing matrices map joint
    coordinates to tendon displacements. ``active_tendon_reference_configuration``
    and ``passive_tendon_reference_configuration`` are joint-space configurations
    with shape ``(num_links,)`` at which the respective tendon displacement
    contribution ``R @ (q - q_ref)`` is zero. Passive tendon impedance fields are
    batched by passive tendon.
    """

    body: PendulumParams
    active_routing_matrix: Array
    passive_routing_matrix: Array
    active_tendon_reference_configuration: Array
    passive_tendon_reference_configuration: Array
    passive_tendon: PassiveTendonParams

    def validate(self) -> None:
        self.body.validate()
        self.passive_tendon.validate()
        passive_routing_matrix = jnp.asarray(self.passive_routing_matrix)
        if passive_routing_matrix.ndim != 2:
            raise ValueError("passive_routing_matrix must be two-dimensional.")
        if self.passive_tendon.num_tendons != passive_routing_matrix.shape[0]:
            raise ValueError(
                "passive_tendon must have one impedance entry per passive routing row."
            )
        n_links = self.body.mass.shape[0]
        if self.active_tendon_reference_configuration.shape != (n_links,):
            raise ValueError(
                "active_tendon_reference_configuration must have shape (num_links,)."
            )
        if self.passive_tendon_reference_configuration.shape != (n_links,):
            raise ValueError(
                "passive_tendon_reference_configuration must have shape (num_links,)."
            )


class ArticulatedSoftRobotParams(BaseArticulatedSoftRobotParams):
    """Dynamic parameters for spatial articulated soft robots.

    Per-joint screw axes and transforms define the dynamic link geometry used by
    the articulated model. The number of joints is fixed by array shapes.
    """

    joint_screw: Array
    parent_to_joint_transform: Array
    tip_position: Array
    center_of_mass_position: Array
    center_of_mass_inertia: Array
    radius: Array


class GVSStructure(eqx.Module):
    """Static GVS segment structure and padded layout choices.

    ``segments`` stores static ``GVSSegment`` specs: joint families, basis
    families/orders/active masks, quadrature counts, and cross-section families.
    ``max_dof`` and ``max_num_gauss_points`` control padding. Changing any field
    here is a structural change and requires reconstruction.
    """

    segments: Sequence[Any] = eqx.field(static=True)
    max_dof: int | None = eqx.field(static=True, default=None)
    max_num_gauss_points: int | None = eqx.field(static=True, default=None)
    scale_rotational_basis_by_length: bool = eqx.field(static=True, default=False)


class GVSLinkParams(BaseSystemParams):
    """Dynamic per-link arrays for all GVS segments.

    Every field has leading shape ``(num_segments,)``; this is not a single-link
    object. It stores the numeric link values paired with the static link specs in
    ``GVSStructure.segments``. ``length`` follows the singular per-link naming
    convention used by the other fields. Reference strain is intentionally stored
    on ``GVSParams`` because it belongs to the strain basis state, not the link
    cross-section/material data.
    """

    length: Array
    young_modulus: Array
    poisson_ratio: Array
    density: Array
    damping_coefficient: Array
    radius_initial: Array
    radius_final: Array
    height_initial: Array
    height_final: Array
    width_initial: Array
    width_final: Array
    semi_major_initial: Array
    semi_major_final: Array
    semi_minor_initial: Array
    semi_minor_final: Array

    def validate(self) -> None:
        n_segments = self.length.shape[0]
        expected_shape = (n_segments,)
        for name in (
            "young_modulus",
            "poisson_ratio",
            "density",
            "damping_coefficient",
            "radius_initial",
            "radius_final",
            "height_initial",
            "height_final",
            "width_initial",
            "width_final",
            "semi_major_initial",
            "semi_major_final",
            "semi_minor_initial",
            "semi_minor_final",
        ):
            value = getattr(self, name)
            if value.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {value.shape}."
                )


class GVSParams(BaseSoftRobotParams):
    """Dynamic parameters for a GVS model.

    ``link`` contains per-segment link arrays. ``reference_strain`` has shape
    ``(num_segments, 6)`` and is kept here, rather than in ``GVSLinkParams``,
    because it parameterizes the strain basis/reference configuration rather
    than link geometry or material properties. ``joint_stiffness`` has shape
    ``(num_segments, max_dof, max_dof)`` and is padded to the static GVS layout.
    """

    link: GVSLinkParams
    base_pose: Array
    reference_strain: Array
    joint_stiffness: Array

    def validate(self) -> None:
        self.link.validate()
        n_segments = self.link.length.shape[0]
        if self.reference_strain.shape[0] != n_segments:
            raise ValueError(
                "reference_strain must have one leading entry per GVS segment."
            )
        if self.joint_stiffness.shape[0] != n_segments:
            raise ValueError(
                "joint_stiffness must have one leading entry per GVS segment."
            )


class TendonActuatedGVSParams(BaseSystemParams):
    """Dynamic parameters for tendon-actuated GVS.

    ``body`` stores the underlying GVS dynamic params. Active and passive routing
    objects are batched by tendon, so each tendon can use distinct linear routing
    parameters and attachment segment. ``passive_tendon`` stores per-passive-
    tendon impedance and rest-length offsets.
    """

    body: GVSParams
    active_tendon_routing: LinearTendonRoutingParams
    passive_tendon_routing: LinearTendonRoutingParams = eqx.field(
        default_factory=LinearTendonRoutingParams.empty
    )
    passive_tendon: PassiveTendonParams = eqx.field(
        default_factory=PassiveTendonParams.empty
    )

    def validate(self) -> None:
        self.body.validate()
        self.active_tendon_routing.validate()
        self.passive_tendon_routing.validate()
        self.passive_tendon.validate()
        if (
            self.passive_tendon.num_tendons
            != self.passive_tendon_routing.num_tendons
        ):
            raise ValueError(
                "passive_tendon must have one impedance entry per "
                "passive_tendon_routing entry."
            )


class PlanarHSAStructure(eqx.Module):
    """Static symbolic and layout choices for planar HSA.

    The symbolic expression path, strain selector, underactuation flag,
    hysteresis mode, and regularization epsilon determine static evaluation
    structure. Dynamic physical values live in ``PlanarHSAParams``.
    """

    symbolic_expression_path: str = eqx.field(static=True)
    strain_selector: Array | None = None
    consider_underactuation: bool = eqx.field(static=True, default=True)
    consider_hysteresis: bool = eqx.field(static=True, default=False)
    eps: float = eqx.field(static=True, default=1e-6)


class PlanarHSAParams(BaseSystemParams):
    """Dynamic parameters for planar HSA systems.

    Field names denote one segment/platform quantity; leading axes store the
    batched values. Arrays store length, rod geometry, reference strain
    components, platform/cap dimensions, end-effector offset, and optional
    hysteresis coefficients used by the symbolic HSA expressions.
    """

    base_angle: Array
    length: Array
    proximal_cap_length: Array
    distal_cap_length: Array
    rod_height: Array
    rod_outer_radius: Array
    rod_inner_radius: Array
    rod_offset: Array
    bending_reference: Array
    shear_reference: Array
    axial_reference: Array
    strain_coupling: Array
    platform_dimension: Array
    rod_density: Array
    platform_density: Array
    end_cap_density: Array
    gravity: Array
    nominal_bending_stiffness: Array
    nominal_shear_stiffness: Array
    nominal_axial_stiffness: Array
    bending_shear_stiffness: Array
    bending_stiffness_correction: Array
    shear_stiffness_correction: Array
    axial_stiffness_correction: Array
    bending_damping: Array
    shear_damping: Array
    axial_damping: Array
    platform_mass: Array
    platform_center_of_gravity: Array
    end_effector_offset: Array
    hysteresis_basis: Array
    hysteresis_alpha: Array
    hysteresis_A: Array
    hysteresis_n: Array
    hysteresis_beta: Array
    hysteresis_gamma: Array
