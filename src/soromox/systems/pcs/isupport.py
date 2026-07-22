import math
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jax import Array, vmap

from soromox.actuation import ThreadlikeActuator, ThreadlikeRouting
from soromox.systems.pcs.params import ISupportParams
from soromox.systems.pcs.structures import ISupportStructure, PCSStructure
from soromox.utils.array_math import blk_diag

from .pcs import PCS

_RIGID_CONNECTOR_PARENT = -1


def _with_default_chamber_azimuth_angles(params: ISupportParams) -> ISupportParams:
    """Resolve the canonical three-chamber layout when no angles are supplied."""
    if params.chamber_azimuth_angles is not None:
        return params
    num_pneumatic_segments = int(params.chamber_inner_radius.shape[0])
    canonical_angles = 2.0 * jnp.pi * jnp.arange(3, dtype=jnp.float64) / 3.0
    return params.replace(
        chamber_azimuth_angles=jnp.tile(
            canonical_angles[None, :], (num_pneumatic_segments, 1)
        )
    )


def _resolve_chamber_effective_pressure_area(params: ISupportParams) -> Array:
    """Return one pressure-coordinate area per physical pneumatic segment."""
    if params.chamber_effective_pressure_area is not None:
        return jnp.asarray(params.chamber_effective_pressure_area, dtype=jnp.float64)
    r_inner = jnp.asarray(params.chamber_inner_radius, dtype=jnp.float64)
    r_outer = jnp.asarray(params.chamber_outer_radius, dtype=jnp.float64)
    return jnp.pi * (r_outer**2 - r_inner**2)


def _pneumatic_chamber_actuator(
    params: ISupportParams,
    pcs_segment_to_pneumatic_segment: Array,
) -> ThreadlikeActuator:
    """Build straight, segment-local chamber paths in pressure-channel order."""
    angles = jnp.asarray(params.chamber_azimuth_angles, dtype=jnp.float64)
    distances = jnp.asarray(params.chamber_distance, dtype=jnp.float64)[:, None]
    y_offsets = (distances * jnp.cos(angles)).reshape(-1)
    z_offsets = (distances * jnp.sin(angles)).reshape(-1)
    intercept = jnp.stack((jnp.zeros_like(y_offsets), y_offsets, z_offsets), axis=-1)

    parents = tuple(
        int(parent) for parent in jnp.asarray(pcs_segment_to_pneumatic_segment).tolist()
    )
    num_pneumatic_segments, num_chambers = angles.shape
    segment_spans = tuple(
        (
            min(index for index, parent in enumerate(parents) if parent == segment),
            max(index for index, parent in enumerate(parents) if parent == segment),
        )
        for segment in range(num_pneumatic_segments)
    )
    starts = tuple(
        segment_spans[segment][0]
        for segment in range(num_pneumatic_segments)
        for _ in range(num_chambers)
    )
    ends = tuple(
        segment_spans[segment][1]
        for segment in range(num_pneumatic_segments)
        for _ in range(num_chambers)
    )
    routing = ThreadlikeRouting.linear(
        intercept=intercept,
        start_segment_index=starts,
        end_segment_index=ends,
    )
    effective_areas = jnp.repeat(
        _resolve_chamber_effective_pressure_area(params), num_chambers
    )
    return ThreadlikeActuator.pressure_chambers(
        routing,
        effective_areas=effective_areas,
    )


def _straight_reference_strain(dtype: jnp.dtype) -> Array:
    return jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=dtype)


def _normalize_structure_for_params(
    params: ISupportParams,
    structure: ISupportStructure,
) -> ISupportStructure:
    num_physical_segments = int(params.length.shape[0])

    rigid_segment_selector = structure.rigid_segment_selector
    if rigid_segment_selector is None:
        rigid_segment_selector = tuple(i % 2 == 0 for i in range(num_physical_segments))
    if len(rigid_segment_selector) != num_physical_segments:
        raise ValueError(
            "rigid_segment_selector must contain one entry per physical segment; "
            f"expected {num_physical_segments}, got {len(rigid_segment_selector)}."
        )
    num_pneumatic_segments = sum(not is_rigid for is_rigid in rigid_segment_selector)
    if num_pneumatic_segments < 1:
        raise ValueError(
            "rigid_segment_selector must contain at least one pneumatic segment."
        )
    num_chamber_segments = int(params.chamber_inner_radius.shape[0])
    if num_chamber_segments != num_pneumatic_segments:
        raise ValueError(
            "Chamber parameter arrays must contain one entry per pneumatic segment; "
            f"expected {num_pneumatic_segments}, got {num_chamber_segments}."
        )

    pcs_segment_counts = structure.pcs_segment_counts
    if pcs_segment_counts is None:
        pcs_segment_counts = tuple(1 for _ in range(num_pneumatic_segments))
    elif len(pcs_segment_counts) == 1 and num_pneumatic_segments > 1:
        pcs_segment_counts = pcs_segment_counts * num_pneumatic_segments
    if len(pcs_segment_counts) != num_pneumatic_segments:
        raise ValueError(
            "pcs_segment_counts must contain one entry per pneumatic segment; "
            f"expected {num_pneumatic_segments}, got {len(pcs_segment_counts)}."
        )
    if any(count < 1 for count in pcs_segment_counts):
        raise ValueError("pcs_segment_counts entries must be positive integers.")

    return ISupportStructure(
        num_gauss_points=structure.num_gauss_points,
        pcs_segment_counts=pcs_segment_counts,
        rigid_segment_selector=rigid_segment_selector,
        strain_selector=structure.strain_selector,
        scale_rotational_basis_by_length=structure.scale_rotational_basis_by_length,
    )


def _resolve_pcs_segment_lengths(
    params: ISupportParams,
    pcs_segment_counts: tuple[int, ...],
    rigid_segment_selector: tuple[bool, ...],
) -> tuple[Array, ...]:
    physical_lengths = jnp.asarray(params.length, dtype=jnp.float64)
    pneumatic_physical_indices = [
        i for i, is_rigid in enumerate(rigid_segment_selector) if not is_rigid
    ]

    if params.pcs_segment_lengths is None:
        return tuple(
            jnp.full(
                (count,),
                physical_lengths[pneumatic_physical_indices[i]] / count,
                dtype=jnp.float64,
            )
            for i, count in enumerate(pcs_segment_counts)
        )

    pcs_segment_lengths = jnp.asarray(params.pcs_segment_lengths, dtype=jnp.float64)
    if pcs_segment_lengths.ndim != 1:
        raise ValueError(
            "pcs_segment_lengths must be one-dimensional with shape "
            "(num_pcs_segments,)."
        )
    expected_num_pcs_segments = sum(pcs_segment_counts)
    if pcs_segment_lengths.shape != (expected_num_pcs_segments,):
        raise ValueError(
            "pcs_segment_lengths must contain one length per pneumatic PCS segment; "
            f"expected {expected_num_pcs_segments}, got {pcs_segment_lengths.shape[0]}."
        )
    if not bool(jnp.all(pcs_segment_lengths > 0.0)):
        raise ValueError("pcs_segment_lengths entries must be positive.")

    segment_lengths_by_pneumatic_segment = []
    start = 0
    for i, count in enumerate(pcs_segment_counts):
        stop = start + count
        child_lengths = pcs_segment_lengths[start:stop]
        child_length_sum = float(jnp.sum(child_lengths))
        pneumatic_length = float(physical_lengths[pneumatic_physical_indices[i]])
        if not math.isclose(
            child_length_sum,
            pneumatic_length,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Each pcs_segment_lengths group must sum to the matching "
                "pneumatic segment length; "
                f"group {i} sums to {child_length_sum}, "
                f"expected {pneumatic_length}."
            )
        segment_lengths_by_pneumatic_segment.append(child_lengths)
        start = stop
    return tuple(segment_lengths_by_pneumatic_segment)


def _expand_isupport_layout(
    params: ISupportParams,
    structure: ISupportStructure,
) -> tuple[ISupportParams, PCSStructure, ISupportStructure, Array, Array]:
    params = _with_default_chamber_azimuth_angles(params)
    params.validate()
    structure = _normalize_structure_for_params(params, structure)

    num_physical_segments = int(params.length.shape[0])
    physical_lengths = jnp.asarray(params.length, dtype=jnp.float64)
    pcs_segment_lengths = _resolve_pcs_segment_lengths(
        params, structure.pcs_segment_counts, structure.rigid_segment_selector
    )
    radius = jnp.asarray(params.radius, dtype=jnp.float64)
    density = jnp.asarray(params.density, dtype=jnp.float64)
    young_modulus = jnp.asarray(params.young_modulus, dtype=jnp.float64)
    shear_modulus = jnp.asarray(params.shear_modulus, dtype=jnp.float64)
    reference_strain = jnp.asarray(params.reference_strain, dtype=jnp.float64).reshape(
        num_physical_segments, 6
    )
    straight_reference_strain = _straight_reference_strain(reference_strain.dtype)
    for physical_index, is_rigid in enumerate(structure.rigid_segment_selector):
        if is_rigid and not bool(
            jnp.allclose(reference_strain[physical_index], straight_reference_strain)
        ):
            raise ValueError(
                "Rigid segment reference_strain entries must equal "
                "[0, 0, 0, 1, 0, 0]; "
                f"physical segment {physical_index} does not."
            )
    has_damping_matrix = params.damping_matrix is not None
    if has_damping_matrix:
        damping_matrix = jnp.asarray(params.damping_matrix, dtype=jnp.float64)
        damping_blocks = jnp.stack(
            [
                damping_matrix[6 * i : 6 * (i + 1), 6 * i : 6 * (i + 1)]
                for i in range(num_physical_segments)
            ]
        )
        if not bool(jnp.allclose(damping_matrix, blk_diag(damping_blocks))):
            raise ValueError(
                "ISupport damping_matrix must be block diagonal by physical segment "
                "before expansion into PCS segments."
            )
    else:
        damping_matrix = None
        damping_blocks = None
        material_damping_coefficient = jnp.asarray(
            params.material_damping_coefficient, dtype=jnp.float64
        )
        if material_damping_coefficient.ndim == 0:
            material_damping_coefficient = jnp.full(
                (num_physical_segments,),
                material_damping_coefficient,
                dtype=jnp.float64,
            )

    r_chamber_in = jnp.asarray(params.chamber_inner_radius, dtype=jnp.float64)
    r_chamber_out = jnp.asarray(params.chamber_outer_radius, dtype=jnp.float64)
    d_chamber = jnp.asarray(params.chamber_distance, dtype=jnp.float64)
    chamber_azimuth_angles = jnp.asarray(
        params.chamber_azimuth_angles, dtype=jnp.float64
    )

    expanded_lengths: list[float] = []
    expanded_radius: list[Array] = []
    expanded_density: list[Array] = []
    expanded_young_modulus: list[Array] = []
    expanded_shear_modulus: list[Array] = []
    expanded_reference_strain: list[Array] = []
    expanded_damping_blocks: list[Array] = []
    expanded_material_damping_coefficient: list[Array] = []
    expanded_chamber_inner_radius: list[Array] = []
    expanded_chamber_outer_radius: list[Array] = []
    expanded_chamber_distance: list[Array] = []
    expanded_chamber_azimuth_angles: list[Array] = []
    default_strain_selector: list[Array] = []
    pcs_segment_to_pneumatic_segment: list[int] = []
    pcs_segment_is_rigid: list[bool] = []

    def append_segment(
        *,
        length: Array,
        source_index: int,
        pneumatic_index: int | None,
        is_rigid: bool,
    ) -> None:
        expanded_lengths.append(jnp.asarray(length, dtype=jnp.float64))
        expanded_radius.append(radius[source_index])
        expanded_density.append(density[source_index])
        expanded_young_modulus.append(young_modulus[source_index])
        expanded_shear_modulus.append(shear_modulus[source_index])
        chamber_index = 0 if pneumatic_index is None else pneumatic_index
        expanded_chamber_inner_radius.append(r_chamber_in[chamber_index])
        expanded_chamber_outer_radius.append(r_chamber_out[chamber_index])
        expanded_chamber_distance.append(d_chamber[chamber_index])
        expanded_chamber_azimuth_angles.append(chamber_azimuth_angles[chamber_index])

        if is_rigid:
            expanded_reference_strain.append(reference_strain[source_index])
            if has_damping_matrix:
                expanded_damping_blocks.append(damping_blocks[source_index])
            else:
                expanded_material_damping_coefficient.append(
                    material_damping_coefficient[source_index]
                )
            default_strain_selector.append(jnp.zeros((6,), dtype=bool))
            pcs_segment_to_pneumatic_segment.append(_RIGID_CONNECTOR_PARENT)
        else:
            length_scale = length / physical_lengths[source_index]
            expanded_reference_strain.append(reference_strain[source_index])
            if has_damping_matrix:
                expanded_damping_blocks.append(
                    damping_blocks[source_index] * length_scale
                )
            else:
                expanded_material_damping_coefficient.append(
                    material_damping_coefficient[source_index]
                )
            default_strain_selector.append(jnp.ones((6,), dtype=bool))
            pcs_segment_to_pneumatic_segment.append(pneumatic_index)
        pcs_segment_is_rigid.append(is_rigid)

    pneumatic_index = 0
    for physical_index, is_rigid in enumerate(structure.rigid_segment_selector):
        if is_rigid:
            append_segment(
                length=physical_lengths[physical_index],
                source_index=physical_index,
                pneumatic_index=None,
                is_rigid=True,
            )
        else:
            for segment_length in pcs_segment_lengths[pneumatic_index]:
                append_segment(
                    length=segment_length,
                    source_index=physical_index,
                    pneumatic_index=pneumatic_index,
                    is_rigid=False,
                )
            pneumatic_index += 1

    strain_selector = jnp.concatenate(default_strain_selector)
    if structure.strain_selector is not None:
        user_strain_selector = jnp.asarray(structure.strain_selector)
        if not jnp.issubdtype(user_strain_selector.dtype, jnp.bool_):
            raise TypeError(
                f"strain_selector must be a boolean array, got {user_strain_selector.dtype}"
            )
        if user_strain_selector.size != strain_selector.size:
            raise ValueError(
                "strain_selector must have one boolean per expanded PCS strain; "
                f"expected {strain_selector.size}, got {user_strain_selector.size}."
            )
        strain_selector = user_strain_selector.reshape(strain_selector.shape) & (
            strain_selector
        )

    expanded_kwargs: dict[str, Array] = {}
    if has_damping_matrix:
        expanded_kwargs["damping_matrix"] = blk_diag(jnp.stack(expanded_damping_blocks))
    else:
        expanded_kwargs["material_damping_coefficient"] = jnp.stack(
            expanded_material_damping_coefficient
        )

    expanded_params = ISupportParams(
        base_pose=params.base_pose,
        length=jnp.stack(expanded_lengths),
        radius=jnp.stack(expanded_radius),
        density=jnp.stack(expanded_density),
        gravity=params.gravity,
        young_modulus=jnp.stack(expanded_young_modulus),
        shear_modulus=jnp.stack(expanded_shear_modulus),
        reference_strain=jnp.concatenate(expanded_reference_strain),
        chamber_inner_radius=jnp.stack(expanded_chamber_inner_radius),
        chamber_outer_radius=jnp.stack(expanded_chamber_outer_radius),
        chamber_distance=jnp.stack(expanded_chamber_distance),
        chamber_azimuth_angles=jnp.stack(expanded_chamber_azimuth_angles),
        **expanded_kwargs,
    )
    pcs_structure = PCSStructure(
        num_gauss_points=structure.num_gauss_points,
        strain_selector=strain_selector,
        scale_rotational_basis_by_length=structure.scale_rotational_basis_by_length,
    )
    return (
        expanded_params,
        pcs_structure,
        structure,
        jnp.asarray(pcs_segment_to_pneumatic_segment, dtype=jnp.int32),
        jnp.asarray(pcs_segment_is_rigid, dtype=bool),
    )


class ISupport(PCS):
    """
    A kinematic and dynamic model for the (AM) I-Support robot based on the Piecewise Constant Strain shape parametrization.

    Attributes:
        num_segments: Number of segments (constant strain sections) along the robot.
        num_actuators: Number of actuators (control inputs) for the robot.
        g0: Initial pose of the robot base as an SE(3) transformation matrix.
        g: Gravitational acceleration vector (embedded in a 6D vector).
            [0, 0, 0, g_x, g_y, g_z]
        length: Length of each segment [m], cached as ``L``.
        radius: Radius of each segment [m], cached as ``r``.
        young_modulus: Elastic modulus of each segment [Pa], cached as ``E``.
        shear_modulus: Shear modulus of each segment [Pa], cached as ``G``.
        density: Density of each segment [kg/m^3], cached as ``rho``.
        damping_matrix: Resolved damping matrix of the flattened strain
            coordinates, cached as ``D_full``. It can be supplied directly or
            derived from a material damping coefficient in Pa*s. Matrix-entry
            units depend on the associated generalized strain coordinates.
        num_active_strains: Number of active strain components (based on strain_selector).
        num_strains: Total number of strain components (6 * num_segments).
        B_xi: Basis matrix for projecting active strains (6 * num_segments, num_active_strains).
        xi_ref: Reference strain (reference configuration) of the robot.
        num_gauss_points: Requested nonzero Gauss-Legendre quadrature nodes.
        num_integration_points: Stored integration nodes, including zero-weight endpoints.
        integration_points, integration_weights: Quadrature nodes and weights.
        chamber_inner_radius: Inner radius of each segment's actuator [m],
            cached as ``r_chamber_in``.
        chamber_outer_radius: Outer radius of each segment's actuator [m],
            cached as ``r_chamber_out``.
        chamber_distance: Radial distance of the center of the actuators from
            the centerline of the backbone [m], cached as ``d_chamber``.
        chamber_azimuth_angles: Explicit chamber azimuths [rad], right-handed
            about local +X from +Y toward +Z. The second axis is pressure-channel order.

    References:
        Arleo, L., Stano, G., Percoco, G., & Cianchetti, M. (2021). I-support soft
        arm for assistance tasks: a new manufacturing approach based on 3D printing
        and characterization. Progress in Additive Manufacturing, 6(2), 243-256.
        https://doi.org/10.1007/s40964-020-00158-y

        Alessi, C., Falotico, E., & Lucantonio, A. (2023). Ablation study of a
        dynamic model for a 3D-printed pneumatic soft robotic arm. IEEE Access, 11,
        37840-37853. https://doi.org/10.1109/ACCESS.2023.3265261

        Alessi, C., Bianchi, D., Stano, G., Cianchetti, M., & Falotico, E. (2024).
        Pushing with soft robotic arms via deep reinforcement learning. Advanced
        Intelligent Systems, 6(8), 2300899. https://doi.org/10.1002/aisy.202300899

    """

    params: ISupportParams
    pcs_params: ISupportParams
    structure: ISupportStructure

    r_chamber_in: Array  # inner radius of each segment's chamber, shape (num_segments,)
    r_chamber_out: (
        Array  # outer radius of each segment's chamber, shape (num_segments,)
    )
    d_chamber: Array  # radial distance of the center of the chambers from the centerline of the backbone, shape (num_segments,)
    # Expanded per-PCS-segment copy of the resolved channel layout.
    chamber_azimuth_angles: Array

    num_chambers_per_segment: int = eqx.field(
        static=True, default=3
    )  # number of pneumatic chambers per segment
    num_pneumatic_segments: int = eqx.field(static=True)
    num_pcs_segments: int = eqx.field(static=True)
    pcs_segment_to_pneumatic_segment: Array
    pcs_segment_is_rigid: Array

    def __init__(
        self,
        params: ISupportParams,
        structure: ISupportStructure | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the ISupport class

        Args:
            params: Dynamic I-SUPPORT parameters.
            structure: Static I-SUPPORT layout. If omitted, physical segments
                alternate rigid and pneumatic from index zero, and each
                pneumatic segment is represented by one PCS segment.
            **kwargs: Additional keyword arguments.
        """
        if not isinstance(params, ISupportParams):
            raise TypeError("params must be an ISupportParams instance.")
        params = _with_default_chamber_azimuth_angles(params)
        if structure is None:
            structure = ISupportStructure()
        if not isinstance(structure, ISupportStructure):
            raise TypeError("structure must be an ISupportStructure instance.")

        self.num_chambers_per_segment = int(params.chamber_azimuth_angles.shape[1])
        (
            pcs_params,
            pcs_structure,
            normalized_structure,
            pcs_segment_to_pneumatic_segment,
            pcs_segment_is_rigid,
        ) = _expand_isupport_layout(params, structure)
        # Geometry-dependent caches are constructed by PCS.__init__, so expose
        # the expanded type mask before delegating to it.
        self.pcs_segment_is_rigid = pcs_segment_is_rigid
        chamber_actuator = _pneumatic_chamber_actuator(
            params, pcs_segment_to_pneumatic_segment
        )
        super().__init__(
            pcs_params,
            structure=pcs_structure,
            actuators=chamber_actuator,
            passive_elements=(),
            **kwargs,
        )

        self.params = params
        self.pcs_params = pcs_params
        self.structure = normalized_structure
        self.num_pneumatic_segments = sum(
            not is_rigid for is_rigid in normalized_structure.rigid_segment_selector
        )
        self.num_pcs_segments = self.num_segments
        self.pcs_segment_to_pneumatic_segment = pcs_segment_to_pneumatic_segment
        self.pcs_segment_is_rigid = pcs_segment_is_rigid

    def _set_params(self, params: ISupportParams):
        """
        Set cached runtime arrays from typed I-SUPPORT parameters.

        Args:
            params: Dynamic parameters for the spatial I-SUPPORT PCS model.
                Expected fields:
                - ``base_pose``: Array of shape ``(7,)``.
                    Scalar-first quaternion pose
                    ``[qw, qx, qy, qz, x0, y0, z0]`` in the inertial frame.
                    The quaternion represents the base orientation and the
                    translation is inserted directly into the homogeneous
                    transform.
                - ``length``: Array of shape ``(num_segments,)``.
                    Length of each segment [m].
                - ``radius``: Array of shape ``(num_segments,)``.
                    Radius of each segment [m].
                - ``density``: Array of shape ``(num_segments,)``.
                    Density of each segment [kg/m^3].
                - ``gravity``: Array of shape ``(3,)``.
                    Gravitational acceleration vector ``[gx, gy, gz]``
                    [m/s^2]. It is cached as the 6D vector
                    ``[0, 0, 0, gx, gy, gz]``.
                - ``young_modulus``: Array of shape ``(num_segments,)``.
                    Elastic modulus of each segment [Pa].
                - ``shear_modulus``: Array of shape ``(num_segments,)``.
                    Shear modulus of each segment [Pa].
                - ``material_damping_coefficient``: Scalar or array of shape
                    ``(num_segments,)``. Material damping coefficient used to
                    derive segment damping from geometry [Pa*s] ([N*s/m^2]).
                - ``damping_matrix``: Optional custom array of shape
                    ``(6 * num_segments, 6 * num_segments)``.
                    Damping matrix of the flattened strain coordinates; its
                    entry units depend on the paired generalized strain
                    coordinates. Exactly one damping input must be provided.
                - ``reference_strain``: Array with ``6 * num_segments``
                    entries. Reference strain of the robot.
                - ``chamber_inner_radius``: Array of shape
                    ``(num_segments,)``. Inner radius of each segment's
                    pneumatic chamber [m].
                - ``chamber_outer_radius``: Array of shape
                    ``(num_segments,)``. Outer radius of each segment's
                    pneumatic chamber [m].
                - ``chamber_distance``: Array of shape ``(num_segments,)``.
                    Radial distance of the center of the chambers from the
                    centerline of the backbone [m].
                - ``chamber_azimuth_angles``: Array of shape
                    ``(num_segments, num_chambers_per_segment)``. Right-handed
                    azimuth about local +X, from local +Y toward local +Z [rad].
        """
        super()._set_params(params)

        # Pneumatic chamber parameters
        r_chamber_in = jnp.asarray(params.chamber_inner_radius, dtype=jnp.float64)
        if r_chamber_in.shape != (self.num_segments,):
            raise ValueError(
                "chamber_inner_radius must have shape "
                f"({self.num_segments},), got {r_chamber_in.shape}."
            )
        self.r_chamber_in = r_chamber_in

        r_chamber_out = jnp.asarray(params.chamber_outer_radius, dtype=jnp.float64)
        if r_chamber_out.shape != (self.num_segments,):
            raise ValueError(
                "chamber_outer_radius must have shape "
                f"({self.num_segments},), got {r_chamber_out.shape}."
            )
        self.r_chamber_out = r_chamber_out

        d_chamber = jnp.asarray(params.chamber_distance, dtype=jnp.float64)
        if d_chamber.shape != (self.num_segments,):
            raise ValueError(
                "chamber_distance must have shape "
                f"({self.num_segments},), got {d_chamber.shape}."
            )
        self.d_chamber = d_chamber

        chamber_azimuth_angles = jnp.asarray(
            params.chamber_azimuth_angles, dtype=jnp.float64
        )
        if chamber_azimuth_angles.shape != (
            self.num_segments,
            self.num_chambers_per_segment,
        ):
            raise ValueError(
                "chamber_azimuth_angles must have shape "
                f"({self.num_segments}, {self.num_chambers_per_segment}), got {chamber_azimuth_angles.shape}."
            )
        self.chamber_azimuth_angles = chamber_azimuth_angles

    def _current_body_params(self) -> ISupportParams:
        """Return the expanded PCS params used by inherited PCS update helpers."""
        return getattr(self, "pcs_params", self.params)

    def with_params(self, params: ISupportParams) -> "ISupport":
        """Return an updated copy with a full pneumatic-segment parameter object."""
        if not isinstance(params, ISupportParams):
            raise TypeError("params must be an ISupportParams instance.")
        params = _with_default_chamber_azimuth_angles(params)
        if params.chamber_azimuth_angles.shape[1] != self.num_chambers_per_segment:
            raise ValueError(
                "Changing the number of chambers per pneumatic segment requires "
                "reconstruction."
            )
        (
            pcs_params,
            _,
            _,
            pcs_segment_to_pneumatic_segment,
            _,
        ) = _expand_isupport_layout(params, self.structure)
        chamber_actuator = _pneumatic_chamber_actuator(
            params, pcs_segment_to_pneumatic_segment
        )
        current_actuator = self.actuators[0]
        chamber_actuator_params = chamber_actuator.params.replace(
            lower_bounds=current_actuator.params.lower_bounds,
            upper_bounds=current_actuator.params.upper_bounds,
        )
        chamber_actuator = current_actuator.with_params(chamber_actuator_params)
        chamber_arrays = (
            jnp.asarray(pcs_params.chamber_inner_radius, dtype=jnp.float64),
            jnp.asarray(pcs_params.chamber_outer_radius, dtype=jnp.float64),
            jnp.asarray(pcs_params.chamber_distance, dtype=jnp.float64),
            jnp.asarray(pcs_params.chamber_azimuth_angles, dtype=jnp.float64),
        )
        updated_self = self._with_pcs_params(pcs_params, stored_params=params)
        updated_self = eqx.tree_at(
            lambda model: (
                model.pcs_params,
                model.r_chamber_in,
                model.r_chamber_out,
                model.d_chamber,
                model.chamber_azimuth_angles,
                model.actuators,
            ),
            updated_self,
            (
                pcs_params,
                *chamber_arrays,
                (chamber_actuator,),
            ),
        )
        return updated_self._with_refreshed_precomputed_matrices()

    def update_params(self, **updates: Array) -> "ISupport":
        """Return an updated copy with selected pneumatic parameter fields replaced."""
        return self.with_params(self.params.replace(**updates))

    @eqx.filter_jit
    def chamber_azimuths(self, pneumatic_segment_idx: Array) -> Array:
        """Return one azimuth per chamber in pressure-channel order.

        This concise accessor name is distinct from the parameter field
        ``chamber_azimuth_angles``. The returned array has shape
        ``(num_chambers_per_segment,)`` and is not sorted or normalized.

        Args:
            pneumatic_segment_idx: Index of the physical pneumatic segment.

        Returns:
            Chamber azimuth angles in radians, indexed exactly like the
            segment's pressure inputs.
        """
        return self.params.chamber_azimuth_angles[pneumatic_segment_idx]

    @eqx.filter_jit
    def local_chamber_offsets(self, pneumatic_segment_idx: Array) -> Array:
        """Return vectors from the local backbone centerline to chamber centers.

        The result has shape ``(num_chambers_per_segment, 3)`` and follows
        pressure-channel order. Each row is expressed in the pneumatic
        segment's local frame as ``[0, d*cos(phi), d*sin(phi)]``; it is an
        offset vector, not an absolute chamber-center position.

        Args:
            pneumatic_segment_idx: Index of the physical pneumatic segment.

        Returns:
            Local chamber-center offset vectors in meters.
        """
        phi = self.chamber_azimuths(pneumatic_segment_idx)
        distance = self.params.chamber_distance[pneumatic_segment_idx]
        return distance * jnp.stack(
            [jnp.zeros_like(phi), jnp.cos(phi), jnp.sin(phi)], axis=-1
        )

    @eqx.filter_jit
    def _local_actuator_centroid(self, i: Array, varphi_angle: Array) -> Array:
        """
        Compute the position of the centroid of one pneumatic actuator in the local reference frame of the i-th segment.

        Args:
            i (Array): index of the segment as array of shape ()
            varphi_angle (Array): polar angle of the actuator center from the local z-axis as Array of shape ()

        Returns:
            centroid_actuator (Array): position of the centroid of one pneumatic actuator in the local reference frame of the i-th segment as array of shape (3, )
        """
        centroid_actuator = jnp.array(
            [
                0.0,  # the actuator centroid is on the cross-section
                self.d_chamber[i] * jnp.cos(varphi_angle),
                self.d_chamber[i] * jnp.sin(varphi_angle),
            ]
        )
        return centroid_actuator

    @eqx.filter_jit
    def _local_actuator_cross_sectional_area(self, i: Array) -> Array:
        """
        Compute the local cross-sectional area of one actuator for the i-th segment.
        This should not be confused with the cross-sectional area of the pneumatic chambers that are located within the actuator and used to define the actuation matrix.
        Instead, it represents the cross-sectional area of the actuator itself that is considered for inertial and stiffness calculations.

        Args:
            i (Array): index of the segment as array of shape ()

        Returns:
            A_one_actuator_i (Array): local cross-sectional area of one actuator of the i-th segment
        """
        A_one_actuator_i = jnp.pi * (
            self.r_chamber_out[i] ** 2 - self.r_chamber_in[i] ** 2
        )

        return A_one_actuator_i

    @eqx.filter_jit
    def _local_cross_sectional_area(self, i: Array) -> Array:
        """
        Compute the local cross-sectional area for the i-th segment.

        Args:
            i (Array): index of the segment as array of shape ()

        Returns:
            A_i (Array): local cross-sectional area of the i-th segment as array of shape ()
        """
        pneumatic_area = (
            self.num_chambers_per_segment * self._local_actuator_cross_sectional_area(i)
        )
        rigid_area = super()._local_cross_sectional_area(i)
        return jnp.where(self.pcs_segment_is_rigid[i], rigid_area, pneumatic_area)

    def _local_actuator_second_moment_of_area(
        self, i: Array, varphi_chamber: Array
    ) -> Array:
        """
        Compute the local second moment of area of one actuator for the i-th segment.
        This should not be confused with the second moment of area of the pneumatic chambers that are located within the actuator and used to define the actuation matrix.
        Instead, it represents the second moment of area of the actuator itself that is considered for inertial and stiffness calculations.

        Args:
            i (Array): index of the segment as array of shape ()
            varphi_chamber (Array): polar angle of the chamber center from the local z-axis as Array of shape ()

        Returns:
            I_one_actuator_i (Array): local second moment of area of one actuator of the i-th segment as array of shape (3, )
        """
        # second moment of area of one pneumatic chamber around its own centroid
        I0 = jnp.array(
            [
                jnp.pi
                * (self.r_chamber_out[i] ** 4 - self.r_chamber_in[i] ** 4)
                / 2,  # twist strain
                jnp.pi
                * (self.r_chamber_out[i] ** 4 - self.r_chamber_in[i] ** 4)
                / 4,  # bending strain around local y-axis
                jnp.pi
                * (self.r_chamber_out[i] ** 4 - self.r_chamber_in[i] ** 4)
                / 4,  # bending strain around local z-axis
            ]
        )

        # position of centroid of actuator in local reference frame
        centroid_actuator = self._local_actuator_centroid(i, varphi_chamber)
        # area of the the actuator
        A = self._local_actuator_cross_sectional_area(i)

        # apply the parallel axis theorem
        I_one_actuator_i = I0 + A * (
            jnp.linalg.norm(centroid_actuator) ** 2 * jnp.ones((3,))
            - centroid_actuator**2
        )

        return I_one_actuator_i

    @eqx.filter_jit
    def _local_second_moment_of_area(self, i: Array) -> Array:
        """
        Compute the local second moment of area for the i-th segment.

        Args:
            i (Array): index of the segment as array of shape ()

        Returns:
            I_i (Array): local second moment of area of the i-th segment as array of shape (3, )
        """

        # Compute the pneumatic-section moments from its actuator annuli.
        varphi_actuators_i = self.chamber_azimuth_angles[i]
        I_actuators_i = vmap(
            lambda varphi: self._local_actuator_second_moment_of_area(i, varphi)
        )(varphi_actuators_i)
        pneumatic_moments = jnp.sum(I_actuators_i, axis=0)
        rigid_moments = super()._local_second_moment_of_area(i)
        return jnp.where(self.pcs_segment_is_rigid[i], rigid_moments, pneumatic_moments)

    def actuator_visual_layers(
        self,
        q: Array,
        s_points: Array,
        *,
        actuator_inputs: Array | None = None,
    ) -> tuple:
        """Keep equivalent chamber-center paths out of generic renderers."""
        del q, s_points, actuator_inputs
        return ()
