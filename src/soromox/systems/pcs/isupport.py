import math
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jax import Array, vmap

from soromox.systems.pcs.params import ISupportParams
from soromox.systems.pcs.structures import ISupportStructure, PCSStructure
from soromox.utils.array_math import blk_diag

from .pcs import PCS

_RIGID_CONNECTOR_PARENT = -1


def _straight_reference_strain(dtype: jnp.dtype) -> Array:
    return jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=dtype)


def _connector_source_index(connector_index: int, num_pneumatic_segments: int) -> int:
    if connector_index <= 0:
        return 0
    if connector_index >= num_pneumatic_segments:
        return num_pneumatic_segments - 1
    return connector_index - 1


def _normalize_structure_for_params(
    params: ISupportParams,
    structure: ISupportStructure,
) -> ISupportStructure:
    num_pneumatic_segments = int(params.length.shape[0])

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

    rigid_connector_selector = structure.rigid_connector_selector
    num_rigid_connectors = num_pneumatic_segments + 1
    if rigid_connector_selector is None:
        rigid_connector_selector = tuple(False for _ in range(num_rigid_connectors))
    elif len(rigid_connector_selector) == 1 and num_rigid_connectors > 1:
        rigid_connector_selector = rigid_connector_selector * num_rigid_connectors
    if len(rigid_connector_selector) != num_rigid_connectors:
        raise ValueError(
            "rigid_connector_selector must contain one more entry than there are "
            "pneumatic segments; "
            f"expected {num_rigid_connectors}, "
            f"got {len(rigid_connector_selector)}."
        )

    return ISupportStructure(
        num_gauss_points=structure.num_gauss_points,
        pcs_segment_counts=pcs_segment_counts,
        rigid_connector_selector=rigid_connector_selector,
        strain_selector=structure.strain_selector,
        scale_rotational_basis_by_length=structure.scale_rotational_basis_by_length,
    )


def _resolve_pcs_segment_lengths(
    params: ISupportParams,
    pcs_segment_counts: tuple[int, ...],
) -> tuple[Array, ...]:
    pneumatic_lengths = jnp.asarray(params.length, dtype=jnp.float64)

    if params.pcs_segment_lengths is None:
        return tuple(
            jnp.full((count,), pneumatic_lengths[i] / count, dtype=jnp.float64)
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
            "pcs_segment_lengths must contain one length per soft PCS segment; "
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
        pneumatic_length = float(pneumatic_lengths[i])
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


def _resolve_rigid_connector_lengths(
    params: ISupportParams,
    rigid_connector_selector: tuple[bool, ...],
) -> Array:
    num_rigid_connectors = len(rigid_connector_selector)
    if params.rigid_connector_lengths is None:
        rigid_connector_lengths = jnp.zeros((num_rigid_connectors,), dtype=jnp.float64)
    else:
        rigid_connector_lengths = jnp.asarray(
            params.rigid_connector_lengths, dtype=jnp.float64
        )
    if rigid_connector_lengths.shape != (num_rigid_connectors,):
        raise ValueError(
            "rigid_connector_lengths must contain one entry per canonical "
            "connector slot; "
            f"expected {num_rigid_connectors}, got {rigid_connector_lengths.shape}."
        )
    if not bool(jnp.all(rigid_connector_lengths >= 0.0)):
        raise ValueError("rigid_connector_lengths entries must be nonnegative.")

    for connector_index, is_selected in enumerate(rigid_connector_selector):
        connector_length = float(rigid_connector_lengths[connector_index])
        if is_selected:
            if connector_length <= 0.0:
                raise ValueError(
                    "Selected rigid connectors must have positive lengths; "
                    f"rigid_connector_lengths[{connector_index}] is "
                    f"{connector_length}."
                )
        elif not math.isclose(connector_length, 0.0, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                "rigid_connector_lengths contains a nonzero length for an "
                "unselected connector; set "
                f"rigid_connector_selector[{connector_index}] to True or set "
                f"rigid_connector_lengths[{connector_index}] to zero."
            )
    return rigid_connector_lengths


def _expand_isupport_layout(
    params: ISupportParams,
    structure: ISupportStructure,
) -> tuple[ISupportParams, PCSStructure, ISupportStructure, Array, Array]:
    params.validate()
    structure = _normalize_structure_for_params(params, structure)

    num_pneumatic_segments = int(params.length.shape[0])
    pneumatic_lengths = jnp.asarray(params.length, dtype=jnp.float64)
    pcs_segment_lengths = _resolve_pcs_segment_lengths(
        params, structure.pcs_segment_counts
    )
    rigid_connector_lengths = _resolve_rigid_connector_lengths(
        params, structure.rigid_connector_selector
    )
    radius = jnp.asarray(params.radius, dtype=jnp.float64)
    density = jnp.asarray(params.density, dtype=jnp.float64)
    young_modulus = jnp.asarray(params.young_modulus, dtype=jnp.float64)
    shear_modulus = jnp.asarray(params.shear_modulus, dtype=jnp.float64)
    reference_strain = jnp.asarray(params.reference_strain, dtype=jnp.float64).reshape(
        num_pneumatic_segments, 6
    )
    damping_matrix = jnp.asarray(params.damping_matrix, dtype=jnp.float64)
    damping_blocks = jnp.stack(
        [
            damping_matrix[6 * i : 6 * (i + 1), 6 * i : 6 * (i + 1)]
            for i in range(num_pneumatic_segments)
        ]
    )
    if not bool(jnp.allclose(damping_matrix, blk_diag(damping_blocks))):
        raise ValueError(
            "ISupport damping_matrix must be block diagonal by pneumatic segment "
            "before expansion into PCS segments."
        )

    r_chamber_in = jnp.asarray(params.chamber_inner_radius, dtype=jnp.float64)
    r_chamber_out = jnp.asarray(params.chamber_outer_radius, dtype=jnp.float64)
    d_chamber = jnp.asarray(params.chamber_distance, dtype=jnp.float64)
    varphi_chamber_off = jnp.asarray(params.chamber_angle_offset, dtype=jnp.float64)

    expanded_lengths: list[float] = []
    expanded_radius: list[Array] = []
    expanded_density: list[Array] = []
    expanded_young_modulus: list[Array] = []
    expanded_shear_modulus: list[Array] = []
    expanded_reference_strain: list[Array] = []
    expanded_damping_blocks: list[Array] = []
    expanded_chamber_inner_radius: list[Array] = []
    expanded_chamber_outer_radius: list[Array] = []
    expanded_chamber_distance: list[Array] = []
    expanded_chamber_angle_offset: list[Array] = []
    default_strain_selector: list[Array] = []
    pcs_segment_to_pneumatic_segment: list[int] = []
    pcs_segment_is_rigid: list[bool] = []

    def append_segment(
        *,
        length: Array,
        source_index: int,
        pneumatic_index: int,
        is_rigid: bool,
    ) -> None:
        expanded_lengths.append(jnp.asarray(length, dtype=jnp.float64))
        expanded_radius.append(radius[source_index])
        expanded_density.append(density[source_index])
        expanded_young_modulus.append(young_modulus[source_index])
        expanded_shear_modulus.append(shear_modulus[source_index])
        expanded_chamber_inner_radius.append(r_chamber_in[source_index])
        expanded_chamber_outer_radius.append(r_chamber_out[source_index])
        expanded_chamber_distance.append(d_chamber[source_index])
        expanded_chamber_angle_offset.append(varphi_chamber_off[source_index])

        if is_rigid:
            expanded_reference_strain.append(
                _straight_reference_strain(reference_strain.dtype)
            )
            expanded_damping_blocks.append(
                jnp.zeros((6, 6), dtype=damping_matrix.dtype)
            )
            default_strain_selector.append(jnp.zeros((6,), dtype=bool))
            pcs_segment_to_pneumatic_segment.append(_RIGID_CONNECTOR_PARENT)
        else:
            length_scale = length / pneumatic_lengths[pneumatic_index]
            expanded_reference_strain.append(reference_strain[pneumatic_index])
            expanded_damping_blocks.append(
                damping_blocks[pneumatic_index] * length_scale
            )
            default_strain_selector.append(jnp.ones((6,), dtype=bool))
            pcs_segment_to_pneumatic_segment.append(pneumatic_index)
        pcs_segment_is_rigid.append(is_rigid)

    for connector_index in range(num_pneumatic_segments + 1):
        if structure.rigid_connector_selector[connector_index]:
            append_segment(
                length=rigid_connector_lengths[connector_index],
                source_index=_connector_source_index(
                    connector_index, num_pneumatic_segments
                ),
                pneumatic_index=_RIGID_CONNECTOR_PARENT,
                is_rigid=True,
            )

        if connector_index < num_pneumatic_segments:
            for segment_length in pcs_segment_lengths[connector_index]:
                append_segment(
                    length=segment_length,
                    source_index=connector_index,
                    pneumatic_index=connector_index,
                    is_rigid=False,
                )

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

    expanded_params = ISupportParams(
        base_pose=params.base_pose,
        length=jnp.stack(expanded_lengths),
        radius=jnp.stack(expanded_radius),
        density=jnp.stack(expanded_density),
        gravity=params.gravity,
        young_modulus=jnp.stack(expanded_young_modulus),
        shear_modulus=jnp.stack(expanded_shear_modulus),
        damping_matrix=blk_diag(jnp.stack(expanded_damping_blocks)),
        reference_strain=jnp.concatenate(expanded_reference_strain),
        chamber_inner_radius=jnp.stack(expanded_chamber_inner_radius),
        chamber_outer_radius=jnp.stack(expanded_chamber_outer_radius),
        chamber_distance=jnp.stack(expanded_chamber_distance),
        chamber_angle_offset=jnp.stack(expanded_chamber_angle_offset),
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
        damping_matrix: Damping matrix of the flattened strain coordinates
            [Pa*s], cached as ``D``.
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
        chamber_angle_offset: Angular offset of the first actuator from the
            local segment frame [rad], cached as ``varphi_chamber_off``.

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
    varphi_chamber_off: Array  # angular offset of the first actuator from the local

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
        num_chambers_per_segment: int = 3,
        **kwargs: Any,
    ):
        """
        Initialize the ISupport class

        Args:
            params: Dynamic I-SUPPORT parameters.
            structure: Static I-SUPPORT layout. If omitted, each pneumatic
                segment is represented by one PCS segment and no rigid
                connectors are inserted.
            num_chambers_per_segment:
                Number of pneumatic chambers per segment. Defaults to 3.
            **kwargs: Additional keyword arguments.
        """
        if not isinstance(params, ISupportParams):
            raise TypeError("params must be an ISupportParams instance.")
        if structure is None:
            structure = ISupportStructure()
        if not isinstance(structure, ISupportStructure):
            raise TypeError("structure must be an ISupportStructure instance.")

        self.num_chambers_per_segment = num_chambers_per_segment
        (
            pcs_params,
            pcs_structure,
            normalized_structure,
            pcs_segment_to_pneumatic_segment,
            pcs_segment_is_rigid,
        ) = _expand_isupport_layout(params, structure)
        super().__init__(pcs_params, structure=pcs_structure, **kwargs)

        self.params = params
        self.pcs_params = pcs_params
        self.structure = normalized_structure
        self.num_pneumatic_segments = int(params.length.shape[0])
        self.num_pcs_segments = self.num_segments
        self.pcs_segment_to_pneumatic_segment = pcs_segment_to_pneumatic_segment
        self.pcs_segment_is_rigid = pcs_segment_is_rigid
        self.num_actuators = (
            self.num_pneumatic_segments * self.num_chambers_per_segment
        )  # each pneumatic segment has one pressure input per chamber

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
                - ``damping_matrix``: Array of shape
                    ``(6 * num_segments, 6 * num_segments)``.
                    Damping matrix of the flattened strain coordinates [Pa*s].
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
                - ``chamber_angle_offset``: Array of shape
                    ``(num_segments,)``. Angular offset of the first actuator
                    from the local z-axis [rad].
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

        varphi_chamber_off = jnp.asarray(params.chamber_angle_offset, dtype=jnp.float64)
        if varphi_chamber_off.shape != (self.num_segments,):
            raise ValueError(
                "chamber_angle_offset must have shape "
                f"({self.num_segments},), got {varphi_chamber_off.shape}."
            )
        self.varphi_chamber_off = varphi_chamber_off

    def _current_body_params(self) -> ISupportParams:
        """Return the expanded PCS params used by inherited PCS update helpers."""
        return self.pcs_params

    def with_params(self, params: ISupportParams) -> "ISupport":
        """Return an updated copy with a full pneumatic-segment parameter object."""
        if not isinstance(params, ISupportParams):
            raise TypeError("params must be an ISupportParams instance.")
        (
            pcs_params,
            _,
            _,
            _,
            _,
        ) = _expand_isupport_layout(params, self.structure)
        chamber_arrays = (
            jnp.asarray(pcs_params.chamber_inner_radius, dtype=jnp.float64),
            jnp.asarray(pcs_params.chamber_outer_radius, dtype=jnp.float64),
            jnp.asarray(pcs_params.chamber_distance, dtype=jnp.float64),
            jnp.asarray(pcs_params.chamber_angle_offset, dtype=jnp.float64),
        )
        updated_self = self._with_pcs_params(pcs_params, stored_params=params)
        updated_self = eqx.tree_at(
            lambda model: (
                model.pcs_params,
                model.r_chamber_in,
                model.r_chamber_out,
                model.d_chamber,
                model.varphi_chamber_off,
            ),
            updated_self,
            (pcs_params, *chamber_arrays),
        )
        return updated_self._with_refreshed_precomputed_matrices()

    def update_params(self, **updates: Array) -> "ISupport":
        """Return an updated copy with selected pneumatic parameter fields replaced."""
        return self.with_params(self.params.replace(**updates))

    @eqx.filter_jit
    def _local_actuator_polar_angles(self, i: Array) -> Array:
        """
        Compute the polar angles of the chambers in the i-th segment.

        Args:
            i (Array): index of the segment as array of shape ()

        Returns:
            varphi_chambers_i (Array): polar angles of the i-th pneumatic chamber as Array of shape (num_chambers_per_segment, )
        """
        varphi_chambers_i = self.varphi_chamber_off[i] + jnp.linspace(
            0, 2 * jnp.pi, self.num_chambers_per_segment, endpoint=False
        )
        return varphi_chambers_i

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
                self.d_chamber[i] * jnp.sin(varphi_angle),  # up along local y-axis
                self.d_chamber[i] * jnp.cos(varphi_angle),  # right along local z-axis
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
        A_one_actuator_i = self._local_actuator_cross_sectional_area(i)
        # we assume that the cross-section only consists of the actuators
        A_i = self.num_chambers_per_segment * A_one_actuator_i

        return A_i

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

        # compute the polar angles of the actuators
        varphi_actuators_i = self._local_actuator_polar_angles(i)
        # compute the second moment of area of each actuator
        I_actuators_i = vmap(
            lambda varphi: self._local_actuator_second_moment_of_area(i, varphi)
        )(varphi_actuators_i)

        # the second moment of area is assumed to be the sum across all actuators
        I_i = jnp.sum(I_actuators_i, axis=0)

        return I_i

    @eqx.filter_jit
    def actuation_matrix(self, q: Array) -> Array:
        """
        Compute the actuation matrix of the robot.
        We assume that each pneumatic segment contains
        self.num_chambers_per_segment identical and symmetric pneumatic chambers
        that are pressurized to p1=u1, p2=u2, p3=u3, etc. If a pneumatic
        segment is split into multiple PCS segments, the same pressure columns
        are applied to each child PCS segment. Rigid connector PCS segments do
        not contribute to the actuation matrix.
        Furthermore, we consider the following geometrical arrangement:
            - The 1st chamber with pressure p1 is located at the
              chamber_angle_offset value for its segment.
            - The 2nd chamber with pressure p2 is located at
              chamber_angle_offset + 1 * 2 * jnp.pi / self.num_chambers_per_segment.
            - The 3rd chamber with pressure p3 is located at
              chamber_angle_offset + 2 * 2 * jnp.pi / self.num_chambers_per_segment.
            - etc.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            A (Array): Actuation matrix of shape (num_active_strains, num_actuators)
        """

        def _actuation_matrix_one_chamber(i: Array, varphi: Array) -> Array:
            # Area of one pneumatic chamber
            A_one_chamber = jnp.pi * self.r_chamber_in[i] ** 2

            # force contribution of the chamber
            force_contrib = jnp.array(
                [A_one_chamber, 0.0, 0.0]
            )  # force along local x-axis

            # compute the centroids of the chambers
            centroid_chamber = self._local_actuator_centroid(i, varphi)

            # compute the contribution of the chamber on the (rotational) backbone torque
            torque_contrib = -jnp.cross(centroid_chamber, force_contrib)

            A_single_chamber = jnp.concatenate(
                [
                    torque_contrib,
                    jnp.array(
                        [
                            A_one_chamber,  # axial strain
                            0.0,  # shear strain local y-dir
                            0.0,  # shear strain local z-dir
                        ]
                    ),
                ]
            )

            return A_single_chamber

        def _actuation_matrix_pcs_segment_i(i: Array) -> Array:
            # compute the local polar chamber angles
            varphi_chambers = self._local_actuator_polar_angles(i)

            # build the actuation matrix
            A_columns_i = vmap(lambda varphi: _actuation_matrix_one_chamber(i, varphi))(
                varphi_chambers
            )
            A_segment_i = jnp.stack(
                A_columns_i, axis=-1
            )  # shape (6, num_chambers_per_segment)

            return A_segment_i

        def _actuation_matrix_pcs_segment_full(i: Array) -> Array:
            A_segment_i = _actuation_matrix_pcs_segment_i(i)
            pneumatic_segment_i = self.pcs_segment_to_pneumatic_segment[i]
            local_column_indices = (
                jnp.arange(self.num_actuators)
                - pneumatic_segment_i * self.num_chambers_per_segment
            )
            valid_columns = (
                (pneumatic_segment_i >= 0)
                & (local_column_indices >= 0)
                & (local_column_indices < self.num_chambers_per_segment)
            )
            local_column_indices = jnp.clip(
                local_column_indices, 0, self.num_chambers_per_segment - 1
            )
            return jnp.where(
                valid_columns[None, :],
                A_segment_i[:, local_column_indices],
                0.0,
            )

        A_blocks_tot = vmap(_actuation_matrix_pcs_segment_full)(
            jnp.arange(self.num_segments),
        )

        # assemble the contributions of the actuation of each PCS segment
        A_full = A_blocks_tot.reshape(self.num_strains, self.num_actuators)
        A = self.B_xi.T @ A_full  # shape (num_active_strains, num_actuators)

        return A
