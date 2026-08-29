"""Construction and material projection for GVS segment specifications."""

from __future__ import annotations

from jax import Array, vmap
from jax import numpy as jnp

from soromox.systems.components import (
    ContinuumLinkParams,
    CrossSectionParams,
    JointParams,
    evaluate_profile,
    section_properties,
)
from soromox.systems.gvs.params import GVSParams
from soromox.systems.gvs.primitives import Basis, Joint
from soromox.systems.gvs.specs import GVSSegment
from soromox.systems.gvs.strain_bases import (
    B_IMQ,
    B_Chebychev,
    B_Fourier,
    B_Gaussian,
    B_LegendrePolynomial,
    B_Monomial,
)
from soromox.systems.gvs.structures import (
    GVSJointStructure,
    GVSLinkStructure,
    GVSSegmentStructure,
    GVSStrainBasisStructure,
    GVSStructure,
)
from soromox.utils.integration import gauss_quadrature

__all__ = ["material_operators_from_params", "params_and_structure_from_segments"]

_BASIS_FUNCTIONS = {
    "monomial": B_Monomial,
    "legendre": B_LegendrePolynomial,
    "chebyshev": B_Chebychev,
    "fourier": B_Fourier,
    "gaussian": B_Gaussian,
    "imq": B_IMQ,
}


def _link_dof(segment: GVSSegmentStructure) -> int:
    return int(
        Basis.DOF_BRANCHES[Basis.BASISTYPE_MAP[segment.basis.type]](
            (
                jnp.asarray(segment.basis.strain_selector),
                jnp.asarray(segment.basis.basis_order),
            )
        )
    )


def _resolve_structure(params: GVSParams, structure: GVSStructure) -> GVSStructure:
    """Return a fully resolved GVS structure compatible with ``params``."""
    params.validate_for_update_against_structure(structure)
    max_dof = (
        int(params.link.stiffness.shape[-1])
        if structure.max_dof is None
        else int(structure.max_dof)
    )
    max_num_gauss_points = (
        max(segment.num_gauss_points for segment in structure.segments)
        if structure.max_num_gauss_points is None
        else int(structure.max_num_gauss_points)
    )
    return GVSStructure(
        segments=structure.segments,
        max_dof=max_dof,
        max_num_gauss_points=max_num_gauss_points,
        scale_rotational_basis_by_length=(structure.scale_rotational_basis_by_length),
    )


def _basis_at_points(
    segment: GVSSegmentStructure, points: Array, max_dof: int
) -> Array:
    function = _BASIS_FUNCTIONS[segment.basis.type]
    selector = jnp.asarray(segment.basis.strain_selector)
    order = jnp.asarray(segment.basis.basis_order)
    return vmap(lambda point: function(point, selector, order, max_dof))(points)


def _dimensions_at_points(
    coefficients: Array, structure: GVSLinkStructure, points: Array
) -> Array:
    dimensions = []
    start = 0
    for profile_type, count in zip(
        structure.cross_section_profile_types,
        structure.cross_section_profile_parameter_counts,
        strict=True,
    ):
        stop = start + count
        dimensions.append(
            evaluate_profile(coefficients[start:stop], profile_type, points)
        )
        start = stop
    return jnp.stack(dimensions, axis=1)


def material_operators_from_params(
    params: GVSParams, structure: GVSStructure
) -> tuple[Array, Array, Array]:
    """Build unit-response material operators for all GVS links.

    The returned arrays project cross-section mechanics through each link's
    configured strain basis and Gauss quadrature rule. They satisfy
    ``K = E * K_E + G * K_G`` and ``D = eta * D_eta`` when the per-link
    material scalars are broadcast over the final two axes.

    Args:
        params: Canonical GVS parameters providing link lengths and packed
            cross-section coefficients.
        structure: Static GVS layout providing cross-section geometries,
            profile metadata, bases, quadrature rules, and the padded DOF size.

    Returns:
        A tuple ``(young_operators, shear_operators, damping_operators)``. Each
        array has shape ``(num_segments, max_dof, max_dof)`` and represents the
        response to a unit Young's modulus, shear modulus, or material damping
        coefficient, respectively.

    Raises:
        ValueError: If the parameters or structure contain incompatible profile
            coefficients, unsupported cross-section data, or invalid quadrature
            configuration.
    """
    params.validate_for_update_against_structure(structure)
    max_dof = (
        int(params.link.stiffness.shape[-1])
        if structure.max_dof is None
        else int(structure.max_dof)
    )
    young_operators = []
    shear_operators = []
    damping_operators = []
    for index, segment in enumerate(structure.segments):
        points, weights, _ = gauss_quadrature(segment.num_gauss_points)
        basis = _basis_at_points(segment, points, max_dof)
        length = params.link.length[index]
        if structure.scale_rotational_basis_by_length:
            basis = basis.at[:, :3, :].divide(length)
        dimensions = _dimensions_at_points(
            params.link.cross_section.coefficients[index], segment.link, points
        )
        geometry = segment.link.cross_section_geometry
        ix, iy, iz, area = vmap(
            lambda value, geometry=geometry: section_properties(geometry, value)
        )(dimensions)
        young_diagonal = jnp.stack(
            [jnp.zeros_like(ix), iy, iz, area, jnp.zeros_like(ix), jnp.zeros_like(ix)],
            axis=1,
        )
        shear_diagonal = jnp.stack(
            [
                ix,
                jnp.zeros_like(ix),
                jnp.zeros_like(ix),
                jnp.zeros_like(ix),
                area,
                area,
            ],
            axis=1,
        )
        damping_diagonal = jnp.stack(
            [ix, 3.0 * iy, 3.0 * iz, 3.0 * area, area, area], axis=1
        )

        def project(
            diagonal: Array,
            basis: Array = basis,
            length: Array = length,
            weights: Array = weights,
        ) -> Array:
            local = vmap(jnp.diag)(diagonal)
            integrand = vmap(lambda b, c: b.T @ c @ b)(basis, local)
            return length * jnp.sum(weights[:, None, None] * integrand, axis=0)

        young_operators.append(project(young_diagonal))
        shear_operators.append(project(shear_diagonal))
        damping_operators.append(project(damping_diagonal))
    return (
        jnp.stack(young_operators),
        jnp.stack(shear_operators),
        jnp.stack(damping_operators),
    )


def _pad_matrix(value: Array | list, dof: int, max_dof: int, name: str) -> Array:
    matrix = jnp.asarray(value)
    if matrix.size == 0:
        return jnp.zeros((max_dof, max_dof))
    if matrix.shape == (max_dof, max_dof):
        return matrix
    if matrix.shape != (dof, dof):
        raise ValueError(
            f"{name} must have shape ({dof}, {dof}) or "
            f"({max_dof}, {max_dof}), got {matrix.shape}."
        )
    return jnp.pad(matrix, ((0, max_dof - dof), (0, max_dof - dof)))


def params_and_structure_from_segments(
    segments: list[GVSSegment] | tuple[GVSSegment, ...],
    *,
    gravity: Array | None = None,
    max_dof: int | None = None,
    max_num_gauss_points: int | None = None,
    scale_rotational_basis_by_length: bool = False,
) -> tuple[GVSParams, GVSStructure]:
    """Convert segment specifications to canonical params and static structure.

    Link and joint matrices expressed in active coordinates are zero-padded to
    the common ``max_dof`` size. Isotropic link material properties are first
    projected through the configured strain bases; explicit link matrices pass
    through unchanged apart from padding.

    Args:
        segments: Non-empty sequence of segment construction specifications.
        gravity: Optional world-frame gravity vector with shape ``(3,)``.
            Defaults to ``[0, 0, 9.81]``.
        max_dof: Optional common padded generalized-coordinate dimension. When
            omitted, the largest joint or link DOF among the segments is used.
        max_num_gauss_points: Optional common quadrature padding size. When
            omitted, the largest segment quadrature rule is used.
        scale_rotational_basis_by_length: Whether rotational strain-basis rows
            are divided by the associated link length during projection.

    Returns:
        A tuple ``(params, structure)`` containing validated canonical
        :class:`GVSParams` and the corresponding static :class:`GVSStructure`.

    Raises:
        KeyError: If a joint or basis type is unsupported.
        ValueError: If ``segments`` is empty; a quadrature rule has fewer than
            five points; an explicit matrix has an invalid shape; a material
            source is incomplete; or either padding size is too small.
    """
    if not segments:
        raise ValueError("GVS requires at least one segment.")
    input_segments = tuple(segments)
    joint_dofs = [Joint.DICT_JOINT_TYPE_DOF[s.joint.type] for s in input_segments]

    static_segments = tuple(
        GVSSegmentStructure(
            link=GVSLinkStructure(
                cross_section_geometry=segment.link.cross_section_geometry,
                cross_section_profile_types=segment.link.cross_section_profile_types,
                cross_section_profile_parameter_counts=(
                    segment.link.cross_section_profile_parameter_counts
                ),
            ),
            joint=GVSJointStructure(
                type=segment.joint.type,
                axis=segment.joint.axis,
                plane=segment.joint.plane,
                pitch=float(segment.joint.pitch),
            ),
            basis=GVSStrainBasisStructure(
                type=segment.basis.type,
                strain_selector=tuple(
                    int(value)
                    for value in jnp.asarray(segment.basis.strain_selector)
                    .reshape(-1)
                    .tolist()
                ),
                basis_order=tuple(
                    int(value)
                    for value in jnp.asarray(segment.basis.basis_order)
                    .reshape(-1)
                    .tolist()
                ),
            ),
            num_gauss_points=int(segment.num_gauss_points),
        )
        for segment in input_segments
    )
    link_dofs = [_link_dof(segment) for segment in static_segments]
    required_max_dof = max(joint_dofs + link_dofs)
    layout_max_dof = required_max_dof if max_dof is None else max_dof
    if layout_max_dof < required_max_dof:
        raise ValueError(
            f"max_dof={layout_max_dof} is smaller than required DOF {required_max_dof}."
        )
    required_max_num_gauss_points = max(
        segment.num_gauss_points for segment in static_segments
    )
    if any(segment.num_gauss_points < 5 for segment in static_segments):
        raise ValueError("Every GVS segment requires at least 5 Gauss points.")
    layout_max_num_gauss_points = (
        required_max_num_gauss_points
        if max_num_gauss_points is None
        else max_num_gauss_points
    )
    if layout_max_num_gauss_points < required_max_num_gauss_points:
        raise ValueError(
            "max_num_gauss_points must cover every segment quadrature rule."
        )
    structure = GVSStructure(
        segments=static_segments,
        max_dof=layout_max_dof,
        max_num_gauss_points=layout_max_num_gauss_points,
        scale_rotational_basis_by_length=scale_rotational_basis_by_length,
    )

    max_coefficients = max(
        len(segment.link.cross_section_coefficients) for segment in input_segments
    )
    coefficients = jnp.zeros((len(input_segments), max_coefficients))
    for index, segment in enumerate(input_segments):
        values = jnp.asarray(segment.link.cross_section_coefficients)
        coefficients = coefficients.at[index, : values.size].set(values)

    zeros = jnp.zeros((len(input_segments), layout_max_dof, layout_max_dof))
    joint_stiffness = []
    joint_damping = []
    for index, segment in enumerate(input_segments):
        joint_stiffness.append(
            _pad_matrix(
                segment.joint.stiffness,
                joint_dofs[index],
                layout_max_dof,
                "joint stiffness",
            )
        )
        joint_damping.append(
            _pad_matrix(
                segment.joint.damping,
                joint_dofs[index],
                layout_max_dof,
                "joint damping",
            )
        )
    params = GVSParams(
        gravity=gravity,
        link=ContinuumLinkParams(
            length=jnp.asarray([segment.link.length for segment in input_segments]),
            density=jnp.asarray([segment.link.density for segment in input_segments]),
            reference_strain=jnp.asarray(
                [segment.link.reference_strain for segment in input_segments],
                dtype=float,
            ),
            cross_section=CrossSectionParams(coefficients=coefficients),
            stiffness=zeros,
            damping=zeros,
        ),
        joint=JointParams(
            stiffness=jnp.stack(joint_stiffness),
            damping=jnp.stack(joint_damping),
        ),
    )
    young_operator, shear_operator, damping_operator = material_operators_from_params(
        params, structure
    )
    link_stiffness = []
    link_damping = []
    for index, segment in enumerate(input_segments):
        link_dof = link_dofs[index]
        if segment.link.stiffness is None:
            stiffness = (
                segment.link.young_modulus * young_operator[index]
                + segment.link._resolved_shear_modulus() * shear_operator[index]
            )
        else:
            stiffness = _pad_matrix(
                segment.link.stiffness, link_dof, layout_max_dof, "link stiffness"
            )
        if segment.link.damping is not None:
            damping = _pad_matrix(
                segment.link.damping, link_dof, layout_max_dof, "link damping"
            )
        elif segment.link.material_damping_coefficient is not None:
            damping = (
                segment.link.material_damping_coefficient * damping_operator[index]
            )
        else:
            damping = jnp.zeros_like(damping_operator[index])
        link_stiffness.append(stiffness)
        link_damping.append(damping)
    params = params.replace(
        link=params.link.replace(
            stiffness=jnp.stack(link_stiffness), damping=jnp.stack(link_damping)
        )
    )
    params.validate_against_structure(structure)
    return params, structure
