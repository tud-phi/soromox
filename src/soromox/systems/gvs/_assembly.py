__all__ = ["assign_gvs_runtime_arrays"]

from typing import Any

import jax.numpy as jnp

from soromox.systems.gvs.params import GVSParams
from soromox.systems.gvs.primitives import Basis, Joint
from soromox.systems.gvs.structures import GVSSegmentStructure, GVSStructure
from soromox.systems.params import validate_quaternion_base_pose
from soromox.utils.dof import build_active_dof_basis
from soromox.utils.geometry import poses


def _set_model_field(model: Any, name: str, value: Any) -> None:
    object.__setattr__(model, name, value)


def _profile_endpoint_pairs(segment, coefficients):
    pairs = []
    start = 0
    for profile_type, count in zip(
        segment.link.cross_section_profile_types,
        segment.link.cross_section_profile_parameter_counts,
        strict=True,
    ):
        values = coefficients[start : start + count]
        pairs.append(
            (values[0], values[0])
            if profile_type == "constant"
            else (values[0], values[1])
        )
        start += count
    return pairs


def _validate_segments(
    segments: list[GVSSegmentStructure] | tuple[GVSSegmentStructure, ...],
    params: GVSParams,
    max_dof: int | None,
    max_num_gauss_points: int | None,
) -> None:
    params.validate_against_structure(
        GVSStructure(
            segments=tuple(segments),
            max_dof=max_dof,
            max_num_gauss_points=max_num_gauss_points,
        )
    )


def assign_gvs_runtime_arrays(
    model: Any,
    *,
    segments: list[GVSSegmentStructure] | tuple[GVSSegmentStructure, ...],
    params: GVSParams,
    max_dof: int | None,
    max_num_gauss_points: int | None,
    g: list[float] | jnp.ndarray,
    p0: jnp.ndarray | None,
) -> None:
    """Assemble padded runtime arrays from public GVS segment specs."""
    _validate_segments(segments, params, max_dof, max_num_gauss_points)

    _set_model_field(model, "num_segments", len(segments))
    num_gauss_points_per_segment = [segment.num_gauss_points for segment in segments]
    _set_model_field(
        model,
        "num_gauss_points",
        jnp.asarray(num_gauss_points_per_segment, dtype=jnp.int32),
    )

    if max_num_gauss_points is None:
        max_num_gauss_points = max(num_gauss_points_per_segment)
    elif max_num_gauss_points < max(num_gauss_points_per_segment):
        raise ValueError(
            "max_num_gauss_points must be greater than or equal to the maximum "
            f"segment num_gauss_points ({max(num_gauss_points_per_segment)}), "
            f"but got {max_num_gauss_points}."
        )
    if max_num_gauss_points < 5:
        raise ValueError(
            f"max_num_gauss_points must be at least 5, got {max_num_gauss_points}."
        )

    _set_model_field(model, "max_num_gauss_points", max_num_gauss_points)
    _set_model_field(model, "max_num_integration_points", max_num_gauss_points + 2)

    dofs_joint = [Joint.DICT_JOINT_TYPE_DOF[s.joint.type] for s in segments]
    dofs_link = [
        int(
            Basis.DOF_BRANCHES[Basis.BASISTYPE_MAP[s.basis.type]](
                (
                    jnp.asarray(s.basis.strain_selector),
                    jnp.asarray(s.basis.basis_order),
                )
            )
        )
        for s in segments
    ]
    real_max_dof = max(dofs_joint + dofs_link)

    if max_dof is None:
        max_dof = real_max_dof
    elif max_dof < real_max_dof:
        raise ValueError(
            "max_dof must be greater than or equal to the maximum DOF in links "
            f"and joints ({real_max_dof}), but got {max_dof}."
        )
    _set_model_field(model, "max_dof", max_dof)

    n_segments = model.num_segments
    max_num_integration_points = model.max_num_integration_points

    segment_lengths = jnp.empty((n_segments,), dtype=float)
    num_integration_points = jnp.empty((n_segments,), dtype=int)
    dofs_per_segment = jnp.empty((n_segments, 2), dtype=int)
    integration_points = jnp.empty(
        (n_segments, max_num_integration_points), dtype=float
    )
    integration_weights = jnp.empty(
        (n_segments, max_num_integration_points), dtype=float
    )

    mass_matrices = jnp.empty(
        (n_segments, max_num_integration_points, 6, 6), dtype=float
    )
    B_joint = jnp.empty((n_segments, 6, max_dof), dtype=float)
    B_Xs = jnp.empty((n_segments, max_num_integration_points, 6, max_dof), dtype=float)
    B_Z1 = jnp.empty(
        (n_segments, max_num_integration_points - 1, 6, max_dof), dtype=float
    )
    B_Z2 = jnp.empty(
        (n_segments, max_num_integration_points - 1, 6, max_dof), dtype=float
    )

    xi_ref_joint = jnp.empty((n_segments, 6), dtype=float)
    xi_ref_Xs = jnp.empty((n_segments, max_num_integration_points, 6), dtype=float)
    xi_ref_Z1 = jnp.empty((n_segments, max_num_integration_points - 1, 6), dtype=float)
    xi_ref_Z2 = jnp.empty((n_segments, max_num_integration_points - 1, 6), dtype=float)
    joint_stiffness = jnp.empty((n_segments, max_dof, max_dof), dtype=float)
    strain_selector = jnp.zeros((n_segments, 2 * max_dof), dtype=jnp.bool_)

    basis_type_index = jnp.empty((n_segments,), dtype=int)
    basis_active_params = jnp.empty((n_segments, 6))
    basis_order_params = jnp.empty((n_segments, 6))
    cross_section_geometry = jnp.empty((n_segments,), dtype=jnp.int32)
    radius_params = jnp.empty((n_segments, 2))
    height_params = jnp.empty((n_segments, 2))
    width_params = jnp.empty((n_segments, 2))
    semi_major_params = jnp.empty((n_segments, 2))
    semi_minor_params = jnp.empty((n_segments, 2))

    for i_segment, segment in enumerate(segments):
        joint_dof = Joint.DICT_JOINT_TYPE_DOF[segment.joint.type]
        pairs = _profile_endpoint_pairs(
            segment, params.link.cross_section.coefficients[i_segment]
        )
        zero_pair = (jnp.array(0.0), jnp.array(0.0))
        geometry = int(segment.link.cross_section_geometry)
        radius_pair = pairs[0] if geometry == 0 else zero_pair
        rectangular_pairs = pairs if geometry == 1 else [zero_pair, zero_pair]
        elliptical_pairs = pairs if geometry == 2 else [zero_pair, zero_pair]
        segment_data = model._build_segment_i(
            max_dof=max_dof,
            max_num_integration_points=max_num_integration_points,
            link_structure=segment.link,
            joint_structure=segment.joint,
            basis_structure=segment.basis,
            num_gauss_points=segment.num_gauss_points,
            length=params.link.length[i_segment],
            density=params.link.density[i_segment],
            radius_initial=radius_pair[0],
            radius_final=radius_pair[1],
            height_initial=rectangular_pairs[0][0],
            height_final=rectangular_pairs[0][1],
            width_initial=rectangular_pairs[1][0],
            width_final=rectangular_pairs[1][1],
            semi_major_initial=elliptical_pairs[0][0],
            semi_major_final=elliptical_pairs[0][1],
            semi_minor_initial=elliptical_pairs[1][0],
            semi_minor_final=elliptical_pairs[1][1],
            reference_strain=params.link.reference_strain[i_segment],
            joint_stiffness=params.joint.stiffness[i_segment, :joint_dof, :joint_dof],
        )

        segment_lengths = segment_lengths.at[i_segment].set(segment_data.L)
        num_integration_points = num_integration_points.at[i_segment].set(
            segment_data.num_integration_points
        )
        dofs_per_segment = dofs_per_segment.at[i_segment].set(
            segment_data.dofs_joint_link
        )
        integration_points = integration_points.at[i_segment].set(
            segment_data.integration_points
        )
        integration_weights = integration_weights.at[i_segment].set(
            segment_data.integration_weights
        )
        mass_matrices = mass_matrices.at[i_segment].set(segment_data.mass_matrices)
        B_joint = B_joint.at[i_segment].set(segment_data.B_joint)
        B_Xs = B_Xs.at[i_segment].set(segment_data.B_Xs)
        B_Z1 = B_Z1.at[i_segment].set(segment_data.B_Z1)
        B_Z2 = B_Z2.at[i_segment].set(segment_data.B_Z2)
        xi_ref_joint = xi_ref_joint.at[i_segment].set(segment_data.xi_ref_joint)
        xi_ref_Xs = xi_ref_Xs.at[i_segment].set(segment_data.xi_ref_Xs)
        xi_ref_Z1 = xi_ref_Z1.at[i_segment].set(segment_data.xi_ref_Z1)
        xi_ref_Z2 = xi_ref_Z2.at[i_segment].set(segment_data.xi_ref_Z2)
        joint_stiffness = joint_stiffness.at[i_segment].set(
            segment_data.joint_stiffness
        )
        strain_selector = strain_selector.at[i_segment].set(
            segment_data.strain_selector
        )

        basis = segment.basis
        basis_type_index = basis_type_index.at[i_segment].set(
            Basis.BASISTYPE_MAP[basis.type]
        )
        basis_active_params = basis_active_params.at[i_segment].set(
            basis.strain_selector
        )
        basis_order_params = basis_order_params.at[i_segment].set(basis.basis_order)
        cross_section_geometry = cross_section_geometry.at[i_segment].set(
            int(segment.link.cross_section_geometry)
        )
        radius_params = radius_params.at[i_segment].set(radius_pair)
        height_params = height_params.at[i_segment].set(rectangular_pairs[0])
        width_params = width_params.at[i_segment].set(rectangular_pairs[1])
        semi_major_params = semi_major_params.at[i_segment].set(elliptical_pairs[0])
        semi_minor_params = semi_minor_params.at[i_segment].set(elliptical_pairs[1])

    _set_model_field(model, "segment_lengths", segment_lengths)
    _set_model_field(model, "num_integration_points", num_integration_points)
    _set_model_field(model, "dofs_per_segment", dofs_per_segment)
    _set_model_field(model, "integration_points", integration_points)
    _set_model_field(model, "integration_weights", integration_weights)
    _set_model_field(model, "mass_matrices", mass_matrices)
    _set_model_field(model, "B_joint", B_joint)
    _set_model_field(model, "B_Xs", B_Xs)
    _set_model_field(model, "B_Z1", B_Z1)
    _set_model_field(model, "B_Z2", B_Z2)
    _set_model_field(model, "xi_ref_joint", xi_ref_joint)
    _set_model_field(model, "xi_ref_Xs", xi_ref_Xs)
    _set_model_field(model, "xi_ref_Z1", xi_ref_Z1)
    _set_model_field(model, "xi_ref_Z2", xi_ref_Z2)
    _set_model_field(model, "joint_stiffness", joint_stiffness)
    _set_model_field(model, "basis_type_index", basis_type_index)
    _set_model_field(model, "basis_active_params", basis_active_params)
    _set_model_field(model, "basis_order_params", basis_order_params)
    _set_model_field(model, "cross_section_geometry_index", cross_section_geometry)
    _set_model_field(model, "radius_params", radius_params)
    _set_model_field(model, "height_params", height_params)
    _set_model_field(model, "width_params", width_params)
    _set_model_field(model, "semi_major_params", semi_major_params)
    _set_model_field(model, "semi_minor_params", semi_minor_params)

    active_selector = jnp.concatenate(strain_selector, axis=0)
    _set_model_field(model, "active_dof_map", build_active_dof_basis(active_selector))
    _set_model_field(
        model, "num_dofs", int(jnp.sum(dofs_per_segment, axis=(0, 1), dtype=int))
    )
    _set_model_field(
        model, "num_padded_dofs", int(jnp.array(n_segments * 2 * max_dof, dtype=int))
    )
    _set_model_field(model, "num_actuators", model.num_dofs)
    _set_model_field(model, "g", jnp.concatenate([jnp.zeros(3), jnp.array(g)]))
    _set_model_field(
        model,
        "segment_end_positions",
        jnp.cumsum(jnp.concatenate([jnp.zeros(1), segment_lengths])),
    )

    if p0 is None:
        p0_arr = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    else:
        p0_arr = jnp.asarray(p0, dtype=jnp.float64)
    validate_quaternion_base_pose("p0", p0_arr, (7,))
    _set_model_field(model, "base_pose", p0_arr)
    _set_model_field(model, "g0", poses.quaternion_pose_to_transform(p0_arr))
