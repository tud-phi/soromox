__all__ = ["params_and_structure_from_segments"]

from jax import Array
from jax import numpy as jnp

from soromox.systems.gvs.params import GVSLinkParams, GVSParams
from soromox.systems.gvs.primitives import Basis, Joint
from soromox.systems.gvs.specs import GVSSegment
from soromox.systems.gvs.structures import (
    GVSJointStructure,
    GVSLinkStructure,
    GVSSegmentStructure,
    GVSStrainBasisStructure,
    GVSStructure,
)


def params_and_structure_from_segments(
    segments: list[GVSSegment] | tuple[GVSSegment, ...],
    *,
    gravity: Array,
    base_pose: Array | None = None,
    max_dof: int | None = None,
    max_num_gauss_points: int | None = None,
    scale_rotational_basis_by_length: bool = False,
) -> tuple[GVSParams, GVSStructure]:
    """Split user-facing GVS segment specs into dynamic params and structure."""
    if not segments:
        raise ValueError("GVS requires at least one segment.")
    input_segments = tuple(segments)
    n_segments = len(input_segments)
    if base_pose is None:
        base_pose = jnp.array([jnp.pi / 2, jnp.pi / 2, 0.0, 0.0, 0.0, 0.0])

    dofs_joint = [
        Joint.DICT_JOINT_TYPE_DOF[segment.joint.type] for segment in input_segments
    ]
    dofs_link = []
    for segment in input_segments:
        basis_type_idx = Basis.BASISTYPE_MAP[segment.basis.type]
        dofs_link.append(
            int(
                Basis.DOF_BRANCHES[basis_type_idx](
                    (
                        jnp.asarray(segment.basis.active),
                        jnp.asarray(segment.basis.orders),
                    )
                )
            )
        )
    inferred_max_dof = max(dofs_joint + dofs_link)
    layout_max_dof = inferred_max_dof if max_dof is None else max_dof
    if layout_max_dof < inferred_max_dof:
        raise ValueError(
            f"max_dof={layout_max_dof} is smaller than the required "
            f"GVS segment DOF {inferred_max_dof}."
        )

    joint_stiffness = jnp.zeros((n_segments, layout_max_dof, layout_max_dof))
    for i, segment in enumerate(input_segments):
        dof = dofs_joint[i]
        stiffness = jnp.asarray(segment.joint.stiffness)
        if dof == 0:
            continue
        if stiffness.size == 0:
            continue
        if stiffness.shape != (dof, dof):
            raise ValueError(
                "joint stiffness must have shape "
                f"({dof}, {dof}) for segment {i}, got {stiffness.shape}."
            )
        joint_stiffness = joint_stiffness.at[i, :dof, :dof].set(stiffness)

    links = [segment.link for segment in input_segments]
    params = GVSParams(
        link=GVSLinkParams(
            length=jnp.asarray([link.L for link in links]),
            young_modulus=jnp.asarray([link.E for link in links]),
            poisson_ratio=jnp.asarray([link.nu for link in links]),
            density=jnp.asarray([link.rho for link in links]),
            damping_coefficient=jnp.asarray([link.eta for link in links]),
            radius_initial=jnp.asarray([link.r_i for link in links]),
            radius_final=jnp.asarray([link.r_f for link in links]),
            height_initial=jnp.asarray([link.h_i for link in links]),
            height_final=jnp.asarray([link.h_f for link in links]),
            width_initial=jnp.asarray([link.w_i for link in links]),
            width_final=jnp.asarray([link.w_f for link in links]),
            semi_major_initial=jnp.asarray([link.a_i for link in links]),
            semi_major_final=jnp.asarray([link.a_f for link in links]),
            semi_minor_initial=jnp.asarray([link.b_i for link in links]),
            semi_minor_final=jnp.asarray([link.b_f for link in links]),
        ),
        gravity=jnp.asarray(gravity),
        base_pose=jnp.asarray(base_pose),
        reference_strain=jnp.asarray(
            [segment.basis.xi_ref for segment in input_segments]
        ),
        joint_stiffness=joint_stiffness,
    )
    static_segments = tuple(
        GVSSegmentStructure(
            link=GVSLinkStructure(
                cross_section_geometry=segment.link.cross_section_geometry
            ),
            joint=GVSJointStructure(
                type=segment.joint.type,
                axis=segment.joint.axis,
                plane=segment.joint.plane,
                pitch=float(segment.joint.pitch),
            ),
            basis=GVSStrainBasisStructure(
                type=segment.basis.type,
                active=tuple(
                    int(value)
                    for value in jnp.asarray(segment.basis.active).reshape(-1).tolist()
                ),
                orders=tuple(
                    int(value)
                    for value in jnp.asarray(segment.basis.orders).reshape(-1).tolist()
                ),
            ),
            num_gauss_points=int(segment.num_gauss_points),
        )
        for segment in input_segments
    )
    structure = GVSStructure(
        segments=static_segments,
        max_dof=layout_max_dof,
        max_num_gauss_points=max_num_gauss_points,
        scale_rotational_basis_by_length=scale_rotational_basis_by_length,
    )
    return params, structure
