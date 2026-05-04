import jax.numpy as jnp
from jax import Array

from soromox.systems import (
    GVS,
    ArticulatedSoftRobotParams,
    BaseTendonRoutingParams,
    GVSParams,
    GVSStructure,
    LinearTendonRoutingParams,
    PassiveTendonParams,
    PCSParams,
    PendulumParams,
    PlanarHSAParams,
    PlanarPCSParams,
    TendonActuatedGVSParams,
    TendonActuatedPCSParams,
    TendonActuatedPendulumParams,
    TendonActuatedPlanarPCSParams,
)


def pcs_params(
    *,
    length: Array,
    radius: Array,
    density: Array,
    young_modulus: Array,
    shear_modulus: Array,
    damping_matrix: Array,
    gravity: Array,
    base_pose: Array | None = None,
    reference_strain: Array | None = None,
) -> PCSParams:
    length = jnp.asarray(length)
    num_segments = length.shape[0]
    if base_pose is None:
        base_pose = jnp.array([jnp.pi / 2, jnp.pi / 2, 0.0, 0.0, 0.0, 0.0])
    if reference_strain is None:
        reference_strain = jnp.tile(
            jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), num_segments
        )
    return PCSParams(
        length=length,
        radius=jnp.asarray(radius),
        density=jnp.asarray(density),
        young_modulus=jnp.asarray(young_modulus),
        shear_modulus=jnp.asarray(shear_modulus),
        damping_matrix=jnp.asarray(damping_matrix),
        gravity=jnp.asarray(gravity),
        base_pose=jnp.asarray(base_pose),
        reference_strain=jnp.asarray(reference_strain),
    )


def planar_pcs_params(
    *,
    length: Array,
    radius: Array,
    density: Array,
    young_modulus: Array,
    shear_modulus: Array,
    damping_matrix: Array,
    gravity: Array,
    base_angle: Array | float = jnp.pi / 2,
    reference_strain: Array | None = None,
) -> PlanarPCSParams:
    length = jnp.asarray(length)
    num_segments = length.shape[0]
    if reference_strain is None:
        reference_strain = jnp.tile(jnp.array([0.0, 1.0, 0.0]), num_segments)
    return PlanarPCSParams(
        length=length,
        radius=jnp.asarray(radius),
        density=jnp.asarray(density),
        young_modulus=jnp.asarray(young_modulus),
        shear_modulus=jnp.asarray(shear_modulus),
        damping_matrix=jnp.asarray(damping_matrix),
        gravity=jnp.asarray(gravity),
        base_angle=jnp.asarray(base_angle),
        reference_strain=jnp.asarray(reference_strain),
    )


def pendulum_params(
    *,
    mass: Array,
    moment_inertia: Array,
    length: Array,
    center_of_mass_length: Array,
    gravity: Array,
    joint_stiffness: Array | None = None,
    joint_damping: Array | None = None,
    joint_rest_configuration: Array | None = None,
    radius: Array | None = None,
) -> PendulumParams:
    mass = jnp.asarray(mass)
    n = mass.shape[0]
    if joint_stiffness is None:
        joint_stiffness = jnp.zeros((n, n))
    if joint_damping is None:
        joint_damping = jnp.zeros((n, n))
    if joint_rest_configuration is None:
        joint_rest_configuration = jnp.zeros((n,))
    if radius is None:
        radius = 0.05 * jnp.asarray(length)
    return PendulumParams(
        mass=mass,
        moment_inertia=jnp.asarray(moment_inertia),
        length=jnp.asarray(length),
        center_of_mass_length=jnp.asarray(center_of_mass_length),
        gravity=jnp.asarray(gravity),
        joint_stiffness=jnp.asarray(joint_stiffness),
        joint_damping=jnp.asarray(joint_damping),
        joint_rest_configuration=jnp.asarray(joint_rest_configuration),
        radius=jnp.asarray(radius),
    )


def articulated_params(
    *,
    joint_screw: Array,
    tip_position: Array,
    center_of_mass_position: Array,
    mass: Array,
    center_of_mass_inertia: Array,
    gravity: Array,
    parent_to_joint_transform: Array | None = None,
    joint_stiffness: Array | None = None,
    joint_damping: Array | None = None,
    joint_rest_configuration: Array | None = None,
    radius: Array | None = None,
) -> ArticulatedSoftRobotParams:
    joint_screw = jnp.asarray(joint_screw)
    n = joint_screw.shape[0]
    if parent_to_joint_transform is None:
        parent_to_joint_transform = jnp.broadcast_to(jnp.eye(4), (n, 4, 4))
    if joint_stiffness is None:
        joint_stiffness = jnp.zeros((n, n))
    if joint_damping is None:
        joint_damping = jnp.zeros((n, n))
    if joint_rest_configuration is None:
        joint_rest_configuration = jnp.zeros((n,))
    if radius is None:
        radius = 0.05 * jnp.linalg.norm(jnp.asarray(tip_position), axis=1)
    return ArticulatedSoftRobotParams(
        joint_screw=joint_screw,
        parent_to_joint_transform=jnp.asarray(parent_to_joint_transform),
        tip_position=jnp.asarray(tip_position),
        center_of_mass_position=jnp.asarray(center_of_mass_position),
        mass=jnp.asarray(mass),
        center_of_mass_inertia=jnp.asarray(center_of_mass_inertia),
        gravity=jnp.asarray(gravity),
        joint_stiffness=jnp.asarray(joint_stiffness),
        joint_damping=jnp.asarray(joint_damping),
        joint_rest_configuration=jnp.asarray(joint_rest_configuration),
        radius=jnp.asarray(radius),
    )


def linear_tendon_routing(
    *,
    y_intercept: Array,
    z_intercept: Array,
    y_slope: Array,
    z_slope: Array,
    attachment_segment_index: Array,
) -> LinearTendonRoutingParams:
    return LinearTendonRoutingParams(
        y_intercept=jnp.asarray(y_intercept),
        z_intercept=jnp.asarray(z_intercept),
        y_slope=jnp.asarray(y_slope),
        z_slope=jnp.asarray(z_slope),
        attachment_segment_index=jnp.asarray(attachment_segment_index, dtype=jnp.int32),
    )


def passive_tendon_params(
    stiffness: Array, damping: Array, rest_length_offset: Array
) -> PassiveTendonParams:
    return PassiveTendonParams(
        stiffness=jnp.asarray(stiffness),
        damping=jnp.asarray(damping),
        rest_length_offset=jnp.asarray(rest_length_offset),
    )


def tendon_actuated_pcs_params(
    *,
    body: PCSParams,
    active_tendon_routing: BaseTendonRoutingParams,
    passive_tendon_routing: BaseTendonRoutingParams | None = None,
    passive_tendon: PassiveTendonParams | None = None,
) -> TendonActuatedPCSParams:
    if passive_tendon_routing is None:
        passive_tendon_routing = LinearTendonRoutingParams.empty()
    if passive_tendon is None:
        passive_tendon = PassiveTendonParams.empty()
    return TendonActuatedPCSParams(
        body=body,
        active_tendon_routing=active_tendon_routing,
        passive_tendon_routing=passive_tendon_routing,
        passive_tendon=passive_tendon,
    )


def tendon_actuated_planar_pcs_params(
    *, body: PlanarPCSParams, tendon_distance: Array
) -> TendonActuatedPlanarPCSParams:
    return TendonActuatedPlanarPCSParams(
        length=body.length,
        radius=body.radius,
        density=body.density,
        young_modulus=body.young_modulus,
        shear_modulus=body.shear_modulus,
        damping_matrix=body.damping_matrix,
        gravity=body.gravity,
        base_angle=body.base_angle,
        reference_strain=body.reference_strain,
        tendon_distance=jnp.asarray(tendon_distance),
    )


def tendon_actuated_pendulum_params(
    *,
    body: PendulumParams,
    active_routing_matrix: Array,
    passive_routing_matrix: Array | None = None,
    active_tendon_reference_configuration: Array | None = None,
    passive_tendon_reference_configuration: Array | None = None,
    passive_tendon: PassiveTendonParams | None = None,
) -> TendonActuatedPendulumParams:
    n = body.mass.shape[0]
    if passive_routing_matrix is None:
        passive_routing_matrix = jnp.zeros((0, n))
    if active_tendon_reference_configuration is None:
        active_tendon_reference_configuration = body.joint_rest_configuration
    if passive_tendon_reference_configuration is None:
        passive_tendon_reference_configuration = body.joint_rest_configuration
    if passive_tendon is None:
        passive_tendon = PassiveTendonParams.empty()
    return TendonActuatedPendulumParams(
        body=body,
        active_routing_matrix=jnp.asarray(active_routing_matrix),
        passive_routing_matrix=jnp.asarray(passive_routing_matrix),
        active_tendon_reference_configuration=jnp.asarray(
            active_tendon_reference_configuration
        ),
        passive_tendon_reference_configuration=jnp.asarray(
            passive_tendon_reference_configuration
        ),
        passive_tendon=passive_tendon,
    )


def gvs_params_from_segments(
    segments,
    *,
    gravity: Array,
    base_pose: Array | None = None,
    max_dof: int | None = None,
    max_num_gauss_points: int | None = None,
    scale_rotational_basis_by_length: bool = False,
) -> tuple[GVSParams, GVSStructure]:
    return GVS.params_from_segments(
        segments,
        gravity=gravity,
        base_pose=base_pose,
        max_dof=max_dof,
        max_num_gauss_points=max_num_gauss_points,
        scale_rotational_basis_by_length=scale_rotational_basis_by_length,
    )


def tendon_actuated_gvs_params(
    *,
    body: GVSParams,
    active_tendon_routing: BaseTendonRoutingParams,
    passive_tendon_routing: BaseTendonRoutingParams | None = None,
    passive_tendon: PassiveTendonParams | None = None,
) -> TendonActuatedGVSParams:
    if passive_tendon_routing is None:
        passive_tendon_routing = LinearTendonRoutingParams.empty()
    if passive_tendon is None:
        passive_tendon = PassiveTendonParams.empty()
    return TendonActuatedGVSParams(
        body=body,
        active_tendon_routing=active_tendon_routing,
        passive_tendon_routing=passive_tendon_routing,
        passive_tendon=passive_tendon,
    )


def planar_hsa_params_from_legacy(params: dict) -> PlanarHSAParams:
    hysteresis = params.get("hysteresis", {})
    return PlanarHSAParams(
        base_angle=jnp.asarray(params["th0"]),
        length=jnp.asarray(params["L"]),
        proximal_cap_length=jnp.asarray(params["lpc"]),
        distal_cap_length=jnp.asarray(params["ldc"]),
        rod_height=jnp.asarray(params["h"]),
        rod_outer_radius=jnp.asarray(params["rout"]),
        rod_inner_radius=jnp.asarray(params["rin"]),
        rod_offset=jnp.asarray(params["roff"]),
        bending_reference=jnp.asarray(params["kappa_b_ref"]),
        shear_reference=jnp.asarray(params["sigma_sh_ref"]),
        axial_reference=jnp.asarray(params["sigma_a_ref"]),
        strain_coupling=jnp.asarray(params["C_varepsilon"]),
        platform_dimension=jnp.asarray(params["pcudim"]),
        rod_density=jnp.asarray(params["rhor"]),
        platform_density=jnp.asarray(params["rhop"]),
        end_cap_density=jnp.asarray(params["rhoec"]),
        gravity=jnp.asarray(params["g"]),
        nominal_bending_stiffness=jnp.asarray(params["S_b_hat"]),
        nominal_shear_stiffness=jnp.asarray(params["S_sh_hat"]),
        nominal_axial_stiffness=jnp.asarray(params["S_a_hat"]),
        bending_shear_stiffness=jnp.asarray(params["S_b_sh"]),
        bending_stiffness_correction=jnp.asarray(params["C_S_b"]),
        shear_stiffness_correction=jnp.asarray(params["C_S_sh"]),
        axial_stiffness_correction=jnp.asarray(params["C_S_a"]),
        bending_damping=jnp.asarray(params["zetab"]),
        shear_damping=jnp.asarray(params["zetash"]),
        axial_damping=jnp.asarray(params["zetaa"]),
        phi_max=jnp.asarray(params["phi_max"]),
        platform_mass=jnp.asarray(params["mpl"]),
        platform_center_of_gravity=jnp.asarray(params["CoGpl"]),
        end_effector_offset=jnp.asarray(params["chiee_off"]),
        hysteresis_basis=jnp.asarray(hysteresis.get("basis", jnp.zeros((0, 0)))),
        hysteresis_alpha=jnp.asarray(hysteresis.get("alpha", jnp.zeros((0,)))),
        hysteresis_A=jnp.asarray(hysteresis.get("A", jnp.zeros((1,)))),
        hysteresis_n=jnp.asarray(hysteresis.get("n", jnp.zeros((1,)))),
        hysteresis_beta=jnp.asarray(hysteresis.get("beta", jnp.zeros((1,)))),
        hysteresis_gamma=jnp.asarray(hysteresis.get("gamma", jnp.zeros((1,)))),
    )
