import jax.numpy as jnp
from jax import Array

from soromox.systems import (
    GVS,
    ArticulatedSoftRobotParams,
    GVSParams,
    GVSStructure,
    PCSParams,
    PendulumParams,
    PlanarHSAParams,
    PlanarPCSParams,
)


def spatial_base_pose(
    x: float | Array = 0.0, y: float | Array = 0.0, z: float | Array = 0.0
) -> Array:
    return jnp.array([1.0, 0.0, 0.0, 0.0, x, y, z])


def planar_base_pose(
    theta: float | Array = 0.0, x: float | Array = 0.0, y: float | Array = 0.0
) -> Array:
    return jnp.array([theta, x, y])


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
        base_pose = spatial_base_pose()
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
    base_pose: Array | None = None,
    reference_strain: Array | None = None,
) -> PlanarPCSParams:
    length = jnp.asarray(length)
    num_segments = length.shape[0]
    if reference_strain is None:
        reference_strain = jnp.tile(jnp.array([0.0, 1.0, 0.0]), num_segments)
    if base_pose is None:
        base_pose = planar_base_pose(jnp.pi / 2)
    return PlanarPCSParams(
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
    base_pose: Array | None = None,
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
    if base_pose is None:
        base_pose = planar_base_pose()
    return PendulumParams(
        base_pose=jnp.asarray(base_pose),
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
    base_pose: Array | None = None,
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
    if base_pose is None:
        base_pose = spatial_base_pose()
    return ArticulatedSoftRobotParams(
        base_pose=jnp.asarray(base_pose),
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


def planar_hsa_params_from_legacy(params: dict) -> PlanarHSAParams:
    hysteresis = params.get("hysteresis", {})
    return PlanarHSAParams(
        base_pose=jnp.array([jnp.asarray(params["th0"]), 0.0, 0.0]),
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
