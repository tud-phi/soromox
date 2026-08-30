"""Factories and simulation utilities for affine soft pendulum models.

This module defines the fixed Soft Inverted Pendulum (SIP), R-SIP, Soft
Cart-Pole, Soft Pendubot, and Soft Furuta Pendulum together with their physical
parameter dataclasses, upright and hanging configuration helpers, constant-input
open-loop rollout helper, and common summary plotting function.

Every soft link uses an affine monomial bending basis with length-scaled
rotational rows, so ``a0`` and ``a1`` are dimensionless bending-angle
coefficients:

``kappa(X) = (a0 + a1 * X / L) / L``.

References:
    Della Santina, C. (2020). The soft inverted pendulum with affine
    curvature. 2020 59th IEEE Conference on Decision and Control (CDC),
    4135-4142. https://doi.org/10.1109/CDC42340.2020.9303976

    Caradonna, D. et al. (2024). Model and Control of R-Soft Inverted
    Pendulum. IEEE Robotics and Automation Letters, 9(6), 5102-5109.
    https://doi.org/10.1109/LRA.2024.3389348

    Caradonna, D. et al. (2026). Soft Swing-up: Benchmarking Model-Based
    Optimal Control for Rigid-Soft Underactuated Systems.
    https://arxiv.org/abs/2602.03435
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from soromox.actuation import AffineJointActuator
from soromox.systems import (
    GVS,
    GVSSegment,
    JointSpec,
    LinkSpec,
    StrainBasisSpec,
    SystemState,
)

jax.config.update("jax_enable_x64", True)

AFFINE_CURVATURE_MATRIX = jnp.array([[1.0, 0.5], [0.5, 1.0 / 3.0]])
REFERENCE_STRAIN = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]


@dataclass(frozen=True)
class SoftInvertedPendulumConfig:
    """Parameters shared by the SIP, R-SIP, and Soft Cart-Pole.

    The soft-link defaults use the affine-curvature SIP stiffness, damping,
    geometry, and total mass. SoRoMoX distributes that mass along the link
    rather than concentrating it at the tip. ``r_sip_joint_damping`` follows
    the R-SIP study and ``cart_hinge_damping`` follows the Soft Swing-up study.
    The cart-body geometry is an explicit SoRoMoX modeling assumption: a short
    rigid cylindrical GVS link carries ``cart_mass`` and places the passive
    hinge at its tip.

    References:
        Della Santina, C. (2020). The soft inverted pendulum with affine
        curvature. 2020 59th IEEE Conference on Decision and Control (CDC),
        4135-4142. https://doi.org/10.1109/CDC42340.2020.9303976

        Caradonna, D. et al. (2024). Model and Control of R-Soft Inverted
        Pendulum. IEEE Robotics and Automation Letters, 9(6), 5102-5109.
        https://doi.org/10.1109/LRA.2024.3389348

        Caradonna, D. et al. (2026). Soft Swing-up: Benchmarking Model-Based
        Optimal Control for Rigid-Soft Underactuated Systems.
        https://arxiv.org/abs/2602.03435
    """

    length: float = 1.0
    thickness: float = 0.1
    link_mass: float = 1.0
    gravity: float = 9.81
    damping_coefficient: float = 0.1
    stiffness_coefficient: float = 1.0
    r_sip_joint_damping: float = 0.5
    cart_hinge_damping: float = 0.05
    cart_mass: float = 1.0
    cart_body_length: float = 0.15
    cart_body_radius: float = 0.06
    num_gauss_points: int = 5

    def __post_init__(self) -> None:
        positive = {
            "length": self.length,
            "thickness": self.thickness,
            "link_mass": self.link_mass,
            "gravity": self.gravity,
            "stiffness_coefficient": self.stiffness_coefficient,
            "cart_mass": self.cart_mass,
            "cart_body_length": self.cart_body_length,
            "cart_body_radius": self.cart_body_radius,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        nonnegative = {
            "damping_coefficient": self.damping_coefficient,
            "r_sip_joint_damping": self.r_sip_joint_damping,
            "cart_hinge_damping": self.cart_hinge_damping,
        }
        for name, value in nonnegative.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.num_gauss_points < 5:
            raise ValueError("num_gauss_points must be at least 5 for GVS.")

    @property
    def density(self) -> float:
        """Density realizing ``link_mass`` for the square soft link."""
        return self.link_mass / (self.length * self.thickness**2)

    @property
    def cart_body_density(self) -> float:
        """Density realizing ``cart_mass`` for the assumed rigid cart body."""
        volume = np.pi * self.cart_body_radius**2 * self.cart_body_length
        return self.cart_mass / volume

    @property
    def stiffness(self) -> Array:
        """Generalized affine-curvature stiffness ``k H``."""
        return self.stiffness_coefficient * AFFINE_CURVATURE_MATRIX

    @property
    def damping(self) -> Array:
        """Generalized affine-curvature damping ``beta H``."""
        return self.damping_coefficient * AFFINE_CURVATURE_MATRIX


@dataclass(frozen=True)
class SoftSwingUpPendulumConfig:
    """Parameters for the Soft Pendubot and Soft Furuta benchmarks.

    ``soft_length`` is split equally between the two Pendubot links and used in
    full by the Furuta soft link. The paper does not provide Furuta rigid-arm
    inertial properties, so its length, radius, and mass are exposed here as
    documented modeling assumptions.

    References:
        Caradonna, D. et al. (2026). Soft Swing-up: Benchmarking Model-Based
        Optimal Control for Rigid-Soft Underactuated Systems.
        https://arxiv.org/abs/2602.03435
    """

    soft_length: float = 1.0
    soft_radius: float = 0.03
    soft_density: float = 1000.0
    young_modulus: float = 1.0e6
    material_damping_coefficient: float = 0.01e6
    gravity: float = 9.81
    joint_damping: float = 0.05
    furuta_arm_length: float = 0.5
    furuta_arm_radius: float = 0.02
    furuta_arm_mass: float = 0.5
    num_gauss_points: int = 5

    def __post_init__(self) -> None:
        positive = {
            "soft_length": self.soft_length,
            "soft_radius": self.soft_radius,
            "soft_density": self.soft_density,
            "young_modulus": self.young_modulus,
            "gravity": self.gravity,
            "furuta_arm_length": self.furuta_arm_length,
            "furuta_arm_radius": self.furuta_arm_radius,
            "furuta_arm_mass": self.furuta_arm_mass,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        nonnegative = {
            "material_damping_coefficient": self.material_damping_coefficient,
            "joint_damping": self.joint_damping,
        }
        for name, value in nonnegative.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.num_gauss_points < 5:
            raise ValueError("num_gauss_points must be at least 5 for GVS.")

    @property
    def shear_modulus(self) -> float:
        """Return ``E / 3``, the incompressible-limit value used in SoRoMoX."""
        return self.young_modulus / 3.0

    @property
    def furuta_arm_density(self) -> float:
        """Density realizing the assumed rigid-arm mass."""
        volume = np.pi * self.furuta_arm_radius**2 * self.furuta_arm_length
        return self.furuta_arm_mass / volume


def _affine_basis(component: str) -> StrainBasisSpec:
    """Return a first-order monomial basis for one bending component."""
    return StrainBasisSpec(type="monomial", strain_selector=(component,), basis_order=1)


def _zero_strain_basis() -> StrainBasisSpec:
    """Return a basis with no elastic generalized coordinates."""
    return StrainBasisSpec(type="monomial", strain_selector=(), basis_order=0)


def _rigid_circular_segment(
    *,
    length: float,
    radius: float,
    density: float,
    joint: JointSpec,
    num_gauss_points: int,
) -> GVSSegment:
    """Construct a finite inertial GVS link with no elastic coordinates."""
    return GVSSegment(
        link=LinkSpec.circular(
            length=length,
            radius=radius,
            density=density,
            reference_strain=REFERENCE_STRAIN,
            stiffness=jnp.zeros((0, 0)),
            damping=jnp.zeros((0, 0)),
        ),
        joint=joint,
        basis=_zero_strain_basis(),
        num_gauss_points=num_gauss_points,
    )


def _sip_soft_segment(
    config: SoftInvertedPendulumConfig, joint: JointSpec
) -> GVSSegment:
    """Construct the explicit-matrix affine SIP link."""
    return GVSSegment(
        link=LinkSpec.rectangular(
            length=config.length,
            height=config.thickness,
            width=config.thickness,
            density=config.density,
            reference_strain=REFERENCE_STRAIN,
            stiffness=config.stiffness,
            damping=config.damping,
        ),
        joint=joint,
        basis=_affine_basis("kappa_z"),
        num_gauss_points=config.num_gauss_points,
    )


def _paper_soft_segment(
    config: SoftSwingUpPendulumConfig,
    *,
    length: float,
    joint: JointSpec,
    bending_component: str,
) -> GVSSegment:
    """Construct a cylindrical paper-parameter link with affine bending only."""
    return GVSSegment(
        link=LinkSpec.circular(
            length=length,
            radius=config.soft_radius,
            density=config.soft_density,
            reference_strain=REFERENCE_STRAIN,
            young_modulus=config.young_modulus,
            shear_modulus=config.shear_modulus,
            material_damping_coefficient=config.material_damping_coefficient,
        ),
        joint=joint,
        basis=_affine_basis(bending_component),
        num_gauss_points=config.num_gauss_points,
    )


def _direct_input(
    num_dofs: int,
    coordinate_index: int,
    *,
    label: str,
    units: str,
    bound: float | None,
    name: str,
) -> AffineJointActuator:
    routing = (
        jnp.zeros((1, num_dofs), dtype=jnp.float64).at[0, coordinate_index].set(1.0)
    )
    return AffineJointActuator.from_routing(
        routing,
        lower_bounds=None if bound is None else -bound,
        upper_bounds=bound,
        labels=(label,),
        units=units,
        name=name,
    )


def make_fixed_pendulum(config: SoftInvertedPendulumConfig) -> GVS:
    """Construct the original two-coordinate fixed-base affine SIP."""
    return GVS.from_segments(
        [_sip_soft_segment(config, JointSpec.fixed())],
        gravity=jnp.array([0.0, 0.0, -config.gravity]),
        scale_rotational_basis_by_length=True,
        actuators=(),
    )


def make_r_soft_inverted_pendulum(config: SoftInvertedPendulumConfig) -> GVS:
    """Construct the torque-actuated R-SIP with coordinates ``[theta_r,a0,a1]``."""
    joint = JointSpec.revolute("z", damping=jnp.array([[config.r_sip_joint_damping]]))
    actuator = _direct_input(
        3,
        0,
        label="base_joint_torque",
        units="N m",
        bound=None,
        name="r_sip_base_joint",
    )
    return GVS.from_segments(
        [_sip_soft_segment(config, joint)],
        gravity=jnp.array([0.0, 0.0, -config.gravity]),
        scale_rotational_basis_by_length=True,
        actuators=(actuator,),
    )


def make_soft_cart_pole(config: SoftInvertedPendulumConfig) -> GVS:
    """Construct the force-actuated, passively hinged Soft Cart-Pole.

    Coordinate order is ``[d, theta, a0, a1]``. The rigid cart/pivot segment is
    a finite zero-strain link, while the second segment contains the passive
    revolute hinge and affine soft pole.
    """
    cart = _rigid_circular_segment(
        length=config.cart_body_length,
        radius=config.cart_body_radius,
        density=config.cart_body_density,
        joint=JointSpec.prismatic("y"),
        num_gauss_points=config.num_gauss_points,
    )
    hinge = JointSpec.revolute("z", damping=jnp.array([[config.cart_hinge_damping]]))
    actuator = _direct_input(
        4,
        0,
        label="cart_force",
        units="N",
        bound=200.0,
        name="soft_cart_pole_cart",
    )
    return GVS.from_segments(
        [cart, _sip_soft_segment(config, hinge)],
        gravity=jnp.array([0.0, 0.0, -config.gravity]),
        scale_rotational_basis_by_length=True,
        actuators=(actuator,),
    )


def make_soft_pendubot(config: SoftSwingUpPendulumConfig) -> GVS:
    """Construct the Soft Pendubot with first-joint torque actuation."""
    damping = jnp.array([[config.joint_damping]])
    link_length = config.soft_length / 2.0
    segments = [
        _paper_soft_segment(
            config,
            length=link_length,
            joint=JointSpec.revolute("z", damping=damping),
            bending_component="kappa_z",
        ),
        _paper_soft_segment(
            config,
            length=link_length,
            joint=JointSpec.revolute("z", damping=damping),
            bending_component="kappa_z",
        ),
    ]
    actuator = _direct_input(
        6,
        0,
        label="first_joint_torque",
        units="N m",
        bound=10.0,
        name="soft_pendubot_first_joint",
    )
    sqrt_half = jnp.sqrt(jnp.asarray(0.5))
    hanging_base_pose = jnp.array([sqrt_half, 0.0, sqrt_half, 0.0, 0.0, 0.0, 0.0])
    return GVS.from_segments(
        segments,
        gravity=jnp.array([0.0, 0.0, -config.gravity]),
        base_pose=hanging_base_pose,
        scale_rotational_basis_by_length=True,
        actuators=(actuator,),
    )


def make_soft_furuta(config: SoftSwingUpPendulumConfig) -> GVS:
    """Construct the Soft Furuta with a horizontal rigid arm and soft pendulum.

    The first revolute axis is world vertical. The second is perpendicular and
    horizontal, so the semantic configuration helpers include the required
    ``+/- pi/2`` second-joint offset.
    """
    damping = jnp.array([[config.joint_damping]])
    arm = _rigid_circular_segment(
        length=config.furuta_arm_length,
        radius=config.furuta_arm_radius,
        density=config.furuta_arm_density,
        joint=JointSpec.revolute("z", damping=damping),
        num_gauss_points=config.num_gauss_points,
    )
    soft_link = _paper_soft_segment(
        config,
        length=config.soft_length,
        joint=JointSpec.revolute("y", damping=damping),
        bending_component="kappa_y",
    )
    actuator = _direct_input(
        4,
        0,
        label="first_joint_torque",
        units="N m",
        bound=80.0,
        name="soft_furuta_first_joint",
    )
    return GVS.from_segments(
        [arm, soft_link],
        gravity=jnp.array([0.0, 0.0, -config.gravity]),
        base_pose=jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        scale_rotational_basis_by_length=True,
        actuators=(actuator,),
    )


def r_sip_upright_configuration() -> Array:
    """Return the straight upright R-SIP configuration."""
    return jnp.zeros((3,), dtype=jnp.float64)


def r_sip_hanging_configuration() -> Array:
    """Return the straight downward R-SIP configuration."""
    return jnp.array([jnp.pi, 0.0, 0.0], dtype=jnp.float64)


def cart_pole_upright_configuration() -> Array:
    """Return the straight upright Soft Cart-Pole configuration."""
    return jnp.zeros((4,), dtype=jnp.float64)


def cart_pole_hanging_configuration() -> Array:
    """Return the straight downward Soft Cart-Pole configuration."""
    return jnp.array([0.0, -jnp.pi, 0.0, 0.0], dtype=jnp.float64)


def pendubot_upright_configuration() -> Array:
    """Return both straight Soft Pendubot links pointing upward."""
    return jnp.array([jnp.pi, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)


def pendubot_hanging_configuration() -> Array:
    """Return both straight Soft Pendubot links hanging downward."""
    return jnp.zeros((6,), dtype=jnp.float64)


def furuta_upright_configuration() -> Array:
    """Return the Soft Furuta link pointing upward at zero arm angle."""
    return jnp.array([0.0, -jnp.pi / 2.0, 0.0, 0.0], dtype=jnp.float64)


def furuta_hanging_configuration() -> Array:
    """Return the Soft Furuta link hanging downward at zero arm angle."""
    return jnp.array([0.0, jnp.pi / 2.0, 0.0, 0.0], dtype=jnp.float64)


def sip_initial_configuration() -> Array:
    """Return the original affine SIP paper's bent initial configuration."""
    return jnp.array([jnp.pi / 4.0, -jnp.pi / 4.0], dtype=jnp.float64)


def sip_tip_torque_generalized_force(torque: float | Array) -> Array:
    """Map a pure tip torque to the fixed SIP affine coordinates."""
    torque_array = jnp.asarray(torque)
    return jnp.stack([torque_array, 0.5 * torque_array], axis=-1)


def split_state(y: Array, num_coordinates: int) -> tuple[Array, Array]:
    """Split a state with trailing layout ``[q, qd]``."""
    expected_size = 2 * num_coordinates
    if y.ndim == 0 or y.shape[-1] != expected_size:
        raise ValueError(
            f"y must have trailing dimension {expected_size}, got {y.shape}."
        )
    return y[..., :num_coordinates], y[..., num_coordinates:]


def simulate_open_loop(
    robot: GVS,
    *,
    initial_configuration: Array,
    control: Array | float | None = None,
    external_force: Array | None = None,
    t1: float = 5.0,
    solver_dt: float = 1e-3,
    save_dt: float = 2e-2,
) -> SystemState:
    """Simulate a constant-input open-loop trajectory."""
    q0 = jnp.asarray(initial_configuration, dtype=jnp.float64)
    if q0.shape != (robot.num_dofs,):
        raise ValueError(f"initial_configuration must have shape ({robot.num_dofs},).")
    if control is None:
        u = jnp.zeros((robot.num_actuators,), dtype=q0.dtype)
    else:
        u = jnp.asarray(control, dtype=q0.dtype)
        if u.ndim == 0:
            u = jnp.full((robot.num_actuators,), u)
        if u.shape != (robot.num_actuators,):
            raise ValueError(f"control must have shape ({robot.num_actuators},).")
    if external_force is None:
        external_force = jnp.zeros((robot.num_dofs,), dtype=q0.dtype)
    initial_state = SystemState(t=0.0, y=jnp.concatenate([q0, jnp.zeros_like(q0)]))
    return robot.rollout_to(
        initial_state=initial_state,
        u=u,
        tau_ext=external_force,
        t1=t1,
        solver_dt=solver_dt,
        save_dt=save_dt,
        max_steps=None,
    )


def save_summary_figure(
    robot: GVS,
    trajectory: SystemState,
    output_path: Path,
    *,
    coordinate_labels: tuple[str, ...],
    cart: bool = False,
    show: bool,
) -> None:
    """Save coordinate, tip, energy, and 3-D backbone plots."""
    import matplotlib.pyplot as plt

    q_ts, v_ts = split_state(trajectory.y, robot.num_dofs)
    if len(coordinate_labels) != robot.num_dofs:
        raise ValueError("coordinate_labels must contain one label per coordinate.")
    s_points = jnp.linspace(0.0, robot.length, 100)
    poses_ts = jax.vmap(robot.forward_kinematics_abscissa_batched, in_axes=(0, None))(
        q_ts, s_points
    )
    positions_ts = np.asarray(poses_ts[:, :, :3, 3])
    tip_positions = positions_ts[:, -1]
    potential = np.asarray(jax.vmap(robot.potential_energy)(q_ts))
    kinetic = np.asarray(jax.vmap(robot.kinetic_energy)(q_ts, v_ts))

    figure = plt.figure(figsize=(11.5, 8.0))
    coordinate_axis = figure.add_subplot(2, 2, 1)
    tip_axis = figure.add_subplot(2, 2, 2)
    energy_axis = figure.add_subplot(2, 2, 3)
    shape_axis = figure.add_subplot(2, 2, 4, projection="3d")

    for index, label in enumerate(coordinate_labels):
        coordinate_axis.plot(trajectory.t, q_ts[:, index], label=label)
    coordinate_axis.set(xlabel="Time [s]", ylabel="Configuration")
    coordinate_axis.legend(fontsize="small")

    for index, label in enumerate(("x", "y", "z")):
        tip_axis.plot(trajectory.t, tip_positions[:, index], label=label)
    tip_axis.set(xlabel="Time [s]", ylabel="Tip position [m]")
    tip_axis.legend()

    energy_axis.plot(trajectory.t, potential, label="potential")
    energy_axis.plot(trajectory.t, kinetic, label="kinetic")
    energy_axis.plot(trajectory.t, potential + kinetic, label="total")
    energy_axis.set(xlabel="Time [s]", ylabel="Energy [J]")
    energy_axis.legend()

    snapshot_indices = np.linspace(0, len(trajectory.t) - 1, 9, dtype=int)
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(snapshot_indices)))
    for color, frame_index in zip(colors, snapshot_indices, strict=True):
        positions = positions_ts[frame_index]
        shape_axis.plot(
            positions[:, 0], positions[:, 1], positions[:, 2], color=color, alpha=0.8
        )
        if cart:
            cart_position = float(q_ts[frame_index, 0])
            shape_axis.scatter(0.0, cart_position, 0.0, color=color, s=28)
    shape_axis.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    shape_axis.set_box_aspect((1.0, 1.0, 1.0))

    for axis in (coordinate_axis, tip_axis, energy_axis):
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(figure)


__all__ = [
    "AFFINE_CURVATURE_MATRIX",
    "SoftSwingUpPendulumConfig",
    "SoftInvertedPendulumConfig",
    "cart_pole_hanging_configuration",
    "cart_pole_upright_configuration",
    "furuta_hanging_configuration",
    "furuta_upright_configuration",
    "make_fixed_pendulum",
    "make_r_soft_inverted_pendulum",
    "make_soft_cart_pole",
    "make_soft_furuta",
    "make_soft_pendubot",
    "pendubot_hanging_configuration",
    "pendubot_upright_configuration",
    "r_sip_hanging_configuration",
    "r_sip_upright_configuration",
    "save_summary_figure",
    "simulate_open_loop",
    "sip_initial_configuration",
    "sip_tip_torque_generalized_force",
    "split_state",
]
