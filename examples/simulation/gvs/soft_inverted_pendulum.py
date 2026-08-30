"""Shared affine-curvature soft inverted-pendulum examples.

The fixed model follows the geometry, generalized constitutive matrices, tip
torque, and parameters reported by Della Santina. Its mass is distributed
along a GVS link rather than concentrated at the tip. The cart model retains
the same soft link and clamps it directly to a passive translating cart.

References:
    Della Santina, C. (2020). The soft inverted pendulum with affine
    curvature. 2020 59th IEEE Conference on Decision and Control (CDC),
    4135-4142. https://doi.org/10.1109/CDC42340.2020.9303976

    Ajithkumar, A. (2023). Control and stabilization of soft inverted
    pendulum on a cart. Master's thesis, University of Maryland, College Park.
    https://doi.org/10.13016/dspace/7jir-df1p
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

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


@dataclass(frozen=True)
class SoftInvertedPendulumConfig:
    """Physical and simulation-independent parameters shared by both models.

    The default soft-link values reproduce Example 1 of Della Santina (2020).
    ``cart_mass`` is the only additional physical parameter of the cart model.

    Attributes:
        length: Undeformed backbone length in meters.
        thickness: Height and width of the square cross-section in meters.
        link_mass: Total distributed link mass in kilograms.
        gravity: Downward gravitational acceleration in meters per second squared.
        damping_coefficient: Scalar ``beta`` multiplying the affine-curvature
            matrix.
        stiffness_coefficient: Scalar ``k`` multiplying the affine-curvature
            matrix.
        cart_mass: Additional passive cart mass in kilograms.
        num_gauss_points: Number of GVS quadrature points.

    References:
        Della Santina, C. (2020). The soft inverted pendulum with affine
        curvature. 2020 59th IEEE Conference on Decision and Control (CDC),
        4135-4142. https://doi.org/10.1109/CDC42340.2020.9303976
    """

    length: float = 1.0
    thickness: float = 0.1
    link_mass: float = 1.0
    gravity: float = 9.81
    damping_coefficient: float = 0.1
    stiffness_coefficient: float = 1.0
    cart_mass: float = 1.0
    num_gauss_points: int = 5

    def __post_init__(self) -> None:
        """Validate concrete example parameters."""
        positive = {
            "length": self.length,
            "thickness": self.thickness,
            "link_mass": self.link_mass,
            "gravity": self.gravity,
            "stiffness_coefficient": self.stiffness_coefficient,
            "cart_mass": self.cart_mass,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not np.isfinite(self.damping_coefficient) or self.damping_coefficient < 0.0:
            raise ValueError("damping_coefficient must be finite and non-negative.")
        if self.num_gauss_points < 5:
            raise ValueError("num_gauss_points must be at least 5 for GVS.")

    @property
    def density(self) -> float:
        """Return the density that realizes ``link_mass`` for the square link."""
        return self.link_mass / (self.length * self.thickness**2)

    @property
    def stiffness(self) -> Array:
        """Return the generalized affine-curvature stiffness matrix ``k H``."""
        return self.stiffness_coefficient * AFFINE_CURVATURE_MATRIX

    @property
    def damping(self) -> Array:
        """Return the generalized affine-curvature damping matrix ``beta H``."""
        return self.damping_coefficient * AFFINE_CURVATURE_MATRIX


class SoftCartPendulumGVS(GVS):
    """GVS pendulum clamped directly to a passive translating cart.

    The prismatic joint already propagates the distributed soft-link inertia.
    This class adds only the cart's rigid translational mass to the first
    generalized coordinate.

    Attributes:
        cart_mass: Passive cart mass in kilograms.

    References:
        Ajithkumar, A. (2023). Control and stabilization of soft inverted
        pendulum on a cart. Master's thesis, University of Maryland, College
        Park. https://doi.org/10.13016/dspace/7jir-df1p
    """

    cart_mass: Array

    def __init__(self, *args: Any, cart_mass: float | Array, **kwargs: Any) -> None:
        """Initialize the GVS model and its additional passive cart mass.

        Args:
            *args: Positional arguments forwarded to :class:`GVS`.
            cart_mass: Finite positive cart mass in kilograms.
            **kwargs: Keyword arguments forwarded to :class:`GVS`.

        Raises:
            ValueError: If ``cart_mass`` is not a finite positive scalar.
        """
        cart_mass_array = jnp.asarray(cart_mass, dtype=jnp.float64)
        if cart_mass_array.shape != () or not bool(
            jnp.isfinite(cart_mass_array) & (cart_mass_array > 0.0)
        ):
            raise ValueError("cart_mass must be a finite positive scalar.")
        super().__init__(*args, **kwargs)
        self.cart_mass = cart_mass_array

    def _augment_cart_inertia(self, inertia: Array) -> Array:
        """Add the constant cart mass without assembling another inertia matrix.

        Args:
            inertia: GVS inertia matrix with shape
                ``(num_coordinates, num_coordinates)``.

        Returns:
            Inertia matrix with ``cart_mass`` added to its first diagonal entry.
        """
        return inertia.at[0, 0].add(self.cart_mass)

    @eqx.filter_jit
    def inertia_matrix(self, q: Array) -> Array:
        """Return the GVS inertia matrix augmented by the passive cart mass.

        Args:
            q: Generalized coordinates ``[x_cart, theta_0, theta_1]``.

        Returns:
            Augmented inertia matrix with shape ``(3, 3)``.
        """
        return self._augment_cart_inertia(super().inertia_matrix(q))

    def dynamics_terms(
        self, q: Array, qd: Array
    ) -> tuple[Array, Array, Array]:
        """Return dynamics terms with one constant cart-mass augmentation.

        Args:
            q: Generalized coordinates ``[x_cart, theta_0, theta_1]``.
            qd: Generalized velocities in the same order.
        Returns:
            Tuple ``(M, Cqd, G)`` with the cart mass included in ``M``.
        """
        inertia, coriolis_velocity, gravity = super().dynamics_terms(q, qd)
        return (
            self._augment_cart_inertia(inertia),
            coriolis_velocity,
            gravity,
        )


def _pendulum_segment(
    config: SoftInvertedPendulumConfig, joint: JointSpec
) -> GVSSegment:
    """Construct the shared affine-curvature GVS segment.

    Args:
        config: Shared physical parameters.
        joint: Fixed or prismatic base-joint specification.

    Returns:
        One GVS segment with affine ``kappa_z`` curvature.

    References:
        Della Santina, C. (2020). The soft inverted pendulum with affine
        curvature. 2020 59th IEEE Conference on Decision and Control (CDC),
        4135-4142. https://doi.org/10.1109/CDC42340.2020.9303976
    """
    link = LinkSpec.rectangular(
        length=config.length,
        height=config.thickness,
        width=config.thickness,
        density=config.density,
        reference_strain=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        stiffness=config.stiffness,
        damping=config.damping,
    )
    basis = StrainBasisSpec(
        type="monomial",
        strain_selector=("kappa_z",),
        basis_order=1,
    )
    return GVSSegment(
        link=link,
        joint=joint,
        basis=basis,
        num_gauss_points=config.num_gauss_points,
    )


def make_fixed_pendulum(config: SoftInvertedPendulumConfig) -> GVS:
    """Construct the fixed-base affine-curvature soft inverted pendulum.

    Args:
        config: Shared physical parameters.

    Returns:
        A two-coordinate JAX GVS model with coordinates
        ``[theta_0, theta_1]``.

    References:
        Della Santina, C. (2020). The soft inverted pendulum with affine
        curvature. 2020 59th IEEE Conference on Decision and Control (CDC),
        4135-4142. https://doi.org/10.1109/CDC42340.2020.9303976
    """
    return GVS.from_segments(
        [_pendulum_segment(config, JointSpec.fixed())],
        gravity=jnp.array([0.0, 0.0, -config.gravity]),
        scale_rotational_basis_by_length=True,
        actuators=(),
    )


def make_cart_pendulum(config: SoftInvertedPendulumConfig) -> SoftCartPendulumGVS:
    """Construct the passive-cart affine-curvature soft inverted pendulum.

    The soft link is clamped directly to a horizontal prismatic joint. There is
    deliberately no revolute joint between the cart and the link.

    Args:
        config: Shared soft-link parameters and the additional cart mass.

    Returns:
        A three-coordinate JAX GVS model with coordinates
        ``[x_cart, theta_0, theta_1]``.

    References:
        Della Santina, C. (2020). The soft inverted pendulum with affine
        curvature. 2020 59th IEEE Conference on Decision and Control (CDC),
        4135-4142. https://doi.org/10.1109/CDC42340.2020.9303976

        Ajithkumar, A. (2023). Control and stabilization of soft inverted
        pendulum on a cart. Master's thesis, University of Maryland, College
        Park. https://doi.org/10.13016/dspace/7jir-df1p
    """
    return SoftCartPendulumGVS.from_segments(
        [_pendulum_segment(config, JointSpec.prismatic("y"))],
        gravity=jnp.array([0.0, 0.0, -config.gravity]),
        scale_rotational_basis_by_length=True,
        actuators=(),
        cart_mass=config.cart_mass,
    )


def initial_configuration(*, cart: bool) -> Array:
    """Return the paper's initial soft configuration, optionally on a cart.

    Args:
        cart: Whether to prepend a zero cart displacement.

    Returns:
        ``[pi/4, -pi/4]`` or ``[0, pi/4, -pi/4]``.
    """
    soft_configuration = jnp.array([jnp.pi / 4.0, -jnp.pi / 4.0])
    if cart:
        return jnp.concatenate([jnp.zeros((1,)), soft_configuration])
    return soft_configuration


def tip_torque_generalized_force(torque: float | Array, *, cart: bool) -> Array:
    """Map the pure tip torque to affine-curvature generalized forces.

    Args:
        torque: Scalar torque or array of torques in newton meters.
        cart: Whether to prepend a zero generalized force for the passive cart.

    Returns:
        ``[tau, tau/2]`` for the fixed model or
        ``[0, tau, tau/2]`` for the cart model.

    References:
        Della Santina, C. (2020). The soft inverted pendulum with affine
        curvature. 2020 59th IEEE Conference on Decision and Control (CDC),
        4135-4142. https://doi.org/10.1109/CDC42340.2020.9303976
    """
    torque_array = jnp.asarray(torque)
    entries = [torque_array, 0.5 * torque_array]
    if cart:
        entries.insert(0, jnp.zeros_like(torque_array))
    return jnp.stack(entries, axis=-1)


def split_pendulum_state(y: Array, *, cart: bool) -> tuple[Array, Array]:
    """Split a fixed or cart pendulum state into configuration and velocity.

    Args:
        y: State with trailing layout ``[q, qd]``.
        cart: Whether the state contains the cart coordinate.

    Returns:
        Configuration and generalized velocity arrays.

    Raises:
        ValueError: If the trailing state dimension does not match the model.
    """
    num_coordinates = 3 if cart else 2
    expected_size = 2 * num_coordinates
    if y.ndim == 0 or y.shape[-1] != expected_size:
        raise ValueError(
            f"y must have trailing dimension {expected_size}, got {y.shape}."
        )
    return y[..., :num_coordinates], y[..., num_coordinates:]


def simulate_open_loop(
    robot: GVS,
    *,
    cart: bool,
    tip_torque: float = 0.0,
    t1: float = 10.0,
    solver_dt: float = 1e-3,
    save_dt: float = 2e-2,
) -> SystemState:
    """Simulate an autonomous or constant-tip-torque trajectory.

    Args:
        robot: Fixed or cart soft inverted-pendulum model.
        cart: Whether ``robot`` is the cart model.
        tip_torque: Constant tip torque in newton meters.
        t1: Final simulation time in seconds.
        solver_dt: Initial solver step in seconds.
        save_dt: Saved trajectory interval in seconds.

    Returns:
        Time-indexed :class:`SystemState` trajectory.
    """
    q0 = initial_configuration(cart=cart)
    v0 = jnp.zeros_like(q0)
    initial_state = SystemState(t=0.0, y=jnp.concatenate([q0, v0]))
    return robot.rollout_to(
        initial_state=initial_state,
        u=jnp.zeros((robot.num_actuators,), dtype=q0.dtype),
        tau_ext=tip_torque_generalized_force(tip_torque, cart=cart),
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
    cart: bool,
    show: bool,
) -> None:
    """Save state, energy, tip-path, and backbone-snapshot plots.

    Args:
        robot: Fixed or cart pendulum model.
        trajectory: Simulated trajectory.
        output_path: Destination PNG path.
        cart: Whether to include the cart coordinate and body.
        show: Whether to display the Matplotlib window after saving.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    q_ts, v_ts = split_pendulum_state(trajectory.y, cart=cart)
    s_points = jnp.linspace(0.0, robot.length, 80)
    poses_ts = jax.vmap(robot.forward_kinematics_abscissa_batched, in_axes=(0, None))(
        q_ts, s_points
    )
    positions_ts = np.asarray(poses_ts[:, :, :3, 3])
    tip_positions = positions_ts[:, -1]
    potential = np.asarray(jax.vmap(robot.potential_energy)(q_ts))
    kinetic = np.asarray(jax.vmap(robot.kinetic_energy)(q_ts, v_ts))

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    coordinate_labels = (
        ("x_cart", "theta_0", "theta_1") if cart else ("theta_0", "theta_1")
    )
    for index, label in enumerate(coordinate_labels):
        axes[0, 0].plot(trajectory.t, q_ts[:, index], label=label)
    axes[0, 0].set(xlabel="Time [s]", ylabel="Configuration")
    axes[0, 0].legend()

    axes[0, 1].plot(trajectory.t, tip_positions[:, 1], label="horizontal y")
    axes[0, 1].plot(trajectory.t, tip_positions[:, 2], label="vertical z")
    axes[0, 1].set(xlabel="Time [s]", ylabel="Tip position [m]")
    axes[0, 1].legend()

    axes[1, 0].plot(trajectory.t, potential, label="potential")
    axes[1, 0].plot(trajectory.t, kinetic, label="kinetic")
    axes[1, 0].plot(trajectory.t, potential + kinetic, label="total")
    axes[1, 0].set(xlabel="Time [s]", ylabel="Energy [J]")
    axes[1, 0].legend()

    snapshot_indices = np.linspace(0, len(trajectory.t) - 1, 9, dtype=int)
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(snapshot_indices)))
    for color, frame_index in zip(colors, snapshot_indices):
        positions = positions_ts[frame_index]
        axes[1, 1].plot(positions[:, 1], positions[:, 2], color=color, alpha=0.8)
        if cart:
            cart_center = float(q_ts[frame_index, 0])
            axes[1, 1].add_patch(
                Rectangle(
                    (cart_center - 0.15, -0.12),
                    0.3,
                    0.12,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.35,
                )
            )
    axes[1, 1].axhline(-0.16, color="0.35", linewidth=1.0)
    axes[1, 1].set(
        xlabel="Horizontal position y [m]",
        ylabel="Vertical position z [m]",
        aspect="equal",
    )

    for axis in axes.flat:
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
    "SoftCartPendulumGVS",
    "SoftInvertedPendulumConfig",
    "initial_configuration",
    "make_cart_pendulum",
    "make_fixed_pendulum",
    "save_summary_figure",
    "simulate_open_loop",
    "tip_torque_generalized_force",
]
