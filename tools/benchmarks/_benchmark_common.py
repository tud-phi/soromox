"""Shared helpers for Soromox benchmarking CLIs."""

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
import sys
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from soromox.systems import (
    GVS,
    PCS,
    ArticulatedSoftRobot,
    ArticulatedSoftRobotParams,
    GVSSegment,
    JointSpec,
    LinkSpec,
    PCSStructure,
    Pendulum,
    PendulumParams,
    PlanarPCS,
    PlanarPCSStructure,
    StrainBasisSpec,
)

Array = jax.Array
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _distribution_version(distribution: str) -> str | None:
    """Return an installed distribution version without importing its package."""

    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_output(*args: str) -> str | None:
    """Return one Git query result, or ``None`` outside a readable checkout."""

    try:
        result = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def benchmark_environment_metadata(device: jax.Device) -> dict[str, Any]:
    """Describe the software and accelerator environment for an artifact row.

    The returned fields are intentionally flat so they can be copied into both
    JSON objects and CSV rows. Hostnames and user-specific paths are excluded;
    the metadata captures the factors most likely to explain benchmark changes
    without exposing machine identity.

    Args:
        device: JAX device on which the benchmark executes.

    Returns:
        Reproducibility metadata containing the UTC start time, source revision,
        installed runtime versions, FP64 setting, and accelerator identity.
    """

    git_revision = _git_output("rev-parse", "HEAD")
    tracked_changes = _git_output("status", "--porcelain", "--untracked-files=no")
    device_kind = getattr(device, "device_kind", None)
    platform_version = getattr(device.client, "platform_version", None)
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_revision": git_revision,
        "git_dirty": None if tracked_changes is None else bool(tracked_changes),
        "python_version": sys.version.split()[0],
        "soromox_version": _distribution_version("soromox"),
        "jax_version": jax.__version__,
        "jaxlib_version": _distribution_version("jaxlib"),
        "x64_enabled": bool(jax.config.x64_enabled),
        "device_kind": device_kind,
        "device_id": f"{device.platform}:{device_kind or 'unknown'}:{device.id}",
        "platform_version": platform_version,
        "device_count": len(jax.devices(device.platform)),
    }


@dataclass(frozen=True)
class SystemConfig:
    """Factory plus context builder for a Soromox system."""

    factory: Callable[[int], Any]
    size_label: str
    build_context: Callable[[Any], MutableMapping[str, Array]]
    gauss_factory: Callable[[int, int], Any] | None = None
    default_gauss_points: int | None = None
    min_gauss_points: int = 1

    @property
    def supports_gauss_points(self) -> bool:
        return self.gauss_factory is not None


def _pendulum_factory(num_links: int) -> Pendulum:
    lengths = jnp.linspace(0.15, 0.25, num_links)
    masses = jnp.linspace(0.8, 1.2, num_links)
    inertias = (1.0 / 12.0) * masses * lengths**2
    lc = lengths * 0.5
    params = PendulumParams(
        mass=masses,
        moment_inertia=inertias,
        length=lengths,
        center_of_mass_length=lc,
        gravity=jnp.array([0.0, -9.81]),
        joint_stiffness=5.0 * jnp.eye(num_links),
        joint_damping=0.1 * jnp.eye(num_links),
        joint_rest_configuration=jnp.zeros((num_links,)),
        radius=0.05 * lengths,
    )
    return Pendulum(params=params)


def _pendulum_context(system: Pendulum) -> MutableMapping[str, Array]:
    n = system.num_links
    q = jnp.linspace(-0.15, 0.15, n)
    qd = jnp.linspace(0.2, -0.2, n)
    ctx: MutableMapping[str, Array] = {
        "q": q,
        "qd": qd,
        "u": jnp.zeros((system.num_actuators,)),
        "tau_ext": jnp.zeros((n,)),
        "y": jnp.concatenate([q, qd]),
        "t": jnp.array(0.0),
    }
    return ctx


def _articulated_soft_robot_factory(
    num_links: int, gauss_points: int = 5
) -> ArticulatedSoftRobot:
    axes = jnp.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    axis_indices = jnp.arange(num_links) % axes.shape[0]
    omega = axes[axis_indices]
    joint_screws = jnp.concatenate([omega, jnp.zeros((num_links, 3))], axis=1)

    lengths = jnp.linspace(0.15, 0.25, num_links)
    lateral = 0.03 * jnp.sin(jnp.arange(num_links, dtype=lengths.dtype))
    vertical = 0.04 * jnp.cos(jnp.arange(num_links, dtype=lengths.dtype))
    p_tip = jnp.stack([lengths, lateral, vertical], axis=1)
    p_com = 0.5 * p_tip

    masses = jnp.linspace(0.8, 1.2, num_links)
    I_diag = jnp.stack(
        [
            0.03 * masses * lengths**2,
            0.04 * masses * lengths**2,
            0.05 * masses * lengths**2,
        ],
        axis=1,
    )
    I_com = jax.vmap(jnp.diag)(I_diag)

    params = ArticulatedSoftRobotParams(
        joint_screw=joint_screws,
        parent_to_joint_transform=jnp.broadcast_to(jnp.eye(4), (num_links, 4, 4)),
        tip_position=p_tip,
        center_of_mass_position=p_com,
        mass=masses,
        center_of_mass_inertia=I_com,
        gravity=jnp.array([0.0, 0.0, -9.81]),
        joint_stiffness=5.0 * jnp.eye(num_links),
        joint_damping=0.1 * jnp.eye(num_links),
        joint_rest_configuration=jnp.zeros((num_links,)),
        radius=0.05 * lengths,
    )
    return ArticulatedSoftRobot(params=params)


def _articulated_soft_robot_context(
    system: ArticulatedSoftRobot,
) -> MutableMapping[str, Array]:
    n = system.num_links
    q = jnp.linspace(-0.15, 0.15, n)
    qd = jnp.linspace(0.2, -0.2, n)
    ctx: MutableMapping[str, Array] = {
        "q": q,
        "qd": qd,
        "u": jnp.zeros((system.num_actuators,)),
        "tau_ext": jnp.zeros((n,)),
        "y": jnp.concatenate([q, qd]),
        "t": jnp.array(0.0),
        "s_tip": system.total_length,
        "g_tips": system.forward_kinematics_tips(q),
    }
    return ctx


def _planar_pcs_factory(num_segments: int, gauss_points: int = 5) -> PlanarPCS:
    return PlanarPCS.from_links(
        [
            LinkSpec.circular(
                length=0.12,
                radius=0.015,
                density=1070.0,
                young_modulus=4.0e5,
                shear_modulus=1.5e5,
                material_damping_coefficient=5.0e-4,
                reference_strain=[0.0, 1.0, 0.0],
            )
            for _ in range(num_segments)
        ],
        base_pose=jnp.array([jnp.pi / 2, 0.0, 0.0]),
        gravity=jnp.array([0.0, 9.81]),
        structure=PlanarPCSStructure(num_gauss_points=gauss_points),
    )


def _planar_pcs_context(system: PlanarPCS) -> MutableMapping[str, Array]:
    dof = system.num_actuators
    q = jnp.linspace(-0.2, 0.2, dof)
    qd = jnp.linspace(0.25, -0.25, dof)
    s_vals = system.L_cum[1:]

    def _forward_kinematics_at_s(s):
        return system.forward_kinematics(q, s)

    chi_tips = jax.vmap(_forward_kinematics_at_s)(s_vals)
    ctx: MutableMapping[str, Array] = {
        "q": q,
        "qd": qd,
        "u": jnp.zeros((system.num_actuators,)),
        "tau_ext": jnp.zeros((dof,)),
        "y": jnp.concatenate([q, qd]),
        "t": jnp.array(0.0),
        "s_tip": jnp.sum(system.L),
        "chi_tips": chi_tips,
    }
    return ctx


def _pcs_factory(num_segments: int, gauss_points: int = 5) -> PCS:
    return PCS.from_links(
        [
            LinkSpec.circular(
                length=0.1,
                radius=0.02,
                density=1050.0,
                young_modulus=6.0e5,
                shear_modulus=2.5e5,
                material_damping_coefficient=5.0e-4,
                reference_strain=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            )
            for _ in range(num_segments)
        ],
        base_pose=jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        gravity=jnp.array([0.0, 0.0, -9.81]),
        structure=PCSStructure(num_gauss_points=gauss_points),
    )


def _pcs_context(system: PCS) -> MutableMapping[str, Array]:
    dof = system.num_actuators
    q = jnp.linspace(-0.15, 0.15, dof)
    qd = jnp.linspace(0.18, -0.18, dof)
    s_vals = system.L_cum[1:]
    g_tips = jax.vmap(lambda s: system.forward_kinematics(q, s))(s_vals)
    midpoints = system.L_cum[:-1] + 0.5 * system.L
    s_points = jnp.concatenate(
        [jnp.array([0.0], dtype=q.dtype), midpoints, system.L_cum[1:]]
    )
    ctx: MutableMapping[str, Array] = {
        "q": q,
        "qd": qd,
        "u": jnp.zeros((system.num_actuators,)),
        "tau_ext": jnp.zeros((dof,)),
        "y": jnp.concatenate([q, qd]),
        "t": jnp.array(0.0),
        "s_tip": jnp.sum(system.L),
        "g_tips": g_tips,
        "s_points": s_points,
    }
    return ctx


def _gvs_segment(strain_basis_order: int, gauss_points: int) -> GVSSegment:
    if strain_basis_order < 0:
        raise ValueError("GVS strain-basis order must be non-negative.")

    return GVSSegment(
        link=LinkSpec.circular(
            young_modulus=1.0e6,
            shear_modulus=1.0e6 / 2.9,
            density=980.0,
            material_damping_coefficient=2.5e3,
            length=0.25,
            radius=0.02,
            reference_strain=[0, 0, 0, 1, 0, 0],
        ),
        joint=JointSpec(type="fixed"),
        basis=StrainBasisSpec(
            type="legendre",
            strain_selector=[1, 1, 1, 1, 1, 1],
            basis_order=int(strain_basis_order),
        ),
        num_gauss_points=gauss_points,
    )


def _gvs_factory(num_segments: int, gauss_points: int = 5) -> GVS:
    segments: list[GVSSegment] = []
    for _ in range(num_segments):
        segments.append(_gvs_segment(strain_basis_order=0, gauss_points=gauss_points))

    return GVS.from_segments(
        segments=segments,
        gravity=jnp.array([0.0, 0.0, 9.81]),
    )


def _gvs_basis_order_factory(strain_basis_order: int, gauss_points: int = 5) -> GVS:
    return GVS.from_segments(
        segments=[
            _gvs_segment(
                strain_basis_order=strain_basis_order,
                gauss_points=gauss_points,
            )
        ],
        gravity=jnp.array([0.0, 0.0, 9.81]),
    )


def get_gvs_basis_order_system_config() -> SystemConfig:
    """Return the GVS configuration that sweeps basis order for one link."""

    return SystemConfig(
        factory=_gvs_basis_order_factory,
        size_label="strain_basis_order",
        build_context=_gvs_context,
        gauss_factory=_gvs_basis_order_factory,
        default_gauss_points=5,
        min_gauss_points=5,
    )


def _gvs_context(system: GVS) -> MutableMapping[str, Array]:
    dof = system.num_dofs
    q = jnp.linspace(-0.12, 0.12, dof)
    qd = jnp.linspace(0.16, -0.16, dof)
    ctx: MutableMapping[str, Array] = {
        "q": q,
        "qd": qd,
        "u": jnp.zeros((system.num_actuators,)),
        "tau_ext": jnp.zeros((dof,)),
        "y": jnp.concatenate([q, qd]),
        "t": jnp.array(0.0),
        "s_tip": jnp.sum(system.segment_lengths),
    }
    return ctx


def get_system_registry() -> Mapping[str, SystemConfig]:
    """Return the standard benchmarking systems."""

    return {
        "pendulum": SystemConfig(
            factory=_pendulum_factory,
            size_label="num_links",
            build_context=_pendulum_context,
        ),
        "articulated_soft_robot": SystemConfig(
            factory=_articulated_soft_robot_factory,
            size_label="num_links",
            build_context=_articulated_soft_robot_context,
        ),
        "planar_pcs": SystemConfig(
            factory=_planar_pcs_factory,
            size_label="num_segments",
            build_context=_planar_pcs_context,
            gauss_factory=_planar_pcs_factory,
            default_gauss_points=5,
        ),
        "pcs": SystemConfig(
            factory=_pcs_factory,
            size_label="num_segments",
            build_context=_pcs_context,
            gauss_factory=_pcs_factory,
            default_gauss_points=5,
        ),
        "gvs": SystemConfig(
            factory=_gvs_factory,
            size_label="num_segments",
            build_context=_gvs_context,
            gauss_factory=_gvs_factory,
            default_gauss_points=5,
            min_gauss_points=5,
        ),
    }


def add_system_selection_args(
    parser: argparse.ArgumentParser,
    registry: Mapping[str, Any],
    *,
    default_segment_counts: Sequence[int],
    default_systems: Sequence[str] | None = None,
) -> None:
    """Attach shared --systems/--segment-counts arguments to a parser."""

    systems = list(registry.keys())
    selected_systems = systems if default_systems is None else list(default_systems)
    selected_systems = normalize_system_names(selected_systems, registry)
    parser.add_argument(
        "--systems",
        nargs="*",
        default=selected_systems,
        metavar="SYSTEM",
        help=(
            "Systems to benchmark. Available: "
            f"{', '.join(systems)}. CamelCase names such as PCS and GVS are accepted."
        ),
    )
    parser.add_argument(
        "--segment-counts",
        nargs="*",
        type=int,
        default=list(default_segment_counts),
        help="Sequence of link/segment counts to benchmark",
    )


def normalize_system_names(
    values: Sequence[str],
    registry: Mapping[str, Any],
) -> list[str]:
    """Normalize CLI system names while accepting snake_case and CamelCase."""

    normalized: list[str] = []
    unknown: list[str] = []
    compact_registry = {name.replace("_", "").lower(): name for name in registry}
    for value in values:
        name = str(value)
        snake = name.strip().replace("-", "_").lower()
        if snake in registry:
            canonical = snake
        else:
            canonical = compact_registry.get(snake.replace("_", ""))

        if canonical is None:
            unknown.append(name)
            continue
        if canonical not in normalized:
            normalized.append(canonical)

    if unknown:
        available = ", ".join(registry.keys())
        raise ValueError(
            "Unknown system name(s): "
            f"{', '.join(unknown)}. Available systems: {available}."
        )
    return normalized


def add_gauss_point_args(parser: argparse.ArgumentParser) -> None:
    """Attach shared Gaussian quadrature sweep arguments to a parser."""

    parser.add_argument(
        "--gauss-points",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional sweep of Gauss-Legendre quadrature orders/interior point "
            "counts. Applies only to systems with quadrature-driven dynamics; "
            "PCS/GVS include two zero-weight endpoints internally."
        ),
    )


def normalize_gauss_point_values(
    values: Sequence[int] | None,
) -> list[int] | None:
    """Return a sorted, de-duplicated Gauss-point sweep."""

    if values is None:
        return None
    return sorted(dict.fromkeys(values))


def gauss_point_sweep_values(
    config: Any,
    requested_values: Sequence[int] | None,
) -> list[int | None]:
    """Return the Gauss-point values that should be evaluated for a system."""

    if requested_values is None:
        return [None]
    if not getattr(config, "supports_gauss_points", False):
        return []

    min_value = int(getattr(config, "min_gauss_points", 1))
    return [int(value) for value in requested_values if value >= min_value]


def build_system_with_gauss_points(
    config: Any,
    size: int,
    gauss_points: int | None,
) -> Any:
    """Build a configured benchmark system, optionally overriding quadrature."""

    if gauss_points is None:
        return config.factory(size)
    if not getattr(config, "supports_gauss_points", False):
        raise ValueError(
            f"{getattr(config, 'size_label', 'system')} does not support "
            "Gaussian quadrature point sweeps."
        )
    return config.gauss_factory(size, gauss_points)


def system_gauss_point_metadata(system: Any) -> tuple[int | None, int | None]:
    """Return ``(gauss_points, integration_points)`` for quadrature systems.

    ``gauss_points`` is the requested Gauss-Legendre order/interior count used
    by constructors. ``integration_points`` includes Soromox's zero-weight
    boundary nodes where the implementation stores them.
    """

    if hasattr(system, "max_num_gauss_points"):
        return int(system.max_num_gauss_points), int(system.max_num_integration_points)
    if hasattr(system, "num_gauss_points") and hasattr(
        system, "num_integration_points"
    ):
        if system.num_gauss_points is None or system.num_integration_points is None:
            return None, None
        return int(system.num_gauss_points), int(system.num_integration_points)
    return None, None


def add_integration_args(
    parser: argparse.ArgumentParser,
    *,
    duration_default: float = 1.0,
    solver_dt_default: float = 1e-4,
    save_dt_default: float = 0.01,
) -> None:
    """Attach shared integration arguments to a parser."""

    parser.add_argument(
        "--duration",
        type=float,
        default=duration_default,
        help="Simulation duration (seconds)",
    )
    parser.add_argument(
        "--solver-dt",
        "--dt",
        dest="solver_dt",
        type=float,
        default=solver_dt_default,
        help="Integration step size",
    )
    parser.add_argument(
        "--save-dt",
        type=float,
        default=save_dt_default,
        help="Save the system state every `save_dt` seconds (must be >= solver_dt)",
    )


def block_until_ready(tree: Any) -> None:
    """Synchronise on all pending JAX work in a PyTree."""

    def _block(x: Any) -> Any:
        if hasattr(x, "block_until_ready"):
            x.block_until_ready()
        return x

    jax.tree_util.tree_map(_block, tree)
