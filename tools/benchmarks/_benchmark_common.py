"""Shared helpers for Soromox benchmarking CLIs."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from soromox.systems import GVS, PCS, CrossSectionGeometry, Pendulum, PlanarPCS
from soromox.systems.gvs import BasisAttributes, JointAttributes, LinkAttributes

Array = jax.Array


@dataclass(frozen=True)
class SystemConfig:
    """Factory plus context builder for a Soromox system."""

    factory: Callable[[int], Any]
    size_label: str
    build_context: Callable[[Any], MutableMapping[str, Array]]


def _pendulum_factory(num_links: int) -> Pendulum:
    lengths = jnp.linspace(0.15, 0.25, num_links)
    masses = jnp.linspace(0.8, 1.2, num_links)
    inertias = (1.0 / 12.0) * masses * lengths**2
    lc = lengths * 0.5
    params = {
        "m": masses,
        "I": inertias,
        "L": lengths,
        "Lc": lc,
        "g": jnp.array([0.0, -9.81]),
        "K": 5.0 * jnp.eye(num_links),
        "D": 0.1 * jnp.eye(num_links),
    }
    return Pendulum(params)


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


def _planar_pcs_factory(num_segments: int) -> PlanarPCS:
    lengths = jnp.full((num_segments,), 0.12)
    radii = jnp.full((num_segments,), 0.015)
    rho = 1070.0 * jnp.ones((num_segments,))
    params: dict[str, Array] = {
        "th0": jnp.array(jnp.pi / 2),
        "L": lengths,
        "r": radii,
        "rho": rho,
        "g": jnp.array([0.0, 9.81]),
        "E": 4.0e5 * jnp.ones((num_segments,)),
        "G": 1.5e5 * jnp.ones((num_segments,)),
    }
    diag_entries = (
        jnp.repeat(jnp.array([[1.0, 200.0, 200.0]]), num_segments, axis=0)
        * lengths[:, None]
    ).reshape(-1)
    params["D"] = 5.0e-4 * jnp.diag(diag_entries)
    return PlanarPCS(num_segments=num_segments, params=params)


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


def _pcs_factory(num_segments: int) -> PCS:
    lengths = jnp.full((num_segments,), 0.1)
    radii = jnp.full((num_segments,), 0.02)
    rho = 1050.0 * jnp.ones((num_segments,))
    params: dict[str, Array] = {
        "p0": jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "L": lengths,
        "r": radii,
        "rho": rho,
        "g": jnp.array([0.0, 0.0, -9.81]),
        "E": 6.0e5 * jnp.ones((num_segments,)),
        "G": 2.5e5 * jnp.ones((num_segments,)),
    }
    diag_entries = (
        jnp.repeat(
            jnp.array([[1.0, 1.0, 1.0, 300.0, 300.0, 300.0]]), num_segments, axis=0
        )
        * lengths[:, None]
    ).reshape(-1)
    params["D"] = 5.0e-4 * jnp.diag(diag_entries)
    return PCS(num_segments=num_segments, params=params)


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


def _gvs_factory(num_segments: int) -> GVS:
    links: Sequence[LinkAttributes] = []
    joints: Sequence[JointAttributes] = []
    bases: Sequence[BasisAttributes] = []
    n_gauss: Sequence[int] = []
    for _ in range(num_segments):
        links.append(
            LinkAttributes(
                cross_section_geometry=CrossSectionGeometry.CIRCULAR,
                E=1.0e6,
                nu=0.45,
                rho=980.0,
                eta=2.5e3,
                L=0.25,
                r_i=0.02,
                r_f=0.02,
            )
        )
        joints.append(JointAttributes(jointtype="Fixed"))
        bases.append(
            BasisAttributes(
                basistype="Monomial",
                Bdof=[1, 1, 1, 1, 1, 1],
                Bodr=[0, 0, 0, 0, 0, 0],
                xi_ref=[0, 0, 0, 1, 0, 0],
            )
        )
        n_gauss.append(5)

    return GVS(
        links_list=list(links),
        joints_list=list(joints),
        basis_list=list(bases),
        n_gauss_list=list(n_gauss),
        gravity_vector=[0.0, 0.0, 9.81],
    )


def _gvs_context(system: GVS) -> MutableMapping[str, Array]:
    dof = system.dof_tot_system
    q = jnp.linspace(-0.12, 0.12, dof)
    qd = jnp.linspace(0.16, -0.16, dof)
    ctx: MutableMapping[str, Array] = {
        "q": q,
        "qd": qd,
        "u": jnp.zeros((system.num_actuators,)),
        "tau_ext": jnp.zeros((dof,)),
        "y": jnp.concatenate([q, qd]),
        "t": jnp.array(0.0),
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
        "planar_pcs": SystemConfig(
            factory=_planar_pcs_factory,
            size_label="num_segments",
            build_context=_planar_pcs_context,
        ),
        "pcs": SystemConfig(
            factory=_pcs_factory,
            size_label="num_segments",
            build_context=_pcs_context,
        ),
        "gvs": SystemConfig(
            factory=_gvs_factory,
            size_label="num_segments",
            build_context=_gvs_context,
        ),
    }


def add_system_selection_args(
    parser: argparse.ArgumentParser,
    registry: Mapping[str, Any],
    *,
    default_segment_counts: Sequence[int],
) -> None:
    """Attach shared --systems/--segment-counts arguments to a parser."""

    systems = list(registry.keys())
    parser.add_argument(
        "--systems",
        nargs="*",
        default=systems,
        choices=systems,
        help="Systems to benchmark (default: all)",
    )
    parser.add_argument(
        "--segment-counts",
        nargs="*",
        type=int,
        default=list(default_segment_counts),
        help="Sequence of link/segment counts to benchmark",
    )


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
