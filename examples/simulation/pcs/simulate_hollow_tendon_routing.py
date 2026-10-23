"""Render and simulate anchored tendons inside two hollow planar sections.

Run with ``JAX_PLATFORMS=cpu MPLBACKEND=Agg python
examples/simulation/pcs/simulate_hollow_tendon_routing.py``.
The output directory contains MP4 animations, PNG figures, NPZ trajectories,
and a JSON report. All dimensions and mechanics are illustrative.

References:
    Mathew, A. T., Alkayas, A. Y., Alshehhi, A. A. S., Adamu, Y. A.,
    Taha, T., & Renda, F. (2026). A Differentiable Framework for Hollow
    Tendon-Driven Continuum Robots With Implicit Internal Routing.
    *IEEE Transactions on Robotics*, 42, 3009-3027.
    [doi:10.1109/TRO.2026.3714660](https://doi.org/10.1109/TRO.2026.3714660).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optimistix as optx

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D

from soromox.actuation import (
    CircularLumenParams,
    PiecewiseLinearRoutingDiscretization,
    RoutingSolverOptions,
    RoutingSpanParams,
    ShortestPathRouting,
    ShortestPathRoutingParams,
)
from soromox.systems import (
    ContinuumLinkParams,
    CrossSectionParams,
    PlanarPCS,
    PlanarPCSParams,
    PlanarPCSStructure,
)

LENGTH = 0.12
OUTER_RADIUS = 0.016
INNER_RADIUS = 0.014
ANCHOR_OFFSET = 0.004
YOUNG_MODULUS = 3e4
DENSITY = 1070.0
VISCOSITY = 1500.0
COLORS = ("#db593b", "#ac7800")
WALL_COLOR = "#4d8c91"
CLEARANCE_TOLERANCE = 1e-4


def build_model(case="same-side", elements=6):
    """Build a bending-only PCS tube with exact annular stiffness and inertia.

    Args:
        case (str): Anchor layout: ``"same-side"`` or ``"opposed"``.
        elements (int): Routing elements per hollow section.

    Returns:
        result (tuple[PlanarPCS, ShortestPathRouting]): PlanarPCS robot with annular
            mechanics and its ShortestPathRouting instance.
    """
    area = np.pi * (OUTER_RADIUS**2 - INNER_RADIUS**2)
    moment = np.pi / 4 * (OUTER_RADIUS**4 - INNER_RADIUS**4)
    stiffness = LENGTH * jnp.diag(
        jnp.array(
            [YOUNG_MODULUS * moment, YOUNG_MODULUS * area, YOUNG_MODULUS / 3 * area]
        )
    )
    damping = LENGTH * VISCOSITY * jnp.diag(jnp.array([moment, area, area]))
    inertia = DENSITY * jnp.diag(jnp.array([moment, area, area]))
    params = PlanarPCSParams(
        gravity=jnp.zeros(2),
        link=ContinuumLinkParams(
            length=jnp.full(2, LENGTH),
            density=jnp.full(2, DENSITY),
            reference_strain=jnp.tile(jnp.array([0.0, 1.0, 0.0]), (2, 1)),
            cross_section=CrossSectionParams(jnp.full((2, 1), OUTER_RADIUS)),
            stiffness=jnp.tile(stiffness[None], (2, 1, 1)),
            damping=jnp.tile(damping[None], (2, 1, 1)),
            sectional_inertia=jnp.tile(inertia[None], (2, 1, 1)),
        ),
    )
    robot = PlanarPCS(
        params,
        PlanarPCSStructure(
            strain_selector=jnp.array([True, False, False] * 2),
            scale_rotational_basis_by_length=True,
            num_gauss_points=7,
        ),
        base_pose=jnp.array([np.pi / 2, 0.0, 0.0]),
    )
    signs = jnp.array([1.0, 1.0 if case == "same-side" else -1.0])
    offsets = jnp.zeros((2, 3)).at[:, 1].set(ANCHOR_OFFSET * signs)
    routing = ShortestPathRouting(
        ShortestPathRoutingParams(
            RoutingSpanParams(offsets, offsets, (0, 1), (0, 1), (0, 1)),
            CircularLumenParams.constant(INNER_RADIUS, 2),
        ),
        PiecewiseLinearRoutingDiscretization(
            elements_per_segment=elements,
            quadrature_order=3,
            constraint_samples_per_element=4,
            validation_samples_per_element=24,
        ),
        RoutingSolverOptions(max_steps=200, tolerance=1e-7),
    )
    return robot, routing


def tensions_at(time_value, case):
    """Positive preload, smooth pulls, and a return to preload for recoil.

    Args:
        time_value (Array): Physical time in seconds.
        case (str): Anchor layout: ``"same-side"`` or ``"opposed"``.

    Returns:
        result (Array): Strictly positive tension vector in newtons, shape ``(2,)``.
    """
    times = jnp.array([0.0, 0.25, 0.7, 1.5, 1.9, 2.35, 3.0, 3.4, 4.0, 6.0])
    if case == "same-side":
        values = jnp.array(
            [
                [0.02, 0.02],
                [0.02, 0.02],
                [0.65, 0.45],
                [0.65, 0.45],
                [0.02, 0.02],
                [0.02, 0.02],
                [0.45, 0.65],
                [0.45, 0.65],
                [0.02, 0.02],
                [0.02, 0.02],
            ]
        )
    else:
        values = jnp.array(
            [
                [0.02, 0.02],
                [0.02, 0.02],
                [0.65, 0.02],
                [0.65, 0.45],
                [0.02, 0.65],
                [0.02, 0.65],
                [0.55, 0.55],
                [0.55, 0.55],
                [0.02, 0.02],
                [0.02, 0.02],
            ]
        )
    index = jnp.clip(
        jnp.searchsorted(times, time_value, side="right") - 1, 0, times.size - 2
    )
    phase = jnp.clip(
        (time_value - times[index]) / (times[index + 1] - times[index]), 0.0, 1.0
    )
    blend = phase**2 * (3 - 2 * phase)
    return 0.35 * (values[index] + blend * (values[index + 1] - values[index]))


def route_diagnostics(solution):
    """Collect independent optimization and dense containment checks.

    Args:
        solution (RoutingSolution): RoutingSolution containing resolved spans and solver
            diagnostics.

    Returns:
        result (Array): Four-entry array: convergence, maximum dense wall penetration in
            metres, maximum stationarity residual, and maximum iteration count.
    """
    return jnp.array(
        [
            solution.converged,
            jnp.max(jnp.array([span.dense_wall_violation for span in solution.spans])),
            jnp.max(jnp.array([span.stationarity_residual for span in solution.spans])),
            jnp.max(jnp.array([span.iterations for span in solution.spans])),
        ]
    )


def solve_equilibrium(robot, routing, tensions):
    """Solve independent static bending balances with Optimistix bisection.

    Args:
        robot (SoftRobot): Continuum host providing material strains and guide
            kinematics.
        routing (ShortestPathRouting): ShortestPathRouting instance for the two hollow
            sections.
        tensions (Array): Positive tendon tensions in newtons, shape ``(2,)``.

    Returns:
        result (tuple[Array, RoutingSolution, Array]): Equilibrium angles,
            RoutingSolution, and force-balance residual, with angle and residual arrays
            shaped ``(2,)``.
    """
    stiffness = robot.stiffness_matrix()
    configurations = []
    for section in range(2):

        def residual(angle, args, section=section):
            """Evaluate one section's static force balance.

            Args:
                angle (Array): Bending coordinate of the section being balanced, in
                    radians.
                args (Any): Unused solver arguments required by the Optimistix callback
                    interface.
                section (int): Zero-based section index whose equilibrium is being
                    solved.

            Returns:
                result (Array): Scalar generalized force imbalance for the selected
                    section.
            """
            del args
            configuration = jnp.zeros(2).at[section].set(angle)
            solution = routing.solve_routing(robot, configuration)
            gradient = routing.path_length_jacobian(robot, configuration, solution)
            return (stiffness @ configuration + gradient.T @ tensions)[section]

        root = optx.root_find(
            residual,
            optx.Bisection(rtol=1e-7, atol=1e-8),
            jnp.array(0.0),
            options={"lower": jnp.array(-0.65), "upper": jnp.array(0.65)},
            max_steps=64,
            throw=True,
        )
        configurations.append(root.value)
    configuration = jnp.stack(configurations)
    solution = routing.solve_routing(robot, configuration)
    gradient = routing.path_length_jacobian(robot, configuration, solution)
    residual = stiffness @ configuration + gradient.T @ tensions
    if not bool(solution.converged) or float(jnp.max(jnp.abs(residual))) > 1e-7:
        raise RuntimeError("Static equilibrium did not satisfy the force balance.")
    return configuration, solution, residual


def make_step(robot, routing, case, timestep):
    """Compile RK4 with a fresh routing solve at every force evaluation.

    Args:
        robot (SoftRobot): Continuum host providing material strains and guide
            kinematics.
        routing (ShortestPathRouting): ShortestPathRouting instance for the two hollow
            sections.
        case (str): Anchor layout: ``"same-side"`` or ``"opposed"``.
        timestep (float): Positive integration time step in seconds.

    Returns:
        result (Callable): JIT-compiled callable advancing time, state, and routing
            guess by one RK4 step.
    """
    stiffness = robot.stiffness_matrix()
    damping = robot.damping_matrix(jnp.zeros(2))
    stage_times = jnp.array([0.0, 0.5, 0.5, 1.0])
    stage_weights = jnp.array(
        [[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]]
    )

    @jax.jit
    def step(time_value, state, guess):
        """Advance the dynamic state by one RK4 time step.

        Args:
            time_value (Array): Physical time in seconds.
            state (Array): Two angles, two velocities, accumulated work, and
                dissipation, shape
                ``(6,)``.
            guess (RoutingSolution): Routing solution from the preceding force
                evaluation.

        Returns:
            result (tuple[Array, RoutingSolution, Array]): Updated six-entry state, last
                routing solution, and per-stage diagnostics with shape ``(4, 4)``.
        """

        def stage(carry, index):
            """Evaluate one RK4 stage with a fresh routing optimization.

            Args:
                carry (tuple): Accumulated RK stage derivatives and the latest routing
                    solution.
                index (Array): Zero-based RK stage index.

            Returns:
                result (tuple[tuple[Array, RoutingSolution], Array]): Updated stage
                    derivatives and routing solution, together with this stage's four
                    diagnostics.
            """
            derivatives, previous = carry
            current = state + timestep * (stage_weights[index] @ derivatives)
            configuration, velocity = current[:2], current[2:4]
            solution = routing.solve_routing(
                robot, configuration, initial_guess=previous
            )
            gradient = routing.path_length_jacobian(robot, configuration, solution)
            tension = tensions_at(time_value + stage_times[index] * timestep, case)
            force = -gradient.T @ tension
            inertia, coriolis, gravity = robot.dynamics_terms(
                configuration, velocity, backend="jax"
            )
            acceleration = jnp.linalg.solve(
                inertia,
                force
                - coriolis
                - gravity
                - stiffness @ configuration
                - damping @ velocity,
            )
            derivative = jnp.concatenate(
                (
                    velocity,
                    acceleration,
                    jnp.array([force @ velocity, velocity @ damping @ velocity]),
                )
            )
            return (derivatives.at[index].set(derivative), solution), route_diagnostics(
                solution
            )

        (derivatives, solution), diagnostics = jax.lax.scan(
            stage, (jnp.zeros((4, state.size)), guess), jnp.arange(4)
        )
        next_state = state + timestep * (jnp.array([1, 2, 2, 1]) @ derivatives) / 6
        return next_state, solution, diagnostics

    return step


def simulate(robot, routing, case, duration, timestep, save_dt=0.04):
    """Integrate physical inertia, Coriolis, elasticity, damping, and tendon work.

    Args:
        robot (SoftRobot): Continuum host providing material strains and guide
            kinematics.
        routing (ShortestPathRouting): ShortestPathRouting instance for the two hollow
            sections.
        case (str): Anchor layout: ``"same-side"`` or ``"opposed"``.
        duration (float): Total simulated duration in seconds.
        timestep (float): Positive integration time step in seconds.
        save_dt (float): Interval between saved samples in seconds; must be divisible by
            timestep.

    Returns:
        result (tuple[np.ndarray, np.ndarray, dict]): Sample times, states shaped
            ``(samples, 6)``, and a dictionary of runtime, convergence, clearance, and
            energy checks.
    """
    initial_configuration, guess, _ = solve_equilibrium(
        robot, routing, tensions_at(0.0, case)
    )
    state = jnp.concatenate((initial_configuration, jnp.zeros(4)))
    step = make_step(robot, routing, case, timestep)
    stride = round(save_dt / timestep)
    count = round(duration / timestep)
    if (
        stride < 1
        or not np.isclose(stride * timestep, save_dt)
        or not np.isclose(count * timestep, duration)
    ):
        raise ValueError("timestep must divide both duration and 0.04 s exactly.")
    states, times = [np.asarray(state)], [0.0]
    maximum = np.zeros(3)
    started = time.perf_counter()
    for index in range(count):
        state, guess, diagnostics = step(index * timestep, state, guess)
        checks = np.asarray(diagnostics)
        if not np.isfinite(np.asarray(state)).all() or not np.all(checks[:, 0] == 1):
            raise RuntimeError(
                f"Invalid dynamics or unconverged routing at t={index * timestep:.6f}: {checks}"
            )
        maximum = np.maximum(maximum, checks[:, 1:].max(axis=0))
        if maximum[0] > CLEARANCE_TOLERANCE:
            raise RuntimeError(
                f"Dense wall penetration {maximum[0]} exceeds the clearance tolerance; refine routing."
            )
        if (index + 1) % stride == 0 or index + 1 == count:
            states.append(np.asarray(state))
            times.append((index + 1) * timestep)
        if (index + 1) % 200 == 0:
            print(
                f"{case}: t={(index + 1) * timestep:.2f} s, wall penetration <= {maximum[0] * 1e6:.1f} um",
                flush=True,
            )
    states = np.array(states)

    @jax.jit
    def mechanical_energy(state_value):
        """Evaluate the mechanical energy of a saved state.

        Args:
            state_value (Array): State containing two angles, two velocities, work, and
                dissipation, shape ``(6,)``.

        Returns:
            result (Array): Scalar kinetic plus elastic energy in joules.
        """
        configuration, velocity = state_value[:2], state_value[2:4]
        inertia = robot.dynamics_terms(configuration, velocity, backend="jax")[0]
        return (
            velocity @ inertia @ velocity
            + configuration @ robot.stiffness_matrix() @ configuration
        ) / 2

    energies = np.asarray(jax.vmap(mechanical_energy)(jnp.asarray(states)))
    energy_error = energies - energies[0] - states[:, 4] + states[:, 5]
    return (
        np.array(times),
        states,
        {
            "runtime_seconds": time.perf_counter() - started,
            "max_dense_wall_violation_m": float(maximum[0]),
            "max_stationarity_residual": float(maximum[1]),
            "max_routing_iterations": int(maximum[2]),
            "routing_solves": count * 4 * routing.spans.num_spans,
            "max_energy_balance_error_J": float(np.max(np.abs(energy_error))),
        },
    )


def sample_frames(robot, routing, configurations):
    """Resolve each saved configuration and transform offsets into world space.

    Args:
        robot (SoftRobot): Continuum host providing material strains and guide
            kinematics.
        routing (ShortestPathRouting): ShortestPathRouting instance for the two hollow
            sections.
        configurations (Array): Saved bending configurations, shape ``(frames, 2)``.

    Returns:
        result (tuple[list[tuple], np.ndarray]): List of resolved drawing frames and
            per-frame diagnostics shaped ``(frames, 4)``.
    """
    points = jnp.stack(
        [
            jnp.linspace(section * LENGTH, (section + 1) * LENGTH, 121)
            for section in range(2)
        ]
    )

    @jax.jit
    def sample(configuration):
        """Resolve and transform one configuration for drawing.

        Args:
            configuration (Array): Generalized configuration, shape
                ``(robot.num_coordinates,)``.

        Returns:
            result (tuple[tuple, Array]): Section centerlines, normals, tendon points,
                reference routes, and offsets, plus four routing diagnostics.
        """
        solution = routing.solve_routing(robot, configuration)
        centerlines, normals, tendons, references, offsets = [], [], [], [], []
        for section in range(2):
            poses = robot.forward_kinematics_abscissa_batched(
                configuration, points[section]
            )
            centers = poses[:, 1:]
            normal = jnp.stack((-jnp.sin(poses[:, 0]), jnp.cos(poses[:, 0])), axis=-1)
            local = routing.span_offsets(solution, section, points[section])[:, 1]
            centerlines.append(centers)
            normals.append(normal)
            tendons.append(centers + normal * local[:, None])
            references.append(centers + normal * routing.spans.start_offset[section, 1])
            offsets.append(local)
        return tuple(
            jnp.stack(value)
            for value in (centerlines, normals, tendons, references, offsets)
        ), route_diagnostics(solution)

    frames, diagnostics = [], []
    for configuration in configurations:
        frame, check = sample(jnp.asarray(configuration))
        frames.append(tuple(np.asarray(value) for value in frame))
        diagnostics.append(np.asarray(check))
    diagnostics = np.asarray(diagnostics)
    if (
        not np.all(diagnostics[:, 0] == 1)
        or diagnostics[:, 1].max() > CLEARANCE_TOLERANCE
    ):
        raise RuntimeError(
            "A rendered route failed its convergence or containment check."
        )
    return frames, diagnostics


def draw_robot(axis, frame, title, limits=None):
    """Draw the actual lumen, optimized paths, anchors, and contact samples.

    Args:
        axis (Axes): Matplotlib axes receiving the drawing.
        frame (tuple): Centerlines, normals, tendon points, reference routes, and
            transverse
            offsets for both sections.
        title (str): Title displayed above the drawing.
        limits (tuple | None): Optional x and y plotting bounds in millimetres.

    Returns:
        result (None): ``None``. Artists are added to the supplied axes.
    """
    centers, normals, tendons, references, offsets = frame
    for section in range(2):
        center, normal = centers[section] * 1000, normals[section]
        for side in (-1, 1):
            outer = center + side * OUTER_RADIUS * 1000 * normal
            inner = center + side * INNER_RADIUS * 1000 * normal
            polygon = np.concatenate((outer, inner[::-1]))
            axis.fill(polygon[:, 0], polygon[:, 1], color=WALL_COLOR, alpha=0.3)
            axis.plot(inner[:, 0], inner[:, 1], color=WALL_COLOR, lw=1.0)
        axis.plot(center[:, 0], center[:, 1], ":", color="#8a999c", lw=0.8)
        reference = references[section] * 1000
        tendon = tendons[section] * 1000
        axis.plot(
            reference[:, 0],
            reference[:, 1],
            "--",
            color=COLORS[section],
            alpha=0.5,
            lw=1,
        )
        axis.plot(tendon[:, 0], tendon[:, 1], color=COLORS[section], lw=2.6)
        axis.scatter(
            tendon[[0, -1], 0],
            tendon[[0, -1], 1],
            color=COLORS[section],
            edgecolor="white",
            linewidth=0.7,
            s=45,
            zorder=6,
        )
        contact = np.abs(offsets[section]) >= INNER_RADIUS - 2e-5
        axis.scatter(
            tendon[contact, 0][::6],
            tendon[contact, 1][::6],
            color="#222b36",
            s=8,
            zorder=7,
        )
        for end in (0, -1):
            rim = (
                center[end]
                + np.array([-1, 1])[:, None] * OUTER_RADIUS * 1000 * normal[end]
            )
            axis.plot(rim[:, 0], rim[:, 1], color=WALL_COLOR, lw=1.8)
    axis.plot([-20, 20], [0, 0], color="#34404d", lw=4)
    axis.set(
        title=title,
        xlabel="Horizontal position [mm]",
        ylabel="Height [mm]",
        aspect="equal",
    )
    axis.grid(alpha=0.18)
    if limits is not None:
        axis.set_xlim(*limits[0])
        axis.set_ylim(*limits[1])


def render(case, times, states, frames, output, static_frame):
    """Save a cutaway montage, diagnostics, and a video at physical playback speed.

    Args:
        case (str): Anchor layout: ``"same-side"`` or ``"opposed"``.
        times (Array): Saved physical times in seconds, shape ``(frames,)``.
        states (Array): Saved angles, velocities, work, and dissipation, shape
            ``(frames, 6)``.
        frames (list[tuple]): Resolved drawing data returned by sample_frames.
        output (Path): Output directory for figures and animation.
        static_frame (tuple): Resolved drawing data for the loaded static equilibrium.

    Returns:
        result (None): ``None``. PNG figures and an MP4 animation are written to the
            output directory.
    """
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#fbfcfd",
            "axes.facecolor": "#fbfcfd",
        }
    )
    points = (
        np.concatenate([frame[0].reshape(-1, 2) for frame in [*frames, static_frame]])
        * 1000
    )
    limits = (
        (min(-30, points[:, 0].min() - 24), max(30, points[:, 0].max() + 24)),
        (-15, 260),
    )
    legend = [
        Line2D([0], [0], color=COLORS[0], lw=2, label="Resolved tendon 1"),
        Line2D([0], [0], color=COLORS[1], lw=2, label="Resolved tendon 2"),
        Line2D([0], [0], color="#777", ls="--", label="Fixed-offset reference"),
    ]
    if any(
        np.any(abs(frame[4]) >= INNER_RADIUS - 2e-5)
        for frame in [*frames, static_frame]
    ):
        legend.append(
            Line2D(
                [0], [0], color="#222b36", marker="o", ls="", label="Near wall contact"
            )
        )
    figure, axes = plt.subplots(1, 4, figsize=(12, 6), layout="constrained")
    for axis, desired in zip(axes[:3], [0.0, 1.5, 3.0]):
        index = np.argmin(abs(times - desired))
        draw_robot(axis, frames[index], f"Dynamic state / {times[index]:.2f} s", limits)
    draw_robot(axes[3], static_frame, "Static / 0.2275 N, 0.1575 N", limits)
    figure.suptitle(
        f"Hollow tendon routing: {case} anchors\nTwo 120 mm annular sections; circles mark physical anchors",
        fontsize=15,
    )
    figure.legend(
        handles=legend, loc="outside lower center", ncol=len(legend), frameon=False
    )
    figure.savefig(output / f"{case}_routing.png", dpi=160)
    plt.close(figure)

    figure = plt.figure(figsize=(8, 8), layout="constrained")
    axis = figure.add_subplot(111, projection="3d")
    centers, normals, tendons, _, _ = static_frame
    azimuth = np.linspace(0, np.pi, 45)
    for section in range(2):
        for radius, alpha in ((OUTER_RADIUS, 0.22), (INNER_RADIUS, 0.1)):
            wall = (
                centers[section, :, None]
                + normals[section, :, None] * (radius * np.cos(azimuth))[None, :, None]
            )
            depth = np.broadcast_to(radius * np.sin(azimuth), wall.shape[:2])
            axis.plot_surface(
                wall[:, :, 0] * 1000,
                depth * 1000,
                wall[:, :, 1] * 1000,
                color=WALL_COLOR,
                alpha=alpha,
                linewidth=0,
                shade=True,
            )
        for end in (0, -1):
            for radius in (INNER_RADIUS, OUTER_RADIUS):
                rim = (
                    centers[section, end]
                    + normals[section, end] * (radius * np.cos(azimuth))[:, None]
                )
                axis.plot(
                    rim[:, 0] * 1000,
                    radius * np.sin(azimuth) * 1000,
                    rim[:, 1] * 1000,
                    color=WALL_COLOR,
                    lw=1,
                )
        tendon = tendons[section] * 1000
        axis.plot(
            tendon[:, 0],
            np.zeros(tendon.shape[0]),
            tendon[:, 1],
            color=COLORS[section],
            lw=3,
        )
        axis.scatter(
            tendon[[0, -1], 0],
            [0, 0],
            tendon[[0, -1], 1],
            color=COLORS[section],
            s=45,
            depthshade=False,
        )
    horizontal = centers[:, :, 0] * 1000
    axis.set(
        xlabel="Horizontal [mm]",
        ylabel="",
        zlabel="Height [mm]",
        ylim=(-15, 20),
        zlim=(-5, 250),
        xlim=(horizontal.min() - 20, horizontal.max() + 20),
    )
    axis.set_box_aspect((np.ptp(horizontal) + 40, 35, 255))
    axis.set_yticks([])
    axis.view_init(elev=17, azim=-65)
    axis.set_title(
        f"{case.capitalize()} anchors / static equilibrium\nFront half of the tube removed to show tendon routing",
        pad=20,
    )
    figure.savefig(
        output / f"{case}_cutaway.png", dpi=160, bbox_inches="tight", pad_inches=0.3
    )
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    peak = frames[np.argmax(np.linalg.norm(states[:, :2], axis=1))]
    for section, axis in enumerate(axes):
        local_positions = np.linspace(0, LENGTH * 1000, static_frame[4].shape[-1])
        axis.plot(
            local_positions,
            static_frame[4][section] * 1000,
            color=COLORS[section],
            label="Static equilibrium",
        )
        axis.plot(
            local_positions,
            peak[4][section] * 1000,
            color=COLORS[section],
            ls="--",
            label="Largest dynamic bend",
        )
        axis.scatter(
            local_positions[[0, -1]],
            static_frame[4][section, [0, -1]] * 1000,
            color=COLORS[section],
            s=30,
            zorder=4,
        )
        for side in (-1, 1):
            axis.axhline(side * INNER_RADIUS * 1000, color=WALL_COLOR, lw=1)
        axis.set(
            title=f"Section {section + 1}",
            xlabel="Position within section [mm]",
            ylabel="Tendon offset from backbone [mm]",
            ylim=(-16, 16),
        )
        axis.grid(alpha=0.2)
    figure.legend(
        handles=[
            Line2D([0], [0], color="#5b6368", label="Static equilibrium"),
            Line2D([0], [0], color="#5b6368", ls="--", label="Largest dynamic bend"),
        ],
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        f"{case.capitalize()} anchors / optimized offsets inside the lumen", fontsize=14
    )
    figure.savefig(output / f"{case}_offsets.png", dpi=160)
    plt.close(figure)

    tensions = np.asarray(
        jax.vmap(lambda instant: tensions_at(instant, case))(jnp.asarray(times))
    )
    figure = plt.figure(figsize=(12, 7), layout="constrained")
    grid = figure.add_gridspec(2, 2, width_ratios=[1.1, 1])
    body_axis = figure.add_subplot(grid[:, 0])
    tension_axis = figure.add_subplot(grid[0, 1])
    angle_axis = figure.add_subplot(grid[1, 1])
    for section in range(2):
        tension_axis.plot(
            times,
            tensions[:, section],
            color=COLORS[section],
            label=f"Section {section + 1}",
        )
        angle_axis.plot(times, np.rad2deg(states[:, section]), color=COLORS[section])
    tension_axis.set(
        ylabel="Tendon tension [N]", xlim=(0, times[-1]), title="Applied input"
    )
    angle_axis.set(
        xlabel="Time [s]",
        ylabel="Section bending angle [deg]",
        xlim=(0, times[-1]),
        title="Simulated response",
    )
    tension_axis.legend(frameon=False)
    for axis in (tension_axis, angle_axis):
        axis.grid(alpha=0.2)
    cursors = [
        axis.axvline(0, color="#34404d", alpha=0.6)
        for axis in (tension_axis, angle_axis)
    ]
    figure.suptitle(
        f"{case.capitalize()} tendon actuation / hollow continuum robot", fontsize=15
    )
    figure.legend(handles=legend, loc="outside lower center", ncol=2, frameon=False)
    writer = FFMpegWriter(
        fps=25, codec="libx264", extra_args=["-crf", "20", "-pix_fmt", "yuv420p"]
    )
    with writer.saving(figure, str(output / f"{case}_dynamics.mp4"), dpi=120):
        for index, instant in enumerate(times):
            body_axis.clear()
            draw_robot(
                body_axis,
                frames[index],
                f"Resolved routing / t = {instant:.2f} s",
                limits,
            )
            for cursor in cursors:
                cursor.set_xdata([instant, instant])
            writer.grab_frame()
    figure.savefig(output / f"{case}_response.png", dpi=160)
    plt.close(figure)


def main():
    """Run the hollow-section simulation and rendering example.

    Returns:
        result (None): ``None``. Command-line arguments control simulation, refinement,
            and generated artifacts.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "figures" / "hollow_tendon_routing",
    )
    parser.add_argument(
        "--case", choices=["same-side", "opposed", "all"], default="all"
    )
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--dt", type=float, default=0.004)
    parser.add_argument("--elements", type=int, default=6)
    parser.add_argument("--check-refinement", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.duration <= 0 or args.dt <= 0 or args.elements < 1:
        parser.error("duration, dt, and elements must be positive")
    jax.config.update("jax_enable_x64", True)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "model": "Two bending-only PlanarPCS annular sections; zero gravity; massless frictionless taut tendons",
        "length_m": LENGTH,
        "outer_radius_m": OUTER_RADIUS,
        "inner_radius_m": INNER_RADIUS,
        "anchor_offset_m": ANCHOR_OFFSET,
        "young_modulus_Pa": YOUNG_MODULUS,
        "density_kg_m3": DENSITY,
        "viscosity_Pa_s": VISCOSITY,
        "dt_s": args.dt,
        "routing_elements_per_section": args.elements,
        "cases": {},
    }
    for case in ["same-side", "opposed"] if args.case == "all" else [args.case]:
        print(f"Preparing {case}", flush=True)
        robot, routing = build_model(case, args.elements)
        if args.render_only:
            saved = np.load(args.output / f"{case}_trajectory.npz")
            states = np.concatenate((saved["configuration"], saved["velocity"]), axis=1)
            frames, _ = sample_frames(robot, routing, saved["configuration"])
            equilibrium, _, _ = solve_equilibrium(
                robot, routing, 0.35 * jnp.array([0.65, 0.45])
            )
            static_frames, _ = sample_frames(robot, routing, equilibrium[None])
            render(case, saved["time"], states, frames, args.output, static_frames[0])
            continue
        equilibrium, static_solution, residual = solve_equilibrium(
            robot, routing, 0.35 * jnp.array([0.65, 0.45])
        )
        print(
            f"Static bending angles: {np.rad2deg(np.asarray(equilibrium))} deg; starting dynamics",
            flush=True,
        )
        times, states, checks = simulate(robot, routing, case, args.duration, args.dt)
        frames, frame_checks = sample_frames(robot, routing, states[:, :2])
        static_frames, _ = sample_frames(robot, routing, equilibrium[None])
        checks.update(
            {
                "static_configuration_rad": np.asarray(equilibrium).tolist(),
                "static_residual_Nm": np.asarray(residual).tolist(),
                "peak_abs_bending_angle_deg": np.rad2deg(
                    np.max(np.abs(states[:, :2]), axis=0)
                ).tolist(),
                "render_max_dense_wall_violation_m": float(frame_checks[:, 1].max()),
            }
        )
        if args.check_refinement:
            fine_times, fine_states, _ = simulate(
                robot, routing, case, args.duration, args.dt / 2
            )
            if not np.allclose(times, fine_times):
                raise RuntimeError("Refinement output times differ.")
            checks["dt_halving_max_angle_error_rad"] = float(
                np.max(np.abs(states[:, :2] - fine_states[:, :2]))
            )
            _, fine_routing = build_model(case, args.elements * 2)
            peak_configuration = jnp.asarray(
                states[np.argmax(np.linalg.norm(states[:, :2], axis=1)), :2]
            )
            coarse_route = routing.solve_routing(robot, peak_configuration)
            fine_route = fine_routing.solve_routing(robot, peak_configuration)
            if not bool(coarse_route.converged & fine_route.converged):
                raise RuntimeError("Routing mesh refinement did not converge.")
            checks["mesh_doubling_max_length_error_m"] = float(
                jnp.max(jnp.abs(coarse_route.path_lengths - fine_route.path_lengths))
            )
            coarse_gradient = routing.path_length_jacobian(
                robot, peak_configuration, coarse_route
            )
            fine_gradient = fine_routing.path_length_jacobian(
                robot, peak_configuration, fine_route
            )
            checks["mesh_doubling_max_length_gradient_error_m_per_rad"] = float(
                jnp.max(jnp.abs(coarse_gradient - fine_gradient))
            )
        np.savez_compressed(
            args.output / f"{case}_trajectory.npz",
            time=times,
            configuration=states[:, :2],
            velocity=states[:, 2:4],
            input_work=states[:, 4],
            dissipated_energy=states[:, 5],
            tension=np.asarray(
                jax.vmap(lambda instant, case=case: tensions_at(instant, case))(
                    jnp.asarray(times)
                )
            ),
            tendon_world=np.array([frame[2] for frame in frames]),
            tendon_offsets=np.array([frame[4] for frame in frames]),
            routing_diagnostics=frame_checks,
        )
        render(case, times, states, frames, args.output, static_frames[0])
        report["cases"][case] = checks
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(checks, indent=2), flush=True)
    print(f"Saved outputs to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
