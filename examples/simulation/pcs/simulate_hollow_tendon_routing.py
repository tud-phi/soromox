"""Render and simulate anchored tendons inside two hollow planar sections.

Run with ``JAX_PLATFORMS=cpu MPLBACKEND=Agg python
examples/simulation/pcs/simulate_hollow_tendon_routing.py``.
The output directory contains MP4 animations, PNG figures, NPZ trajectories,
and a JSON report. Select ``--backend matplotlib|open3d|viser|opencv`` or ``all``.
``--render-only`` reuses the complete routing solutions in the NPZ archive;
``--snapshot-only`` skips video export. Viser needs a browser at its printed URL.
All dimensions and mechanics are illustrative.

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
from dataclasses import fields
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from PIL import Image

from soromox.actuation import (
    CircularLumenParams,
    PiecewiseLinearRoutingDiscretization,
    RoutingSolution,
    RoutingSolverOptions,
    RoutingSpanParams,
    RoutingSpanSolution,
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
ARCHIVE_VERSION = 1


class HollowPlanarPCS(PlanarPCS):
    """Expose this example's annular section to the shared renderer adapters."""

    def cross_section_geometry(self, q, s):
        """Return the outer radius and the cavity used by the example's mechanics."""
        tag, outer, _ = super().cross_section_geometry(q, s)
        return tag, outer, jnp.full_like(outer, INNER_RADIUS)


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
    robot = HollowPlanarPCS(
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
        result (tuple): Sample times, states shaped ``(samples, 6)``, a
            RoutingSolution with a leading sample axis on every array, and runtime,
            convergence, clearance, and energy checks. Solutions are evaluated at
            the saved state, not at the last RK4 stage's approximate endpoint.
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
    saved_solutions = [guess]
    resolve_saved = jax.jit(
        lambda configuration, previous: routing.solve_routing(
            robot, configuration, initial_guess=previous
        )
    )
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
        if (index + 1) % stride == 0 or index + 1 == count:
            # RK4's last stage is not the final integrated state. Resolve once
            # here so the archived solution matches the saved configuration.
            guess = resolve_saved(state[:2], guess)
            if not bool(guess.converged):
                raise RuntimeError("Routing failed at a saved configuration.")
            saved_solutions.append(guess)
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
        jax.tree.map(lambda *values: jnp.stack(values), *saved_solutions),
        {
            "runtime_seconds": time.perf_counter() - started,
            "max_dense_wall_violation_m": float(maximum[0]),
            "max_stationarity_residual": float(maximum[1]),
            "max_routing_iterations": int(maximum[2]),
            "routing_solves": (count * 4 + len(saved_solutions) - 1)
            * routing.spans.num_spans,
            "max_energy_balance_error_J": float(np.max(np.abs(energy_error))),
        },
    )


def save_trajectory(path, case, elements, times, states, solutions):
    """Save this two-span example's complete solutions as named, non-pickle arrays.

    Every solution array includes a leading time axis. Keeping solver diagnostics
    and interpolation coefficients allows render-only runs without a solve or a
    template solution. This example has two independent paths and no joint guides.
    """
    values = {
        "archive_version": ARCHIVE_VERSION,
        "case": case,
        "elements": elements,
        "time": times,
        "state": states,
        "path_lengths": solutions.path_lengths,
    }
    for index, span in enumerate(solutions.spans):
        for field in fields(RoutingSpanSolution):
            values[f"span_{index}_{field.name}"] = getattr(span, field.name)
    np.savez_compressed(path, **values)


def load_trajectory(path):
    """Load saved configurations and exact routing solutions without optimization.

    Returns:
        Case, discretization size, times, states, and the batched RoutingSolution.
        Archives from the former custom drawing example must be regenerated.
    """
    with np.load(path, allow_pickle=False) as saved:
        if (
            "archive_version" not in saved
            or int(saved["archive_version"]) != ARCHIVE_VERSION
        ):
            raise ValueError("Regenerate this trajectory to save routing solutions.")
        spans = tuple(
            RoutingSpanSolution(
                **{
                    field.name: jnp.asarray(saved[f"span_{index}_{field.name}"])
                    for field in fields(RoutingSpanSolution)
                }
            )
            for index in range(2)
        )
        solution = RoutingSolution(spans, jnp.asarray(saved["path_lengths"]))
        times, states = np.asarray(saved["time"]), np.asarray(saved["state"])
        if states.shape != (len(times), 6) or solution.path_lengths.shape != (
            len(times),
            2,
        ):
            raise ValueError(
                "Saved states and routing solutions must match the time axis."
            )
        return str(saved["case"]), int(saved["elements"]), times, states, solution


def render_shared(
    robot,
    routing,
    case,
    times,
    states,
    solutions,
    output,
    backend,
    *,
    video=True,
    port=8094,
):
    """Render a peak-bending snapshot and playback using only saved solutions.

    Viser requires a connected browser at the printed local URL. Other backends
    export directly. Snapshot and video share the same camera and tendon geometry.
    """
    from soromox.rendering import (
        ActuatorStyleConfig,
        BackboneColorConfig,
        CameraConfig,
        GeometryConfig,
        MatplotlibRenderer,
        Open3DRenderer,
        OpenCVPlanarRenderer,
        RendererColorConfig,
        RendererConfig,
        RenderOutputConfig,
        SceneConfig,
        ViserRenderer,
    )

    scene = SceneConfig.studio()
    scene.ground.visible = False
    scene.backdrop.enabled = False
    scene.shadows = False
    config = RendererConfig(
        scene=scene,
        camera=CameraConfig(
            position=(0.32, 0.17, 0.48),
            look_at=(0.02, 0.115, 0.0),
            up=(0.0, 1.0, 0.0),
            fov=38.0,
        ),
        geometry=GeometryConfig(
            num_points=65,
            cross_section_resolution=32,
            actuator_line_width=3.0,
            line_width=5.0,
        ),
        output=RenderOutputConfig(width=960, height=720),
        colors=RendererColorConfig(
            backbone=BackboneColorConfig(
                segment_colors=[(0.21, 0.58, 0.65, 0.22), (0.38, 0.70, 0.72, 0.22)]
            ),
            actuators=ActuatorStyleConfig(
                default_color=(0.85, 0.12, 0.06), default_radius=0.0008
            ),
        ),
    )
    if backend == "viser":
        renderer = ViserRenderer(
            robot, config=config, host="127.0.0.1", port=port, open_browser=False
        )
    elif backend == "opencv":
        renderer = OpenCVPlanarRenderer(
            robot, config=config, length_scale=1.3, origin_uv=(480, 660)
        )
    else:
        cls = {"matplotlib": MatplotlibRenderer, "open3d": Open3DRenderer}[backend]
        if cls is None:
            raise ImportError(f"Install the rendering dependencies for {backend}.")
        renderer = cls(robot, config=config)
    try:
        if backend == "viser":
            print(f"Open {renderer.url} for Viser capture", flush=True)
            deadline = time.monotonic() + 300
            while not renderer.server.get_clients():
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "No Viser browser connected within five minutes."
                    )
                time.sleep(0.2)
        peak = int(np.argmax(np.linalg.norm(states[:, :2], axis=1)))
        solution = jax.tree.map(lambda value: value[peak], solutions)
        pixels = renderer.render_frame(
            states[peak, :2], routing=routing, routing_solution=solution
        )
        if backend == "opencv":
            pixels = pixels[..., ::-1]
        Image.fromarray(pixels).save(output / f"{case}_{backend}.png")
        print(f"Saved {backend} snapshot at t={times[peak]:.2f} s", flush=True)
        if video:
            kwargs = {
                "routing": routing,
                "routing_solution": solutions,
                "record_path": str(output / f"{case}_{backend}.mp4"),
            }
            if backend == "matplotlib":
                kwargs["interval"] = round(1000 * float(np.mean(np.diff(times))))
            if backend == "viser":
                kwargs.update(
                    loop=False,
                    autoplay=True,
                    stop_when_recording_done=True,
                    plot_configurations=False,
                    plot_actuator_positions=False,
                )
            renderer.render_sequence(times, states[:, :2], **kwargs)
    finally:
        if backend == "viser":
            renderer.stop()


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
    parser.add_argument(
        "--backend",
        choices=["none", "matplotlib", "open3d", "viser", "opencv", "all"],
        default="matplotlib",
    )
    parser.add_argument("--snapshot-only", action="store_true")
    parser.add_argument("--port", type=int, default=8094)
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
        archive = args.output / f"{case}_trajectory.npz"
        if args.render_only:
            saved_case, saved_elements, times, states, solutions = load_trajectory(
                archive
            )
            if saved_case != case:
                raise ValueError(
                    "Saved anchor layout does not match the requested case."
                )
            robot, routing = build_model(case, saved_elements)
        else:
            robot, routing = build_model(case, args.elements)
            times, states, solutions, checks = simulate(
                robot, routing, case, args.duration, args.dt
            )
            save_trajectory(archive, case, args.elements, times, states, solutions)
            checks["peak_abs_bending_angle_deg"] = np.rad2deg(
                np.max(np.abs(states[:, :2]), axis=0)
            ).tolist()
            checks["render_max_dense_wall_violation_m"] = max(
                float(jnp.max(span.dense_wall_violation)) for span in solutions.spans
            )
            report["cases"][case] = checks
        if args.check_refinement and not args.render_only:
            fine_times, fine_states, _, _ = simulate(
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
        if not args.render_only:
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(json.dumps(checks, indent=2), flush=True)
        backends = (
            ["matplotlib", "opencv", "open3d", "viser"]
            if args.backend == "all"
            else [args.backend]
        )
        for backend in backends:
            if backend != "none":
                render_shared(
                    robot,
                    routing,
                    case,
                    times,
                    states,
                    solutions,
                    args.output,
                    backend,
                    video=not args.snapshot_only,
                    port=args.port,
                )
    print(f"Saved outputs to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
