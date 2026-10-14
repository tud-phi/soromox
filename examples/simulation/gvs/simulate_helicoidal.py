"""Simulate the cylindrical helicoidal example with quadratic rotational strains and JAX.

Run from the checkout with ``JAX_PLATFORMS=cpu python
examples/simulation/gvs/simulate_helicoidal.py --duration 0.02``.
Use ``--duration 0`` to inspect the model without a rollout. Tension is in newtons.
The fixture describes the toolkit's illustrative cylindrical arm.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from diffrax import ImplicitEuler

from soromox.actuation import ThreadlikeActuator, ThreadlikeRouting
from soromox.systems import (
    GVSSegment,
    Helicoidal,
    HelicoidalSectionParams,
    JointSpec,
    LinkSpec,
    StrainBasisSpec,
    SystemState,
)

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "assets/robot_parameters/helicoidal/demo_cyl.json"
)


MODULE_BASIS_ORDER = (2, 2, 2, 1, 1, 1)


def build_robot(num_gauss_points: int = 16) -> Helicoidal:
    """Build six deformable links and two massless, inactive connector links."""
    data = json.loads(FIXTURE.read_text())
    links = data["links"]
    segments = []
    for link in links:
        active = link["role"] != "connector"
        basis_order = MODULE_BASIS_ORDER if link["role"] == "module" else (0,) * 6
        dof = sum(order + 1 for order in basis_order) if active else 0
        segments.append(
            GVSSegment(
                link=LinkSpec.circular(
                    length=link["length"],
                    radius=link["radius"],
                    density=1.0,
                    reference_strain=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    stiffness=jnp.zeros((dof, dof)),
                    damping=jnp.zeros((dof, dof)),
                ),
                joint=JointSpec(type="fixed"),
                basis=StrainBasisSpec(
                    type="legendre",
                    strain_selector=[active] * 6,
                    basis_order=basis_order,
                ),
                num_gauss_points=num_gauss_points if link["role"] == "module" else 8,
            )
        )
    law = HelicoidalSectionParams(
        compliance=jnp.array(
            [
                [
                    1 / link["torsional_stiffness"],
                    *([link["bending_compliance"][0]] * 2),
                    1 / link["axial_stiffness"],
                    *([1 / link["shear_stiffness"]] * 2),
                ]
                for link in links
            ]
        ),
        compliance_gradient=jnp.array(
            [
                [0.0, *([link["bending_compliance"][1]] * 2), 0.0, 0.0, 0.0]
                for link in links
            ]
        ),
        damping_time_constant=jnp.full(len(links), data["damping_time_constant"]),
    )
    routing = ThreadlikeRouting.linear(
        intercept=jnp.array([tendon["intercept"] for tendon in data["tendons"]]),
        slope=jnp.zeros((len(data["tendons"]), 3)),
        start_segment_index=0,
        end_segment_index=tuple(
            tendon["end_segment_index"] for tendon in data["tendons"]
        ),
    )
    return Helicoidal.from_segments(
        segments,
        mass_per_length=jnp.array([link["mass_per_length"] for link in links]),
        rotational_inertia_per_length=jnp.array(
            [link["rotational_inertia_per_length"] for link in links]
        ),
        tip_mass=jnp.array([link["tip_mass"] for link in links]),
        sectional_law=law,
        gravity=jnp.array(data["gravity"]),
        base_pose=jnp.array(data["base_pose"]),
        actuators=ThreadlikeActuator.tendons(routing),
        backend="jax",
    )


def build_renderer(robot: Helicoidal, **kwargs):
    """Build the cylindrical example's Viser lattice with three helical modules."""
    from soromox.rendering import (
        CameraConfig,
        HelicoidalModuleVisual,
        HelicoidalViserRenderer,
        RendererConfig,
        SceneConfig,
    )

    config = kwargs.pop(
        "config",
        RendererConfig(
            camera=CameraConfig(
                position=(0.035, -0.62, 0.31),
                look_at=(0.035, 0.0, 0.31),
                up=(0.0, 0.0, -1.0),
            ),
            scene=SceneConfig.studio(),
        ),
    )
    return HelicoidalViserRenderer(
        robot,
        modules=tuple(HelicoidalModuleVisual(index) for index in (1, 4, 7)),
        config=config,
        **kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=0.02)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument(
        "--tension", type=float, default=0.1, help="Distal tendon tension [N]."
    )
    parser.add_argument("--output", type=Path, help="Optional trajectory .npz output.")
    parser.add_argument(
        "--viser", action="store_true", help="Display the helicoidal lattice in Viser."
    )
    args = parser.parse_args()
    if args.duration < 0 or args.dt <= 0 or args.tension < 0:
        parser.error("duration and tension must be nonnegative; dt must be positive")
    jax.config.update("jax_enable_x64", True)
    robot = build_robot()
    q0 = jnp.zeros(robot.num_coordinates)
    tension = jnp.zeros(robot.num_actuators).at[6].set(args.tension)
    print(
        f"{robot.num_segments} links, {robot.num_coordinates} coordinates, {robot.num_actuators} tendons"
    )
    print("Undeformed tip [m]:", robot.forward_kinematics_tips(q0)[-1, :3, 3])
    print("Contiguous tendon lengths [m]:", robot.actuators[0].path_lengths(robot, q0))
    if args.duration == 0:
        if args.viser:
            build_renderer(robot).show(q0)
        return
    # Large effective shear stiffness calls for an implicit integrator.
    trajectory = robot.rollout_to(
        SystemState(t=jnp.array(0.0), y=jnp.concatenate((q0, q0)), u=tension),
        t1=args.duration,
        solver_dt=args.dt,
        save_ts=jnp.linspace(
            0.0, args.duration, max(11, int(np.ceil(30 * args.duration)) + 1)
        ),
        solver=ImplicitEuler(root_finder=optx.Newton(rtol=1e-7, atol=1e-9)),
        max_steps=10000,
    )
    trajectory.y.block_until_ready()
    print(
        "Final tip [m]:",
        robot.forward_kinematics_tips(trajectory.y[-1, : robot.num_coordinates])[
            -1, :3, 3
        ],
    )
    if args.output:
        np.savez(args.output, t=trajectory.t, y=trajectory.y, tension=tension)
        print("Saved:", args.output)
    if args.viser:
        build_renderer(robot).render_sequence(
            trajectory.t,
            trajectory.y[:, : robot.num_coordinates],
        )


if __name__ == "__main__":
    main()
