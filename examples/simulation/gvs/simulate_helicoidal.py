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
from soromox.rendering import (
    CameraConfig,
    HelicoidalModuleVisual,
    HelicoidalViserRenderer,
    RendererConfig,
    SceneConfig,
)
from soromox.systems import (
    GVSSegment,
    Helicoidal,
    HelicoidalBackboneParams,
    JointSpec,
    LinkSpec,
    StrainBasisSpec,
    SystemState,
    spatial_inertia,
)

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "assets/robot_parameters/helicoidal/demo_cyl.json"
)


MODULE_BASIS_ORDER = (2, 2, 2, 1, 1, 1)


def _merge_rigid_interfaces(source_links):
    """Merge each massless connector gap with its following rigid end ring.

    The ring's distributed inertia is integrated into an attached-body matrix
    at the merged link tip, preserving its mass, center of mass, and rotational
    inertia. The fixture's module-tip masses retain their attachment locations.
    """
    links, source_to_link, attached_inertia = [], [], []
    for source in source_links:
        if (
            source["role"] == "transition"
            and links
            and links[-1]["role"] == "connector"
        ):
            length = source["length"]
            mass = length * source["mass_per_length"]
            inertia_at_center = jnp.diag(
                length * jnp.array(source["rotational_inertia_per_length"])
                + mass * length**2 / 12 * jnp.array([0.0, 1.0, 1.0])
            )
            attached_inertia[-1] = spatial_inertia(
                mass, jnp.array([-length / 2, 0.0, 0.0]), inertia_at_center
            )
            links[-1]["length"] += length
        else:
            links.append(dict(source))
            attached_inertia.append(jnp.zeros((6, 6)))
        source_to_link.append(len(links) - 1)
    return links, source_to_link, jnp.stack(attached_inertia)


def build_robot(num_gauss_points: int = 16) -> Helicoidal:
    """Build three deformable modules with rigid base and inter-module rings."""
    data = json.loads(FIXTURE.read_text())
    links, source_to_link, attached_inertia = _merge_rigid_interfaces(data["links"])
    segments = []
    for link in links:
        active = link["role"] == "module"
        shear_active = active and link["shear_stiffness"] is not None
        strain_selector = [active] * 4 + [shear_active] * 2
        basis_order = MODULE_BASIS_ORDER if link["role"] == "module" else (0,) * 6
        dof = sum(
            order + 1
            for order, selected in zip(basis_order, strain_selector, strict=True)
            if selected
        )
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
                    strain_selector=strain_selector,
                    basis_order=basis_order,
                ),
                num_gauss_points=num_gauss_points if link["role"] == "module" else 8,
            )
        )
    law = HelicoidalBackboneParams.from_compliance(
        lengths=jnp.array([link["length"] for link in links]),
        compliance=jnp.array(
            [
                [
                    1 / link["torsional_stiffness"],
                    *([link["bending_compliance"][0]] * 2),
                    1 / link["axial_stiffness"],
                    # Inactive shear rows have no basis columns. Their finite
                    # padding is ignored by the sectional projection.
                    *(
                        [
                            1 / link["shear_stiffness"]
                            if link["shear_stiffness"] is not None
                            else 1.0
                        ]
                        * 2
                    ),
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
            source_to_link[tendon["end_segment_index"]] for tendon in data["tendons"]
        ),
    )
    robot = Helicoidal.from_segments(
        segments,
        mass_per_length=jnp.array([link["mass_per_length"] for link in links]),
        rotational_inertia_per_length=jnp.array(
            [link["rotational_inertia_per_length"] for link in links]
        ),
        tip_mass=jnp.array([link["tip_mass"] for link in links]),
        constitutive_law=law,
        gravity=jnp.array(data["gravity"]),
        base_pose=jnp.array(data["base_pose"]),
        actuators=ThreadlikeActuator.tendons(routing),
        backend="jax",
    )
    return robot.update_link_params(
        tip_inertia=robot.params.link.tip_inertia + attached_inertia
    )


def build_renderer(robot: Helicoidal, **kwargs):
    """Build the cylindrical example's Viser lattice with three helical modules."""
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
        modules=tuple(HelicoidalModuleVisual(index) for index in (1, 3, 5)),
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
    # The remaining stiff, damped modes still call for an implicit integrator.
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
