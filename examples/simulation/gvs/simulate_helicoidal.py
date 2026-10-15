"""Simulate the cylindrical helicoidal example with quadratic rotational strains and JAX.

Run from the checkout with ``JAX_PLATFORMS=cpu python
examples/simulation/gvs/simulate_helicoidal.py --duration 0.02``.
Use ``--duration 0`` to inspect the model without a rollout. Tension is in newtons.
The fixture describes the toolkit's illustrative cylindrical arm. Pass
``--parameter-domain helix`` to prepare its stiffness from strand material and
geometry using ``HelicoidalHelixParams``. Both parameter modes derive module
inertia from strand geometry and density; crossings use additive strand volumes.
This example uses normalized square-root sectional damping and fixed-step Tsit5
at 250 microseconds. The damping is an illustrative choice, not an identified
material law: it preserves the selected law's bending/extension damping and reduces
torsional damping to permit explicit integration.
"""

import argparse
import json
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from diffrax import Tsit5

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
    HelicoidalHelixParams,
    JointSpec,
    LinkSpec,
    StrainBasisSpec,
    SystemState,
    spatial_inertia,
)
from soromox.systems.gvs.construction import project_sectional_matrices

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "assets/robot_parameters/helicoidal/demo_cyl.json"
)


MODULE_BASIS_ORDER = (2, 2, 2, 1, 1, 1)


def _merge_rigid_interfaces(source_links):
    """Merge ring/plate spans and distribute their combined mass in cylinders.

    Each internal cylinder includes the upstream module's platform mass and the
    following transition's distributed mass. This preserves the fixture's total
    mass while replacing its point-mass and thin-ring inertia approximations for
    rigid parts. Each rigid assembly gets its own effective cylinder density.
    """
    links, source_to_link = [], []
    for source in source_links:
        if (
            source["role"] == "transition"
            and links
            and links[-1]["role"] == "connector"
        ):
            length = source["length"]
            mass = (
                length * source["mass_per_length"]
                + links[-1]["length"] * links[-1]["mass_per_length"]
                + links[-2]["tip_mass"]
            )
            links[-2]["tip_mass"] = 0.0
            links[-1]["length"] += length
            links[-1]["mass_per_length"] = mass / links[-1]["length"]
        else:
            links.append(dict(source))
        source_to_link.append(len(links) - 1)
    for link in links:
        if link["role"] != "module":
            # Preserve the known assembly mass with an effective solid-cylinder
            # density. GVS computes its sectional inertia from this geometry.
            link["density"] = link["mass_per_length"] / (np.pi * link["radius"] ** 2)
    return links, source_to_link


def build_helix_params() -> HelicoidalHelixParams:
    """Strand geometry and material of the reference cylindrical demonstration."""
    return HelicoidalHelixParams.circular(
        length=0.2,
        radius_base=0.025,
        radius_tip=0.025,
        num_turns=3.0,
        helices_per_direction=3,
        winding_family_phase_offset=np.pi / 6,
        strand_radius=0.002,
        density=1220.0,
        young_modulus=30e6,
        shear_modulus=30e6 / (2 * (1 + 0.48)),
        fusion_stiffness_multiplier=4.0,
        tip_attachment_stiffness_multiplier=10.0,
        damping_time_constant=0.02,
    )


def build_robot(
    num_gauss_points: int = 16,
    *,
    parameter_domain: Literal["backbone", "helix"] = "backbone",
) -> Helicoidal:
    """Build three modules from backbone or helix inputs with square-root damping."""
    if parameter_domain not in {"backbone", "helix"}:
        raise ValueError("parameter_domain must be 'backbone' or 'helix'.")
    helix = build_helix_params()
    backbone = helix.to_backbone() if parameter_domain == "helix" else None
    data = json.loads(FIXTURE.read_text())
    links, source_to_link = _merge_rigid_interfaces(data["links"])
    segments = []
    for link in links:
        active = link["role"] == "module"
        shear_active = active and link["shear_stiffness"] is not None
        strain_selector = (
            list(backbone.admissible_strains)
            if active and backbone is not None
            else [active] * 4 + [shear_active] * 2
        )
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
                    density=float(helix.density) if active else link["density"],
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
    laws = {}
    for index, link in enumerate(links):
        if link["role"] != "module":
            continue
        # Convert shared helix inputs once, then use the same law for selecting
        # admissible basis strains and preparing the example's damping.
        laws[index] = (
            backbone
            if backbone is not None
            else HelicoidalBackboneParams.from_compliance(
                length=link["length"],
                compliance=jnp.array(
                    [
                        1 / link["torsional_stiffness"],
                        *([link["bending_compliance"][0]] * 2),
                        1 / link["axial_stiffness"],
                        *(
                            [
                                1 / link["shear_stiffness"]
                                if link["shear_stiffness"] is not None
                                else 1.0
                            ]
                            * 2
                        ),
                    ]
                ),
                compliance_gradient=jnp.array(
                    [0.0, *([link["bending_compliance"][1]] * 2), 0.0, 0.0, 0.0]
                ),
                damping_time_constant=data["damping_time_constant"],
                admissible_strains=(True,) * 4
                + (link["shear_stiffness"] is not None,) * 2,
            )
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
        constitutive_params=laws,
        gravity=jnp.array(data["gravity"]),
        base_pose=jnp.array(data["base_pose"]),
        actuators=ThreadlikeActuator.tendons(routing),
        backend="jax",
    )
    robot = robot.with_helix_inertia(dict.fromkeys(laws, helix))
    # The terminal rigid assembly extends beyond the backbone tip. Attach its full
    # cylinder inertia there, with its center halfway along the extension.
    tip = links[-1]
    mass, radius, length = tip["tip_mass"], tip["radius"], tip["tip_body_length"]
    axial = mass * radius**2 / 2
    transverse = mass * (3 * radius**2 + length**2) / 12
    tip_inertia = spatial_inertia(
        mass,
        jnp.array([length / 2, 0.0, 0.0]),
        jnp.diag(jnp.array([axial, transverse, transverse])),
    )
    return robot.update_link_params(
        tip_inertia=robot.params.link.tip_inertia.at[-1].set(tip_inertia),
        damping=_square_root_damping(robot, links, laws),
    )


def _square_root_damping(robot, links, laws):
    """Project ``D_i(s) = tau * sqrt(K_ref,i(s) * K_i(s))`` in the backbone frame.

    The reference is the selected law's bending rigidity for rotational strains and
    axial rigidity for translational strains, which have different units. A
    common ``tau`` in seconds therefore preserves bending and extension damping
    while reducing torsional damping. Apply the square root to sectional
    rigidities before Gauss projection, not to the generalized stiffness matrix.
    This does not impose equal modal damping ratios. Subsequent constitutive-law
    updates replace these prepared damping matrices with that law's damping.
    """

    def sectional(index, s):
        link = links[index]
        if link["role"] != "module":
            return jnp.zeros((len(s), 6, 6))
        law = laws[index]
        u = s[:, None] / link["length"]
        stiffness = 1 / ((1 - u) / law.stiffness_base + u / law.stiffness_tip)
        bending, axial = stiffness[:, 1], stiffness[:, 3]
        reference = jnp.stack((bending, bending, bending, axial, axial, axial), axis=-1)
        return jax.vmap(jnp.diag)(
            law.damping_time_constant * jnp.sqrt(reference * stiffness)
        )

    return project_sectional_matrices(robot.params, robot.structure, sectional)


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
    parser.add_argument("--dt", type=float, default=250e-6, help="Tsit5 step size [s].")
    parser.add_argument(
        "--tension", type=float, default=0.1, help="Distal tendon tension [N]."
    )
    parser.add_argument("--output", type=Path, help="Optional trajectory .npz output.")
    parser.add_argument(
        "--viser", action="store_true", help="Display the helicoidal lattice in Viser."
    )
    parser.add_argument(
        "--parameter-domain",
        choices=("backbone", "helix"),
        default="backbone",
        help="Prepare stiffness from backbone coefficients or helix material/geometry.",
    )
    args = parser.parse_args()
    if args.duration < 0 or args.dt <= 0 or args.tension < 0:
        parser.error("duration and tension must be nonnegative; dt must be positive")
    jax.config.update("jax_enable_x64", True)
    robot = build_robot(parameter_domain=args.parameter_domain)
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
    trajectory = robot.rollout_to(
        SystemState(t=jnp.array(0.0), y=jnp.concatenate((q0, q0)), u=tension),
        t1=args.duration,
        solver_dt=args.dt,
        save_ts=jnp.linspace(
            0.0, args.duration, max(11, int(np.ceil(30 * args.duration)) + 1)
        ),
        solver=Tsit5(),
        max_steps=max(10000, int(np.ceil(args.duration / args.dt)) + 2),
    )
    trajectory.y.block_until_ready()
    print(
        "Final tip [m]:",
        robot.forward_kinematics_tips(trajectory.y[-1, : robot.num_coordinates])[
            -1, :3, 3
        ],
    )
    if args.output:
        np.savez(
            args.output,
            t=trajectory.t,
            y=trajectory.y,
            tension=tension,
            parameter_domain=args.parameter_domain,
            solver="Tsit5",
            dt=args.dt,
        )
        print("Saved:", args.output)
    if args.viser:
        build_renderer(robot).render_sequence(
            trajectory.t,
            trajectory.y[:, : robot.num_coordinates],
        )


if __name__ == "__main__":
    main()
