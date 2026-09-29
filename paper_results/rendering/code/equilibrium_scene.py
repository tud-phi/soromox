"""Reproduce a tendon-actuated equilibrium of the Section Va soft tentacle."""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import eigvalsh
from scipy.optimize import root
from scipy.spatial.transform import Rotation

from soromox.actuation import ThreadlikeActuator, ThreadlikeRouting
from soromox.systems import (
    GVS,
    GVSSegment,
    JointSpec,
    LinearProfile,
    LinkSpec,
    StrainBasisSpec,
)

jax.config.update("jax_enable_x64", True)
CASE = Path(__file__).resolve().parents[1]
DATA = CASE.parent / "secVa_system_identification" / "data"
SOLUTION = CASE / "data" / "equilibrium.json"


def make_robot():
    """Use the fitted material, basis, tendon routing, and hanging mounting."""
    E, nu, rho = [float(np.load(DATA / f"{n}_hat.npy")) for n in ("E", "nu", "rho")]
    links = [
        LinkSpec.circular(
            length=length,
            radius=LinearProfile(base=base, tip=tip),
            density=rho,
            young_modulus=E,
            shear_modulus=E / (2 * (1 + nu)),
            material_damping_coefficient=1e4,
            reference_strain=[0, 0, 0, 1, 0, 0],
        )
        for length, base, tip in [(0.305, 0.01541, 0.00642), (0.055, 0.00642, 0.00480)]
    ]
    bases = [
        StrainBasisSpec(
            type="monomial",
            strain_selector=[1, 1, 1, 1, 0, 0],
            basis_order=[0, 1, 1, 1, 0, 0],
        ),
        StrainBasisSpec(
            type="monomial",
            strain_selector=[0, 1, 1, 0, 0, 0],
            basis_order=[0, 0, 0, 0, 0, 0],
        ),
    ]
    angles = jnp.deg2rad(jnp.array([30.0, 150.0]))
    directions = jnp.stack(
        (jnp.zeros_like(angles), jnp.cos(angles), jnp.sin(angles)), axis=-1
    )
    routing = ThreadlikeRouting.linear(
        intercept=0.0114 * directions,
        slope=-0.0295 * directions,
        start_segment_index=0,
        end_segment_index=(0, 0),
    )
    # Rotate the experimental hanging mount around gravity so tendon 1 bends
    # in the world x-z plane. This yaw does not change the physical equilibrium.
    rotation = Rotation.from_euler("z", -60, degrees=True) * Rotation.from_euler(
        "y", 90, degrees=True
    )
    return GVS.from_segments(
        [
            GVSSegment(
                link=link,
                joint=JointSpec(type="fixed"),
                basis=basis,
                num_gauss_points=8,
            )
            for link, basis in zip(links, bases, strict=True)
        ],
        gravity=jnp.array([0.0, 0.0, -9.81]),
        base_pose=jnp.asarray(np.r_[rotation.as_quat(), [0.0, 0.0, 0.0]]),
        actuators=ThreadlikeActuator.tendons(routing),
        scale_rotational_basis_by_length=True,
    )


def load_scene():
    """Load the verified equilibrium without repeating the solve for captures."""
    data = json.loads(SOLUTION.read_text())
    for name, expected in data["parameter_files_sha256"].items():
        if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Fitted parameters changed; rerun equilibrium_scene.py")
    return make_robot(), np.asarray(data["q"]), data


def solve():
    """Follow stable equilibria from zero tension through experimental inputs."""
    robot = make_robot()
    K = robot.stiffness_matrix()

    @jax.jit
    def residual(q, u):
        return (
            robot.elastic_force(q)
            + robot.gravitational_force(q)
            - robot.actuation_matrix(q) @ u
        )

    jacobian = jax.jit(jax.jacfwd(residual, argnums=0))
    scale = np.sqrt(np.diag(np.asarray(K)))
    q = np.zeros(int(np.sum(robot.dofs_per_segment)))
    candidates = []
    # Continuation from the hanging rest state to the paper's largest tension.
    for tension in np.linspace(0, 0.25 / 0.0325, 21):
        u = np.array([tension, 0.0])
        result = root(
            lambda v, u=u: np.asarray(residual(v, u)) / scale,
            q,
            jac=lambda v, u=u: np.asarray(jacobian(v, u)) / scale[:, None],
            tol=1e-10,
        )
        q = result.x
        err = np.asarray(residual(q, u))
        if np.linalg.norm(err) > 1e-8:
            raise RuntimeError(
                f"Equilibrium did not converge: {result.message}; residual {err}"
            )
        if any(np.isclose(tension, value / 0.0325) for value in [0.1, 0.15, 0.2, 0.25]):
            tangent = np.asarray(jacobian(q, u))
            mass = np.asarray(robot.inertia_matrix(q))
            damping = np.asarray(robot.damping_matrix(q))
            frequencies_squared = eigvalsh((tangent + tangent.T) / 2, mass)
            dynamics = np.block(
                [
                    [np.zeros_like(mass), np.eye(len(q))],
                    [-np.linalg.solve(mass, tangent), -np.linalg.solve(mass, damping)],
                ]
            )
            max_real = float(np.max(np.linalg.eigvals(dynamics).real))
            curve = np.asarray(
                robot.forward_kinematics_abscissa_batched(
                    q, jnp.linspace(0, robot.length, 200)
                )
            )[:, :3, 3]
            sample = {
                "tension_N": u.tolist(),
                "q": q.tolist(),
                "residual_l2": float(np.linalg.norm(err)),
                "min_squared_natural_frequency": float(frequencies_squared.min()),
                "max_linearized_eigenvalue_real": max_real,
                "max_out_of_plane_m": float(np.max(np.abs(curve[:, 1]))),
                "curve_m": curve.tolist(),
            }
            if frequencies_squared.min() <= 0 or max_real >= 0:
                raise RuntimeError(f"Candidate is not locally stable: {sample}")
            candidates.append(sample)
            print(
                {k: v for k, v in sample.items() if k not in ("curve_m", "q")},
                flush=True,
            )
    # A clearly bent shape at a tension used in the identification experiment.
    selected = candidates[2]
    metadata = {k: v for k, v in selected.items() if k != "curve_m"}
    metadata.update(
        {
            "model": "Section Va fitted two-segment GVS soft tentacle",
            "mounting": "hanging; yaw -60 degrees about gravity",
            "equation": "elastic_force(q) + gravitational_force(q) - actuation_matrix(q) @ tension = 0",
            "tendon_termination": "both tendons end at segment 1; segment 2 is passive",
            "parameter_files_sha256": {
                f"{n}_hat.npy": hashlib.sha256(
                    (DATA / f"{n}_hat.npy").read_bytes()
                ).hexdigest()
                for n in ("E", "nu", "rho")
            },
            "solver": "SciPy hybr with JAX Jacobian; 21 continuation steps from zero tension",
            "material": {
                n: float(np.load(DATA / f"{n}_hat.npy")) for n in ("E", "nu", "rho")
            },
        }
    )
    SOLUTION.parent.mkdir(parents=True, exist_ok=True)
    SOLUTION.write_text(json.dumps(metadata, indent=2) + "\n")
    (SOLUTION.parent / "equilibrium_candidates.json").write_text(
        json.dumps(
            [{k: v for k, v in c.items() if k != "curve_m"} for c in candidates],
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    solve()
