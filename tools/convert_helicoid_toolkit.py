"""Convert the pinned toolkit's cylindrical demo into a native analytic fixture.

Run with a checkout of https://github.com/CharlesQueen/helicoid-toolkit.
The conversion follows ``rod/equiv_rod.py:load_calibration`` (Apache-2.0).
No upstream Python code is imported; conversion requires only NumPy.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

UPSTREAM_COMMIT = "76c659d7900fb1a5ed01830a4fc1c6c6d473aa59"


def convert(checkout: Path) -> dict:
    """Read the pinned cylindrical demo and recover its analytic law in SI units."""
    commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"Expected upstream commit {UPSTREAM_COMMIT}, got {commit}.")
    folder = checkout / "deliver/demo_cyl"
    card = json.loads((folder / "helicoid_arm_spec.json").read_text())
    if (
        card["schema"] != "helicoid_arm_spec/1.0"
        or card["design"]["name"] != "demo_cyl"
    ):
        raise ValueError("Only the pinned cylindrical demo schema is supported.")
    lattice = json.loads((folder / "generated_lattice_model.json").read_text())
    sections = card["sections"]
    if len(sections) != 6:
        raise ValueError("Expected three modules and three compliant transitions.")
    links, diagnostics, source_to_native = [], [], {}
    with np.load(folder / "module_compliance.npz", allow_pickle=False) as archive:
        for index, (name, module) in enumerate(lattice["modules"].items()):
            length = (module["z_lattice"][1] - module["z_lattice"][0]) * 1e-3
            compliance = archive[f"C_{name}"]
            if compliance.shape != (6, 6) or not np.isfinite(compliance).all():
                raise ValueError("Invalid module interface compliance.")
            moments = [
                (compliance[0, 0] + compliance[1, 1]) / 2,
                (abs(compliance[0, 4]) + abs(compliance[1, 3])) / 2,
            ]
            c_a, c_b = np.linalg.solve(
                [[length, length**2 / 2], [length**2 / 2, length**3 / 3]], moments
            )
            residual = (compliance[3, 3] + compliance[4, 4]) / 2 - (
                c_a * length**3 / 3 + c_b * length**4 / 4
            )
            gj, ea = length / compliance[2, 2], length / compliance[5, 5]
            ga = length / residual if residual > 1e-9 else 1e6
            radius = module["taper"]["r_start"] * 1e-3
            if module["taper"]["r_end"] * 1e-3 != radius:
                raise ValueError("This converter requires cylindrical modules.")
            for offset, role in enumerate(("transition", "module")):
                source_index = 2 * index + offset
                section = sections[source_index]
                stations = section["stations"]
                inertia = np.array([s["rot_inertia_per_length"] for s in stations])
                line_mass = np.array([s["mass_per_length"] for s in stations])
                if not (
                    np.allclose(inertia, inertia[0], rtol=0, atol=1e-15)
                    and np.allclose(line_mass, line_mass[0], rtol=0, atol=1e-15)
                ):
                    raise ValueError("Expected constant sectional inertia.")
                section_law = (
                    ([c_a / 50, 0.0], 50 * gj, 50 * ea, 50 * ea)
                    if offset == 0
                    else ([c_a, c_b], gj, ea, ga)
                )
                coefficients, torsion, axial, shear = section_law
                # Verify the recovered law against the export, without fitting its samples.
                for station in stations:
                    np.testing.assert_allclose(
                        [
                            1
                            / (
                                coefficients[0]
                                + coefficients[1] * (section["length"] - station["s"])
                            ),
                            torsion,
                            axial,
                            shear,
                        ],
                        [station["EI"], station["GJ"], station["EA"], station["GA"]],
                        rtol=1e-11,
                    )
                source_to_native[source_index] = len(links)
                links.append(
                    {
                        "role": role,
                        "source_section": section["name"],
                        "length": section["length"],
                        "radius": radius,
                        "bending_compliance": list(coefficients),
                        "torsional_stiffness": torsion,
                        "axial_stiffness": axial,
                        "shear_stiffness": shear,
                        "mass_per_length": line_mass[0],
                        "rotational_inertia_per_length": inertia[0, [2, 0, 1]].tolist(),
                        "tip_mass": 0.0,
                    }
                )
            diagnostics.append(
                {
                    "source_module": name,
                    "interface_compliance": compliance.tolist(),
                    "shear_compliance_residual": residual,
                    "shear_fallback_used": bool(residual <= 1e-9),
                }
            )
            if index < 2:
                transform = np.array(card["connectors"][2 * index + 1]["transform"])
                gap = transform[2, 3]
                expected = np.eye(4)
                expected[2, 3] = gap
                if gap <= 0 or not np.allclose(transform, expected, rtol=0, atol=1e-15):
                    raise ValueError("Expected a positive axial connector gap.")
                links.append(
                    {
                        "role": "connector",
                        "length": gap,
                        "radius": radius,
                        "bending_compliance": [1.0, 0.0],
                        "torsional_stiffness": 1.0,
                        "axial_stiffness": 1.0,
                        "shear_stiffness": 1.0,
                        "mass_per_length": 0.0,
                        "rotational_inertia_per_length": [0.0, 0.0, 0.0],
                        "tip_mass": 0.0,
                    }
                )
    for body in card["rigid_bodies"]:
        links[source_to_native[body["at_tip_of_section"]]]["tip_mass"] += body["mass"]
    tendons = []
    for tendon in card["tendons"]:
        first = tendon["parts"][0]
        for part in tendon["parts"]:
            if not (
                np.isclose(part["offset_radius_start"], first["offset_radius_start"])
                and np.isclose(part["offset_radius_end"], first["offset_radius_start"])
                and np.isclose(part["azimuth"], first["azimuth"])
            ):
                raise ValueError(
                    "Expected constant offsets along each cylindrical route."
                )
        angle, radius = first["azimuth"], first["offset_radius_start"]
        tendons.append(
            {
                "intercept": [0.0, radius * np.cos(angle), radius * np.sin(angle)],
                "start_segment_index": 0,
                "end_segment_index": source_to_native[
                    tendon["anchored_at_tip_of_section"]
                ],
            }
        )
    if not np.array_equal(card["base_transform"], np.eye(4)):
        raise ValueError("Expected identity source mounting.")
    return {
        "schema": "soromox.helicoidal/1",
        "source": {
            "repository": "https://github.com/CharlesQueen/helicoid-toolkit",
            "commit": commit,
            "dataset": "deliver/demo_cyl",
            "license": "Apache-2.0",
            "paper": "https://arxiv.org/abs/2609.25264",
        },
        # Q maps source-local (z tangent) vectors to native-local (x tangent).
        # Mount with Q.T to retain source world positions and gravity.
        "base_pose": [-0.5, -0.5, -0.5, 0.5, 0.0, 0.0, 0.0],
        "gravity": card["gravity"],
        "damping_time_constant": card["dynamics"]["kv_time_constant"],
        "links": links,
        "tendons": tendons,
        "compliance_diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = convert(args.checkout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
