"""Validate saved GVS terms and condition-aware FP64 forward-solve accuracy."""

from __future__ import annotations

import argparse
import json

import numpy as np

from tools.benchmarks.native_gvs import robot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    with open(args.baseline) as stream:
        baseline = {(c["case"], c["worlds"]): c for c in json.load(stream)["cases"]}
    with open(args.candidate) as stream:
        candidate = json.load(stream)
    models = {}
    for c in candidate["cases"]:
        case, worlds = c["case"], c["worlds"]
        if case not in models:
            r = robot(
                8 if case == "large8" else 16 if case == "large16" else 2,
                2 if case.startswith("large") else 1,
                case.startswith("large"),
                case == "moving",
            )
            models[case] = np.asarray(r.K), np.asarray(r.D_active)
        stiffness, damping = models[case]
        reference = np.load(baseline[case, worlds]["outputs"])
        actual = np.load(c["outputs"])
        for name in ("M", "C", "G"):
            np.testing.assert_array_equal(reference[name], actual[name])
        n = c["dofs"]
        rhs = (
            -reference["C"]
            - reference["G"]
            - stiffness @ np.linspace(-0.002, 0.002, n)
            - damping @ np.linspace(-0.003, 0.003, n)
        )
        condition = float(np.linalg.cond(reference["M"][0]))
        scale = np.max(abs(reference["a"]))
        error = float(np.max(abs(reference["a"] - actual["a"])))
        assert error <= 10 * condition * np.finfo(np.float64).eps * scale
        residuals = []
        for result in (reference, actual):
            residual = np.einsum("wij,wj->wi", result["M"], result["a"]) - rhs
            denominator = np.linalg.norm(
                result["M"], ord=np.inf, axis=(1, 2)
            ) * np.linalg.norm(result["a"], ord=np.inf, axis=1) + np.linalg.norm(
                rhs, ord=np.inf, axis=1
            )
            backward = float(
                np.max(np.linalg.norm(residual, ord=np.inf, axis=1) / denominator)
            )
            assert backward < 2e-14
            residuals.append(backward)
        c["validation"] = {
            "terms_bitwise_equal": True,
            "max_acceleration_abs_error": error,
            "relative_acceleration_inf_error": error / scale,
            "inertia_condition_number": condition,
            "normalized_solve_residuals": residuals,
        }
        print(case, worlds, c["validation"], flush=True)
    with open(args.candidate, "w") as stream:
        json.dump(candidate, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
