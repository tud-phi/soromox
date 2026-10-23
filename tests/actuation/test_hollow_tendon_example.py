"""Mechanics and integration checks for the runnable hollow-tendon example."""

import runpy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def example():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/simulation/pcs/simulate_hollow_tendon_routing.py"
    )
    return runpy.run_path(str(path))


def test_annular_stiffness_mass_and_anchor_topology(example):
    robot, routing = example["build_model"]("opposed")
    outer, inner, length = (
        example[name] for name in ("OUTER_RADIUS", "INNER_RADIUS", "LENGTH")
    )
    area = np.pi * (outer**2 - inner**2)
    moment = np.pi / 4 * (outer**4 - inner**4)
    expected_inertia = example["DENSITY"] * np.diag([moment, area, area])
    assert_allclose(
        robot.params.link.sectional_inertia, np.tile(expected_inertia, (2, 1, 1))
    )
    assert_allclose(
        robot.stiffness_matrix(), np.eye(2) * example["YOUNG_MODULUS"] * moment / length
    )
    assert routing.spans.start_segment_index == (0, 1)
    assert routing.spans.end_segment_index == (0, 1)
    assert_allclose(
        routing.spans.start_offset[:, 1],
        [example["ANCHOR_OFFSET"], -example["ANCHOR_OFFSET"]],
    )
    assert_allclose(routing.spans.start_offset, routing.spans.end_offset)


@pytest.mark.parametrize("case", ["same-side", "opposed"])
def test_control_keeps_tendons_taut(example, case):
    tensions = jax.vmap(lambda instant: example["tensions_at"](instant, case))(
        jnp.linspace(0, 6, 101)
    )
    assert bool(jnp.all(tensions > 0))
    assert float(jnp.max(tensions)) <= 0.2275 + 1e-12


def test_rk4_step_energy_balance_and_refinement(example):
    robot, routing = example["build_model"]()
    state = jnp.array([0.2, 0.1, 0.03, -0.02, 0.0, 0.0])
    solution = routing.solve_routing(robot, state[:2])
    coarse_step = example["make_step"](robot, routing, "same-side", 0.004)
    fine_step = example["make_step"](robot, routing, "same-side", 0.002)
    coarse, _, diagnostics = coarse_step(0.9, state, solution)
    half, warm, first_checks = fine_step(0.9, state, solution)
    fine, _, second_checks = fine_step(0.902, half, warm)
    assert bool(jnp.all(diagnostics[:, 0] == 1))
    assert bool(jnp.all(first_checks[:, 0] == 1) & jnp.all(second_checks[:, 0] == 1))
    assert_allclose(coarse[:2], fine[:2], atol=1e-8, rtol=1e-6)
    assert_allclose(coarse[2:4], fine[2:4], atol=2e-6, rtol=1e-5)
    assert_allclose(coarse[4:], fine[4:], atol=1e-9, rtol=1e-5)

    def energy(value):
        inertia = robot.dynamics_terms(value[:2], value[2:4], backend="jax")[0]
        return (
            value[2:4] @ inertia @ value[2:4]
            + value[:2] @ robot.stiffness_matrix() @ value[:2]
        ) / 2

    assert_allclose(energy(coarse) - energy(state), coarse[4] - coarse[5], atol=1e-9)
