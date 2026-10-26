"""Saved routing solutions must match output states and bypass replay optimization."""

from unittest.mock import Mock

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from examples.simulation.pcs import simulate_hollow_tendon_routing as example
from soromox.actuation import ShortestPathRouting

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def trajectory():
    robot, routing = example.build_model("opposed", elements=2)
    configurations = jnp.array([[0.1, -0.2], [0.3, -0.1]])
    solutions = eqx.filter_jit(jax.vmap(lambda q: routing.solve_routing(robot, q)))(
        configurations
    )
    states = np.pad(np.asarray(configurations), ((0, 0), (0, 4)))
    return robot, routing, np.array([0.0, 0.04]), states, solutions


def test_archive_round_trip_preserves_all_solution_arrays(trajectory, tmp_path):
    robot, _, times, states, solutions = trajectory
    archive = tmp_path / "opposed_trajectory.npz"
    example.save_trajectory(archive, "opposed", 2, times, states, solutions)
    case, elements, loaded_times, loaded_states, loaded = example.load_trajectory(
        archive
    )
    assert case == "opposed" and elements == 2
    np.testing.assert_array_equal(loaded_times, times)
    np.testing.assert_array_equal(loaded_states, states)
    for expected, actual in zip(
        jax.tree.leaves(solutions), jax.tree.leaves(loaded), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    _, outer, inner = robot.cross_section_geometry(jnp.asarray(states[0, :2]), 0.05)
    assert_allclose(outer, [example.OUTER_RADIUS])
    assert_allclose(inner, [example.INNER_RADIUS])


def test_render_only_uses_saved_discretization_without_solving(
    trajectory, tmp_path, monkeypatch
):
    _, _, times, states, solutions = trajectory
    example.save_trajectory(
        tmp_path / "opposed_trajectory.npz", "opposed", 2, times, states, solutions
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("render-only must not solve routing or equilibrium")

    monkeypatch.setattr(ShortestPathRouting, "solve_routing", forbidden)
    monkeypatch.setattr(example, "solve_equilibrium", forbidden)
    render = Mock()
    monkeypatch.setattr(example, "render_shared", render)
    monkeypatch.setattr(
        "sys.argv",
        [
            "example",
            "--render-only",
            "--case",
            "opposed",
            "--backend",
            "all",
            "--output",
            str(tmp_path),
        ],
    )
    example.main()
    assert [call.args[7] for call in render.call_args_list] == [
        "matplotlib",
        "opencv",
        "open3d",
        "viser",
    ]
    for call in render.call_args_list:
        assert call.args[1].discretization.elements_per_segment == 2
        assert_allclose(call.args[4], states)
        for expected, actual in zip(
            jax.tree.leaves(solutions), jax.tree.leaves(call.args[5]), strict=True
        ):
            np.testing.assert_array_equal(actual, expected)


def test_saved_routes_match_integrated_state_not_last_rk_stage(trajectory, monkeypatch):
    robot, routing, _, states, solutions = trajectory
    initial = jax.tree.map(lambda x: x[0], solutions)
    monkeypatch.setattr(
        example,
        "solve_equilibrium",
        lambda *args: (jnp.asarray(states[0, :2]), initial, jnp.zeros(2)),
    )

    # Mimic a final RK stage whose route differs from the integrated state.
    def make_step(*args):
        def step(t, state, guess):
            return (
                state.at[:2].add(jnp.array([0.01, -0.02])),
                initial,
                jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (4, 1)),
            )

        return step

    monkeypatch.setattr(example, "make_step", make_step)
    times, output, saved, _ = example.simulate(
        robot, routing, "opposed", duration=0.08, timestep=0.02
    )
    assert len(times) == 3
    reference = eqx.filter_jit(jax.vmap(lambda q: routing.solve_routing(robot, q)))(
        jnp.asarray(output[:, :2])
    )
    assert_allclose(saved.path_lengths, reference.path_lengths, atol=1e-9)
    assert not np.allclose(
        saved.path_lengths[-1], initial.path_lengths, atol=1e-7, rtol=0
    )


def test_old_archives_require_regeneration(tmp_path):
    path = tmp_path / "old.npz"
    np.savez(path, time=[0.0], configuration=[[0.0, 0.0]])
    with pytest.raises(ValueError, match="Regenerate"):
        example.load_trajectory(path)


def test_shared_renderer_receives_saved_solutions_and_physical_frame_rate(
    trajectory, tmp_path, monkeypatch
):
    from soromox import rendering

    robot, routing, times, states, solutions = trajectory
    renderer = Mock()
    renderer.render_frame.return_value = np.zeros((8, 8, 3), dtype=np.uint8)
    monkeypatch.setattr(rendering, "MatplotlibRenderer", Mock(return_value=renderer))
    example.render_shared(
        robot, routing, "opposed", times, states, solutions, tmp_path, "matplotlib"
    )
    frame = renderer.render_frame.call_args
    peak = int(np.argmax(np.linalg.norm(states[:, :2], axis=1)))
    assert_allclose(frame.args[0], states[peak, :2])
    assert_allclose(
        frame.kwargs["routing_solution"].path_lengths, solutions.path_lengths[peak]
    )
    sequence = renderer.render_sequence.call_args
    assert sequence.kwargs["routing_solution"] is solutions
    assert sequence.kwargs["interval"] == 40
    assert (tmp_path / "opposed_matplotlib.png").exists()
