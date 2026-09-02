"""Tests for the shared Section Vd problem builder.

Both generators and the tuning study build their optimization problem from
``secvd_case.build_evaluator``. Before that existed, each generator constructed
its controller, reference and objective inline and the tuning study held a
hand-copy of the collocated setup, so the study could silently tune a different
problem than the generator ran. These tests pin the contract that replaced it.

The expensive check -- that the builder reproduces each committed archive's
initial loss -- is marked ``slow``: it needs two full 5 s rollouts. Everything
else runs against a synthetic setpoint, avoiding the 100000-step rollout that
:func:`build_sec_vd_case` performs to find the real one.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

SECTION_DIR = (
    Path(__file__).resolve().parents[3]
    / "paper_results"
    / "secVd_control_gain_optimization"
)
CODE_DIR = SECTION_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from secvd_case import (  # noqa: E402
    COMMITTED_ALPHA,
    COMMITTED_LR_RATIO,
    METHODS,
    SecVdCase,
    build_evaluator,
    build_sec_vd_robot,
    end_effector_pose,
)
from secvd_objective import (  # noqa: E402
    configuration_space_scales,
    operational_space_scales,
)
from secvd_results import load_results  # noqa: E402

SHORT_HORIZON = 0.01


@pytest.fixture(scope="module")
def cheap_case():
    """A case with a synthetic setpoint, so no 10 s settling rollout is needed.

    ``build_sec_vd_case`` finds the real setpoint by integrating to equilibrium,
    which costs 100000 solver steps. Nothing here depends on the setpoint being
    the physical one -- only on it being non-degenerate.
    """
    robot, routing, total_length = build_sec_vd_robot()
    q0 = jnp.zeros(robot.num_active_strains)
    q_des = q0.at[0].set(0.5)
    return SecVdCase(
        robot=robot,
        routing=routing,
        total_length=total_length,
        q0=q0,
        qd0=q0,
        q_des=q_des,
        x_des=end_effector_pose(robot, q_des, total_length),
        solver_dt=1e-4,
        save_ts=robot._compute_save_times(
            0.0, SHORT_HORIZON, solver_dt=1e-4, save_dt=1e-4
        ),
    )


@pytest.fixture(scope="module")
def problems(cheap_case):
    return {
        method: build_evaluator(method, case=cheap_case, horizon=SHORT_HORIZON)
        for method in METHODS
    }


def test_both_methods_build(problems):
    assert set(problems) == set(METHODS)
    for method, problem in problems.items():
        assert problem.method == method
        assert problem.horizon == SHORT_HORIZON
        assert callable(problem.gradient_fn)


def test_saturation_support_is_a_field_not_a_method_check(problems):
    """Consumers gate on this rather than on the method name.

    The synergistic controller's ``PIDControl`` takes no ``saturation_fn``, so
    follow-up item 2 has nothing to sweep there. A stage that branched on the
    method name instead would need editing for every controller added.
    """
    assert problems["collocated"].supports_saturation is True
    assert problems["synergistic"].supports_saturation is False
    assert problems["synergistic"].saturation_config == "none"
    assert "e_sat" in problems["collocated"].saturation_config


def test_saturation_scale_reaches_the_archived_config(cheap_case):
    problem = build_evaluator(
        "collocated", case=cheap_case, horizon=SHORT_HORIZON, e_sat=3.2e-2
    )
    assert "0.032" in problem.saturation_config


def test_objective_scales_come_from_the_shared_objective_module(problems, cheap_case):
    """The scales must be the objective module's, not a private copy."""
    robot = cheap_case.robot
    np.testing.assert_allclose(
        np.asarray(problems["collocated"].objective_scales),
        np.asarray(configuration_space_scales(robot.params.link.length)),
        rtol=1e-15,
    )
    np.testing.assert_allclose(
        np.asarray(problems["synergistic"].objective_scales),
        np.asarray(
            operational_space_scales(
                jnp.array([False, False, False, True, True, True]),
                cheap_case.total_length,
            )
        ),
        rtol=1e-15,
    )


def test_nominal_gains_are_sized_for_their_own_control_space(problems):
    # Collocated acts in actuator space, synergistic in the selected task space;
    # both happen to be three-dimensional here, but for different reasons.
    for problem in problems.values():
        assert set(problem.nominal_gains) == {"Kp", "Ki", "Kd"}
        for value in problem.nominal_gains.values():
            assert value.shape == (3,)
    assert float(problems["collocated"].nominal_gains["Kp"][0]) == 50.0
    assert float(problems["synergistic"].nominal_gains["Kp"][0]) == 10.0


def test_committed_rates_factor_into_a_magnitude_and_a_ratio(problems):
    """(0.5, 0.25, 0.05) must be exactly alpha * ratio.

    The tuning study searches the magnitude and the ratio separately, and gates
    on how far the committed magnitude sits from the located optimum, so the
    factorization has to reproduce what the generators actually use.
    """
    expected = {"Kp": 0.5, "Ki": 0.25, "Kd": 0.05}
    for name, rate in expected.items():
        assert COMMITTED_ALPHA * COMMITTED_LR_RATIO[name] == pytest.approx(rate)
    for problem in problems.values():
        assert problem.committed_alpha == COMMITTED_ALPHA
        assert problem.committed_lr_ratio == COMMITTED_LR_RATIO


def test_reference_is_exposed_over_the_time_grid(problems, cheap_case):
    timesteps = len(cheap_case.save_ts)
    for problem in problems.values():
        assert problem.reference_ts.shape[0] == timesteps


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"horizon": 0.0}, "horizon must be positive"),
        ({"horizon": -1.0}, "horizon must be positive"),
        ({"e_sat": 0.0}, "e_sat must be positive"),
    ],
)
def test_degenerate_requests_are_rejected(cheap_case, kwargs, match):
    with pytest.raises(ValueError, match=match):
        build_evaluator("collocated", case=cheap_case, **kwargs)


def test_unknown_method_is_rejected(cheap_case):
    with pytest.raises(ValueError, match="Unknown Section Vd method"):
        build_evaluator("cartesian", case=cheap_case)


def test_gradient_is_finite_and_shaped_like_the_gains(problems):
    """One short rollout per method, enough to prove the problem differentiates."""
    for problem in problems.values():
        opt_vars = {
            "opt_ctr_params": problem.nominal_gains,
            "opt_atr_params": {},
        }
        (loss, aux), grad = problem.gradient_fn(opt_vars)
        assert jnp.isfinite(loss)
        assert sorted(aux) == [
            "integral_error_ts",
            "q_ts",
            "qd_ts",
            "t_ts",
            "u_ts",
        ]
        for name, value in grad["opt_ctr_params"].items():
            assert value.shape == problem.nominal_gains[name].shape
            assert bool(jnp.all(jnp.isfinite(value)))


def test_control_error_is_the_controllers_own_not_the_objectives(problems, cheap_case):
    """The error ``e_sat`` saturates, in the controller's space and units.

    This exists because the saturation sweep first measured the wrong signal.
    ``e_sat`` is a metre scale on the error the PID integrates -- actuated
    coordinates for the collocated regulator -- while the loss is built from the
    configuration-space strain error. Comparing that to ``e_sat`` compares
    different spaces in different units, and reports a saturated fraction that
    means nothing.
    """
    for problem in problems.values():
        opt_vars = {"opt_ctr_params": problem.nominal_gains, "opt_atr_params": {}}
        (_loss, aux), _grad = problem.gradient_fn(opt_vars)
        error = problem.control_error_fn(aux)
        assert error.shape[0] == aux["q_ts"].shape[0]
        # One column per gain the controller carries, which is what e_sat and
        # the integral state are sized against.
        assert error.shape[1] == problem.nominal_gains["Kp"].shape[0]
        assert error.shape == aux["integral_error_ts"].shape
        assert bool(jnp.all(jnp.isfinite(error)))


def test_collocated_control_error_is_not_the_strain_error(problems):
    """Pin the specific confusion: actuation space is not configuration space.

    The regulator is underactuated -- three tendons against six active strains --
    so the two errors do not even share a shape. The sweep's original metric
    thresholded the six-column strain error against ``e_sat``, a metre scale on
    the three-column actuated error, and averaged the result.
    """
    problem = problems["collocated"]
    opt_vars = {"opt_ctr_params": problem.nominal_gains, "opt_atr_params": {}}
    (_loss, aux), _grad = problem.gradient_fn(opt_vars)
    control_error = np.asarray(problem.control_error_fn(aux))
    strain_error = np.asarray(problem.reference_ts - aux["q_ts"])
    assert control_error.shape[1] == problem.case.robot.num_actuators
    assert strain_error.shape[1] == problem.case.robot.num_active_strains
    assert control_error.shape[1] != strain_error.shape[1]
    # Tendon lengths, so metres and small; the strain error carries 1/m
    # components and is orders larger on this setpoint.
    assert np.abs(control_error).max() < 0.1
    assert np.abs(strain_error).max() > np.abs(control_error).max()


@pytest.mark.slow
@pytest.mark.parametrize("method", METHODS)
def test_builder_reproduces_the_committed_archive_initial_loss(method):
    """The check that makes the shared builder provably the generators' problem.

    Batch entry 0 of every initialization scheme is the nominal gain set, so the
    archive's ``history_loss[0, 0]`` is this problem evaluated at the nominal.

    Bit-identity is not expected: the archives were produced under ``vmap`` at
    six starts, this evaluates one start unbatched, and batched matmuls
    reassociate floating-point additions. Across 50000 solver steps that is
    worth about 1e-10 relative, measured.
    """
    archive = SECTION_DIR / "data" / method / "optimization_results.npz"
    if not archive.exists():
        pytest.skip(f"no committed {method} archive to compare against")
    problem = build_evaluator(method)
    (loss, _aux), _grad = problem.gradient_fn(
        {"opt_ctr_params": problem.nominal_gains, "opt_atr_params": {}}
    )
    stored = float(np.asarray(load_results(archive)["history_loss"])[0, 0])
    assert float(loss) == pytest.approx(stored, rel=1e-8)
