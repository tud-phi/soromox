import warnings

import jax.numpy as jnp
import pytest

from soromox.control.actuation_matrix_utils import (
    ActuationScenario,
    analyze_actuation_matrix,
    emit_actuation_warnings,
)


@pytest.mark.parametrize(
    ("matrix", "expected_scenario", "expected_rank"),
    [
        (jnp.eye(2), ActuationScenario.FULL_ACTUATION, 2),
        (
            jnp.array([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]]),
            ActuationScenario.OVERACTUATION,
            2,
        ),
        (jnp.array([[1.0], [0.0]]), ActuationScenario.UNDERACTUATION, 1),
    ],
)
def test_analyze_actuation_matrix_detects_scenario_and_rank(
    fake_robot_factory, matrix, expected_scenario, expected_rank
):
    scenario, shape, rank = analyze_actuation_matrix(fake_robot_factory(matrix))

    assert scenario is expected_scenario
    assert shape == matrix.shape
    assert rank == expected_rank


def test_analyze_actuation_matrix_falls_back_when_robot_cannot_be_evaluated(
    broken_actuation_robot,
):
    scenario, shape, rank = analyze_actuation_matrix(broken_actuation_robot)

    assert scenario is ActuationScenario.FULL_ACTUATION
    assert shape == (2, 2)
    assert rank == 2


def test_emit_actuation_warnings_rejects_singular_full_actuation(fake_robot_factory):
    robot = fake_robot_factory(jnp.array([[1.0, 0.0], [0.0, 0.0]]))

    with pytest.raises(ValueError, match="singular"):
        emit_actuation_warnings("Controller", robot)


def test_emit_actuation_warnings_handles_full_rank_full_actuation(fake_robot_factory):
    robot = fake_robot_factory(jnp.eye(2))

    with pytest.warns(UserWarning, match="configuration-independent"):
        scenario = emit_actuation_warnings("Controller", robot)

    with warnings.catch_warnings(record=True) as caught:
        scenario_computed_torque = emit_actuation_warnings(
            "Controller", robot, is_computed_torque=True
        )

    assert scenario is ActuationScenario.FULL_ACTUATION
    assert scenario_computed_torque is ActuationScenario.FULL_ACTUATION
    assert caught == []


def test_emit_actuation_warnings_handles_overactuation(fake_robot_factory):
    robot = fake_robot_factory(jnp.array([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]]))

    with pytest.warns(UserWarning, match="overactuated"):
        scenario = emit_actuation_warnings("Controller", robot)

    assert scenario is ActuationScenario.OVERACTUATION


def test_emit_actuation_warnings_rejects_rank_deficient_overactuation(
    fake_robot_factory,
):
    robot = fake_robot_factory(jnp.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))

    with pytest.raises(ValueError, match="less than the number of DOFs"):
        emit_actuation_warnings("Controller", robot)


@pytest.mark.parametrize(
    ("is_regulator", "message"),
    [
        (True, "underactuated regulation"),
        (False, "Trajectory tracking"),
    ],
)
def test_emit_actuation_warnings_handles_underactuation(
    fake_robot_factory, is_regulator, message
):
    robot = fake_robot_factory(jnp.array([[1.0], [0.0]]))

    with pytest.warns(UserWarning, match=message):
        scenario = emit_actuation_warnings(
            "Controller", robot, is_regulator=is_regulator
        )

    assert scenario is ActuationScenario.UNDERACTUATION
