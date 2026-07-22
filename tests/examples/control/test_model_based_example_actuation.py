import sys
from pathlib import Path

import jax.numpy as jnp
from numpy.testing import assert_allclose

from soromox.actuation import IdentityActuator

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONFIGURATION_SPACE_DIR = (
    REPOSITORY_ROOT / "examples" / "control" / "configuration_space"
)
OPERATIONAL_SPACE_DIR = REPOSITORY_ROOT / "examples" / "control" / "operational_space"

for module_dir in (CONFIGURATION_SPACE_DIR, OPERATIONAL_SPACE_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import configuration_space_comparison_simulation  # noqa: E402
import pcs_impedance_example  # noqa: E402
from trajectory_primitives import TaskSpaceTrajectoryConfig  # noqa: E402


def assert_identity_actuation(robot) -> None:
    assert robot.num_actuators == robot.num_dofs
    assert len(robot.actuators) == 1
    assert isinstance(robot.actuators[0], IdentityActuator)
    assert_allclose(
        robot.actuation_matrix(jnp.zeros((robot.num_dofs,))),
        jnp.eye(robot.num_dofs),
    )


def test_configuration_space_comparison_uses_identity_actuation():
    robot, num_dofs, _ = configuration_space_comparison_simulation.create_robot()

    assert num_dofs == robot.num_dofs
    assert_identity_actuation(robot)


def test_operational_space_impedance_uses_identity_actuation():
    problem = pcs_impedance_example.build_problem(TaskSpaceTrajectoryConfig())

    assert problem.num_dofs == problem.robot.num_dofs
    assert_identity_actuation(problem.robot)
