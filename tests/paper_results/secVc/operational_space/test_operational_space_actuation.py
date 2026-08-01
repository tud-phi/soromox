import sys
from pathlib import Path

import jax.numpy as jnp
from numpy.testing import assert_allclose

from soromox.actuation import IdentityActuator

MODULE_DIR = (
    Path(__file__).resolve().parents[4]
    / "paper_results"
    / "secVc_model_based_control"
    / "operational_space_impedance_control"
    / "code"
)
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import operational_space_impedance_common  # noqa: E402
from trajectory_primitives import TaskSpaceTrajectoryConfig  # noqa: E402


def test_operational_space_impedance_uses_identity_actuation():
    problem = operational_space_impedance_common.build_problem(
        TaskSpaceTrajectoryConfig()
    )

    assert problem.num_dofs == problem.robot.num_dofs
    assert problem.robot.num_actuators == problem.robot.num_dofs
    assert len(problem.robot.actuators) == 1
    assert isinstance(problem.robot.actuators[0], IdentityActuator)
    assert_allclose(
        problem.robot.actuation_matrix(jnp.zeros((problem.robot.num_dofs,))),
        jnp.eye(problem.robot.num_dofs),
    )
