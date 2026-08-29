"""
Utility functions for actuation matrix analysis and validation.

This module provides functions to check and warn about actuation matrix properties
for configuration-space controllers. The actuation matrix A maps actuator inputs u
to generalized forces tau:
    tau = A(q) @ u

The actuation scenarios are:
- Full actuation: A is square (n_dof × n_actuators, n_dof = n_actuators)
- Overactuation: More actuators than DOFs (n_actuators > n_dof)
- Underactuation: Fewer actuators than DOFs (n_actuators < n_dof)
"""

__all__ = [
    "ActuationScenario",
    "analyze_actuation_matrix",
    "emit_actuation_warnings",
]

import warnings
from enum import Enum
from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from soromox.systems.soft_robot import SoftRobot


class ActuationScenario(Enum):
    """Enum representing the actuation scenario of a robot."""

    FULL_ACTUATION = "full_actuation"
    OVERACTUATION = "overactuation"
    UNDERACTUATION = "underactuation"


def analyze_actuation_matrix(
    robot: "SoftRobot",
) -> tuple[ActuationScenario, tuple[int, int], int]:
    """
    Analyze the actuation matrix of a robot at zero configuration.

    Args:
        robot: The dynamical system to analyze.

    Returns:
        scenario: The actuation scenario (full, over, or under actuation).
        shape: The shape of the actuation matrix (n_dof, n_actuators).
        rank: The rank of the actuation matrix.
    """
    fixed_view = getattr(robot, "_fixed_evaluation_view", None)
    evaluation_robot = robot if fixed_view is None else fixed_view()
    num_dofs = getattr(robot, "num_internal_dofs", robot.num_dofs)
    num_actuators = robot.num_actuators
    q_test = jnp.zeros((num_dofs,))

    try:
        A = evaluation_robot.actuation_matrix(q_test)
        n_dof, n_act = A.shape
        rank = int(jnp.linalg.matrix_rank(A))

        if n_act > n_dof:
            scenario = ActuationScenario.OVERACTUATION
        elif n_act < n_dof:
            scenario = ActuationScenario.UNDERACTUATION
        else:
            scenario = ActuationScenario.FULL_ACTUATION

        return scenario, A.shape, rank
    except Exception:
        # If we can't analyze, assume full actuation
        return (
            ActuationScenario.FULL_ACTUATION,
            (num_dofs, num_actuators),
            num_actuators,
        )


def emit_actuation_warnings(
    controller_name: str,
    robot: "SoftRobot",
    is_regulator: bool = False,
    is_computed_torque: bool = False,
    stacklevel: int = 3,
) -> ActuationScenario:
    """
    Emit appropriate warnings about actuation matrix properties.

    This function analyzes the actuation matrix and emits warnings/exceptions
    appropriate for the actuation scenario and controller type.

    Args:
        controller_name: Name of the controller class for warning messages.
        robot: The soft robot system to be controlled.
        is_regulator: Whether the controller is a regulator (vs. tracker).
        is_computed_torque: Whether this is the ComputedTorqueTracker (which handles
            configuration-dependent actuation via feedback linearization).
        stacklevel: Stack level for warning messages.

    Returns:
        The detected actuation scenario.

    Raises:
        ValueError: If the actuation matrix is rank-deficient for full actuation,
            or if the rank is less than the number of DOFs for overactuation.
    """
    scenario, shape, rank = analyze_actuation_matrix(robot)
    n_dof, n_actuators = shape

    if scenario == ActuationScenario.FULL_ACTUATION:
        # Square actuation matrix
        if rank < n_dof:
            raise ValueError(
                f"{controller_name}: The actuation matrix is singular "
                f"(rank {rank} < {n_dof}). The actuation matrix must be "
                f"full rank (positive-definite) for configuration-space control."
            )

        # Warn about configuration-dependent actuation (except for computed torque)
        if not is_computed_torque:
            warnings.warn(
                f"{controller_name}: For full theoretical guarantees, the actuation "
                f"matrix should be configuration-independent (constant). If A(q) is "
                f"configuration-dependent, consider using the ComputedTorqueTracker "
                f"(with full feedback linearization) or actuation-space controllers.",
                UserWarning,
                stacklevel=stacklevel,
            )

    elif scenario == ActuationScenario.OVERACTUATION:
        # More actuators than DOFs
        if rank < n_dof:
            raise ValueError(
                f"{controller_name}: The actuation matrix has rank {rank}, which is "
                f"less than the number of DOFs ({n_dof}). Even with overactuation "
                f"({n_actuators} actuators > {n_dof} DOFs), the actuation matrix "
                f"must have rank at least {n_dof} to control all DOFs."
            )

        warnings.warn(
            f"{controller_name}: The system is overactuated ({n_actuators} actuators > "
            f"{n_dof} DOFs). Overactuation has been rarely investigated in literature. "
            f"The pseudo-inverse will be used to compute actuator inputs, which minimizes "
            f"the actuator effort while achieving the desired generalized forces. "
            f"This should generally work well as the residual can be minimized to "
            f"(almost) zero.",
            UserWarning,
            stacklevel=stacklevel,
        )

    elif scenario == ActuationScenario.UNDERACTUATION:
        # Fewer actuators than DOFs
        _emit_underactuation_warnings(
            controller_name=controller_name,
            n_dof=n_dof,
            n_actuators=n_actuators,
            is_regulator=is_regulator,
            stacklevel=stacklevel,
        )

    return scenario


def _emit_underactuation_warnings(
    controller_name: str,
    n_dof: int,
    n_actuators: int,
    is_regulator: bool,
    stacklevel: int,
) -> None:
    """
    Emit warnings specific to underactuated systems.

    Args:
        controller_name: Name of the controller class.
        n_dof: Number of degrees of freedom.
        n_actuators: Number of actuators.
        is_regulator: Whether the controller is a regulator.
        stacklevel: Stack level for warnings.
    """
    base_warning = (
        f"{controller_name}: The system is underactuated ({n_actuators} actuators < "
        f"{n_dof} DOFs). "
    )

    if is_regulator:
        warning_msg = base_warning + (
            "For underactuated regulation, several conditions must be met:\n"
            "  1. The desired configuration must be statically feasible (reachable by "
            "the actuated DOFs).\n"
            "  2. The null-space dynamics must be stable (usually the case for soft robots "
            "due to natural damping).\n"
            "  3. The (proportional) feedback gains must be sufficiently high to locally "
            "stabilize the neighborhood of the setpoint.\n"
            "  4. The system must be initialized in the region of attraction of the setpoint.\n"
            "If these conditions are met, regulators can work adequately. However, we "
            "recommend actuation-space controllers for underactuated settings, as they "
            "can also handle configuration-dependent actuation matrices.\n"
            "For deeper analysis, see: Borja (2022), Pustina (2022), "
            "Della Santina (2023), and Pustina (2025)."
        )
    else:
        warning_msg = base_warning + (
            "Trajectory tracking for underactuated soft robots generally lacks formal "
            "stability guarantees. The conditions for regulation still apply:\n"
            "  1. The desired trajectory must be statically feasible at each point.\n"
            "  2. The null-space dynamics must be stable.\n"
            "  3. The feedback gains must be sufficiently high.\n"
            "  4. The system must remain in the region of attraction.\n"
            "We strongly recommend using actuation-space controllers instead for "
            "underactuated trajectory tracking.\n"
            "For deeper analysis, see: Borja (2022), Pustina (2022), "
            "Della Santina (2023), and Pustina (2025)."
        )

    warnings.warn(warning_msg, UserWarning, stacklevel=stacklevel)
