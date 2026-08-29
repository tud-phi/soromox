__all__ = ["ActuationSpaceBaseController"]

from abc import ABC

import equinox as eqx
from jax import Array

from soromox.control.base_controller import BaseController
from soromox.control.reference_trajectory import ReferenceTrajectory
from soromox.coordinate_transformations.actuation_space import ActuationSpaceDynamics


class ActuationSpaceBaseController(BaseController, ABC):
    """
    Abstract base class for actuation-space controllers.

    Actuation-space control, as developed in the fundamental work by Pustina et al.
    (2024) and Pustina (2025), represents the robot dynamics in a **collocated form**
    (also called input-decoupled form), where the actuation matrix becomes a simple
    identity structure:

        A_y = [[I_{n_a}], [0_{n_u, n_a}]]

    where n_a is the number of actuators and n_u = n - n_a is the number of unactuated
    degrees of freedom. In this form:

    - The first n_a/num_actuators rows (actuated coordinates) are directly actuated via an identity
      actuation matrix, meaning actuator inputs directly map to generalized forces
      on these coordinates.
    - The remaining n_u (i.e., num_dofs - num_actuators) rows (unactuated coordinates) receive no direct actuation
      input—they are driven only by dynamic coupling.

    Benefits of Actuation-Space Formulation
    ----------------------------------------

    1. **Natural handling of actuation coupling and underactuation**: For robots with
       full (i.e., non-diagonal) and on-square actuation matrices A(q), the actuation-space
       formulation provides a clean separation between the actuated and unactuated dynamics.


    2. **Simplified stability/convergence proofs**: Control design and stability
       analysis become significantly easier when the actuation structure is diagonal.
       Standard Lyapunov-based arguments can be applied directly to the actuated
       subsystem without needing to account for the structure of A(q).

    Zero Dynamics and Stability
    ---------------------------

    In any underactuated control setting—regardless of whether the controller operates
    in configuration space, operational space, or actuation space—a key assumption is
    that the **zero dynamics** (i.e., the dynamics of the unactuated coordinates when
    the actuated coordinates are perfectly controlled) are stable or at least bounded.

    We emphasize this assumption here because actuation-space controllers are
    **particularly well-suited for underactuated robots**, and the clean separation
    between actuated and unactuated dynamics in this formulation makes the role of
    zero dynamics especially transparent.

    This assumption is often satisfied for soft robots because:

    - Elastic and damping forces provide natural stabilization of the unactuated modes
    - The internal dynamics of soft materials tend to be dissipative
    - Gravitational potential energy often has stabilizing effects

    When designing controllers, keep in mind:
    - Controllers typically only directly control the first n_a (actuated) coordinates
    - The unactuated coordinates evolve according to the zero dynamics
    - If the zero dynamics are unstable, trajectory tracking may fail despite
      perfect tracking on the actuated coordinates
    - This limitation is fundamental to underactuation and cannot be overcome by any
      control strategy—whether in configuration space, operational space, or actuation space

    Feasibility of Reference Trajectories in Underactuated Settings
    ----------------------------------------------------------------

    In underactuated settings (n_u > 0), the reference trajectory for the unactuated
    coordinates cannot be arbitrarily specified. The unactuated coordinates are not
    directly controlled and evolve according to the system's internal dynamics.

    - **Regulation (setpoint control)**: The unactuated coordinates of the setpoint
      must be **statically feasible**, meaning there must exist an equilibrium of the
      system at the desired configuration. Otherwise, the system cannot remain at rest
      at the setpoint, even with perfect control of the actuated coordinates.

    - **Tracking (trajectory following)**: The unactuated coordinates of the reference
      trajectory must be at least **dynamically feasible**, meaning the trajectory
      must be consistent with the system's equations of motion. In other words, the
      reference must correspond to a valid solution of the zero dynamics.

    If these feasibility conditions are not satisfied, the controller will be unable
    to achieve the desired behavior, regardless of control gains or design.

    Reference Trajectory Conversion
    --------------------------------

    This base class provides a `convert_reference_trajectory_to_actuation_space`
    method that transforms a configuration-space reference trajectory to actuation
    space using the coordinate transformations provided by `ActuationSpaceDynamics`.
    This ensures that:
    - Position setpoints are mapped through the (generally nonlinear) actuated
      coordinate map for the actuated coordinates
    - Velocity and acceleration references are properly transformed using the
      Jacobian and its time derivatives

    Attributes:
        robot: The soft robot system to be controlled (set to
            actuation_space_dynamics.robot).
        reference_trajectory: The desired trajectory in actuation space.
        reference_trajectory_config_space: The original configuration-space trajectory,
            or None if trajectory was provided directly in actuation space. This is
            needed for model-based control terms that evaluate dynamics at q_des,
            since the inverse mapping from actuation space y back to configuration
            space q is generally nonlinear and not trivially invertible.
        actuation_space_dynamics: The ActuationSpaceDynamics instance that provides
            coordinate transformations between configuration and actuation space.

    References:
        Pustina, P., Della Santina, C., Boyer, F., De Luca, A., & Renda, F. (2024).
        Input decoupling of lagrangian systems via coordinate transformation:
        General characterization and its application to soft robotics.
        IEEE Transactions on Robotics, 40, 2098-2110.

        Pustina, P. (2025). Analysis and control of the underactuation in continuum
        soft robots: a kinematic independent approach.
        PhD Thesis, Sapienza University of Rome.
    """

    actuation_space_dynamics: ActuationSpaceDynamics
    reference_trajectory_config_space: ReferenceTrajectory | None

    def _controller_dynamics_and_state(
        self, y: Array
    ) -> tuple[ActuationSpaceDynamics, Array, Array]:
        """Return current-pose fixed dynamics and internal controller state.

        Args:
            y: Complete system state with trailing dimension
                ``robot.state_size``.

        Returns:
            A tuple containing an actuation-space dynamics view mounted at the
            current base pose, internal coordinates, and internal velocities.
        """
        robot, q_internal, qd_internal = super()._controller_model_and_state(y)
        dynamics = self.actuation_space_dynamics
        if dynamics.robot is not robot:
            dynamics = eqx.tree_at(lambda value: value.robot, dynamics, robot)
        return dynamics, q_internal, qd_internal
