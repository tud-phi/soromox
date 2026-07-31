# Operational-Space Controllers

Operational-space controllers work in task space (e.g., end-effector positions, orientations) rather than configuration space. They are ideal when the control objective is naturally expressed in Cartesian coordinates or other task-related quantities.

## Overview

Operational-space control, pioneered by Khatib (1987), formulates robot control directly in the task space of interest. The key insight is to use the operational-space dynamics:

$$
\Lambda(q) \ddot{x} + \mu(q, \dot{q}) \dot{x} + p(q) = f
$$

where:

- \(\Lambda(q) = (J M^{-1} J^T)^{-1}\) is the operational-space inertia matrix
- \(\mu(q, \dot{q})\) is the operational-space Coriolis matrix
- \(p(q)\) represents operational-space forces from potential energy
- \(f\) is the control force in operational space
- \(J\) is the task Jacobian

### Base Class

All operational-space controllers inherit from `OperationalSpaceBaseController`:

::: soromox.control.OperationalSpaceBaseController
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4
      group_by_category: true
      docstring_section_style: table
      members_order: source

---

## Trajectory Trackers

### OperationalSpaceImpedanceControlTracker

!!! note "Full Actuation Required"
    This controller requires the system to be **fully actuated** (number of actuators equals number of DOFs, \(n = m\)). For under-actuated systems, consider using `OperationalSpaceSynergisticController`.

Operational-space impedance control combines selectable feedback linearization
with moving-reference feedforward. Set `feedback_linearization="full"` (the
default) or `"partial"`.

Both modes cancel task-projected elasticity and damping, cancel gravity, feed
forward \(\Lambda\ddot{x}^d\), and inject the requested stiffness and damping.
They differ only in their Coriolis/centrifugal compensation:

| Mode | Compensated task force | Result |
| --- | --- | --- |
| `full` | \(\mu_x\dot{q}\) | Cancels the complete task-space Coriolis force |
| `partial` | \(\mu_x(I-\bar{J}J)\dot{q}\) | Cancels only null-space Coriolis coupling and retains \(\mu_x\bar{J}\dot{x}\) |

!!! info "Historical provenance"
    The `partial` mode follows the Cartesian impedance structure proposed in
    Section 3.3, equations 49 and 56–60, of
    [Della Santina et al. (2020)](https://doi.org/10.1177/0278364919897292).
    That controller removes dynamic coupling from the residual/null-space
    degrees of freedom while deliberately retaining the natural task-space
    Coriolis term. The implementation here extends the paper's set-point law
    with desired velocity and acceleration feedforward for moving-reference
    tracking.

    The `full` mode instead uses a Khatib-style operational-space nonlinear
    dynamic-decoupling structure. Section IV, equations 29–31, of
    [Khatib (1987)](https://doi.org/10.1109/JRA.1987.1087068) compensates the
    complete operational-space centrifugal/Coriolis and gravity terms before
    applying desired acceleration and tracking feedback. The implementation
    here is not a verbatim reproduction: its stiffness and damping are physical
    task-space impedance forces, so the closed-loop error dynamics retain
    \(\Lambda\) rather than being presented as unit-mass dynamics.

With full linearization and \(e_x\) denoting the geometric correction from the
current pose to the desired pose, the local closed-loop error dynamics become:

$$
\Lambda \ddot{e}_x + D_x \dot{e}_x + K_x e_x = 0
$$

where:

- \(\Lambda\) is the operational-space inertia matrix (preserved)
- \(D_x\) is the desired damping matrix
- \(K_x\) is the desired stiffness matrix
- \(x^d\), \(\dot{x}^d\), and \(\ddot{x}^d\) define the moving reference

The control law implements six key terms:

1. **Cancel elastic and damping forces** projected to task space
2. **Cancel gravity**
3. **Apply the selected Coriolis cancellation**
4. **Feed forward desired acceleration**: \(\Lambda\ddot{x}^d\)
5. **Inject desired stiffness**: \(K_x e_x\)
6. **Inject desired damping**: \(D_x (\dot{x}^d - \dot{x})\)

!!! tip "Choosing the mode"
    Use `full` when moving-reference tracking accuracy is the priority and the
    model is sufficiently accurate. Use `partial` when retaining the natural
    task-space velocity-dependent dynamics is desired. Partial mode generally
    exhibits more tracking error on fast trajectories because that term is
    intentionally not cancelled.

!!! tip "Gain scaling"
    Translational and rotational gains have different units and should be tuned
    separately. For approximately decoupled modes at a representative
    configuration, a useful initialization is
    \(K_i = \lambda_i\omega_{n,i}^2\) and
    \(D_i = 2\zeta_i\lambda_i\omega_{n,i}\), where \(\lambda_i\) is the
    corresponding operational-space inertia. Do not use the same numeric gains
    for position and orientation without accounting for this scaling.

!!! info "Assumptions"
    The impedance controller assumes:
    
    - **Full actuation**: Number of actuators equals number of DOFs (\(n = m\))
    - **Invertible actuation matrix**: \(A(q)\) must be full rank
    - **Stable null space**: Typically satisfied when stiffness and damping matrices are positive definite

::: soromox.control.OperationalSpaceImpedanceControlTracker
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4
      group_by_category: true
      docstring_section_style: table
      members_order: source

---

### OperationalSpaceSynergisticController

Synergistic control enables exact task execution in highly under-actuated soft robots by exploiting the dynamic coupling between actuation and operational spaces.

The control law is:

$$
\tau = P_{AM}(q) J^T(q) \left(K_p e_x + K_i \int e_x \, dt + K_d e_{\dot{x}}\right)
$$

with:

$$
P_{AM}(q) = (J(q) M^{-1}(q) A(q))^{-1} J(q) M^{-1}(q),
\quad e_x = x^d - x,
\quad e_{\dot{x}} = \dot{x}^d - \dot{x}
$$

where:
- \(A(q)\) is the actuation matrix
- \(M(q)\) is the inertia matrix
- \(J(q)\) is the operational space Jacobian
- \(K_p\) is the operational space proportional gain matrix
- \(K_i\) is the operational space integral gain matrix
- \(K_d\) is the operational space derivative gain matrix
- \(e_x\) is the operational space pose error (geometric error for orientation)
- \(x^d\) is the desired operational space pose
- \(x\) is the current operational space pose
- \(\dot{x}^d\) is the desired operational space velocity
- \(\dot{x}\) is the current operational space velocity

The key insight is the use of a dynamically-consistent synergistic projector \(P_{AM}\) that maps operational-space PID control forces to actuator inputs while respecting the system's dynamic structure.

!!! info "Assumptions"
    The synergistic controller assumes:
    
    - **Under-actuation**: The actuation space has lower dimensionality than the configuration space (\(m < n\)). For fully actuated systems, consider using the impedance control tracker.
    - **Equal dimensions**: The operational space dimension equals the actuation space dimension (\(o = m\))
    - **Full-rank matrix**: The matrix \(J(q) M^{-1}(q) A(q) \in \mathbb{R}^{m \times m}\) must be full-rank (invertible)

!!! tip "When to Use Synergistic Control"
    Synergistic control is ideal for:
    
    - **Under-actuated soft robots** where the number of actuators is less than the number of DOFs
    - **Exact task execution** when the task dimension matches the actuation dimension

::: soromox.control.OperationalSpaceSynergisticController
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4
      group_by_category: true
      docstring_section_style: table
      members_order: source

---

## Usage Examples

### Impedance Control (Fully Actuated Systems)

```python
import jax.numpy as jnp
from soromox.control import OperationalSpaceImpedanceControlTracker, ReferenceTrajectory
from soromox.coordinate_transformations import OperationalSpaceDynamics

# Create operational space dynamics (defines the task space)
osd = OperationalSpaceDynamics(
    robot=robot,
    forward_kinematics_fn=lambda q: robot.end_effector_position(q),
)

# Define reference trajectory in operational space
ts = jnp.linspace(0, 10, 1000)
x_des = jnp.zeros((len(ts), osd.n_operational_space))
# ... fill in desired trajectory ...

ref_traj = ReferenceTrajectory(ts=ts, x_des_ts=x_des)

# Create impedance controller (requires full actuation: n = m)
controller = OperationalSpaceImpedanceControlTracker(
    operational_space_dynamics=osd,
    reference_trajectory=ref_traj,
    K_x=100.0,  # Stiffness (scalar, vector, or matrix)
    D_x=10.0,   # Damping (scalar, vector, or matrix)
    feedback_linearization="full",  # Or "partial"
)
```

### Synergistic Control (Under-Actuated Systems)

```python
import jax.numpy as jnp
from soromox.control import (
    OperationalSpaceSynergisticController,
    PIDControl,
    ReferenceTrajectory,
)
from soromox.coordinate_transformations import OperationalSpaceDynamics

# Create operational space dynamics (defines the task space)
# IMPORTANT: n_operational_space must equal n_actuators (o = m)
osd = OperationalSpaceDynamics(
    robot=robot,
    forward_kinematics_fn=lambda q: robot.end_effector_position(q),
)

# Verify that operational space dimension matches actuation dimension
assert osd.n_operational_space == robot.num_actuators, \
    "SynergisticController requires o = m (operational space = actuation space)"

# Define reference trajectory in operational space
ts = jnp.linspace(0, 10, 1000)
x_des = jnp.zeros((len(ts), osd.n_operational_space))
# ... fill in desired trajectory ...

ref_traj = ReferenceTrajectory(ts=ts, x_des_ts=x_des)

# Define PID gains in operational space
Kp = 100.0 * jnp.ones((osd.n_operational_space,))
Ki = 10.0 * jnp.ones((osd.n_operational_space,))
Kd = 5.0 * jnp.ones((osd.n_operational_space,))
pid_control = PIDControl(Kp, Ki, Kd)

# Create synergistic controller (for under-actuated systems: m < n, o = m)
controller = OperationalSpaceSynergisticController(
    operational_space_dynamics=osd,
    reference_trajectory=ref_traj,
    pid_control=pid_control,
)
```

## References

The operational-space formulation for robot control was developed in the following foundational works:

- Khatib, O. (1987). A unified approach for motion and force control of robot manipulators: The operational space formulation. *IEEE Journal on Robotics and Automation*, 3(1), 43-53.

- Ott, C. (2008). *Cartesian impedance control of redundant and flexible-joint robots*. Springer.

For soft robot applications:

- Della Santina, C., Katzschmann, R. K., Bicchi, A., & Rus, D. (2020). Model-based dynamic feedback control of a planar soft robot: trajectory tracking and interaction with the environment. *The International Journal of Robotics Research*, 39(4-5), 490-513.

For synergistic control of under-actuated soft robots:

- Della Santina, C., Pallottino, L., Rus, D., & Bicchi, A. (2019). Exact task execution in highly under-actuated soft limbs: an operational space based approach. *IEEE Robotics and Automation Letters*, 4(3), 2508-2515.
