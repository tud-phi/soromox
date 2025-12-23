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

Operational-space impedance control implements partial feedback linearization to cancel the original task dynamics and replace them with desired linear impedance behavior.

The closed-loop dynamics in operational space become:

$$
\Lambda \ddot{x} + D_x \dot{x} + K_x (x - x^d) = 0
$$

where:

- \(\Lambda\) is the operational-space inertia matrix (preserved)
- \(D_x\) is the desired damping matrix
- \(K_x\) is the desired stiffness matrix
- \(x^d\) is the reference position

The control law implements five key terms:

1. **Cancel elastic and damping forces** projected to task space
2. **Cancel gravity**
3. **Cancel null-space Coriolis coupling** to prevent null-space motion from affecting the task
4. **Inject desired stiffness**: \(K_x (x^d - x)\)
5. **Inject desired damping**: \(D_x (\dot{x}^d - \dot{x})\)

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

## Usage Example

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

# Create impedance controller
controller = OperationalSpaceImpedanceControlTracker(
    operational_space_dynamics=osd,
    reference_trajectory=ref_traj,
    K_x=100.0,  # Stiffness (scalar, vector, or matrix)
    D_x=10.0,   # Damping (scalar, vector, or matrix)
)
```

## References

The operational-space formulation for robot control was developed in the following foundational works:

- Khatib, O. (1987). A unified approach for motion and force control of robot manipulators: The operational space formulation. *IEEE Journal on Robotics and Automation*, 3(1), 43-53.

- Ott, C. (2008). *Cartesian impedance control of redundant and flexible-joint robots*. Springer.

For soft robot applications:

- Della Santina, C., Katzschmann, R. K., Bicchi, A., & Rus, D. (2020). Model-based dynamic feedback control of a planar soft robot: trajectory tracking and interaction with the environment. *The International Journal of Robotics Research*, 39(4-5), 490-513.

