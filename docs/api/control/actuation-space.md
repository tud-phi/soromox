# Actuation-Space Controllers

Actuation-space controllers operate in a transformed coordinate system where the actuation matrix becomes a simple identity structure. This formulation, developed by [Pustina et al. (2024)](#references) and further elaborated in [Pustina (2025)](#references), provides a clean separation between actuated and unactuated dynamics, making it particularly well-suited for underactuated soft robots.

## Overview

The key insight of actuation-space control is to transform the robot dynamics into a **collocated form** (also called input-decoupled form), where the actuation matrix becomes:

$$
A_y = \begin{bmatrix} I_{n_a} \\ 0_{n_u \times n_a} \end{bmatrix}
$$

where:

- $n_a$ is the number of actuators
- $n_u = n - n_a$ is the number of unactuated degrees of freedom

In this representation:

- The first $n_a$ rows (actuated coordinates) are directly controlled via an identity actuation matrix
- The remaining $n_u$ rows (unactuated coordinates) receive no direct actuation—they evolve according to dynamic coupling only

### Benefits of Actuation-Space Formulation

!!! success "Key Advantages"
    1. **Natural handling of underactuation**: Clean separation between actuated and unactuated dynamics without needing pseudo-inverses
    2. **No actuation matrix inversion in the control law**: Control inputs directly map to actuator forces without requiring $A(q)^{-1}$ or $A(q)^{\dagger}$ in the control law
    3. **Simplified stability analysis**: Standard Lyapunov-based arguments can be applied directly to the actuated subsystem
    4. **Handles configuration-dependent actuation**: Works seamlessly with non-constant, non-diagonal, and non-square actuation matrices as long as mapping from configuration- to actuation-space exists (i.e., integrability condition from (Pustina et al., 2024) is satisfied)

### Controller Summary

| Controller | Type | Model-Based Term | Description |
|------------|------|------------------|-------------|
| [`PIDController`](#pidcontroller) | Tracker | None | Pure PID control in actuation space |
| [`FeedforwardCompensationTracker`](#feedforwardcompensationtracker) | Tracker | Full dynamics at $q_\mathrm{des}$ | Open-loop feedforward + feedback |
| [`MixedStateFeedbackTracker`](#mixedstatefeedbacktracker) | Tracker | Mixed state evaluation | Hybrid feedforward strategy |
| [`GravityCancellationRegulator`](#gravitycancellationregulator) | Regulator | $G_y(q) + \tau_{\mathrm{el},y}(q_\mathrm{des})$ | Real-time gravity compensation |
| [`PotentialCancellationRegulator`](#potentialcancellationregulator) | Regulator | $G_y(q) + \tau_{\mathrm{el},y}(q)$ | Full potential cancellation |
| [`PotentialCompensationRegulator`](#potentialcompensationregulator) | Regulator | $G_y(q_\mathrm{des}) + \tau_{\mathrm{el},y}(q_\mathrm{des})$ | Potential shaping |

---

## Zero Dynamics and Stability

!!! warning "Zero Dynamics Assumption"
    In any underactuated control setting—regardless of whether the controller operates in configuration space, operational space, or actuation space—a key assumption is that the **zero dynamics** (the dynamics of unactuated coordinates when actuated coordinates are perfectly controlled) are stable or at least bounded.

!!! note "Why emphasized here?"
    While the zero dynamics stability assumption applies universally to all underactuated systems, we emphasize it in the actuation-space context because actuation-space controllers are **particularly well-suited for underactuated robots**. The clean separation between actuated and unactuated dynamics in this formulation makes the role of zero dynamics especially transparent.

For soft robots, this assumption is often satisfied because:

- **Elastic forces** provide natural restoring behavior
- **Damping forces** dissipate energy from unactuated modes  
- **Gravitational potential** often has stabilizing effects

However, if the zero dynamics are unstable, trajectory tracking may fail despite perfect tracking on the actuated coordinates. This limitation is fundamental to underactuation and cannot be overcome by any control strategy—whether in configuration space, operational space, or actuation space.

---

## Reference Trajectory Feasibility

In underactuated settings ($n_u > 0$), the reference trajectory cannot be arbitrarily specified:

### Regulation (Setpoint Control)

For setpoint regulation, the unactuated coordinates of the setpoint must be **statically feasible**—there must exist an equilibrium of the system at the desired configuration. Otherwise, the system cannot remain at rest at the setpoint, even with perfect control of the actuated coordinates.

### Trajectory Tracking

For trajectory tracking, the unactuated coordinates of the reference trajectory must be **dynamically feasible**—the trajectory must be consistent with the system's equations of motion. In other words, the reference must correspond to a valid solution of the zero dynamics.

!!! tip "Practical Recommendation"
    For underactuated systems, **setpoint regulation** is generally more robust than trajectory tracking since it only requires static feasibility of the unactuated coordinates.

---

## Base Class

All actuation-space controllers inherit from `ActuationSpaceBaseController`:

::: soromox.control.actuation_space.ActuationSpaceBaseController
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source

---

## Trajectory Trackers

Trajectory trackers are designed for dynamic trajectory tracking where the reference includes position, velocity, and acceleration profiles.

### PIDController

Basic PID controller in actuation space without model-based feedforward. By working in actuation space, there is no need to invert the actuation matrix—the control input is directly the actuator force/torque.

The control law is:

$$
\tau = K_p e_a + K_i \int e_a \, dt + K_d \dot{e}_a
$$

where $e_a = y_{a,\mathrm{des}} - y_a$ is the error in the actuated coordinates (first $n_a$ entries of the actuation-space coordinates).

!!! note "Direct Actuation"
    Unlike configuration-space controllers that require $\tau = A(q)^{-1} \cdot (\text{control law})$, the actuation-space PID output $\tau$ is directly the actuator input without any matrix inversion.

::: soromox.control.actuation_space.PIDController
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4
      group_by_category: true
      docstring_section_style: table
      members_order: source

---

### FeedforwardCompensationTracker

Feedforward compensation evaluates the complete inverse dynamics at the *desired* configuration in actuation space. This provides open-loop feedforward plus feedback correction.

The control law is:

$$
\tau_\mathrm{model} = M_y(q_\mathrm{des}) \ddot{y}_\mathrm{des} + \eta_y(q_\mathrm{des}, \dot{q}_\mathrm{des}) \dot{y}_\mathrm{des} + G_y(q_\mathrm{des}) + \tau_{\mathrm{el},y}(q_\mathrm{des}) + D_y(q_\mathrm{des}) \dot{y}_\mathrm{des}
$$

$$
\tau = \tau_\mathrm{model}[:n_a] + \tau_\mathrm{feedback}
$$

where $[:n_a]$ denotes extracting only the first $n_a$ (actuated) rows.

!!! warning "Underactuation"
    In underactuated settings, trajectory tracking is an open research problem. The reference trajectory for unactuated coordinates must be dynamically feasible. The controller will emit a warning when used with underactuated systems.

::: soromox.control.actuation_space.FeedforwardCompensationTracker
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4
      group_by_category: true
      docstring_section_style: table
      members_order: source

---

### MixedStateFeedbackTracker

Mixed state feedback uses a hybrid evaluation strategy: dynamical matrices (inertia, Coriolis, gravitational, damping) are evaluated at the *current* state, while the desired velocities and accelerations are used for computing forces. Elastic forces are evaluated at the *desired* configuration.

The control law is:

$$
\tau_\mathrm{model} = M_y(q) \ddot{y}_\mathrm{des} + \eta_y(q, \dot{q}) \dot{y}_\mathrm{des} + G_y(q) + \tau_{\mathrm{el},y}(q_\mathrm{des}) + D_y(q) \dot{y}_\mathrm{des}
$$

$$
\tau = \tau_\mathrm{model}[:n_a] + \tau_\mathrm{feedback}
$$

This approach often provides better control performance than pure feedforward compensation since it uses the actual state for configuration-dependent matrices.

!!! warning "Coriolis Matrix Requirements"
    For theoretical stability guarantees, the Coriolis matrix $C(q, \dot{q})$ must be derived using Christoffel symbols of the first kind, ensuring that $N = \dot{M} - 2C$ is skew-symmetric.

::: soromox.control.actuation_space.MixedStateFeedbackTracker
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4
      group_by_category: true
      docstring_section_style: table
      members_order: source

---

## Setpoint Regulators

Setpoint regulators are specialized for regulation tasks (constant setpoints) or quasi-static trajectories. They do not consider inertial, Coriolis, or damping forces.

### GravityCancellationRegulator

Gravity cancellation evaluates gravity at the current configuration (for real-time compensation) and elastic forces at the desired configuration (to set the equilibrium), all in actuation space.

The control law is:

$$
\tau_\mathrm{model} = G_y(q) + \tau_{\mathrm{el},y}(q_\mathrm{des})
$$

$$
\tau = \tau_\mathrm{model}[:n_a] + \tau_\mathrm{feedback}
$$

::: soromox.control.actuation_space.GravityCancellationRegulator
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4
      group_by_category: true
      docstring_section_style: table
      members_order: source

---

### PotentialCancellationRegulator

Potential cancellation evaluates both gravitational and elastic forces at the current configuration, providing real-time cancellation of all potential energy forces.

This is closely related to the **PD+ controller** in rigid robotics, which was shown to be globally asymptotically stable by Paden & Panja (1988) and Kelly & Carelli (1996).

The control law is:

$$
\tau_\mathrm{model} = G_y(q) + \tau_{\mathrm{el},y}(q)
$$

$$
\tau = \tau_\mathrm{model}[:n_a] + \tau_\mathrm{feedback}
$$

::: soromox.control.actuation_space.PotentialCancellationRegulator
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4
      group_by_category: true
      docstring_section_style: table
      members_order: source

---

### PotentialCompensationRegulator

Potential compensation (or "potential shaping") evaluates both forces at the *desired* configuration, effectively reshaping the potential energy landscape to have a minimum at the setpoint.

The control law is:

$$
\tau_\mathrm{model} = G_y(q_\mathrm{des}) + \tau_{\mathrm{el},y}(q_\mathrm{des})
$$

$$
\tau = \tau_\mathrm{model}[:n_a] + \tau_\mathrm{feedback}
$$

::: soromox.control.actuation_space.PotentialCompensationRegulator
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
from soromox.control import ReferenceTrajectory
from soromox.control.actuation_space import (
    PIDController,
    MixedStateFeedbackTracker,
    GravityCancellationRegulator,
)
from soromox.control import PIDControl
from soromox.coordinate_transformations import ActuationSpaceDynamics

# Assume `robot` is a soft robot system (e.g., TendonActuatedPlanarPcs)

# Create actuation-space dynamics
asd = ActuationSpaceDynamics(robot=robot)

# Define reference trajectory in configuration space
ts = jnp.linspace(0, 10, 1000)
q_des_ts = jnp.zeros((len(ts), robot.num_dofs))
# ... fill in desired trajectory ...

ref_traj = ReferenceTrajectory(ts=ts, x_des_ts=q_des_ts)

# Create PID control gains (sized for actuated coordinates)
pid = PIDControl(
    Kp=jnp.ones(robot.num_actuators) * 100.0,
    Ki=jnp.ones(robot.num_actuators) * 10.0,
    Kd=jnp.ones(robot.num_actuators) * 20.0,
)

# Option 1: Pure PID in actuation space
controller_pid = PIDController(
    actuation_space_dynamics=asd,
    reference_trajectory=ref_traj,
    pid_control=pid,
    reference_in_configuration_space=True,  # Trajectory is in config space
)

# Option 2: Gravity cancellation regulator (for setpoint regulation)
controller_regulator = GravityCancellationRegulator(
    actuation_space_dynamics=asd,
    reference_trajectory=ref_traj,
    pid_control=pid,
    reference_in_configuration_space=True,
)

# Option 3: Mixed state feedback tracker (for trajectory tracking)
controller_tracker = MixedStateFeedbackTracker(
    actuation_space_dynamics=asd,
    reference_trajectory=ref_traj,
    pid_control=pid,
    reference_in_configuration_space=True,
)
```

---

## References

The actuation-space control formulation was developed in the following foundational works:

- Pustina, P., Della Santina, C., Boyer, F., De Luca, A., & Renda, F. (2024). Input decoupling of Lagrangian systems via coordinate transformation: General characterization and its application to soft robotics. *IEEE Transactions on Robotics*, 40, 2098-2110.

- Pustina, P. (2025). *Analysis and control of the underactuation in continuum soft robots: a kinematic independent approach*. PhD Thesis, Sapienza University of Rome.

For related soft robot control approaches:

- Della Santina, C., Katzschmann, R. K., Bicchi, A., & Rus, D. (2020). Model-based dynamic feedback control of a planar soft robot: trajectory tracking and interaction with the environment. *The International Journal of Robotics Research*, 39(4-5), 490-513.

- Della Santina, C., Duriez, C., & Rus, D. (2023). Model-based control of soft robots: A survey of the state of the art and open challenges. *IEEE Control Systems Magazine*, 43(3), 30-65.

For stability analysis of PD-type controllers:

- Paden, B., & Panja, R. (1988). Globally asymptotically stable 'PD+' controller for robot manipulators. *International Journal of Control*, 47(6), 1697-1712.

- Kelly, R., & Carelli, R. (1996). A class of nonlinear PD-type controllers for robot manipulators. *Journal of Robotic Systems*, 13(12), 793-802.

