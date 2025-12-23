# Configuration-Space Controllers

Configuration-space controllers operate directly in the robot's generalized coordinates (joint angles, strains, curvatures, etc.). These controllers are suitable for applications where the desired trajectory is naturally expressed in configuration space and the actuation matrix is easily invertible (i.e., full actuation) and (preferably) configuration-independent (i.e., constant).

## Overview

All configuration-space controllers in SoRoMoX inherit from [`ClosedFormModelBasedController`](index.md#closedformmodelbasedcontroller-base-class) and decompose the control action into:

$$
u = u_\mathrm{model} + u_\mathrm{feedback}
$$

The controllers differ in how they compute the model-based term:

| Controller | Model-Based Term | Evaluation Point |
|------------|------------------|------------------|
| `PIDController` | None | - |
| `ComputedTorqueTracker` | Full inverse dynamics | Current state (q, q̇) |
| `FeedforwardCompensationTracker` | Full inverse dynamics | Desired state (q_d, q̇_d, q̈_d) |
| `MixedStateFeedbackTracker` | Mixed evaluation | Dynamic matrices at current, elastic at desired |
| `GravityCancellationRegulator` | G(q) + τ_el(q_des) | Gravity at current, elastic at desired |
| `PotentialCancellationRegulator` | G(q) + τ_el(q) | Both at current |
| `PotentialCompensationRegulator` | G(q_des) + τ_el(q_des) | Both at desired |

## Trajectory Trackers

Trajectory trackers are designed for dynamic trajectory tracking where the reference includes position, velocity, and acceleration profiles.

### PIDController

Basic PID controller in configuration space without model-based feedforward.

The control law is:

$$
\tau = K_p e + K_i \int e \, dt + K_d \dot{e}
$$

$$
u = A(q)^T \tau
$$

where \(e = q_d - q\) is the configuration error.

::: soromox.control.configuration_space.PIDController
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4
      group_by_category: true
      docstring_section_style: table
      members_order: source

---

### ComputedTorqueTracker

Computed torque (inverse dynamics) control achieves exact input-output linearization through feedback. This is the recommended choice for fully actuated systems as it provides the strongest theoretical guarantees.

The control law is:

$$
\ddot{q}_\mathrm{ref} = \ddot{q}_d + K_p e + K_i \int e \, dt + K_d \dot{e}
$$

$$
\tau = M(q) \ddot{q}_\mathrm{ref} + C(q, \dot{q}) \dot{q} + G(q) + \tau_\mathrm{el}(q) + D(q) \dot{q}
$$

$$
u = A(q)^{-1} \tau
$$

With perfect model knowledge, the closed-loop error dynamics become linear:

$$
\ddot{e} + K_d \dot{e} + K_p e + K_i \int e \, dt = 0
$$

::: soromox.control.configuration_space.ComputedTorqueTracker
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4
      group_by_category: true
      docstring_section_style: table
      members_order: source

---

### FeedforwardCompensationTracker

Feedforward compensation evaluates the complete inverse dynamics at the *desired* trajectory rather than the current state. This provides open-loop feedforward plus feedback correction.

The control law is:

$$
\tau_\mathrm{model} = M(q_d) \ddot{q}_d + C(q_d, \dot{q}_d) \dot{q}_d + G(q_d) + \tau_\mathrm{el}(q_d) + D(q_d) \dot{q}_d
$$

$$
u = A(q)^{-1} \tau_\mathrm{model} + u_\mathrm{feedback}
$$

::: soromox.control.configuration_space.FeedforwardCompensationTracker
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4
      group_by_category: true
      docstring_section_style: table
      members_order: source

---

### MixedStateFeedbackTracker

Mixed state feedback uses a hybrid evaluation strategy: dynamic matrices (inertia, Coriolis, damping) at the current state, but desired velocities/accelerations for computing forces. Elastic forces are evaluated at the desired configuration.

The control law is:

$$
\tau_\mathrm{model} = M(q) \ddot{q}_d + C(q, \dot{q}) \dot{q}_d + G(q) + \tau_\mathrm{el}(q_d) + D(q) \dot{q}_d
$$

$$
u = A(q)^{-1} \tau_\mathrm{model} + u_\mathrm{feedback}
$$

This approach often provides better control performance than pure feedforward compensation since it uses the actual state for configuration-dependent matrices.

!!! warning "Coriolis Matrix Requirements"
    For theoretical stability guarantees, the Coriolis matrix \(C(q, \dot{q})\) must be derived using Christoffel symbols of the first kind, ensuring that \(N = \dot{M} - 2C\) is skew-symmetric.

::: soromox.control.configuration_space.MixedStateFeedbackTracker
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

Gravity cancellation evaluates gravity at the current configuration (for real-time compensation) and elastic forces at the desired configuration (to set the equilibrium).

The control law is:

$$
\tau_\mathrm{model} = G(q) + \tau_\mathrm{el}(q_d)
$$

$$
u = A(q)^{-1} \tau_\mathrm{model} + u_\mathrm{feedback}
$$

::: soromox.control.configuration_space.GravityCancellationRegulator
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

This is related to the PD+ controller in rigid robotics, which was shown to be globally asymptotically stable by Paden & Panja (1988).

The control law is:

$$
\tau_\mathrm{model} = G(q) + \tau_\mathrm{el}(q)
$$

$$
u = A(q)^{-1} \tau_\mathrm{model} + u_\mathrm{feedback}
$$

::: soromox.control.configuration_space.PotentialCancellationRegulator
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
\tau_\mathrm{model} = G(q_d) + \tau_\mathrm{el}(q_d)
$$

$$
u = A(q)^{-1} \tau_\mathrm{model} + u_\mathrm{feedback}
$$

::: soromox.control.configuration_space.PotentialCompensationRegulator
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4
      group_by_category: true
      docstring_section_style: table
      members_order: source

---

## Actuation Matrix Considerations

!!! warning "Configuration-Dependent Actuation Matrices"
    For full theoretical stability guarantees, most controllers assume the actuation matrix $A(q)$ is configuration-independent (constant). If your system has a configuration-dependent actuation matrix, consider:
    
    1. **ComputedTorqueTracker**: Achieves full feedback linearization regardless of $A(q)$ dependence
    2. **[Actuation-space controllers](actuation-space.md)**: Work directly in actuator coordinates without requiring actuation matrix inversion

The controllers handle different actuation scenarios:

- **Full actuation** ($n = m$, square $A$): Uses matrix inverse $A^{-1}$
- **Overactuation** ($m > n$): Uses pseudo-inverse $A^{\dagger}$ to minimize actuator effort
- **Underactuation** ($m < n$): Only supported via pseudo-inverse $A^{\dagger}$ and transpose $A^T$; Preferably use [actuation-space controllers](actuation-space.md) instead

