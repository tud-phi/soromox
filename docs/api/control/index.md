# Control API

This module provides model-based controllers for soft robots implemented in JAX. The controllers are designed for high-performance simulation and real-time control applications.

## Overview

SoRoMoX includes a comprehensive suite of model-based controllers organized into:

- **[Configuration-Space Controllers](configuration-space.md)**: Controllers operating in the robot's generalized coordinates (joint angles, strains, etc.)
- **[Operational-Space Controllers](operational-space.md)**: Controllers operating in task space (end-effector positions, orientations)
- **[Utilities](utilities.md)**: Common components like PID control and reference trajectories

!!! note "Future Development"
    In addition to the `configuration_space` and `operational_space` controllers, future releases will include `actuation_space` controllers that operate directly in the actuator space (e.g., tendon tensions, pressures).

## Controller Architecture

Most controllers in SoRoMoX follow a unified architecture that decomposes the control action into two terms:

$$
u = u_\mathrm{model} + u_\mathrm{feedback}
$$

where:

- **Model-based term** (\(u_\mathrm{model}\)): Computes control inputs using the robot's dynamics model to achieve desired behavior (e.g., inverse dynamics, gravity compensation, potential shaping)
- **Error-based feedback term** (\(u_\mathrm{feedback}\)): Provides corrective action based on tracking errors (e.g., PID control)

This decomposition is implemented in the [`ClosedFormModelBasedController`](#closedformmodelbasedcontroller-base-class) base class, which provides a clean interface for implementing new controllers:

```python
class MyController(ClosedFormModelBasedController):
    def model_based_term(self, system_state):
        # Compute feedforward control based on dynamics model
        # e.g., inverse dynamics, gravity compensation
        ...
        return u_model, None
    
    def error_based_feedback_term(self, system_state):
        # Compute feedback control based on tracking error
        # e.g., PID control
        ...
        return u_feedback, control_state_dot
```

### Benefits of This Architecture

1. **Modularity**: Model-based and feedback terms can be designed and tuned independently
2. **Flexibility**: Different feedback strategies (PD, PID, etc.) can be combined with various model-based terms
3. **Theoretical foundation**: Matches the structure used in passivity-based and Lyapunov-based control design
4. **JAX compatibility**: Both terms can be JIT-compiled for high-performance execution

### Example: Computed Torque Control

The [Computed Torque Tracker](configuration-space.md#computedtorquetracker) exemplifies this architecture:

- **Model-based term**: Full inverse dynamics to cancel all forces and achieve desired acceleration
- **Feedback term**: PID control on tracking error, mapped through inertia matrix

```python
# The model-based term computes:
# tau_model = M(q) @ qdd_des + C(q, qd) @ qd + G(q) + tau_el(q) + D(q) @ qd

# The feedback term computes:
# tau_fb = M(q) @ PID(e, ed, integral_error)

# Combined:
# u = A^{-1}(q) @ (tau_model + tau_fb)
```

## Available Controllers

### Configuration-Space Controllers

| Controller | Type | Model-Based Term | Description |
|------------|------|------------------|-------------|
| [`PIDController`](configuration-space.md#pidcontroller) | Tracker | None | Pure PID control in configuration space |
| [`ComputedTorqueTracker`](configuration-space.md#computedtorquetracker) | Tracker | Full inverse dynamics | Exact feedback linearization |
| [`FeedforwardCompensationTracker`](configuration-space.md#feedforwardcompensationtracker) | Tracker | Inverse dynamics at desired state | Open-loop feedforward + feedback |
| [`MixedStateFeedbackTracker`](configuration-space.md#mixedstatefeedbacktracker) | Tracker | Mixed state evaluation | Hybrid feedforward strategy |
| [`GravityCancellationRegulator`](configuration-space.md#gravitycancellationregulator) | Regulator | G(q) + τ_el(q_des) | Real-time gravity compensation |
| [`PotentialCancellationRegulator`](configuration-space.md#potentialcancellationregulator) | Regulator | G(q) + τ_el(q) | Full potential cancellation |
| [`PotentialCompensationRegulator`](configuration-space.md#potentialcompensationregulator) | Regulator | G(q_des) + τ_el(q_des) | Potential shaping |

### Operational-Space Controllers

| Controller | Type | Description |
|------------|------|-------------|
| [`ImpedanceControlTracker`](operational-space.md#impedancecontroltracker) | Tracker | Operational-space impedance control with partial feedback linearization |

## References and Citation

!!! info "PhD Thesis Reference"
    An overview of the theory and implementation of (most of) these model-based controllers for soft robots can be found in **Chapter II** of the following PhD thesis:
    
    > Stölzle, M. (2025). *Safe yet Precise Soft Robots: Incorporating Physics into Learned Models for Control*. Dissertation, Delft University of Technology.  
    > [https://doi.org/10.4233/uuid:24c1f667-8fd6-431a-bb78-11d22f8cb3da](https://doi.org/10.4233/uuid:24c1f667-8fd6-431a-bb78-11d22f8cb3da)

!!! tip "Citation Request"
    If you find this implementation of model-based controllers useful for your research, please consider citing the following work:

    ```bibtex
    @phdthesis{stolzle2025phdthesis,
      title = "Safe yet Precise Soft Robots: Incorporating Physics into Learned Models for Control",
      keywords = "Soft Robotics, Nonlinear Control, Machine Learning, Artificial Intelligence",
      author = "Maximilian St{\"o}lzle",
      year = "2025",
      month = "9",
      day = "15",
      language = "English",
      type = "Dissertation (TU Delft)",
      school = "Mechanical Engineering, Delft University of Technology",
      doi = "10.4233/uuid:24c1f667-8fd6-431a-bb78-11d22f8cb3da",
      isbn = "978-94-6384-836-7",
    }
    ```

## Base Classes

### BaseController

The abstract base class for all controllers.

::: soromox.control.BaseController
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source

### ClosedFormModelBasedController Base Class

This base class implements the model-based + feedback architecture described above. Subclasses override `model_based_term()` and/or `error_based_feedback_term()` to implement specific control strategies.

::: soromox.control.ClosedFormModelBasedController
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source

