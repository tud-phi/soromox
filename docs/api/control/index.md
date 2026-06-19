# Control API

This module provides model-based controllers for soft robots implemented in JAX. The controllers are designed for high-performance simulation and real-time control applications.

## Overview

SoRoMoX includes a comprehensive suite of model-based controllers organized into three control spaces:

- **[Configuration-Space Controllers](configuration-space.md)**: Controllers operating in the robot's generalized coordinates (joint angles, strains, etc.)
- **[Operational-Space Controllers](operational-space.md)**: Controllers operating in task space (end-effector positions, orientations)
- **[Actuation-Space Controllers](actuation-space.md)**: Controllers operating in a transformed coordinate system where the actuation matrix becomes an identity structure—ideal for underactuated robots
- **[Utilities](utilities.md)**: Common components like PID control and reference trajectories

---

## Choosing the Right Control Space

Selecting the appropriate control space is crucial for achieving good performance. The table below summarizes when to use each approach:

| Criterion | Configuration-Space | Operational-Space | Actuation-Space |
|-----------|---------------------|-------------------|-----------------|
| **Best for** | Fully actuated systems with constant, invertible $A$ | Task-space objectives | Underactuated systems, config-dependent $A(q)$ |
| **Reference** | Configurations ($q$) | Task-space poses ($x$) | Configurations ($q$) or actuated coords ($y$) |
| **Underactuation** | Limited | Not supported | Native support |
| **Stability Guarantees** | Well-established (full act., constant $A$) | Challenging | Easy (full act.), local/high-gain (underact.) |

### Configuration-Space Control

**Use when:**

- The trajectory is naturally expressed in generalized coordinates (strains, curvatures, joint angles)
- The robot is **fully actuated** ($n = m$)
- The actuation matrix $A$ is **constant** (configuration-independent) and easily invertible

**Avoid when:**

- The system is underactuated ($m < n$)
- The actuation matrix $A(q)$ is configuration-dependent, non-square, or poorly conditioned
- The control objective is primarily in task space

**Example applications:**

- Shape control of fully-actuated continuum robots
- Strain regulation in soft manipulators with identity/diagonal actuation
- Joint-level position/velocity control

### Operational-Space Control

**Use when:**

- The control objective is naturally expressed in task space (end-effector position, orientation)
- You want to specify **impedance behavior** in Cartesian coordinates
- The robot is **fully actuated** and the task Jacobian is well-conditioned

**Avoid when:**

- Stability guarantees are strictly required
- The task space has kinematic singularities or is ill-conditioned
- You are looking for well-established, thoroughly-analyzed controllers in underactuated settings
- You need direct control over internal DOFs, not just task coordinates

**Example applications:**

- End-effector position tracking
- Compliant manipulation and interaction control
- Cartesian impedance control for safe human-robot interaction

### Actuation-Space Control

**Use when:**

- Most importantly: a mapping from configuration- to actuation-space exists (i.e., integrability condition from (Pustina et al., 2024) is satisfied)
- The robot is **underactuated** ($m < n$)
- The actuation matrix $A(q)$ is configuration-dependent, non-square, or poorly conditioned
- You want to avoid matrix inversions in the control law
- You need clean separation between actuated and unactuated dynamics

**Also works well for:**

- Fully actuated systems with configuration-dependent $A(q)$—stability proofs remain straightforward

**Avoid when:**

- The zero dynamics (unactuated modes) are unstable
- The reference trajectory for unactuated coordinates is not feasible
- You need precise control over all DOFs simultaneously (impossible with underactuation)

**Example applications:**

- Tendon-actuated soft robots with configuration-dependent routing
- Pressure-actuated robots with fewer chambers than DOFs
- Any soft robot where underactuation or complex actuation coupling is intrinsic

!!! info "Zero Dynamics: A Universal Consideration for Underactuation"
    The stability of **zero dynamics** (the dynamics of unactuated DOFs when actuated DOFs are perfectly controlled) is a fundamental concern for **all underactuated systems**, regardless of control space. Whether you use configuration-space, operational-space, or actuation-space controllers, unstable zero dynamics will prevent successful control of the full system. We emphasize this in the [actuation-space documentation](actuation-space.md#zero-dynamics-and-stability) because actuation-space controllers are particularly well-suited for underactuated robots, making the zero dynamics consideration especially relevant.

---

## Control Space Decision Flowchart

```mermaid
graph TD
    A[Start] --> B{Is the objective<br/>in task space?}
    B -->|Yes| C{Is the system<br/>fully actuated?}
    C -->|Yes| D[Operational-Space Control]
    C -->|No| E[Actuation-Space Control<br/>+ task-space mapping]
    B -->|No| F{Is integrability conditioned<br/>satisfied?}
    F -->|Yes| I[Actuation-Space Control]
    F -->|No|  G{Configuration-Space Control}
```

---

## Controller Architecture

Most controllers in SoRoMoX follow a unified architecture that decomposes the control action into two terms:

$$
u = u_\mathrm{model} + u_\mathrm{feedback}
$$

where:

- **Model-based term** ($u_\mathrm{model}$): Computes control inputs using the robot's dynamics model to achieve desired behavior (e.g., inverse dynamics, gravity compensation, potential shaping)
- **Error-based feedback term** ($u_\mathrm{feedback}$): Provides corrective action based on tracking errors (e.g., PID control)

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

1. **Modularity**: Model-based and error-based feedback terms can be designed and tuned independently
2. **Flexibility**: Different feedback strategies (PD, PID, etc.) can be combined with various model-based terms
3. **Theoretical foundation**: Matches the structure used in passivity-based and Lyapunov-based control design
4. **JAX compatibility**: Both terms can be JIT-compiled for high-performance execution

### Example: Feedforward Compensation Tracker

The [Feedforward Compensation Tracker](actuation-space.md#feedforwardcompensationtracker) exemplifies this architecture:

- **Model-based term**: Pure feedforward inverse dynamics evaluated entirely at the *desired* state—no dependence on current $q$ or $\dot{q}$
- **Feedback term**: Standard PID control on tracking error, fully independent of the model

```python
# The model-based term computes (all at desired trajectory):
# tau_model = M(q_des) @ ydd_des + C(q_des, qd_des) @ yd_des 
#           + G(q_des) + tau_el(q_des) + D(q_des) @ yd_des

# The feedback term computes (independent of model):
# tau_fb = PID(e, ed, integral_error)

# Combined:
# u = tau_model + tau_fb
```

This clean separation means the two terms can be designed, tuned, and analyzed independently.

---

## Available Controllers

### Configuration-Space Controllers

| Controller | Type | Model-Based Term | Description |
|------------|------|------------------|-------------|
| [`PIDController`](configuration-space.md#pidcontroller) | Regulator/Tracker | None | Pure PID control in configuration space |
| [`GravityCancellationRegulator`](configuration-space.md#gravitycancellationregulator) | Regulator | $G(q) + \tau_\mathrm{el}(q_\mathrm{des})$ | Real-time gravity compensation |
| [`PotentialCancellationRegulator`](configuration-space.md#potentialcancellationregulator) | Regulator | $G(q) + \tau_\mathrm{el}(q)$ | Full potential cancellation |
| [`PotentialCompensationRegulator`](configuration-space.md#potentialcompensationregulator) | Regulator | $G(q_\mathrm{des}) + \tau_\mathrm{el}(q_\mathrm{des})$ | Potential shaping |
| [`ComputedTorqueTracker`](configuration-space.md#computedtorquetracker) | Tracker | Full inverse dynamics | Exact feedback linearization |
| [`FeedforwardCompensationTracker`](configuration-space.md#feedforwardcompensationtracker) | Tracker | Inverse dynamics at desired state | Open-loop feedforward + feedback |
| [`MixedStateFeedbackTracker`](configuration-space.md#mixedstatefeedbacktracker) | Tracker | Mixed state evaluation | Hybrid feedforward strategy |

### Operational-Space Controllers

| Controller                                                                                                            | Type    | Description                                                              |
|-----------------------------------------------------------------------------------------------------------------------|---------|--------------------------------------------------------------------------|
| [`OperationalSpaceSynergisticController`](operational-space.md#operationalspacesynergisticcontroller)                | Regulator | Synergistic control for under-actuated systems (requires $o = m$)       |
| [`OperationalSpaceImpedanceControlTracker`](operational-space.md#operationalspaceimpedancecontroltracker)            | Tracker | Operational-space impedance control with partial feedback linearization |

### Actuation-Space Controllers

| Controller | Type | Model-Based Term | Description |
|------------|------|------------------|-------------|
| [`PIDController`](actuation-space.md#pidcontroller) | Regulator/Tracker | None | Pure PID control in actuation space |
| [`GravityCancellationRegulator`](actuation-space.md#gravitycancellationregulator) | Regulator | $G_y(q) + \tau_{\mathrm{el},y}(q_\mathrm{des})$ | Real-time gravity compensation |
| [`PotentialCancellationRegulator`](actuation-space.md#potentialcancellationregulator) | Regulator | $G_y(q) + \tau_{\mathrm{el},y}(q)$ | Full potential cancellation |
| [`PotentialCompensationRegulator`](actuation-space.md#potentialcompensationregulator) | Regulator | $G_y(q_\mathrm{des}) + \tau_{\mathrm{el},y}(q_\mathrm{des})$ | Potential shaping |
| [`FeedforwardCompensationTracker`](actuation-space.md#feedforwardcompensationtracker) | Tracker | Full dynamics at $q_\mathrm{des}$ | Open-loop feedforward + feedback |
| [`MixedStateFeedbackTracker`](actuation-space.md#mixedstatefeedbacktracker) | Tracker | Mixed state evaluation | Hybrid feedforward strategy |

---

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

---

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
