# Control Utilities

This page documents utility classes and components used by controllers in SoRoMoX.

## PID Control

The `PIDControl` class provides a flexible PID control implementation that can be used as the error-based feedback term in model-based controllers.

### Features

- **Flexible gain specification**: Scalar, diagonal vector, or full matrix gains
- **Anti-windup**: Optional saturation functions for integral error
- **JAX-compatible**: Fully jittable for high-performance execution

### Control Law

The PID control law is:

$$
u = K_p e + K_i \int_0^t e(\tau) d\tau + K_d \dot{e}
$$

The integral error dynamics include optional saturation for anti-windup:

$$
\frac{d}{dt}\left(\int e \, dt\right) = \mathrm{sat}(e)
$$

For the built-in hyperbolic-tangent saturation, `gamma` is an inverse error
scale. The componentwise scalar/vector form is

$$
\mathrm{sat}_{\gamma}(e)
= \frac{\tanh(\gamma e)}{\gamma}
= e_{\mathrm{sat}}\tanh\!\left(\frac{e}{e_{\mathrm{sat}}}\right),
\qquad e_{\mathrm{sat}}=\frac{1}{\gamma}.
$$

This keeps the saturation argument dimensionless, preserves the units of the
integrated error, and has unit slope at the origin. A vector `gamma` can assign
different physical error scales to heterogeneous coordinates. Scalar and
vector values of `gamma` must be strictly positive; a matrix `gamma` must be
symmetric positive definite.

### PIDControl

::: soromox.control.PIDControl
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4
      group_by_category: true
      docstring_section_style: table
      members_order: source

### PIDControllerState

Container for PID controller state (integral error).

::: soromox.control.PIDControllerState
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4
      group_by_category: true
      docstring_section_style: table
      members_order: source

### Usage Example

```python
import jax.numpy as jnp
from soromox.control import PIDControl, PIDControllerState

# Create PID controller with diagonal gains
pid = PIDControl(
    Kp=jnp.array([100.0, 100.0]),  # Proportional gains
    Ki=jnp.array([10.0, 10.0]),    # Integral gains
    Kd=jnp.array([20.0, 20.0]),    # Derivative gains
    saturation_fn="tanh",          # Anti-windup saturation
    gamma=1.0,                     # Inverse saturation-error scale
)

# Initialize controller state
control_state = PIDControllerState.zero(num_dofs=2)

# Compute control output
e = q_des - q          # Position error
ed = qd_des - qd       # Velocity error
u, integral_error_dot = pid(e, ed, control_state.integral_error)
```

---

## Reference Trajectory

The `ReferenceTrajectory` class provides a flexible way to specify desired trajectories for controllers. It supports both discrete time-series data and continuous functions, automatically deriving velocities and accelerations when not provided.

### Features

- **Dual representation**: Discrete time series and continuous functions
- **Auto-derivation**: Automatically computes velocity and acceleration using JAX autodiff
- **Linear interpolation**: Creates smooth continuous trajectories from discrete data
- **JAX PyTree**: Compatible with JAX transformations and ODE solvers

### ReferenceTrajectory

::: soromox.control.ReferenceTrajectory
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4
      group_by_category: true
      docstring_section_style: table
      members_order: source

### Usage Examples

#### From Discrete Time Series

```python
import jax.numpy as jnp
from soromox.control import ReferenceTrajectory

# Time array
ts = jnp.linspace(0, 10, 1000)

# Desired positions at each time step
x_des_ts = jnp.column_stack([
    jnp.sin(ts),           # First DOF
    jnp.cos(ts),           # Second DOF
])

# Create trajectory - velocities and accelerations auto-derived
ref_traj = ReferenceTrajectory(ts=ts, x_des_ts=x_des_ts)

# Access continuous function (interpolated)
x_at_5 = ref_traj.x_des_fn(5.0)
xd_at_5 = ref_traj.xd_des_fn(5.0)
xdd_at_5 = ref_traj.xdd_des_fn(5.0)
```

#### From Continuous Function

```python
import jax.numpy as jnp
from soromox.control import ReferenceTrajectory

# Define trajectory as a function
def x_des_fn(t):
    return jnp.array([jnp.sin(t), jnp.cos(t)])

# Time array for discrete evaluation
ts = jnp.linspace(0, 10, 1000)

# Create trajectory - discrete arrays and derivatives auto-computed
ref_traj = ReferenceTrajectory(ts=ts, x_des_fn=x_des_fn)

# Access discrete arrays
x_des_ts = ref_traj.x_des_ts    # Shape: (1000, 2)
xd_des_ts = ref_traj.xd_des_ts  # Auto-derived via JAX autodiff
```

#### Constant Setpoint (Regulation)

```python
import jax.numpy as jnp
from soromox.control import ReferenceTrajectory

# For regulation, create a constant trajectory
q_setpoint = jnp.array([0.5, 0.3])

ts = jnp.linspace(0, 10, 100)
x_des_ts = jnp.tile(q_setpoint, (len(ts), 1))

ref_traj = ReferenceTrajectory(ts=ts, x_des_ts=x_des_ts)
# xd_des and xdd_des will be (approximately) zero
```
