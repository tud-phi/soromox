# Extending SoRoMoX

This guide explains how to extend SoRoMoX by implementing custom robot systems and renderers.

## Adding Custom Systems

### Understanding the Base Classes

SoRoMoX uses a two-level inheritance hierarchy for robot systems:

```text
DynamicalSystem (abstract)
└── SoftRobot (abstract)
    ├── PCS, PlanarPCS, ...
    ├── GVS, ...
    ├── Pendulum, ...
    └── YourCustomSystem
```

#### DynamicalSystem

The base class for all dynamical systems. Key interface:

```python
from soromox.systems import DynamicalSystem

class DynamicalSystem:
    # Static attributes (set at creation, used for JIT compilation)
    num_dofs: int       # Degrees of freedom
    num_actuators: int  # Number of control inputs

    # Core methods
    def forward_dynamics(self, t, y, tau_ext, actuation_args):
        """Compute state derivatives yd = f(t, y, tau_ext, u)."""
        ...

    def rollout_to(self, ts, y0, actuation_fn, ...):
        """Simulate open-loop trajectory."""
        ...

    def rollout_closed_loop_to(self, ts, y0, controller, ...):
        """Simulate closed-loop trajectory."""
        ...
```

#### SoftRobot

Extends `DynamicalSystem` with soft robot-specific methods:

```python
from soromox.systems import SoftRobot

class SoftRobot(DynamicalSystem):
    # Properties
    @property
    def length(self) -> float:
        """Total backbone length."""
        ...

    @property
    def is_planar(self) -> bool:
        """True for SE(2) robots, False for SE(3)."""
        ...

    # Kinematics (arc-length parameterized)
    def forward_kinematics(self, q, s):
        """Pose at arc length s along the backbone."""
        ...

    def jacobian(self, q, s):
        """Jacobian at arc length s."""
        ...

    # Dynamics
    def inertia_matrix(self, q):
        """Mass/inertia matrix M(q)."""
        ...

    def coriolis_matrix(self, q, qd):
        """Coriolis/centrifugal matrix C(q, qd)."""
        ...

    def elastic_force(self, q):
        """Elastic restoring force."""
        ...

    def gravitational_force(self, q):
        """Gravitational force."""
        ...
```

### Step-by-Step: Implementing a New System

#### 1. Choose Your Base Class

- **Extend `SoftRobot`** for continuum or articulated soft robots
- **Extend an existing system** (e.g., `PCS`, `PlanarPCS`) for variants

#### 2. Define Your Class

```python
import equinox as eqx
import jax.numpy as jnp
from soromox.systems import SoftRobot

class MyCustomRobot(SoftRobot):
    # Static fields (compile-time constants)
    num_segments: int = eqx.field(static=True)

    # Array fields (dynamic parameters)
    L: jax.Array  # Segment lengths

    def __init__(self, num_segments: int, L: jax.Array, ...):
        self.num_segments = num_segments
        self.L = L
        # Initialize other parameters...
```

#### 3. Implement Required Properties

```python
@property
def num_dofs(self) -> int:
    """Number of degrees of freedom."""
    return self.num_segments * 3  # Example: 3 DOFs per segment

@property
def num_actuators(self) -> int:
    """Number of control inputs."""
    return self.num_segments * 2  # Example: 2 actuators per segment

@property
def length(self) -> float:
    """Total backbone length."""
    return float(jnp.sum(self.L))

@property
def is_planar(self) -> bool:
    """True for 2D robots."""
    return True  # or False for 3D
```

#### 4. Implement Kinematics

Soft robots use arc-length parametrization for kinematics:

```python
def forward_kinematics(self, q: jax.Array, s: float) -> jax.Array:
    """
    Compute pose at arc length s along the backbone.

    Args:
        q: Configuration vector (num_dofs,) - strains or joint angles
        s: Arc length along backbone [0, length]

    Returns:
        Pose: [theta, x, y] for 2D or 4x4 SE(3) matrix for 3D
    """
    # For PCS-style robots: integrate constant strains
    # For other models: use appropriate kinematic model
    ...

def jacobian(self, q: jax.Array, s: float) -> jax.Array:
    """
    Compute Jacobian at arc length s.

    Args:
        q: Configuration vector
        s: Arc length

    Returns:
        Jacobian matrix J(q,s) mapping qd to spatial velocity
        Shape: (3, num_dofs) for 2D or (6, num_dofs) for 3D
    """
    # Typically computed via automatic differentiation
    # or analytical derivation
    ...

def jacobian_and_time_derivative(
    self, q: jax.Array, qd: jax.Array, s: float
) -> tuple[jax.Array, jax.Array]:
    """
    Compute Jacobian and its time derivative at arc length s.

    This is useful for operational-space control and analysis.

    Args:
        q: Configuration vector
        qd: Configuration velocity
        s: Arc length

    Returns:
        J: Jacobian matrix
        Jd: Time derivative of Jacobian
    """
    # Can use JAX automatic differentiation:
    # J = self.jacobian(q, s)
    # Jd = jax.jacfwd(lambda q: self.jacobian(q, s))(q) @ qd
    ...
```

#### 5. Implement Dynamics

Soft robot dynamics follow the equation: **M(q)q̈ + C(q,q̇)q̇ + τ_elastic(q) + τ_gravity(q) + D(q)q̇ = τ_actuation(q,u) + τ_external**

```python
def inertia_matrix(self, q: jax.Array) -> jax.Array:
    """
    Mass/inertia matrix M(q) for the soft robot.

    For continuum robots, typically computed by integrating
    along the backbone using the mass density distribution.

    Returns:
        M: Inertia matrix (num_dofs, num_dofs)
    """
    # Example: Integrate using quadrature
    # M = sum over integration points of (m_i * J_i^T @ J_i)
    ...

def coriolis_matrix(self, q: jax.Array, qd: jax.Array) -> jax.Array:
    """
    Coriolis/centrifugal matrix C(q, qd).

    Should use the Christoffel/energy-consistent convention associated
    with inertia_matrix(q): C(q, qd) @ qd is the convective force and
    M_dot(q, qd) - 2 * C(q, qd) is skew-symmetric.

    Returns:
        C: Coriolis matrix (num_dofs, num_dofs)
    """
    ...

def elastic_force(self, q: jax.Array) -> jax.Array:
    """
    Elastic restoring force for soft robot.

    Typically linear in strains: tau_el = K @ (q - q_ref)
    where K is the stiffness matrix.

    Returns:
        tau_el: Elastic force vector (num_dofs,)
    """
    ...

def gravitational_force(self, q: jax.Array) -> jax.Array:
    """
    Gravitational force on soft robot.

    Computed by integrating gravity effect along backbone:
    tau_g = integral of (J^T @ [0, 0, m*g])

    Returns:
        tau_g: Gravitational force vector (num_dofs,)
    """
    ...

def damping_matrix(self, q: jax.Array) -> jax.Array:
    """
    Damping matrix D(q) for soft robot (optional).

    Can be configuration-dependent for complex materials.
    Often constant for simple models.

    Returns:
        D: Damping matrix (num_dofs, num_dofs)
    """
    ...

def actuation_matrix(self, q: jax.Array) -> jax.Array:
    """
    Actuation matrix A(q) mapping control inputs to forces.

    For tendon-actuated: based on tendon routing geometry
    For pressure chambers: based on chamber geometry and pressure

    Returns:
        A: Actuation matrix (num_dofs, num_actuators)
    """
    ...

def forward_dynamics(self, t, y, tau_ext, actuation_args):
    """
    State derivatives for simulation: yd = [qd, qdd]

    Implements the equation of motion for soft robots.
    """
    q, qd = jnp.split(y, 2)
    u = actuation_args.get("u", jnp.zeros(self.num_actuators))

    # Dynamic matrices
    M = self.inertia_matrix(q)
    C = self.coriolis_matrix(q, qd)
    D = self.damping_matrix(q)

    # Forces
    tau_el = self.elastic_force(q)
    tau_g = self.gravitational_force(q)
    A = self.actuation_matrix(q)
    tau_act = A @ u

    # Solve for acceleration: M qdd = tau_total
    tau_total = tau_act + tau_ext - C @ qd - D @ qd - tau_el - tau_g
    qdd = jnp.linalg.solve(M, tau_total)

    return jnp.concatenate([qd, qdd])
```

**Energy and control consistency:** `coriolis_matrix(q, qd)` must be consistent
with the same `inertia_matrix(q)` used by the model dynamics. In the standard
Christoffel convention, `C` itself is not generally skew-symmetric; instead
`M_dot(q, qd) - 2 * C(q, qd)` is skew-symmetric. This structure is important for
the mechanical energy balance, passivity arguments, and model-based control. It
also keeps derivative-based APIs consistent. If you override an optimized
`dynamics_terms(q, qd)` method, its returned `Cqd` vector must equal the
convective force `C(q, qd) @ qd` for this same convention.

#### 6. Implement Energy Methods (Optional but Recommended)

Energy methods are useful for analysis and energy-based control:

```python
def kinetic_energy(self, q: jax.Array, qd: jax.Array) -> jax.Array:
    """
    Compute kinetic energy: T = 0.5 * qd^T @ M(q) @ qd

    Returns:
        T: Kinetic energy (scalar)
    """
    M = self.inertia_matrix(q)
    return 0.5 * qd.T @ M @ qd

def elastic_energy(self, q: jax.Array) -> jax.Array:
    """
    Compute elastic potential energy stored in deformation.

    For linear elasticity: U_elastic = 0.5 * q^T @ K @ q

    Returns:
        U_el Elastic potential energy (scalar)
    """
    K = self.stiffness_matrix()
    return 0.5 * q.T @ K @ q

def gravitational_energy(self, q: jax.Array) -> jax.Array:
    """
    Compute gravitational potential energy.

    Computed by integrating along backbone:
    U_g = integral of (m * g * h(s, q))

    Returns:
        U_g: Gravitational potential energy (scalar)
    """
    ...

def potential_energy(self, q: jax.Array) -> jax.Array:
    """
    Total potential energy: U = U_elastic + U_gravity

    Useful for energy-based control and analysis.

    Returns:
        U: Total potential energy (scalar)
    """
    U_el = self.elastic_energy(q)
    U_g = self.gravitational_energy(q)
    U = U_el + U_g
    return U

def total_energy(self, q: jax.Array, qd: jax.Array) -> jax.Array:
    """
    Total mechanical energy: E = T + U_elastic + U_gravity

    Useful for energy-based control and passivity analysis.

    Returns:
        E: Total energy (scalar)
    """
    T = self.kinetic_energy(q, qd)
    U = self.potential_energy(q)
    E = T + U
    return E
```

#### 7. Autodiff and analytical custom JVPs

JAX makes it natural for users to differentiate through model methods directly.
Optimization, control, and sensitivity code can be written against the public
SoRoMoX API:

```python
import jax
import jax.numpy as jnp

# Energy gradients
G = jax.grad(lambda q: robot.gravitational_energy(q))(q)
K = jax.grad(lambda q: robot.elastic_energy(q))(q)
dE_dq, dE_dqd = jax.grad(
    lambda q, qd: robot.total_energy(q, qd),
    argnums=(0, 1),
)(q, qd)

# Forward-kinematics directional derivatives
pose, posed = jax.jvp(
    lambda q: robot.forward_kinematics(q, s),
    (q,),
    (qd,),
)

# Jacobian directional derivatives
J, Jd = jax.jvp(
    lambda q: robot.jacobian(q, s),
    (q,),
    (qd,),
)
```

This user-facing code is convenient, but plain automatic differentiation is not
always the most efficient implementation. Many soft-robot quantities have
closed-form derivatives:

- `grad(gravitational_energy)` is the gravitational generalized force.
- `grad(elastic_energy)` is the elastic generalized force.
- A `q`-directional JVP of `kinetic_energy(q, qd)` uses the same
  Christoffel/energy-consistent convective force convention as the model
  dynamics.
- A `q`-directional JVP of `jacobian(q, s)` is the Jacobian time derivative
  `dJ/dq @ qd`.
- An `s`-directional JVP is an arc-length derivative along the backbone.

SoRoMoX uses JAX `custom_jvp` rules on the public `SoftRobot` methods to keep the
user-facing code simple while selecting analytical derivatives internally when a
system provides them. In other words, users can still call `jax.grad`,
`jax.jvp`, `jax.jacfwd`, or `jax.jacrev` on the public methods; the library
routes the derivative through the best available implementation.

The base-class pattern looks like this for `jacobian`. The public method stays
ordinary Python, while the static custom-JVP entry point delegates primal
evaluation and tangent evaluation to protected hooks:

```python
import equinox as eqx
import jax
import jax.numpy as jnp

from soromox.autodiff import custom_jvp_enabled


class SoftRobot:
    def jacobian(self, q, s):
        if custom_jvp_enabled():
            return SoftRobot._jacobian_custom_jvp(self, q, s)
        return self._jacobian(q, s)

    @eqx.filter_custom_jvp
    @staticmethod
    def _jacobian_custom_jvp(robot, q, s):
        return robot._jacobian(q, s)

    @_jacobian_custom_jvp.def_jvp
    def _jacobian_custom_jvp_jvp(primals, tangents):
        robot, q, s = primals
        _, qd, sd = tangents

        if qd is None and sd is None:
            J = robot._jacobian(q, s)
            return J, jnp.zeros_like(J)

        if qd is None:
            J, Js = robot.jacobian_and_arc_length_derivative(q, s)
            return J, Js * sd

        if sd is None:
            return robot.jacobian_and_time_derivative(q, qd, s)

        return jax.jvp(
            lambda q_, s_: robot._jacobian(q_, s_),
            (q, s),
            (qd, sd),
        )
```

The first two derivative branches use analytical hooks. The final mixed branch
falls back to autodiff through `_jacobian`, because that path has not justified a
separate analytical custom-JVP rule in benchmarks.

As an implementer, define analytical protected hooks rather than writing custom
JVP rules in each system class. For example, a spatial PCS-like system can
provide closed-form arc-length kinematics and energy gradients:

```python
from soromox.utils.lie_algebra import se3, so3


def _angular_cross_matrix(self, xi_i):
    return so3.skew(xi_i[:3])


def _forward_kinematics_arc_length_derivative(self, q, s):
    i_segment, _ = self.classify_segment(s)
    xi_i = self.segment_strain(q, i_segment)
    g = self._forward_kinematics(q, s)
    return g @ se3.hat(xi_i)


def _jacobian_and_time_derivative(self, q, qd, s):
    J = self._jacobian(q, s)
    Jd = self._jacobian_time_derivative_recurrence(q, qd, s)
    return J, Jd


def _gravitational_force(self, q):
    return self._integrate_gravitational_generalized_force(q)
```

Planar systems follow the same pattern, but return derivatives in the planar pose
representation:

```python
import jax.numpy as jnp


def _forward_kinematics_arc_length_derivative(self, q, s):
    i_segment, _ = self.classify_segment(s)
    kappa_i, sigma_i = self.segment_strain(q, i_segment)
    theta, _, _ = self._forward_kinematics(q, s)
    R = jnp.array(
        [
            [jnp.cos(theta), -jnp.sin(theta)],
            [jnp.sin(theta), jnp.cos(theta)],
        ]
    )
    return jnp.concatenate([jnp.atleast_1d(kappa_i), R @ sigma_i])
```

For the Jacobian custom JVP, the dispatch is split by tangent type. A
configuration-only tangent uses `jacobian_and_time_derivative(q, qd, s)`. An
arc-length-only tangent uses `jacobian_and_arc_length_derivative(q, s)`. A mixed
`q` and `s` tangent falls back to JAX autodiff through `_jacobian`; higher-order
benchmarks showed that maintaining a fused mixed custom-JVP path was not worth
the additional code complexity.

For forward kinematics, configuration and mixed tangents use JAX autodiff
through `_forward_kinematics`, while arc-length-only tangents use
`forward_kinematics_and_arc_length_derivative(q, s)` so systems can still share
the primal pose computation with an analytical `d pose / ds`.

When implementing a new system, prefer overriding the protected hooks below and
let the public methods on `SoftRobot` provide the custom-JVP wrappers:

| Hook | Purpose |
|------|---------|
| `_forward_kinematics(q, s)` | Primal pose computation |
| `_forward_kinematics_arc_length_derivative(q, s)` | Analytical `d pose / ds` |
| `_forward_kinematics_and_arc_length_derivative(q, s)` | Optional fused pose and `d pose / ds` |
| `_jacobian(q, s)` | Primal Jacobian computation |
| `_jacobian_and_time_derivative(q, qd, s)` | Analytical `J` and `dJ/dq @ qd` |
| `_jacobian_and_arc_length_derivative(q, s)` | Optional fused `J` and `dJ/ds` |
| `_gravitational_energy(q)` | Primal gravitational potential energy |
| `_gravitational_force(q)` | Analytical gradient of gravitational energy |
| `_elastic_energy(q)` / `elastic_force(q)` | Primal elastic energy and analytical gradient |

If a hook is not available, the base class falls back to JAX autodiff through the
protected primal methods. Analytical hooks are therefore an optimization and a
numerical robustness tool, not a different public API.

Avoid exact-class checks such as `type(self) is PCS` in these hooks. Subclasses
often share the same kinematics and should inherit the same protected-hook
behavior. If a subclass changes the kinematic model, override the affected hook
in that subclass.

Custom JVPs are enabled by default. They can be disabled globally before tracing
JAX code:

```python
from soromox import custom_jvp_mode

with custom_jvp_mode(False):
    J, Jd = jax.jvp(lambda q: robot.jacobian(q, s), (q,), (qd,))
```

#### 8. Register as JAX PyTree (if using Equinox)

Equinox modules are automatically registered as JAX PyTrees. Ensure:

- Static fields use `eqx.field(static=True)`
- Dynamic fields are JAX arrays

### Required Methods Reference

| Method | Required | Description |
|--------|----------|-------------|
| **Kinematics** | | |
| `forward_kinematics(q, s)` | Yes | Pose at arc length s |
| `jacobian(q, s)` | Yes | Jacobian at arc length s |
| `jacobian_and_time_derivative(q, qd, s)` | Recommended | Jacobian and time derivative |
| **Dynamics** | | |
| `inertia_matrix(q)` | Yes | Mass/inertia matrix M(q) |
| `coriolis_matrix(q, qd)` | Yes | Coriolis matrix C(q,qd) |
| `elastic_force(q)` | Yes | Elastic restoring force |
| `gravitational_force(q)` | Yes | Gravitational force |
| `damping_matrix(q)` | Recommended | Damping matrix D(q) |
| `actuation_matrix(q)` | Yes | Actuation mapping A(q) |
| `forward_dynamics(...)` | Yes | State derivatives |
| **Energy Methods** | | |
| `kinetic_energy(q, qd)` | Recommended | Kinetic energy T |
| `elastic_energy(q)` | Recommended | Elastic potential energy |
| `gravitational_energy(q)` | Recommended | Gravitational potential energy |
| `potential_energy(q)` | Recommended | Total potential energy |
| `total_energy(q, qd)` | Recommended | Total mechanical energy |
| **Optional** | | |
| `stiffness_matrix()` | Optional | Linear stiffness matrix K |

---

## Adding Custom Renderers

### Understanding BaseSoftRobotRenderer

All renderers inherit from `BaseSoftRobotRenderer`:

```text
BaseSoftRobotRenderer (abstract)
├── MatplotlibRenderer
├── Open3DRenderer
├── ViserRenderer
├── OpenCVPlanarRenderer
└── YourCustomRenderer
```

### Base Class Interface

```python
from soromox.rendering import BaseSoftRobotRenderer

class BaseSoftRobotRenderer:
    # Attributes
    robot: SoftRobot           # The robot to render
    width: int                 # Image width
    height: int                # Image height
    num_points: int            # Backbone discretization points
    color_config: RendererColorConfig

    # Helper methods (already implemented)
    def compute_backbone_curve(self, q):
        """Compute backbone points from configuration."""
        ...

    def compute_actuator_visual_layers(self, q):
        """Compute actuator visual layers if supported."""
        ...

    # Abstract methods (must implement)
    def render_frame(self, q, **kwargs) -> np.ndarray:
        """Render single frame to image array."""
        ...

    def show(self, q, **kwargs) -> None:
        """Display single frame interactively."""
        ...

    def render_sequence(self, ts, q_ts, **kwargs) -> None:
        """Render animated sequence."""
        ...
```

### Step-by-Step: Implementing a New Renderer

#### 1. Create Your Class

```python
import numpy as np
from soromox.rendering import BaseSoftRobotRenderer, RendererColorConfig
from soromox.systems import SoftRobot

class MyCustomRenderer(BaseSoftRobotRenderer):
    def __init__(
        self,
        robot: SoftRobot,
        width: int = 800,
        height: int = 600,
        num_points: int = 50,
        color_config: RendererColorConfig = None,
        # Your custom parameters
        my_param: float = 1.0,
    ):
        super().__init__(
            robot=robot,
            width=width,
            height=height,
            num_points=num_points,
            color_config=color_config,
        )
        self.my_param = my_param
```

#### 2. Implement render_frame

```python
def render_frame(
    self,
    q: np.ndarray,
    color_config: RendererColorConfig = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    Render a single frame.

    Args:
        q: Configuration vector or batch (N, DOF)
        color_config: Optional color override

    Returns:
        RGB image array (height, width, 3) uint8
    """
    # Use helper to get backbone curve
    backbone_points = self.compute_backbone_curve(q)

    # Create your visualization
    image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    # Draw the robot...
    # Your rendering logic here

    return image
```

#### 3. Implement show

```python
def show(self, q: np.ndarray, **kwargs) -> None:
    """Display single frame interactively."""
    image = self.render_frame(q, **kwargs)

    # Display using your preferred method
    # e.g., matplotlib, OpenCV, PIL, etc.
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.show()
```

#### 4. Implement render_sequence

```python
def render_sequence(
    self,
    ts: np.ndarray,
    q_ts: np.ndarray,
    record_path: str = None,
    **kwargs: Any,
) -> None:
    """
    Render animated sequence.

    Args:
        ts: Time points (T,)
        q_ts: Configurations (T, DOF) or (N, T, DOF)
        record_path: Optional path to save video
    """
    for i, (t, q) in enumerate(zip(ts, q_ts)):
        frame = self.render_frame(q, **kwargs)

        # Display or record frame
        # Your animation logic here

    if record_path:
        # Save video
        ...
```

### Using Helper Methods

The base class provides useful helpers:

```python
# Get backbone curve points
backbone = self.compute_backbone_curve(q)
# Returns: (num_points, 2) for 2D or (num_points, 3) for 3D

# Get actuator visual layers (if robot exposes actuator_visual_layers)
actuator_layers = self.compute_actuator_visual_layers(q)
# Returns: tuple of ActuatorVisualLayer objects

# Batched and trajectory actuator visual helpers use optional robot fast paths
# named actuator_visual_layers_batched(...) and
# actuator_visual_layers_trajectory(...) when available. Otherwise they fall
# back to repeated calls to the single-configuration actuator_visual_layers(...).

# Resolve colors with hierarchy
colors = self.resolve_backbone_colors(num_robots=1, color_config=color_config)

# Check robot dimension
if self.is_planar:
    # 2D rendering
else:
    # 3D rendering
```

### Best Practices

1. **Support batched input**: Handle both single configs `(DOF,)` and batches `(N, DOF)`

2. **Use color configuration**: Respect the `color_config` parameter and hierarchy

3. **Support recording**: Implement `record_path` parameter for video export

4. **Handle both 2D and 3D**: Check `self.is_planar` for dimension-appropriate rendering

5. **Cache expensive computations**: Use the base class caching mechanisms

---

## Adding Custom Controllers

### Understanding the Controller Hierarchy

SoRoMoX provides two base classes for implementing controllers:

```text
BaseController (abstract)
└── ClosedFormModelBasedController (abstract)
    ├── Configuration-space controllers
    ├── Operational-space controllers
    └── Actuation-space controllers
```

### BaseController Interface

The `BaseController` class provides the fundamental interface for all controllers:

```python
from soromox.control import BaseController
from soromox.systems import SystemState

class BaseController(eqx.Module):
    """
    Abstract base class for controllers.

    Controllers compute actuation inputs based on the current system state
    and a reference trajectory.
    """

    robot: SoftRobot
    reference_trajectory: ReferenceTrajectory

    def __call__(
        self, system_state: SystemState
    ) -> tuple[Array, Any | None]:
        """
        Compute control action given the current system state.

        Args:
            system_state: Current state of the system containing:
                - t (Array): Current simulation time
                - y (Array): Robot state [q, qd] concatenated
                - u (Optional[Array]): Previous actuation input
                - control_state (Optional[Any]): Internal controller state

        Returns:
            u_control (Array): Control input to apply, shape (num_actuators,)
            control_state_dot (Optional[Any]): Time derivative of internal
                controller state (e.g., integrator error), or None if stateless
        """
        raise NotImplementedError
```

### ClosedFormModelBasedController Interface

For model-based controllers that decompose control into feedforward and feedback terms:

```python
from soromox.control import ClosedFormModelBasedController
from soromox.systems import SystemState

class ClosedFormModelBasedController(BaseController):
    """
    Base class for closed-form model-based controllers.

    Combines a model-based feedforward term with an error-based feedback term.
    The control law is: u_control = model_based_term + error_based_feedback_term
    """

    # Override these methods
    def model_based_term(
        self, system_state: SystemState
    ) -> tuple[Array, Any | None]:
        """
        Compute the model-based feedforward control term.

        This term typically computes the control input required to achieve
        the desired trajectory based on the system dynamics model (e.g.,
        inverse dynamics, gravity compensation).

        Args:
            system_state: Current system state containing:
                - t (Array): Current simulation time
                - y (Array): Robot state [q, qd] concatenated
                - u (Optional[Array]): Previous actuation input
                - control_state (Optional[Any]): Internal controller state

        Returns:
            u_model (Array): Model-based control input, shape (num_actuators,)
            control_state_dot (Optional[Any]): Time derivative of internal
                controller state contributed by this term, or None
        """
        # Default: returns zero (override in subclass)
        return jnp.zeros((self.robot.num_actuators,)), None

    def error_based_feedback_term(
        self, system_state: SystemState
    ) -> tuple[Array, Any | None]:
        """
        Compute the error-based feedback control term.

        This term typically computes a corrective control input based on
        the tracking error (e.g., PD control, PID control).

        Args:
            system_state: Current system state

        Returns:
            u_feedback (Array): Feedback control input, shape (num_actuators,)
            control_state_dot (Optional[Any]): Time derivative of internal
                controller state contributed by this term, or None
        """
        # Default: returns zero (override in subclass)
        return jnp.zeros((self.robot.num_actuators,)), None
```

### Step-by-Step: Implementing a Custom Controller

#### 1. Choose Your Base Class

- **Extend `BaseController`** for general controllers (e.g., learning-based, hybrid)
- **Extend `ClosedFormModelBasedController`** for model-based controllers with feedforward/feedback structure

#### 2. Define Your Controller Class

```python
import equinox as eqx
import jax.numpy as jnp
from jax import Array
from soromox.control import ClosedFormModelBasedController, ReferenceTrajectory
from soromox.systems import SoftRobot, SystemState

class MyCustomController(ClosedFormModelBasedController):
    """Custom model-based controller implementation."""

    # Required attributes (from BaseController)
    robot: SoftRobot
    reference_trajectory: ReferenceTrajectory

    # Controller gains
    K_p: Array
    K_d: Array

    def __init__(
        self,
        robot: SoftRobot,
        reference_trajectory: ReferenceTrajectory,
        K_p: Array,
        K_d: Array,
    ):
        self.robot = robot
        self.reference_trajectory = reference_trajectory
        self.K_p = K_p
        self.K_d = K_d
```

#### 3. Implement Required Methods

```python
@eqx.filter_jit
def model_based_term(
    self,
    system_state: SystemState,
) -> tuple[Array, None]:
    """
    Compute model-based feedforward term.

    Implements inverse dynamics feedforward control.
    """
    # Extract state
    t = system_state.t
    y = system_state.y
    num_dofs = self.robot.num_dofs
    q, qd = y[:num_dofs], y[num_dofs:]

    # Get desired trajectory from reference
    q_des, qd_des, qdd_des = self.reference_trajectory.get_state(t)

    # Compute inverse dynamics
    M = self.robot.inertia_matrix(q_des)
    C = self.robot.coriolis_matrix(q_des, qd_des)
    G = self.robot.gravitational_force(q_des)
    tau_el = self.robot.elastic_force(q_des)

    # Feedforward torque
    tau_ff = M @ qdd_des + C @ qd_des + G + tau_el

    # Map to actuation space
    A = self.robot.actuation_matrix(q_des)
    u_model = jnp.linalg.lstsq(A, tau_ff)[0]

    return u_model, None  # None = no internal state

@eqx.filter_jit
def error_based_feedback_term(
    self,
    system_state: SystemState,
) -> tuple[Array, None]:
    """
    Compute PD feedback term based on tracking error.
    """
    # Extract state
    t = system_state.t
    y = system_state.y
    num_dofs = self.robot.num_dofs
    q, qd = y[:num_dofs], y[num_dofs:]

    # Get desired trajectory
    q_des, qd_des, _ = self.reference_trajectory.get_state(t)

    # Compute errors
    e = q_des - q
    ed = qd_des - qd

    # PD feedback torque
    tau_fb = self._apply_gain(self.K_p, e) + self._apply_gain(self.K_d, ed)

    # Map to actuation space
    A = self.robot.actuation_matrix(q)
    u_feedback = jnp.linalg.lstsq(A, tau_fb)[0]

    return u_feedback, None  # None = no internal state

# The __call__ method is already implemented in the base class
# It combines: u_control = model_based_term + error_based_feedback_term
```

#### 4. Usage Example

```python
from soromox.systems import PlanarPCS, PlanarPCSParams
from soromox.control import ReferenceTrajectory

# Create robot
params = PlanarPCSParams(
    length=jnp.array([0.1, 0.1, 0.1]),
    radius=jnp.array([0.01, 0.01, 0.01]),
    density=jnp.array([1000.0, 1000.0, 1000.0]),
    young_modulus=jnp.array([1e6, 1e6, 1e6]),
    shear_modulus=jnp.array([1e5, 1e5, 1e5]),
    damping_matrix=jnp.eye(9),
    gravity=jnp.array([0.0, -9.81]),
    reference_strain=jnp.tile(jnp.array([0.0, 1.0, 0.0]), 3),
    base_pose=jnp.array([jnp.pi / 2, 0.0, 0.0]),
)
robot = PlanarPCS(params=params)

# Create reference trajectory (defines desired motion)
ref_traj = ReferenceTrajectory(...)  # Your trajectory

# Create controller
controller = MyCustomController(
    robot=robot,
    reference_trajectory=ref_traj,
    K_p=jnp.diag(jnp.array([100.0, 100.0, 100.0])),
    K_d=jnp.diag(jnp.array([10.0, 10.0, 10.0])),
)

# Use in closed-loop simulation
ts, ys = robot.rollout_closed_loop_to(
    ts=jnp.linspace(0, 5, 100),
    y0=jnp.zeros(2 * robot.num_dofs),
    controller=controller,
)
```

### Controller Design Patterns

#### Configuration-Space Controllers

Operate directly in joint/strain coordinates:

```python
class ConfigurationSpaceController(ClosedFormModelBasedController):
    def model_based_term(self, system_state: SystemState):
        # Extract state
        num_dofs = self.robot.num_dofs
        q, qd = system_state.y[:num_dofs], system_state.y[num_dofs:]

        # Compute in configuration space
        tau = ...  # Your control law

        # Map to actuation space
        A = self.robot.actuation_matrix(q)
        u = jnp.linalg.lstsq(A, tau)[0]
        return u, None
```

#### Operational-Space Controllers

Operate in Cartesian/task space:

```python
class OperationalSpaceController(ClosedFormModelBasedController):
    s: float  # Arc length for end-effector control

    def model_based_term(self, system_state: SystemState):
        # Extract state
        num_dofs = self.robot.num_dofs
        q, qd = system_state.y[:num_dofs], system_state.y[num_dofs:]

        # Get task-space state
        chi = self.robot.forward_kinematics(q, s=self.s)
        J = self.robot.jacobian(q, s=self.s)

        # Compute task-space force
        F = ...  # Your operational-space control law

        # Map to joint torques then actuation
        tau = J.T @ F
        A = self.robot.actuation_matrix(q)
        u = jnp.linalg.lstsq(A, tau)[0]
        return u, None
```

#### Actuation-Space Controllers

Operate directly in actuation coordinates (tendon lengths, pressures):

```python
class ActuationSpaceController(ClosedFormModelBasedController):
    def model_based_term(self, system_state: SystemState):
        # Extract state
        num_dofs = self.robot.num_dofs
        q = system_state.y[:num_dofs]

        # Get actuation coordinates (e.g., tendon lengths)
        if hasattr(self.robot, 'tendon_length'):
            l = self.robot.tendon_length(q)
            # Compute directly in actuation space
            u = ...  # Your actuation-space control law

        return u, None
```

### Best Practices

1. **Use JAX transformations**: Decorate methods with `@eqx.filter_jit` for performance
3. **Handle singularities**: Use `jnp.linalg.lstsq` instead of matrix inversion
4. **Test thoroughly**: Verify stability with different robot configurations
5. **Document control law**: Clearly explain the mathematical formulation in docstrings

---

## API Reference

For detailed API documentation of the base classes, see:

- **[DynamicalSystem](../api/overview.md)** - Base class for all dynamical systems
- **[SoftRobot](../api/overview.md)** - Base class for soft robot systems (extends DynamicalSystem)
- **[BaseSoftRobotRenderer](../api/rendering/renderers.md#soromox.rendering.base.BaseSoftRobotRenderer)** - Base class for all renderers
