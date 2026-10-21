# Simulation Integrators

Soromox provides Diffrax-compatible solvers for state layouts and dynamics that
need integration behavior beyond Diffrax's generic solver collection.

## Diffrax solvers

Soromox rollouts accept any compatible `diffrax.AbstractSolver` through the
`solver` argument. This includes standard Diffrax methods such as `Tsit5`,
`Dopri5`, `Bosh3`, `Heun`, and `Euler`; `Tsit5` remains the default when no
solver is supplied. See Diffrax's
[ODE solver catalog](https://docs.kidger.site/diffrax/api/solvers/ode_solvers/)
for the complete collection. Use a compatible
[step-size controller](https://docs.kidger.site/diffrax/api/stepsize_controller/)
when adaptive stepping is desired.

Standard Diffrax solvers operate on the complete first-order Soromox state, so
they support `[q, qd, auxiliary]` layouts, including unequal spatial
floating-base configuration and velocity dimensions. Before every dynamics
evaluation and on saved output, Soromox projects the state through the system's
`project_state` hook. Spatial floating-base robots therefore present a
normalized quaternion to the dynamics and return normalized saved states.

!!! warning "Floating-base quaternion integration"
    Standard Diffrax solvers advance the stored quaternion in its ambient
    coordinates. Soromox projects the quaternion before dynamics evaluations
    and on saved output, but these solvers do not perform a Lie-group retraction
    at each Runge--Kutta stage. Validate the timestep or adaptive tolerances for
    the floating-base model and operating range.

```python
from diffrax import PIDController, Tsit5

trajectory = robot.rollout_to(
    initial_state=initial_state,
    t1=1.0,
    solver_dt=1e-4,
    save_dt=1e-2,
    solver=Tsit5(),
    stepsize_controller=PIDController(rtol=1e-5, atol=1e-7),
)
```

## Semi-implicit Euler

`SemiImplicitEuler` is a deterministic, first-order kick–drift integrator for
Soromox systems whose state is ordered as

\[
y = \begin{bmatrix}q & qd & auxiliary\end{bmatrix}^{\mathsf T}.
\]

For a fixed step \(h\), it evaluates the complete forward dynamics at the old
state and updates velocity before configuration:

\[
qd_{n+1} = qd_n + h\,qdd(q_n, qd_n),
\qquad
q_{n+1} = \operatorname{retract}(q_n, h\,qd_{n+1}).
\]

This kick–drift ordering is generally more stable for oscillatory mechanical
systems than forward Euler. It remains a conditionally stable, first-order
method, so the timestep must still be validated for the model family and
operating range.

```python
from soromox.simulation import SemiImplicitEuler
from soromox.systems import SystemState

trajectory = robot.rollout_to(
    initial_state=SystemState(t=0.0, y=y0),
    t1=1.0,
    solver_dt=1e-5,
    save_dt=1e-2,
    solver=SemiImplicitEuler(robot),
)
```

!!! warning "Soromox-specific state and dynamics contract"
    The primary integrated state must follow the system's `[q, qd, auxiliary]`
    contract. The solver calls the system's state and retraction helpers, which
    support unequal floating-base configuration and velocity dimensions and
    preserve spatial quaternion geometry. It is not a general-purpose
    semi-implicit method for arbitrary Diffrax PyTrees. Unlike
    `diffrax.SemiImplicitEuler`, which requires a separable two-term vector
    field, the Soromox implementation evaluates the full derivative because its
    acceleration can depend on both configuration and velocity. It then applies
    the mechanical kick--drift update to the `[q, qd, auxiliary]` leaf. Optional
    Diffrax-level control and environment states are advanced with forward Euler.

::: soromox.simulation.SemiImplicitEuler
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2
      members_order: source
