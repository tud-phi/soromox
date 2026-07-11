# Pendulum

The `Pendulum` system is the planar benchmark entry in the articulated systems
family. It provides N-link rigid-body chain dynamics for benchmarking and
comparison with soft robot systems.

## Overview

The pendulum system implements classical rigid-body dynamics for articulated robots with revolute joints. This serves as a baseline for comparing with continuum soft robot models.

## Usage

```python
import jax.numpy as jnp
from soromox.systems import Pendulum, PendulumParams

# Parameters for an N-link planar pendulum
params = PendulumParams(
    mass=jnp.array([1.0, 0.8]),
    moment_inertia=jnp.array([0.1, 0.05]),
    length=jnp.array([0.5, 0.3]),
    center_of_mass_length=jnp.array([0.25, 0.15]),
    joint_stiffness=jnp.zeros((2, 2)),
    joint_damping=jnp.zeros((2, 2)),
    joint_rest_configuration=jnp.zeros(2),
    radius=jnp.array([0.025, 0.015]),
)

robot = Pendulum(params=params)
q = jnp.array([0.2, -0.3])
qd = jnp.array([0.0, 0.0])

# Kinematics
chi_tips = robot.forward_kinematics_tips(q)   # (N, 3): [theta, px, py] per link tip
J_tips    = robot.jacobians_tips(q)           # (N, 3, N)

# Dynamics
B = robot.inertia_matrix(q)
C = robot.coriolis_matrix(q, qd)
G = robot.gravitational_force(q)

# Forward dynamics derivative of state y=[q, qd]
u = jnp.zeros_like(q)
y = jnp.concatenate([q, qd])
yd = robot.forward_dynamics(jnp.zeros(()), y, (u,))
```

## API Reference

::: soromox.systems.pendulum.pendulum
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source
