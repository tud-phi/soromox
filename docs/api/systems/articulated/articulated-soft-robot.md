# Articulated Soft Robot

The `ArticulatedSoftRobot` system models a spatial serial chain with rigid
links and optional joint stiffness and damping. It is the spatial counterpart to
the planar `Pendulum` benchmark, while keeping the `SoftRobot` interface used by
the rest of SoRoMoX.

## Overview

The model uses one screw-axis joint per link. Dense matrix dynamics are exposed
for controller and coordinate-transformation APIs, and forward dynamics is
implemented through a dedicated articulated-body path that is validated against
the dense equations of motion.

## Usage

```python
import jax.numpy as jnp
from soromox.systems import ArticulatedSoftRobot

params = {
    "joint_screws": jnp.array([
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ]),
    "p_tip": jnp.array([
        [0.5, 0.0, 0.0],
        [0.4, 0.0, 0.0],
    ]),
    "p_com": jnp.array([
        [0.25, 0.0, 0.0],
        [0.20, 0.0, 0.0],
    ]),
    "m": jnp.array([1.0, 0.8]),
    "I_com": jnp.array([
        jnp.diag(jnp.array([0.02, 0.03, 0.04])),
        jnp.diag(jnp.array([0.01, 0.02, 0.03])),
    ]),
    "g": jnp.array([0.0, 0.0, -9.81]),
    "K": jnp.diag(jnp.array([0.5, 0.3])),
    "D": jnp.diag(jnp.array([0.02, 0.01])),
}

robot = ArticulatedSoftRobot(params)
q = jnp.array([0.2, -0.1])
qd = jnp.array([0.0, 0.0])

g_tips = robot.forward_kinematics_tips(q)
J_tip = robot.jacobians_tips(q)[-1]

M = robot.inertia_matrix(q)
C = robot.coriolis_matrix(q, qd)
G = robot.gravitational_force(q)
```

## API Reference

::: soromox.systems.articulated.articulated_soft_robot
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source
