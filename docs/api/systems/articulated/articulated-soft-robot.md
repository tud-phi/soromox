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

The constructor accepts the shared `actuators=` and `passive_elements=`
interface. `actuators=None` installs identity joint-effort actuation, while an
empty tuple creates an unactuated chain. Affine joint transmissions and
articulated tendons use the same API as the planar `Pendulum`; see
[Joint-space actuation](../../actuation/joint-space.md).

## Usage

```python
import jax.numpy as jnp
from soromox.systems import ArticulatedSoftRobot, ArticulatedSoftRobotParams

params = ArticulatedSoftRobotParams(
    joint_screw=jnp.array([
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ]),
    tip_position=jnp.array([
        [0.5, 0.0, 0.0],
        [0.4, 0.0, 0.0],
    ]),
    center_of_mass_position=jnp.array([
        [0.25, 0.0, 0.0],
        [0.20, 0.0, 0.0],
    ]),
    mass=jnp.array([1.0, 0.8]),
    center_of_mass_inertia=jnp.array([
        jnp.diag(jnp.array([0.02, 0.03, 0.04])),
        jnp.diag(jnp.array([0.01, 0.02, 0.03])),
    ]),
    joint_stiffness=jnp.diag(jnp.array([0.5, 0.3])),
    joint_damping=jnp.diag(jnp.array([0.02, 0.01])),
    joint_rest_configuration=jnp.zeros(2),
    parent_to_joint_transform=jnp.broadcast_to(jnp.eye(4), (2, 4, 4)),
    radius=jnp.array([0.025, 0.02]),
)

robot = ArticulatedSoftRobot(params=params)
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
