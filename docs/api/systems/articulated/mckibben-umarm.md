# McKibben UMArm

The `McKibbenActuatedUMArm` system models the UMArm as a spatial articulated
chain with pressure-actuated McKibben muscles. Cached robot parameters are kept
outside the Python wheel under `assets/robot_parameters/` in the source
repository. Pass an explicit `.npz` path to
`McKibbenActuatedUMArmParams.from_cached_npz(path)`.

The cached MuJoCo scene placement is normalized to the SoRoMoX base convention:
the root is placed at the origin and the proximal physical segment points along
positive x.

## Usage

```python
from pathlib import Path

import jax.numpy as jnp

from soromox.systems import McKibbenActuatedUMArm, McKibbenActuatedUMArmParams

params_path = Path("assets/robot_parameters/mckibben_umarm/reference_parameters.npz")
params = McKibbenActuatedUMArmParams.from_cached_npz(params_path)
robot = McKibbenActuatedUMArm(params)

q = jnp.zeros(robot.num_dofs)
p = jnp.zeros(robot.num_actuators)

lengths = robot.effective_lengths(q)
forces = robot.actuator_forces(q, p)
tau = robot.actuation_force(q, p)
```

For actuator-aware visualization, use `UMArmViserRenderer`:

```python
from soromox.rendering import UMArmViserRenderer

renderer = UMArmViserRenderer(robot, actuator_color_mode="pressure")
renderer.show(q, pressures=p)
```

## API Reference

::: soromox.systems.articulated.mckibben_actuated_umarm
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source

::: soromox.rendering.umarm.viser_renderer
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source
