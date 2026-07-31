# McKibben UMArm

The `McKibbenActuatedUMArm` system models the UMArm as a spatial articulated
chain composed with a direct-pressure `ArticulatedMcKibbenActuator`. Body dynamics and
actuator geometry have independent immutable parameter objects. Cached robot
parameters are kept outside the Python wheel under `assets/robot_parameters/`
in the source repository; the convenience factory splits the existing cache
format internally.

`ArticulatedMcKibbenActuator` is specifically a spatial articulated transmission: its
grouped fixed/moving attachments span two-axis joint pairs and depend on the
host's kinematic frames. It is not installed on PCS or GVS rods. Continuum
models can use threadlike muscle or equivalent pressure-chamber coordinates;
generalizing the full McKibben attachment geometry would require a separate
continuum transmission contract.

The cached MuJoCo scene placement is normalized to the SoRoMoX base convention:
the root is placed at the origin and the proximal physical segment points along
positive x.

## Usage

```python
from pathlib import Path

import jax.numpy as jnp

from soromox.systems import McKibbenActuatedUMArm

params_path = Path("assets/robot_parameters/mckibben_umarm/reference_parameters.npz")
robot = McKibbenActuatedUMArm.from_cached_parameters(params_path)

q = jnp.zeros(robot.num_dofs)
p = jnp.zeros(robot.num_actuators)

actuator = robot.actuators[0]
lengths = actuator.effective_lengths(q)
segments = actuator.segments(robot, q)
forces = actuator.axial_forces(q, p)
tau = robot.actuation_force(q, p)
```

Manual construction keeps body and actuator parameters explicit:

```python
from soromox.actuation import ArticulatedMcKibbenActuator
from soromox.systems import McKibbenActuatedUMArmParams

body_params = McKibbenActuatedUMArmParams.from_cached_npz(params_path)
actuator = ArticulatedMcKibbenActuator.from_cached_npz(params_path)
robot = McKibbenActuatedUMArm(body_params, actuator=actuator)
```

McKibben actuator coordinates are pressure-normalized volume coordinates,

\[
y_{\mathrm{a},i}
= \frac{(B_i^2-\ell_i^2)\ell_i}{4\pi N_i^2},
\]

so pressure is the direct work-conjugate effort. The analytic moment matrix is
the transpose of the coordinate Jacobian and retains the cache's group and
channel ordering.

For actuator-aware visualization, use `UMArmViserRenderer`:

```python
from soromox.rendering import UMArmViserRenderer

renderer = UMArmViserRenderer(robot, actuator_color_mode="pressure")
renderer.show(q, actuator_inputs=p)
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
