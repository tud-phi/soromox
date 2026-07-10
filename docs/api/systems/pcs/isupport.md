# I-Support

3D pneumatically actuated soft robot based on the I-Support platform.

## Overview

`ISupport` is a specialized PCS implementation for 3D pneumatic soft robots with chamber-based actuation. It extends the base `PCS` class with:

- **3 chambers per segment**: Radially arranged pneumatic chambers
- **Pressure-based control**: Direct pressure input for actuation
- **I-Support platform**: Specific configuration for the I-Support robot

## Specialized Viser Rendering

Use `ISupportViserRenderer` to display the physical chamber layout instead of
the generic swept backbone. The renderer follows the model's chamber radii,
radial distances, angle offsets, pneumatic-segment grouping, and rigid connector
topology. It adds corrugated chamber surfaces, six translucent spacers per
pneumatic segment, and the base, intermediate, and tip interfaces.

```python
from soromox.rendering import ISupportViserRenderer, ISupportVisualConfig

renderer = ISupportViserRenderer(
    robot,
    visual_config=ISupportVisualConfig(),
    num_points=50,
)
renderer.show(q)
```

`ISupportVisualConfig` exposes the interface and spacer dimensions, chamber
ellipticity, bellows pitch and amplitude, mesh resolution, colors, and spacer
opacity. When an `ISupport` topology omits a rigid connector, the renderer adds
a thin visual-only interface plate without changing the robot kinematics.

## Model Quick Start

```python
import jax.numpy as jnp
from soromox.systems import ISupport, ISupportParams, ISupportStructure

params = ISupportParams(
    base_pose=jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    length=jnp.array([0.18]),
    radius=jnp.array([0.03]),
    density=jnp.array([1104.0]),
    gravity=jnp.array([0.0, 0.0, -9.81]),
    young_modulus=jnp.array([1.6464e6]),
    shear_modulus=jnp.array([0.5488e6]),
    damping_matrix=1e-3 * jnp.eye(6),
    reference_strain=jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    chamber_inner_radius=jnp.array([0.00639]),
    chamber_outer_radius=jnp.array([0.00779]),
    chamber_distance=jnp.array([0.020]),
    chamber_angle_offset=jnp.array([0.0]),
)
robot = ISupport(params, structure=ISupportStructure(num_gauss_points=3))

q = jnp.zeros(robot.num_dofs)
u = jnp.array([2.0e4, 0.0, 0.0])
g_tip = robot.forward_kinematics(q, s=jnp.sum(robot.L))
```

## Key Features

### Chamber Configuration

The I-Support robot uses 3 pneumatic chambers per segment, arranged radially at 120-degree intervals. This configuration enables:

- Bending in any direction
- Extension/compression along the backbone
- Complex 3D deformations

### Actuation Mapping

Pressure inputs are mapped to strains via the actuation basis matrix, converting chamber pressures to the 6 strain components (curvatures and linear strains).

## API Reference

::: soromox.systems.pcs.isupport.ISupport
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source
