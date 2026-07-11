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
    length=jnp.array([0.18]),
    radius=jnp.array([35.6e-3]),
    density=jnp.array([1104.0]),
    young_modulus=jnp.array([1.6464e6]),
    shear_modulus=jnp.array([0.5488e6]),
    material_damping_coefficient=1.96e3,
    reference_strain=jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    chamber_inner_radius=jnp.array([6.39e-3]),
    chamber_outer_radius=jnp.array([7.79e-3]),
    chamber_distance=jnp.array([20e-3]),
    chamber_azimuth_angles=(2.0 * jnp.pi * jnp.arange(3) / 3)[None, :],
)
robot = ISupport(params, structure=ISupportStructure(num_gauss_points=3))

q = jnp.zeros(robot.num_dofs)

# Pressure actuation (3 chambers per pneumatic segment)
u = jnp.array([2.0e4, 0.0, 0.0])

# Forward kinematics
g_tip = robot.forward_kinematics(q, s=jnp.sum(robot.L))
```

## Key Features

### Chamber Configuration

The I-Support robot uses 3 pneumatic chambers per segment, arranged radially at 120-degree intervals. This configuration enables:

`chamber_azimuth_angles` has shape
`(num_pneumatic_segments, num_chambers_per_segment)`. Azimuth is a
right-handed rotation about local `+X`, measured in radians from local `+Y`
toward local `+Z`, so a chamber center is
`[0, d*cos(phi), d*sin(phi)]`. Entry `j` is pressure channel `j`; the model
does not sort or generate angles. Each row may be rotated, wrapped, and listed
in any channel order, but its wrapped circular gaps must be uniformly
`2*pi/num_chambers_per_segment` (within `rtol=1e-6`, `atol=1e-8`). Asymmetric
layouts are currently rejected because the passive cross-section model assumes
rotational symmetry.

If `chamber_azimuth_angles` is omitted, `ISupport` supplies the canonical
three-chamber layout `[0, 2*pi/3, 4*pi/3]`, or `[0°, 120°, 240°]`, for every
pneumatic segment. Pass an explicit array whenever physical pressure-channel
order differs from that default.

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
