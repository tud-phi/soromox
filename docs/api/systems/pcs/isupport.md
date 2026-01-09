# I-Support

3D pneumatically actuated soft robot based on the I-Support platform.

## Overview

`ISupport` is a specialized PCS implementation for 3D pneumatic soft robots with chamber-based actuation. It extends the base `PCS` class with:

- **3 chambers per segment**: Radially arranged pneumatic chambers
- **Pressure-based control**: Direct pressure input for actuation
- **I-Support platform**: Specific configuration for the I-Support robot

## Quick Start

```python
import jax.numpy as jnp
from soromox.systems import ISupport

# Create an I-Support robot
robot = ISupport(
    num_segments=2,
    L=jnp.array([0.1, 0.1]),           # Segment lengths
    r=jnp.array([0.02, 0.02]),         # Cross-section radii
    rho=jnp.array([1000.0, 1000.0]),   # Density
    E=jnp.array([1e5, 1e5]),           # Young's modulus
    G=jnp.array([1e4, 1e4]),           # Shear modulus
)

# Configuration
q = jnp.zeros(robot.num_dofs)

# Pressure actuation (3 chambers per segment)
u = jnp.array([0.1, 0.0, -0.1, 0.1, 0.0, -0.1])

# Forward kinematics
chi = robot.forward_kinematics(q, s=robot.length)
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
