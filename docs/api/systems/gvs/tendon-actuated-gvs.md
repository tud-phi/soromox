# Tendon Actuated GVS

3D GVS continuum soft robots with tendon actuation.

## Overview

`TendonActuatedGVS` extends the base `GVS` class to add cable-driven actuation. It combines:

- **Flexible strain basis functions**: All GVS basis types available
- **Tendon routing**: Configurable tendon paths along the backbone
- **Force mapping**: Tendon tensions mapped to generalized forces

## Quick Start

```python
import jax.numpy as jnp
from soromox.systems import TendonActuatedGVS

# Create a tendon-actuated GVS robot
robot = TendonActuatedGVS(
    num_segments=2,
    max_dof=6,
    V_L=jnp.array([0.1, 0.1]),
    V_basistype_idx=jnp.array([3, 3]),  # Legendre basis
    tendon_routing_params=...,           # Tendon configuration
)

# Configuration and actuation
q = jnp.zeros(robot.dof_tot_system)
u = jnp.array([1.0, -1.0])  # Tendon tensions

# Forward kinematics
chi = robot.forward_kinematics(q, s=robot.length)
```

## Key Features

### Tendon Routing

Similar to `TendonActuatedPCS`, tendons are defined by offset functions:

- `d_s(s)`: Tendon offset vector at arc length `s`
- `dd_s_ds(s)`: Derivative of tendon offset

### Combined with GVS Features

All GVS basis function types and joint configurations are available:

- Multiple basis functions (Legendre, Chebyshev, Fourier, etc.)
- Arbitrary joint types at segment boundaries
- Flexible strain parametrization

## When to Use

Use `TendonActuatedGVS` when you need:

- Cable-driven actuation with complex kinematics
- Flexible strain basis for accurate deformation modeling
- Custom joint configurations between segments

For simpler applications, consider `TendonActuatedPCS` which uses piecewise constant strain.

## API Reference

::: soromox.systems.gvs.tendon_actuated_gvs.TendonActuatedGVS
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source
