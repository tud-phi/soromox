# Tendon Actuated PCS

3D PCS continuum soft robots with active and passive tendon actuation.

## Overview

`TendonActuatedPCS` extends the base `PCS` class to add cable-driven actuation for spatial (3D) continuum robots. It supports:

- **Active tendons**: Tension-controlled cables for robot actuation
- **Passive tendons**: Spring-damper elements for compliance and stability
- **Configurable routing**: Tendon paths defined by offset functions along the backbone

## Quick Start

```python
import jax.numpy as jnp
from soromox.systems import TendonActuatedPCS

# Create a tendon-actuated 3D PCS robot
robot = TendonActuatedPCS(
    num_segments=2,
    L=jnp.array([0.1, 0.1]),           # Segment lengths
    r=jnp.array([0.01, 0.01]),         # Cross-section radii
    rho=jnp.array([1000.0, 1000.0]),   # Density
    E=jnp.array([1e6, 1e6]),           # Young's modulus
    G=jnp.array([1e5, 1e5]),           # Shear modulus
    active_tendon_routing_params=...,   # Tendon routing configuration
)

# Configuration and actuation
q = jnp.zeros(robot.num_dofs)
u = jnp.array([1.0, -1.0])  # Tendon tensions

# Forward kinematics
chi = robot.forward_kinematics(q, s=robot.length)
```

## Key Features

### Tendon Routing

Tendons are defined by their offset from the backbone centerline as a function of arc length:

- `active_d_s(s)`: Active tendon offset vector at arc length `s`
- `active_dd_s_ds(s)`: Derivative of tendon offset (for Jacobian computation)

### Passive Tendons

Optional passive tendons provide:

- Elastic restoring forces via stiffness `K_pt`
- Damping via coefficient `D_pt`
- Initial displacement `l_pt0`

## API Reference

::: soromox.systems.pcs.tendon_actuated_pcs.TendonActuatedPCS
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source
