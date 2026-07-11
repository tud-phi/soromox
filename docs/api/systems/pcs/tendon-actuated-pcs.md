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
from soromox.systems import (
    LinearTendonRoutingParams,
    PCSParams,
    PCSStructure,
    PassiveTendonParams,
    TendonActuatedPCS,
    TendonActuatedPCSParams,
)

# Create a tendon-actuated 3D PCS robot
body_params = PCSParams(
    length=jnp.array([0.1, 0.1]),
    radius=jnp.array([0.01, 0.01]),
    density=jnp.array([1000.0, 1000.0]),
    young_modulus=jnp.array([1e6, 1e6]),
    shear_modulus=jnp.array([1e5, 1e5]),
    material_damping_coefficient=jnp.array([362.0, 362.0]),
    reference_strain=jnp.tile(jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), 2),
)

active_routing = LinearTendonRoutingParams(
    y_intercept=jnp.array([0.01, -0.01]),
    y_slope=jnp.array([0.0, 0.0]),
    z_intercept=jnp.array([0.0, 0.0]),
    z_slope=jnp.array([0.0, 0.0]),
    attachment_segment_index=jnp.array([1, 1]),
)

passive_routing = LinearTendonRoutingParams(
    y_intercept=jnp.array([0.005]),
    y_slope=jnp.array([0.0]),
    z_intercept=jnp.array([0.005]),
    z_slope=jnp.array([0.0]),
    attachment_segment_index=jnp.array([1]),
)

passive_tendon = PassiveTendonParams(
    stiffness=jnp.array([10.0]),
    damping=jnp.array([0.1]),
    rest_length_offset=jnp.array([0.0]),
)

robot = TendonActuatedPCS(
    params=TendonActuatedPCSParams(
        body=body_params,
        active_tendon_routing=active_routing,
        passive_tendon_routing=passive_routing,
        passive_tendon=passive_tendon,
    ),
    structure=PCSStructure(num_gauss_points=5),
)

# Configuration and actuation
q = jnp.zeros(robot.num_dofs)
u = jnp.array([1.0, -1.0])  # Tendon tensions

# Forward kinematics
chi = robot.forward_kinematics(q, s=robot.L_cum[-1])
```

## Key Features

### Tendon Routing

Tendons are defined by their offset from the backbone centerline as a function of
arc length. `LinearTendonRoutingParams` stores all tendons in one batched PyTree:
each field has leading shape `(num_tendons,)`.

- `active_d_s(s)`: Active tendon offset vector at arc length `s`
- `active_dd_s_ds(s)`: Derivative of tendon offset (for Jacobian computation)

### Passive Tendons

Optional passive tendons provide:

- Per-tendon elastic restoring forces via `PassiveTendonParams.stiffness`
- Per-tendon damping via `PassiveTendonParams.damping`
- Per-tendon rest-length offsets via `PassiveTendonParams.rest_length_offset`

## API Reference

::: soromox.systems.pcs.tendon_actuated_pcs.TendonActuatedPCS
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source
