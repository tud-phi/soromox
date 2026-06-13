# Tendon Actuated GVS

3D GVS continuum soft robots with tendon actuation.

## Overview

`TendonActuatedGVS` extends `GVS` with cable-driven actuation. The usual
constructor is segment-first via `TendonActuatedGVS.from_segments(...)`. The
factory splits GVS segment specs into static structure and dynamic params, then
stores tendon routing and passive tendon impedance in typed PyTree params.

## Quick Start

```python
import jax.numpy as jnp
from soromox.systems import (
    GVSSegment,
    JointSpec,
    LinearTendonRoutingParams,
    LinkSpec,
    StrainBasisSpec,
    TendonActuatedGVS,
)

segment = GVSSegment(
    link=LinkSpec.circular(E=3e5, nu=0.45, rho=1300.0, eta=1e4, L=0.2, r=0.015),
    joint=JointSpec.fixed(),
    basis=StrainBasisSpec(type="monomial", active=[1, 1, 1, 1, 0, 0], orders=1),
    num_gauss_points=8,
)

active_tendon_routing = LinearTendonRoutingParams(
    y_intercept=jnp.array([0.005, -0.005]),
    y_slope=jnp.array([0.0, 0.0]),
    z_intercept=jnp.array([0.0, 0.0]),
    z_slope=jnp.array([0.0, 0.0]),
    attachment_segment_index=jnp.array([0, 0]),
)

robot = TendonActuatedGVS.from_segments(
    [segment],
    gravity=jnp.array([0.0, 0.0, -9.81]),
    base_pose=jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    max_dof=4,
    active_tendon_routing=active_tendon_routing,
)

q = jnp.zeros(robot.num_dofs)
u = jnp.array([1.0])
A = robot.actuation_matrix(q)
```

As for plain `GVS`, the segment-first factory builds `GVSParams` and
`GVSStructure` internally. Static structure stores only choices such as joint
family, basis layout, quadrature count, and cross-section family; numeric values
like material constants, reference strain, and joint stiffness live in
`robot.params.body`. `robot.params.body.link.length` is available for
optimization/update workflows.

## Tendon Routing

The default routing is linear and uses `LinearTendonRoutingParams`. Each field
has leading shape `(num_tendons,)`, so each tendon can have different intercepts,
slopes, and an attachment segment. Optional passive tendons use
`passive_tendon_routing=LinearTendonRoutingParams(...)` plus
`passive_tendon=PassiveTendonParams(...)`.

Custom routing can be supplied with `active_tendon_routing_basis` or
`passive_tendon_routing_basis`. The `d_s` and `dd_s_ds` callables receive one
single-tendon typed routing PyTree and an arc-length value.

## Combined With GVS Features

All GVS segment features remain available: per-segment link geometry, joint type, strain basis family, active strain components, and quadrature resolution.

## API Reference

::: soromox.systems.gvs.tendon_actuated_gvs.TendonActuatedGVS
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source
