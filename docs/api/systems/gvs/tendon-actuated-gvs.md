# Tendon Actuated GVS

3D GVS continuum soft robots with tendon actuation.

## Overview

`TendonActuatedGVS` extends `GVS` with cable-driven actuation. Static GVS
segment choices live in `GVSStructure`; numeric values and tendon layouts live in
typed PyTree params.

## Quick Start

```python
import jax.numpy as jnp
from soromox.systems import (
    GVSLinkParams,
    GVSParams,
    GVSStructure,
    GVSSegment,
    JointSpec,
    LinearTendonRoutingParams,
    LinkSpec,
    StrainBasisSpec,
    TendonActuatedGVS,
    TendonActuatedGVSParams,
)

segment = GVSSegment(
    link=LinkSpec.circular(E=3e5, nu=0.45, rho=1300.0, eta=1e4, L=0.2, r=0.015),
    joint=JointSpec.fixed(),
    basis=StrainBasisSpec(type="monomial", active=[1, 1, 1, 1, 0, 0], orders=1),
    num_gauss_points=8,
)

body_params = GVSParams(
    link=GVSLinkParams(
        length=jnp.array([0.2]),
        young_modulus=jnp.array([3e5]),
        poisson_ratio=jnp.array([0.45]),
        density=jnp.array([1300.0]),
        damping_coefficient=jnp.array([1e4]),
        radius_initial=jnp.array([0.015]),
        radius_final=jnp.array([0.015]),
        height_initial=jnp.array([0.0]),
        height_final=jnp.array([0.0]),
        width_initial=jnp.array([0.0]),
        width_final=jnp.array([0.0]),
        semi_major_initial=jnp.array([0.0]),
        semi_major_final=jnp.array([0.0]),
        semi_minor_initial=jnp.array([0.0]),
        semi_minor_final=jnp.array([0.0]),
    ),
    gravity=jnp.array([0.0, 0.0, -9.81]),
    base_pose=jnp.zeros(6),
    reference_strain=jnp.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]),
    joint_stiffness=jnp.zeros((1, 4, 4)),
)

active_tendon_routing = LinearTendonRoutingParams(
    y_intercept=jnp.array([0.005, -0.005]),
    y_slope=jnp.array([0.0, 0.0]),
    z_intercept=jnp.array([0.0, 0.0]),
    z_slope=jnp.array([0.0, 0.0]),
    attachment_segment_index=jnp.array([0, 0]),
)

robot = TendonActuatedGVS(
    params=TendonActuatedGVSParams(
        body=body_params,
        active_tendon_routing=active_tendon_routing,
    ),
    structure=GVSStructure(segments=[segment], max_dof=4),
)

q = jnp.zeros(robot.num_dofs)
u = jnp.array([1.0])
A = robot.actuation_matrix(q)
```

As for plain `GVS`, `GVSLinkParams.length` is a per-segment link-length array.
`reference_strain` remains on `GVSParams` because it belongs to the strain basis,
not to the link geometry/material subtree.

## Tendon Routing

The default routing is linear and uses `LinearTendonRoutingParams`. Each field
has leading shape `(num_tendons,)`, so each tendon can have different intercepts,
slopes, and an attachment segment. Optional passive tendons use
`passive_tendon_routing=LinearTendonRoutingParams(...)` plus
`passive_tendon=PassiveTendonParams(...)` in `TendonActuatedGVSParams`.

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
