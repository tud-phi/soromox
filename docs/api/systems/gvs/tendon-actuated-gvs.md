# Tendon Actuated GVS

3D GVS continuum soft robots with tendon actuation.

## Overview

`TendonActuatedGVS` extends `GVS` with cable-driven actuation. Construction uses the same `segments`, `g`, and `p0` arguments as `GVS`, plus active tendon routing parameters.

## Quick Start

```python
import jax.numpy as jnp
from soromox.systems import (
    GVSSegment,
    JointSpec,
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

active_tendon_routing_params = {
    "ry": jnp.array([0.005]),
    "my": jnp.array([0.0]),
    "rz": jnp.array([0.0]),
    "mz": jnp.array([0.0]),
    "idx_seg_att": jnp.array([0]),
}

robot = TendonActuatedGVS(
    segments=[segment],
    g=[0.0, 0.0, -9.81],
    active_tendon_routing_params=active_tendon_routing_params,
)

q = jnp.zeros(robot.num_dofs)
u = jnp.array([1.0])
A = robot.actuation_matrix(q)
```

## Tendon Routing

The default routing is linear and uses the same key convention as `TendonActuatedPCS`: `ry`, `my`, `rz`, `mz`, and `idx_seg_att`. Custom routing can be supplied with `active_tendon_routing_basis`, whose `d_s` and `dd_s_ds` callables receive one tendon parameter dictionary and an arc-length value.

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
