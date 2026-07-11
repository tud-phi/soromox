# Tendon Actuated Planar PCS

Tendon actuated Planar PCS systems with active and passive cable-driven actuation mechanisms, extending the discrete Cosserat approach by Renda et al. (2018).

## Overview

This module extends the planar PCS model to include routed tendons. Its public API mirrors `TendonActuatedPCS` and `TendonActuatedGVS`: the body parameters live in `PlanarPCSParams`, active tendons are provided through `active_tendon_routing`, and optional passive tendons are provided through `passive_tendon_routing` plus `PassiveTendonParams`.

## Quick Start

```python
import jax.numpy as jnp
from soromox.systems import (
    LinearTendonRoutingParams,
    PlanarPCSParams,
    TendonActuatedPlanarPCS,
    TendonActuatedPlanarPCSParams,
)

num_segments = 2
body = PlanarPCSParams(
    length=0.1 * jnp.ones((num_segments,)),
    radius=0.02 * jnp.ones((num_segments,)),
    density=1070.0 * jnp.ones((num_segments,)),
    young_modulus=5e3 * jnp.ones((num_segments,)),
    shear_modulus=1e3 * jnp.ones((num_segments,)),
    damping_matrix=jnp.eye(3 * num_segments),
    reference_strain=jnp.tile(jnp.array([0.0, 1.0, 0.0]), num_segments),
)

offsets = 0.02 * jnp.array([[1.0, -1.0], [1.0, -1.0]])
active_routing = LinearTendonRoutingParams(
    y_intercept=offsets.reshape(-1),
    y_slope=jnp.zeros((4,)),
    z_intercept=jnp.zeros((4,)),
    z_slope=jnp.zeros((4,)),
    attachment_segment_index=jnp.array([0, 0, 1, 1]),
)

robot = TendonActuatedPlanarPCS(
    params=TendonActuatedPlanarPCSParams(
        body=body,
        active_tendon_routing=active_routing,
    )
)
```

## Planar Routing Return Shapes

Custom routing basis functions may be compact planar functions or shared with the spatial PCS/GVS models. For a single tendon at arc length `s`, both `d_s(params, s)` and `dd_s_ds(params, s)` may return:

- a scalar, interpreted as the planar offset `d(s)` or derivative `d'(s)`;
- shape `(1,)`, where the first entry is the planar offset or derivative;
- shape `(3,)`, using the PCS/GVS-compatible `[0, y, z]` convention. The planar model uses `y` and requires `z == 0`.

Shape `(2,)` is rejected because its coordinate convention is ambiguous. Nonzero `z` routing or nonzero `z` derivative is invalid for `TendonActuatedPlanarPCS`.

The built-in `LinearTendonRoutingParams` returns `[0, y, z]`. For planar use, set `z_intercept = z_slope = 0`.

### Compact Scalar Routing

```python
def d_s(params, s):
    return params.offset + params.slope * s


def dd_s_ds(params, s):
    return params.slope + jnp.zeros_like(s)


robot = TendonActuatedPlanarPCS(
    params=params,
    active_tendon_routing_basis={"d_s": d_s, "dd_s_ds": dd_s_ds},
)
```

## API Reference

::: soromox.systems.pcs.tendon_actuated_planar_pcs
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source

## References

The PCS (Piecewise Constant Strain) model was originally proposed in:

Renda, F., Boyer, F., Dias, J., & Seneviratne, L. (2018). Discrete cosserat approach for multisection soft manipulator dynamics. *IEEE Transactions on Robotics*, 34(6), 1518-1533.
