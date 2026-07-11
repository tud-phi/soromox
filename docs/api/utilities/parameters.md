# Parameters

System parameters are represented as typed Equinox PyTrees. Shared base classes
and cross-system tendon params live in `soromox.systems.params`; concrete params
and structures live next to their system family, for example
`soromox.systems.gvs.params` and `soromox.systems.gvs.structures`.

## Overview

Each system separates dynamic numeric values from static model structure:

- **Params objects** store JAX arrays that may be optimized, differentiated,
  vmapped, and replaced without changing the PyTree layout.
- **Structure objects** store static choices such as quadrature counts, active
  strain masks, GVS joint/basis/cross-section choices, symbolic expression
  paths, and padding sizes. Changing structure means constructing a new system
  and may recompile jitted methods.
- **Spec objects** are ergonomic construction inputs for model families that
  need richer setup. For GVS, `GVSSegment`, `LinkSpec`, `JointSpec`, and
  `StrainBasisSpec` may contain both static choices and numeric values; factory
  methods split them into params and structure objects.

The top-level `soromox.systems` package re-exports the public params,
structures, and specs for convenient imports. Internally, concrete containers
are family-local:

| System family | Dynamic params | Static structures | Construction specs |
|---------------|----------------|-------------------|--------------------|
| PCS | `soromox.systems.pcs.params` | `soromox.systems.pcs.structures` | - |
| GVS | `soromox.systems.gvs.params` | `soromox.systems.gvs.structures` | `soromox.systems.gvs.specs` |
| HSA | `soromox.systems.hsa.params` | `soromox.systems.hsa.structures` | - |
| Pendulum | `soromox.systems.pendulum.params` | - | - |
| Articulated | `soromox.systems.articulated.params` | - | - |

The public construction pattern is:

```python
robot = PCS(params=PCSParams(...), structure=PCSStructure(...))
robot = robot.update_params(length=new_length)
robot = robot.with_params(new_params)
```

Same-shape, same-dtype parameter updates preserve the JAX compilation layout.
Changing the number of segments, tendons, active strains, GVS basis layout, or
quadrature layout is a structural change and requires reconstruction.

For GVS specifically, `GVS.from_segments(...)` is the recommended constructor.
It accepts user-facing segment specs, stores numeric values only in `GVSParams`,
and stores stripped static choices in `GVSStructure`. This avoids stale
duplicates when updating values such as Young's modulus or link length.

## Naming

Typed params use singular physical quantity names. A field name denotes the
quantity for one indexed entity; array axes describe batching. For example,
`length` stores one length per segment or link when its shape is `(num_segments,)`
or `(num_links,)`.

| Field | Meaning |
|-------|---------|
| `length` | Per-segment or per-link length |
| `radius` | Per-segment circular cross-section radius |
| `density` | Per-segment material density |
| `young_modulus` | Per-segment Young's modulus |
| `shear_modulus` | Per-segment shear modulus |
| `damping_matrix` | Generalized damping matrix |
| `gravity` | Gravity vector |
| `base_pose` | Base configuration as scalar-first quaternion pose |
| `reference_strain` | Reference strain vector |
| `joint_rest_configuration` | Joint coordinates where elastic joint force is zero |

## World Frame, Mounting, and Gravity Defaults

Soft-robot parameter objects use an upright mounting and Earth gravity when
`base_pose` or `gravity` is omitted. Vertical is world y for planar systems and
world z for spatial systems. Gravity is always expressed in the inertial/world
frame; changing the base mounting does not rotate the gravity vector.

| Mounting constructor | Planar backbone | Spatial backbone | Default gravity |
|----------------------|-----------------|------------------|-----------------|
| `horizontal` | +x | +x | negative vertical |
| `upright` | +y | +z | negative vertical |
| `hanging` | -y | -z | negative vertical |

The exact default values are:

- Planar upright pose: `[pi / 2, 0, 0]`; gravity: `[0, -9.81]` m/s².
- Spatial upright pose: `[sqrt(0.5), 0, -sqrt(0.5), 0, 0, 0, 0]`;
  gravity: `[0, 0, -9.81]` m/s².
- Planar poses use `[theta, x, y]`. Spatial poses use scalar-first Hamilton
  quaternions in `[qw, qx, qy, qz, x, y, z]` order.

Omitting both fields selects the defaults:

```python
params = PlanarPCSParams(
    length=length,
    radius=radius,
    density=density,
    young_modulus=young_modulus,
    shear_modulus=shear_modulus,
    damping_matrix=damping_matrix,
    reference_strain=reference_strain,
)
```

Use the inherited mounting constructors to make another common mounting
explicit. `base_position` translates the mounting without changing its
orientation:

```python
horizontal = PlanarPCSParams.horizontal(**planar_params)
upright = PlanarPCSParams.upright(
    **planar_params, base_position=jnp.array([0.2, 0.1])
)
hanging = PCSParams.hanging(
    **spatial_params, base_position=jnp.array([0.0, 0.0, 0.5])
)
```

Pass an explicit vector for custom or zero gravity. Pass an explicit
`base_pose` through the ordinary constructor for arbitrary orientations:

```python
zero_gravity = PCSParams(..., gravity=jnp.zeros(3))
custom = PlanarPCSParams(
    ...,
    gravity=jnp.array([1.0, -9.7]),
    base_pose=jnp.array([0.3, 0.2, 0.1]),
)
```

Calling `replace(base_pose=None)` or `replace(gravity=None)` restores the
dimension-appropriate default. The former identity fallback is available as
`.horizontal(...)` or by passing an explicit identity pose.

## Tendon Parameters

Continuum tendon routing uses `LinearTendonRoutingParams`. It is a batched
PyTree: the leading axis indexes tendons, so each tendon can have distinct
intercepts, slopes, and an attachment segment.

```python
import jax.numpy as jnp
from soromox.systems import LinearTendonRoutingParams, PassiveTendonParams

active_routing = LinearTendonRoutingParams(
    y_intercept=jnp.array([0.01, -0.01]),
    y_slope=jnp.array([0.0, 0.002]),
    z_intercept=jnp.array([0.0, 0.0]),
    z_slope=jnp.array([0.0, 0.0]),
    attachment_segment_index=jnp.array([0, 1]),
)

passive_routing = LinearTendonRoutingParams(
    y_intercept=jnp.array([0.005, -0.005]),
    y_slope=jnp.array([0.0, 0.0]),
    z_intercept=jnp.array([0.004, 0.004]),
    z_slope=jnp.array([0.0, 0.0]),
    attachment_segment_index=jnp.array([0, 1]),
)

passive_impedance = PassiveTendonParams(
    stiffness=jnp.array([10.0, 25.0]),
    damping=jnp.array([0.1, 0.3]),
    rest_length_offset=jnp.array([0.0, 0.01]),
)
```

The number of active or passive tendons is fixed by these leading dimensions.
Changing it changes actuator layout and requires reconstruction.

## Structure Naming

The public static objects are named `PCSStructure`, `GVSStructure`, and
`PlanarHSAStructure`. `Topology` was considered, but it is too narrow for
objects that also contain quadrature counts, active strain masks, basis padding,
and symbolic evaluation choices. `Layout` was also considered, but it reads as
array-shape-only. `Structure` is the least misleading umbrella for static model
choices that affect compilation.

## Example

```python
import jax.numpy as jnp
from soromox.systems import PCS, PCSParams, PCSStructure

params = PCSParams.upright(
    length=jnp.array([0.1, 0.1]),
    radius=jnp.array([0.01, 0.01]),
    density=jnp.array([1000.0, 1000.0]),
    young_modulus=jnp.array([1e6, 1e6]),
    shear_modulus=jnp.array([1e5, 1e5]),
    damping_matrix=jnp.eye(12),
    reference_strain=jnp.tile(jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), 2),
)

robot = PCS(params=params, structure=PCSStructure(num_gauss_points=5))
updated_robot = robot.update_params(length=jnp.array([0.12, 0.1]))
```

## API Reference

::: soromox.systems.params
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source

::: soromox.systems.pcs.params
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source

::: soromox.systems.pcs.structures
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source

::: soromox.systems.gvs.params
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source

::: soromox.systems.gvs.structures
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source

::: soromox.systems.hsa.params
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source

::: soromox.systems.hsa.structures
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source

::: soromox.systems.pendulum.params
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source

::: soromox.systems.articulated.params
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source
