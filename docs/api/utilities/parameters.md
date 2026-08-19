# Parameters

Shared link, joint, cross-section, and isotropic-material mechanics are
described in [Continuum Robot Components](../systems/continuum-components.md).
Construction, immutable replacement, and gradient-based workflows are covered
in [Parameters, Updates, and Optimization](../../user-guide/parameters-and-optimization.md).

Actuator and passive-element parameters follow the same immutable replacement
style through indexed robot delegates. See
[Actuation parameter updates](../actuation/index.md#parameter-updates).

System parameters are represented as typed Equinox PyTrees. Shared continuum
link, joint, cross-section, material params, and construction specs live in
`soromox.systems.components`. System params and static structures remain next
to their system family.

## Overview

Each system separates dynamic numeric values from static model structure:

- **Params objects** store JAX arrays that may be optimized, differentiated,
  vmapped, and replaced without changing the PyTree layout.
- **Structure objects** store static choices such as quadrature counts, active
  strain masks, GVS joint/basis/cross-section choices, and padding sizes.
  Changing structure means constructing a new system
  and may recompile jitted methods.
- **Spec objects** are ergonomic construction inputs. Shared `LinkSpec` and
  `JointSpec` objects live in `soromox.systems.components`; `GVSSegment` and
  `StrainBasisSpec` remain GVS-specific. Factory methods split specs into
  runtime params and static structures.

The top-level `soromox.systems` package re-exports the public params,
structures, and specs for convenient imports. Internally, concrete containers
are family-local:

| System family | Dynamic params | Static structures | Construction specs |
|---------------|----------------|-------------------|--------------------|
| Shared components | `soromox.systems.components` | - | `LinkSpec`, `JointSpec` |
| PCS | `soromox.systems.pcs.params` | `soromox.systems.pcs.structures` | shared `LinkSpec` |
| GVS | `soromox.systems.gvs.params` | `soromox.systems.gvs.structures` | shared specs plus `soromox.systems.gvs.specs` |
| Pendulum | `soromox.systems.pendulum.params` | - | - |
| Articulated | `soromox.systems.articulated.params` | - | - |

The public construction pattern is:

```python
robot = PCS.from_links(links, structure=PCSStructure(...))
robot = robot.update_link_params(length=new_length)
robot = robot.with_params(new_params)
```

Same-shape, same-dtype parameter updates preserve the JAX compilation layout.
Changing the number of segments, tendons, active strains, GVS basis layout, or
quadrature layout is a structural change and requires reconstruction.

For GVS specifically, `GVS.from_segments(...)` is the recommended constructor.
It accepts user-facing segment specs, stores canonical link and joint matrices
in `GVSParams`, and stores stripped static choices in `GVSStructure`.
`IsotropicMaterialParams` remains a separate caller-owned optimization PyTree,
so runtime and material representations are not duplicated.

## Naming

Typed params use singular physical quantity names. A field name denotes the
quantity for one indexed entity; array axes describe batching. For example,
`length` stores one length per segment or link when its shape is `(num_segments,)`
or `(num_links,)`.

| Field | Meaning |
|-------|---------|
| `length` | Per-segment or per-link length |
| `cross_section.coefficients` | Batched cross-section profile coefficients |
| `density` | Per-segment material density |
| `stiffness` | Canonical per-link or per-joint generalized stiffness matrices |
| `damping` | Canonical per-link or per-joint generalized damping matrices |
| `gravity` | Gravity vector |
| `base_pose` | Base configuration as scalar-first quaternion pose |
| `reference_strain` | Reference strain vector |
| `joint_rest_configuration` | Joint coordinates where elastic joint force is zero |

Young's modulus, either shear modulus or Poisson's ratio, and optional material
damping use the separate `IsotropicMaterialParams` fields `young_modulus`,
`shear_modulus`, `poisson_ratio`, and `material_damping_coefficient`. See the
linked components page for their mapping to canonical matrices and the user
guide for optimization examples.

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

Omitting both fields selects the defaults. For example, the factory below
constructs an upright PlanarPCS under Earth gravity:

```python
robot = PlanarPCS.from_links(planar_links)
```

Use the inherited mounting constructors to make another common mounting
explicit when constructing a param tree directly. `base_position` translates
the mounting without changing its orientation:

```python
horizontal = PlanarPCSParams.horizontal(link=planar_link_params)
upright = PlanarPCSParams.upright(
    link=planar_link_params, base_position=jnp.array([0.2, 0.1])
)
hanging = PCSParams.hanging(
    link=spatial_link_params, base_position=jnp.array([0.0, 0.0, 0.5])
)
```

Pass an explicit vector for custom or zero gravity. Pass an explicit
`base_pose` through the ordinary constructor for arbitrary orientations:

```python
zero_gravity = PCSParams(link=spatial_link_params, gravity=jnp.zeros(3))
custom = PlanarPCSParams(
    link=planar_link_params,
    gravity=jnp.array([1.0, -9.7]),
    base_pose=jnp.array([0.3, 0.2, 0.1]),
)
```

Calling `replace(base_pose=None)` or `replace(gravity=None)` restores the
dimension-appropriate default. The former identity fallback is available as
`.horizontal(...)` or by passing an explicit identity pose.

## Threadlike Actuation Parameters

Continuum routed actuation uses `ThreadlikeRouting` and explicit active or
passive components. The leading array axis indexes paths, so each path can have
distinct offsets, slopes, and a contiguous segment span.

```python
import jax.numpy as jnp
from soromox.actuation import (
    ThreadlikeActuator,
    ThreadlikeImpedance,
    ThreadlikeRouting,
)

active_routing = ThreadlikeRouting.linear(
    intercept=jnp.array([[0.0, 0.01, 0.0], [0.0, -0.01, 0.0]]),
    slope=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.002, 0.0]]),
    start_segment_index=(0, 0),
    end_segment_index=(0, 1),
)
active_tendons = ThreadlikeActuator.tendons(active_routing)

passive_routing = ThreadlikeRouting.linear(
    intercept=jnp.array([[0.0, 0.005, 0.004], [0.0, -0.005, 0.004]]),
    slope=jnp.zeros((2, 3)),
    start_segment_index=(0, 0),
    end_segment_index=(0, 1),
)

passive_impedance = ThreadlikeImpedance(
    routing=passive_routing,
    stiffness=jnp.array([10.0, 25.0]),
    damping=jnp.array([0.1, 0.3]),
    rest_length=jnp.array([0.2, 0.21]),
)
```

The number of paths and their segment-span topology are structural. Changing
either requires reconstructing the component and robot; numeric coefficients
and mechanical parameters use immutable component updates.

## Structure Naming

The public static objects are named `PCSStructure`, `GVSStructure`, and
`PlanarHSAStructure`. `Topology` was considered, but it is too narrow for
objects that also contain quadrature counts, active strain masks, basis padding,
and numerical evaluation choices. `Layout` was also considered, but it reads as
array-shape-only. `Structure` is the least misleading umbrella for static model
choices that affect compilation.

## Example

```python
import jax.numpy as jnp
from soromox.systems import LinkSpec, PCS, PCSStructure

links = [
    LinkSpec.circular(
        length=0.1,
        radius=0.01,
        density=1000.0,
        young_modulus=1e6,
        shear_modulus=1e5,
        material_damping_coefficient=1e4,
        reference_strain=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    )
    for _ in range(2)
]

robot = PCS.from_links(links, structure=PCSStructure(num_gauss_points=5))
updated_robot = robot.update_link_params(length=jnp.array([0.12, 0.1]))
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
