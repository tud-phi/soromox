# GVS (Geometric Variable Strain)

3D continuum soft robots with generalized strain basis functions.

## Overview

`GVS` extends the PCS approach by letting each segment declare its own link,
joint, strain basis, and quadrature resolution. Static segment choices live in
`GVSStructure`; dynamic numeric arrays live in `GVSParams`.

## Quick Start

```python
import jax.numpy as jnp
from soromox.systems import (
    GVS,
    GVSLinkParams,
    GVSParams,
    GVSStructure,
    GVSSegment,
    JointSpec,
    LinkSpec,
    StrainBasisSpec,
)

segment = GVSSegment(
    link=LinkSpec.circular(E=1e6, nu=0.45, rho=1000.0, eta=1e4, L=0.3, r=0.03),
    joint=JointSpec.fixed(),
    basis=StrainBasisSpec(
        type="monomial",
        active=[1, 1, 1, 1, 0, 0],
        orders=[1, 1, 1, 1, 0, 0],
        xi_ref=[0, 0, 0, 1, 0, 0],
    ),
    num_gauss_points=5,
)

params = GVSParams(
    link=GVSLinkParams(
        length=jnp.array([0.3]),
        young_modulus=jnp.array([1e6]),
        poisson_ratio=jnp.array([0.45]),
        density=jnp.array([1000.0]),
        damping_coefficient=jnp.array([1e4]),
        radius_initial=jnp.array([0.03]),
        radius_final=jnp.array([0.03]),
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
robot = GVS(params=params, structure=GVSStructure(segments=[segment], max_dof=4))
q = jnp.zeros(robot.num_dofs)
base_transform = robot.forward_kinematics(q, s=robot.segment_end_positions[-1])
```

`GVSLinkParams` uses singular per-link field names because every array has the
same leading `(num_segments,)` axis. The link length array is therefore named
`length`. `reference_strain` is stored on `GVSParams`, not `GVSLinkParams`,
because it parameterizes the strain basis/reference configuration rather than
link geometry or material.

## Segment Specs

- `LinkSpec`: link geometry, material properties, and length. Use `LinkSpec.circular`, `LinkSpec.rectangular`, or `LinkSpec.elliptical` for common cross sections.
- `JointSpec`: joint type and optional axis, plane, pitch, and stiffness. Factory methods include `fixed`, `revolute`, `prismatic`, `helical`, `cylindrical`, `planar`, `spherical`, and `free`.
- `StrainBasisSpec`: basis family, active strain components, basis orders, and reference strain.
- `GVSSegment`: combines one link, one preceding joint, one strain basis, and `num_gauss_points`.

## Basis And Joint Names

Basis names are lower-case strings: `monomial`, `legendre`, `chebyshev`, `fourier`, `gaussian`, and `imq`.

Joint names are lower-case strings: `fixed`, `revolute`, `prismatic`, `helical`, `cylindrical`, `planar`, `spherical`, and `free`.

Active strain components can be passed either as a six-entry mask or as names: `kappa_x`, `kappa_y`, `kappa_z`, `sigma_x`, `sigma_y`, and `sigma_z`.

## Runtime Arrays

After construction, GVS exposes canonical runtime arrays:

- `segment_lengths`, `segment_end_positions`
- `num_gauss_points`, `num_integration_points`
- `integration_points`, `integration_weights`
- `dofs_per_segment`, `num_dofs`, `num_padded_dofs`, `active_dof_map`
- `B_joint`, `B_Xs`, `B_Z1`, `B_Z2`
- `xi_ref_joint`, `xi_ref_Xs`, `xi_ref_Z1`, `xi_ref_Z2`
- `mass_matrices`, `stiffness_matrices`, `damping_matrices`, `joint_stiffness`

## When To Use GVS vs PCS

| Use GVS when... | Use PCS when... |
|-----------------|-----------------|
| Complex deformation patterns need higher-order basis functions | Piecewise constant strain is sufficient |
| Segment joints differ or include hybrid soft-rigid structure | The robot is a standard serial continuum chain |
| You are comparing strain parametrizations | Runtime simplicity is the priority |

## API Reference

::: soromox.systems.gvs.core.GVS
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source

::: soromox.systems.gvs.specs.GVSSegment
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: soromox.systems.gvs.specs.LinkSpec
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: soromox.systems.gvs.specs.JointSpec
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: soromox.systems.gvs.specs.StrainBasisSpec
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
