# GVS (Geometric Variable Strain)

3D continuum soft robots with generalized strain basis functions.

## Overview

`GVS` extends the PCS approach by letting each segment declare its own link,
joint, strain basis, and quadrature resolution. For ordinary use, construct from
segment specs with `GVS.from_segments(...)`. The factory splits those specs into
`GVSStructure`, which stores only static choices, and `GVSParams`, which stores
all dynamic numeric arrays. The explicit `GVS(params=..., structure=...)`
constructor remains available for advanced optimization and
system-identification workflows.

## Quick Start

The shared link, cross-section, material, and joint model is described in
[Continuum Robot Components](../continuum-components.md). For full immutable
replacement and optimization workflows, see
[Parameters, Updates, and Optimization](../../../user-guide/parameters-and-optimization.md).

```python
import jax.numpy as jnp
from soromox.systems import (
    GVS,
    GVSSegment,
    JointSpec,
    LinearProfile,
    LinkSpec,
    StrainBasisSpec,
)

segment = GVSSegment(
    link=LinkSpec.rectangular(
        length=0.3,
        height=LinearProfile(base=0.03, tip=0.02),
        width=0.025,
        density=1000.0,
        young_modulus=1e6,
        shear_modulus=3.45e5,
        material_damping_coefficient=1e4,
        reference_strain=[0, 0, 0, 1, 0, 0],
    ),
    joint=JointSpec.revolute(
        axis="z",
        stiffness=jnp.array([[0.3]]),
        damping=jnp.array([[0.02]]),
    ),
    basis=StrainBasisSpec(
        type="legendre",
        strain_selector=("kappa_y", "sigma_x"),
        basis_order=1,
    ),
    num_gauss_points=5,
)

robot = GVS.from_segments(
    [segment],
    max_dof=4,
)
q = jnp.zeros(robot.num_coordinates)
base_transform = robot.forward_kinematics(q, s=robot.segment_end_positions[-1])
```

`GVS.from_segments(...)` creates typed `GVSParams` and `GVSStructure`
internally, so `robot.params` can be optimized or partially replaced later.
Static structure contains no copied material constants, lengths, joint
matrices, or reference strains. For workflows that need the split without
constructing a robot, use `GVS.params_from_segments(...)`.

## Segment Specs

- `LinkSpec`: shared construction input for link geometry, reference strain, material properties or explicit generalized matrices, and length.
- `JointSpec`: shared construction input for joint type, kinematic choices, stiffness, and damping. Matrices are copied into `GVSParams.joint`.
- `StrainBasisSpec`: GVS-specific basis family, active strain components, and basis orders.
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
- `dofs_per_segment`, `num_internal_dofs`, `num_padded_dofs`, `active_dof_map`
- `B_joint`, `B_Xs`, `B_Z1`, `B_Z2`
- `xi_ref_joint`, `xi_ref_Xs`, `xi_ref_Z1`, `xi_ref_Z2`
- per-quadrature `mass_matrices`
- canonical `params.link.stiffness`, `params.link.damping`
- canonical `params.joint.stiffness`, `params.joint.damping`

## Immutable updates

```python
robot = robot.update_link_params(
    stiffness=1.1 * robot.params.link.stiffness,
    damping=0.9 * robot.params.link.damping,
)
robot = robot.update_joint_params(
    damping=1.2 * robot.params.joint.damping,
)

replacement = robot.params.replace(
    link=robot.params.link.replace(
        density=1.05 * robot.params.link.density,
    ),
    joint=robot.params.joint.replace(
        stiffness=1.1 * robot.params.joint.stiffness,
    ),
)
robot = robot.with_params(replacement)
```

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

::: soromox.systems.components.links.LinkSpec
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: soromox.systems.components.joints.JointSpec
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: soromox.systems.gvs.specs.StrainBasisSpec
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## References

Key literature on the geometric variable-strain formulation and its applications:

- Renda, F., Armanini, C., Lebastard, V., Candelier, F., & Boyer, F. (2020). A geometric variable-strain approach for static modeling of soft manipulators with tendon and fluidic actuation. *IEEE Robotics and Automation Letters*, 5(3), 4006-4013.
- Mathew, A. T., Hmida, I. B., Armanini, C., Boyer, F., & Renda, F. (2022). Sorosim: A MATLAB toolbox for hybrid rigid-soft robots based on the geometric variable-strain approach. *IEEE Robotics & Automation Magazine*, 30(3), 106-122.
- Boyer, F., Lebastard, V., Candelier, F., & Renda, F. (2020). Dynamics of continuum and soft robots: A strain parameterization based approach. *IEEE Transactions on Robotics*, 37(3), 847-863.
