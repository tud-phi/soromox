# Continuum Robot Components

PCS and GVS use the same public vocabulary for continuum links, cross-sections,
isotropic materials, and joints. This page describes the physical meaning and
ownership of those shared components. For construction, immutable replacement,
and gradient-based identification workflows, see
[Parameters and Optimization](../../user-guide/parameters-and-optimization.md).

## Model composition

Continuum-system parameters use shallow composition:

```text
PCSParams / PlanarPCSParams
└── link: ContinuumLinkParams
    └── cross_section: CrossSectionParams

GVSParams
├── link: ContinuumLinkParams
│   └── cross_section: CrossSectionParams
└── joint: JointParams
```

The hierarchy expresses physical ownership rather than a different material
model for every system. A link owns its geometry, reference strain, mass
properties, stiffness, and damping. A GVS joint separately owns joint
stiffness and damping. System-level parameters own the base pose and gravity.

## Cross-sections

`CrossSectionParams.coefficients` is a two-dimensional array with one row per
link. The static system structure determines how each row is interpreted:

- a scalar dimension defines a constant profile;
- `LinearProfile(base, tip)` defines a base-to-tip linear profile;
- rows are zero-padded when different GVS links require different numbers of
  coefficients.

The supported solid cross-sections are circular, rectangular, and elliptical.
`CrossSectionGeometry` identifies the family, while shared geometry utilities
calculate area and second moments of area. Geometry dimensions are finite and
strictly positive. Shared evaluators and `robot.cross_section_geometry(...)`
use `[height, width]` for rectangular sections and
`[semi_major, semi_minor]` for elliptical sections.

`LinkSpec` provides the ergonomic construction interface:

```python
from soromox.systems import LinearProfile, LinkSpec

circular = LinkSpec.circular(
    length=0.2,
    radius=0.012,
    density=1000.0,
    young_modulus=1.0e6,
    shear_modulus=3.4e5,
    material_damping_coefficient=1.0e4,
    reference_strain=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
)

tapered_rectangle = LinkSpec.rectangular(
    length=0.25,
    height=LinearProfile(base=0.03, tip=0.02),
    width=0.025,
    density=1000.0,
    young_modulus=1.0e6,
    shear_modulus=3.4e5,
    material_damping_coefficient=1.0e4,
    reference_strain=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
)
```

PCS currently accepts constant circular links. GVS accepts constant or linear
circular, rectangular, and elliptical cross-sections.

## Continuum links

`ContinuumLinkParams` stores the canonical numeric values used at runtime:

| Field | Shape | Meaning |
|-------|-------|---------|
| `length` | `(N,)` | Link backbone lengths |
| `density` | `(N,)` | Volumetric mass densities |
| `reference_strain` | `(N, strain_dimension)` | Stress-free strain fields |
| `cross_section.coefficients` | `(N, max_coefficients)` | Packed geometry profiles |
| `stiffness` | `(N, generalized_dimension, generalized_dimension)` | Generalized link stiffness |
| `damping` | same as `stiffness` | Generalized link damping |

Stiffness and damping are finite symmetric matrices. They are not required to
be diagonal or positive definite, which permits anisotropic, coupled, fitted,
and learned constitutive models. Link lengths and densities must be finite and
strictly positive; reference strains must be finite.

The generalized dimension depends on the system:

| System | Canonical link matrix shape |
|--------|-----------------------------|
| Spatial PCS | `(N, 6, 6)` |
| PlanarPCS | `(N, 3, 3)` |
| GVS | `(N, max_dof, max_dof)` |

GVS matrices are zero-padded beyond each link's active basis coordinates.

## Isotropic material model

Young's modulus, either shear modulus or Poisson's ratio, and optional material
damping are useful construction and identification variables, but they are not
duplicated in the runtime robot parameters. Instead, callers keep an
`IsotropicMaterialParams` PyTree:

```python
from soromox.systems import IsotropicMaterialParams

young = 1.0e6
material = IsotropicMaterialParams(
    young_modulus=young,
    poisson_ratio=0.45,
)
```

Each active field may be scalar or contain one value per link. Scalar values
are broadcast when the material is applied. Exactly one of `shear_modulus` and
`poisson_ratio` is required. Young's and shear moduli must be finite and
strictly positive, Poisson's ratio must lie in `(-1, 0.5)`, and material
damping must be finite and nonnegative when supplied. Omitting material damping
produces a zero damping matrix.

The optional-field pattern is structural: shear-modulus, Poisson-ratio, damped,
and undamped materials have different PyTree structures. JIT compilation
therefore specializes on these choices, while the active arrays remain normal
differentiable leaves.

Geometry, link length, strain basis, rotational scaling, and quadrature are
projected into unit-response operators. Material matrices are then evaluated
as

\[
K_i = E_i K_{i,E} + G_i K_{i,G},
\qquad
D_i = \eta_i D_{i,\eta}\quad\text{or}\quad 0,
\]

For the Young/Poisson parameterization,
\(G_i = E_i / (2 (1 + \nu_i))\) is evaluated inside this mapping so gradients
propagate through both \(E_i\) and \(\nu_i\).

Equivalently, the unit operators represent the discretized integrals

\[
K_i = \int B_i(s)^\mathsf{T} C_i(s) B_i(s)\,\mathrm{d}s,
\qquad
D_i = \int B_i(s)^\mathsf{T} V_i(s) B_i(s)\,\mathrm{d}s.
\]

`link_matrices_from_material` evaluates the mapping without modifying the
robot. `with_isotropic_material` returns a robot containing the resulting
canonical matrices. It does not store the material PyTree.

Geometry updates refresh the unit-response operators but leave explicitly
stored matrices unchanged. Reapply `with_isotropic_material` when matrices
should follow updated material or geometry values. This prevents geometry
updates from silently overwriting an explicitly supplied constitutive model.

## Joints

GVS segments may start with a fixed, revolute, prismatic, helical,
cylindrical, planar, spherical, or free joint. `JointSpec` accepts stiffness
and damping in active joint coordinates. Omitted values create zero matrices.

At runtime, `JointParams` stores both matrices with the same GVS padding as the
link matrices:

```python
import jax.numpy as jnp

from soromox.systems import JointSpec

joint = JointSpec.revolute(
    axis="z",
    stiffness=jnp.array([[0.3]]),
    damping=jnp.array([[0.02]]),
)
```

The GVS global matrices interleave joint and link contributions before
projection into active coordinates. PCS has no separate joint component.

## Construction specs, params, and structure

The three object categories have different lifetimes:

| Category | Purpose | Typical examples |
|----------|---------|------------------|
| Specs | Convenient user input during construction | `LinkSpec`, `JointSpec`, `GVSSegment`, `StrainBasisSpec` |
| Params | Dynamic numeric JAX PyTrees | `ContinuumLinkParams`, `JointParams`, `PCSParams`, `GVSParams` |
| Structure | Static choices affecting layout or compilation | `PCSStructure`, `GVSStructure` |

Specs may contain either isotropic material values or explicit generalized
matrices. Construction resolves them to one canonical runtime representation.
Changing numeric params with the same layout supports JAX transformations;
changing segment count, basis order, active strains, or padding requires
reconstruction with a new structure.

## API reference

::: soromox.systems.components
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source
