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
strictly positive. `robot.cross_section_geometry(q, s)` returns
`(geometry, outer_dimensions, inner_dimensions)`, with `None` for entirely
solid models and zero inner dimensions at solid links in mixed models.
Shared evaluators and this query use
`[height, width]` for rectangular sections and `[semi_major, semi_minor]` for
elliptical sections. Height and semi-major radius span local material y; width
and semi-minor radius span local material z. These axes rotate with the link
and do not imply a world-up direction.

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

PCS currently accepts constant solid circular links. GVS accepts constant or
linear circular, rectangular, and elliptical cross-sections. Add
`wall_thickness=0.001` or `wall_thickness=LinearProfile(0.002, 0.001)` to
`LinkSpec.circular(...)` or `LinkSpec.rectangular(...)` for a tube, together
with an explicit [rod constitutive law](#rod-constitutive-laws). Elliptical
sections are currently solid only.

Wall thickness must be positive and leave an open cavity throughout the link.
The same geometry supplies rendering, inertia, and material-derived stiffness
and damping. Tubes require effective shear areas and Saint-Venant torsion; rectangular
tubes use Marshall within that choice.
Switching between solid and hollow sections requires constructing a new model.
For geometry updates, follow the [material update guidance](#isotropic-material-model).

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
| `sectional_inertia` | `(N, d, d)` or `(N, Q, d, d)`, optional | Angular-first inertia per reference length |
| `sectional_inertia_mask` | `(N,)`, optional boolean | Select supplied sectional inertia per link; other links use geometry and density |
| `tip_inertia` | `(N, d, d)`, optional | Attached-body inertia about each link tip |

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

## Distributed and attached-body inertia

Optional `sectional_inertia` and `tip_inertia` matrices describe mass and
rotational inertia in each link's local material frame, including an offset
center of mass. Use angular-first order with `d=3` for planar mechanics and
`d=6` for spatial mechanics. Sectional matrices are per unit reference length;
tip matrices describe the whole attached body about the link tip. Omitting
sectional inertia uses geometry and density; omitting tip inertia adds no attached
body. With supplied sectional matrices, `sectional_inertia_mask` selects which
links use them: `True` selects the override and `False` keeps geometry-and-density
evaluation. Omitting the mask selects every supplied row. This allows mixed
geometric and prepared inertia distributions, including subsequent density updates.

A sectional matrix may be constant along a link or sampled at its interior
Gauss nodes. Sampled arrays must match the model's quadrature grid, including
GVS padding. Matrices must be finite, symmetric, positive semidefinite, and
have an isotropic translational mass block.

`spatial_inertia` constructs these matrices from mass, center of mass, and
rotational inertia about that center, including parallel-axis and coupling terms:

```python
from soromox.systems import spatial_inertia

tip = spatial_inertia(masses, centers_of_mass, inertias_about_centers)
robot = robot.update_link_params(tip_inertia=tip)
```

Numeric updates preserve the model structure when shapes and optional-field
presence remain fixed. Allocate zero tip matrices before compilation if masses
will be attached later. See the helper's API reference below for input shapes
and units.

## Isotropic material model

`IsotropicMaterialParams` holds Young's modulus, either shear modulus or
Poisson's ratio, and optional scalar material damping. Each field may be a
scalar or one value per link. These are caller-owned construction and
identification inputs; the robot stores the resulting stiffness and damping
matrices.

```python
from soromox.systems import IsotropicMaterialParams

material = IsotropicMaterialParams(
    young_modulus=1.0e6,
    poisson_ratio=0.45,
    material_damping_coefficient=100.0,
)
```

Supply exactly one of `shear_modulus` and `poisson_ratio`. Moduli must be
positive, explicit Poisson ratios must lie in `(-1, 0.5)`, and damping must be
nonnegative. Omitting damping produces a zero damping matrix. The relation
\(G=E/[2(1+\nu)]\) is evaluated inside the material mapping so gradients
include both inputs. Changing which optional fields are present changes the
PyTree structure and requires recompilation.

### Rod constitutive laws

The default GVS law uses uniform transverse shear and polar-moment torsion
for solid sections. Shear and torsion can be selected independently. Hollow
links require effective shear and Saint-Venant torsion.

A GVS link's `RodConstitutiveLaw` selects how the isotropic material and
cross-section produce rod stiffness. For strain order
`[twist, bend_y, bend_z, extension, shear_y, shear_z]`, the local matrix is

\[
C=\mathrm{diag}(GJ,EI_y,EI_z,EA,GA_{s,y},GA_{s,z}).
\]

Area \(A\) and bending moments \(I_y,I_z\) subtract the cavity contribution
exactly. The model choices determine the effective shear areas \(A_s\) and
torsion constant \(J\). Rotational mass inertia uses \(I_y+I_z\), which can
differ from \(J\).

| Choice | Supported sections and behavior |
|---|---|
| `TransverseShearRodModel.UNIFORM` (default) | Solid sections; shear stiffness is \(GA\). |
| `TransverseShearRodModel.EFFECTIVE_AREA` | Solid/hollow circles use material-dependent Cowper factors; solid/hollow rectangles use Jourawski shear-energy matching. |
| `TorsionRodModel.POLAR_MOMENT` (default) | Solid sections; torsional stiffness is \(G(I_y+I_z)\). |
| `TorsionRodModel.SAINT_VENANT` | Marshall for solid/hollow rectangles; exact circular/annular and solid elliptical torsion. |

For a rectangular tube, select both alternatives explicitly:

```python
from soromox.systems import (
    LinkSpec, TorsionRodModel, RodConstitutiveLaw,
    TransverseShearRodModel,
)

link = LinkSpec.rectangular(
    length=0.2, height=0.02, width=0.04, wall_thickness=0.003,
    density=1000.0, young_modulus=1e6, poisson_ratio=0.45,
    reference_strain=[0, 0, 0, 1, 0, 0],
    constitutive_law=RodConstitutiveLaw(
        transverse_shear=TransverseShearRodModel.EFFECTIVE_AREA,
        torsion=TorsionRodModel.SAINT_VENANT,
    ),
)
```

These choices are retained in `GVSLinkStructure`; reconstruct the model to
change them. Hollow links require effective shear areas and Saint-Venant
torsion, using Marshall for rectangles. PCS/PlanarPCS currently accept only the default
law and solid circular geometry. Helicoidal strands retain their own
[sectional assumptions](gvs/helicoidal.md).

The effective shear laws connect continuously to their solid counterparts:
Cowper's circular factor becomes \(6(1+\nu)/(7+6\nu)\), and rectangular
shear area becomes \(5A/6\). These differ from uniform shear, so opting in also
changes solid-link stiffness. Circular torsion is exact at every wall thickness.
Marshall approaches Bredt for thin walls; its solid-square extension is about
3.74% above the exact solid solution. Rectangular shear approximates stress
redistribution, and neither torsion option models restrained warping or wall
buckling. See the
[transverse-shear][soromox.systems.components.TransverseShearRodModel] and
[torsion][soromox.systems.components.TorsionRodModel]
API references below for formulas, limits, and publications.

The scalar Kelvin–Voigt damping convention uses
\(V=\eta\,\mathrm{diag}(J,3I_y,3I_z,3A,A_{s,y},A_{s,z})\), including the
selected shear and torsion models. Cowper shear uses the elastic Poisson ratio
for both stiffness and damping. With explicit stiffness and Cowper shear,
supply an explicit damping matrix or omit damping: a damping coefficient alone
does not determine Poisson's ratio.

### Applying and updating material

`link_matrices_from_material(material)` returns generalized stiffness and
damping without changing the robot. `with_isotropic_material(material)` returns
a robot containing those matrices, projected through its strain bases and
quadrature. Cowper shear is reevaluated at the supplied Poisson ratio, including
its dependence on \(E\) and \(G\).

Geometry updates refresh inertia but preserve stored stiffness and damping.
Reapply the material when those matrices should follow new dimensions:

```python
robot = robot.update_link_params(cross_section=new_section)
robot = robot.with_isotropic_material(material)
```

Explicit generalized matrices remain independent inputs. For complete update
and differentiation examples, see
[Parameters and Optimization](../../user-guide/parameters-and-optimization.md).

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
