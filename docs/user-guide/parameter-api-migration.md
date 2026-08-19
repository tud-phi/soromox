# PCS/GVS Parameter Migration

The harmonized parameter API is a clean break: no compatibility aliases are
provided. PCS, PlanarPCS, and GVS now share link parameters and construction
specifications, while GVS additionally owns joint parameters.

## Import shared components

Import reusable link, joint, cross-section, and material types from the common
systems API:

```python
# Before
from soromox.systems.gvs import JointSpec, LinkSpec

# After
from soromox.systems import JointSpec, LinkSpec
# Equivalent: from soromox.systems.components import JointSpec, LinkSpec
```

`GVSSegment` and `StrainBasisSpec` remain GVS concepts.

## Construct PCS and PlanarPCS from links

Replace flat `PCSParams` and `PlanarPCSParams` construction with `from_links`
or `params_from_links`:

```python
from soromox.systems import PCS, LinkSpec

robot = PCS.from_links([
    LinkSpec.circular(
        length=0.2,
        radius=0.012,
        density=1000.0,
        young_modulus=1.0e6,
        shear_modulus=3.4e5,
        material_damping_coefficient=1.0e4,
        reference_strain=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    )
])
```

Each link may instead supply explicit generalized `stiffness` and `damping`
matrices. Do not mix an explicit matrix with its material-property source.

## Update field paths

Link properties are nested consistently:

| Before | After |
| --- | --- |
| `params.length` | `params.link.length` |
| `params.density` | `params.link.density` |
| `params.reference_strain` | `params.link.reference_strain` |
| `params.stiffness` | `params.link.stiffness` |
| `params.damping` | `params.link.damping` |

Use `update_link_params` for common changes or nested `replace` for complete
parameter replacement:

```python
robot = robot.update_link_params(
    damping=0.9 * robot.params.link.damping
)

robot = robot.with_params(
    robot.params.replace(
        link=robot.params.link.replace(density=new_density)
    )
)
```

PCS damping is now link-local. Split a former block-diagonal global damping
matrix into one block per link; cross-link damping terms are unsupported.

## Rename GVS construction fields

| Before | After |
| --- | --- |
| `E` | `young_modulus` |
| `nu` | `poisson_ratio` (or derive and pass `shear_modulus`) |
| `rho` | `density` |
| `eta` | `material_damping_coefficient` |
| `L` | `length` |
| `r_i`, `r_f` | `radius=LinearProfile(base=..., tip=...)` |

Move `reference_strain` from `StrainBasisSpec` to `LinkSpec`. Replace legacy
`active` and `orders` basis arguments with `strain_selector` and `basis_order`.

```python
from soromox.systems import (
    GVSSegment,
    JointSpec,
    LinearProfile,
    LinkSpec,
    StrainBasisSpec,
)

segment = GVSSegment(
    link=LinkSpec.circular(
        length=0.2,
        radius=LinearProfile(base=0.015, tip=0.010),
        density=1000.0,
        young_modulus=1.0e6,
        poisson_ratio=0.45,
        material_damping_coefficient=1.0e4,
        reference_strain=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ),
    joint=JointSpec.fixed(),
    basis=StrainBasisSpec(
        type="legendre",
        strain_selector=("kappa_y", "sigma_x"),
        basis_order=1,
    ),
    num_gauss_points=7,
)
```

GVS joint matrices are available at `params.joint.stiffness` and
`params.joint.damping` and now contribute to the assembled dynamics.

## Update isotropic material values

Material values are caller-owned optimization variables, not duplicated in
runtime system params. Supply exactly one of `shear_modulus` and
`poisson_ratio`; omit `material_damping_coefficient` to disable damping:

```python
from soromox.systems import IsotropicMaterialParams

material = IsotropicMaterialParams(
    young_modulus=[1.0e6],
    poisson_ratio=[0.45],
    material_damping_coefficient=[1.0e4],
)
material = material.replace(
    young_modulus=1.1 * material.young_modulus
)
robot = robot.with_isotropic_material(material)
```

For full construction, update, JAX, Optax, geometry co-optimization, and direct
matrix examples, see [Parameters and Optimization](parameters-and-optimization.md).
For component ownership and validation rules, see
[Continuum Robot Components](../api/systems/continuum-components.md).
