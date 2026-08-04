# Parameters, Updates, and Optimization

SoRoMoX separates convenient construction specifications, dynamic numeric
parameters, and static model structure. This guide shows how to construct PCS
and GVS systems, update their parameter PyTrees immutably, and use the same
interfaces in gradient-based optimization.

For the mechanical meaning of links, joints, cross-sections, stiffness,
damping, and isotropic materials, see
[Continuum Robot Components](../api/systems/continuum-components.md).

## Choosing the right parameter layer

| Task | Interface |
|------|-----------|
| Construct a robot from physical descriptions | `LinkSpec`, `JointSpec`, `GVSSegment` |
| Replace runtime numeric values | `update_params`, `update_link_params`, `update_joint_params` |
| Replace several nested components together | nested `.replace(...)` followed by `with_params(...)` |
| Identify isotropic material properties | caller-owned `IsotropicMaterialParams` and `with_isotropic_material(...)` |
| Fit anisotropic or coupled mechanics | optimize `params.link.stiffness` and `params.link.damping` directly |
| Change basis order, segment count, or padding | reconstruct with a new structure |

All update methods return new objects. They do not mutate the original robot or
parameter PyTree.

## Constructing PCS and GVS

The shared `LinkSpec` accepts either isotropic material properties or explicit
generalized matrices. This PCS example derives shear modulus from Poisson's
ratio:

```python
import jax.numpy as jnp

from soromox.systems import (
    PCS,
    LinkSpec,
    shear_modulus_from_poisson_ratio,
)

young = 1.0e6
shear = shear_modulus_from_poisson_ratio(
    young,
    poisson_ratio=0.45,
)
reference = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

pcs = PCS.from_links([
    LinkSpec.circular(
        length=0.20,
        radius=0.012,
        density=1000.0,
        young_modulus=young,
        shear_modulus=shear,
        material_damping_coefficient=1.0e4,
        reference_strain=reference,
    )
])
```

GVS uses the same link description and adds joint, strain-basis, and
quadrature specifications:

```python
from soromox.systems import (
    GVS,
    GVSSegment,
    JointSpec,
    StrainBasisSpec,
)

gvs = GVS.from_segments([
    GVSSegment(
        link=LinkSpec.circular(
            length=0.20,
            radius=0.012,
            density=1000.0,
            young_modulus=young,
            shear_modulus=shear,
            material_damping_coefficient=1.0e4,
            reference_strain=reference,
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
        num_gauss_points=7,
    )
])
```

An explicit `stiffness=` or `damping=` on `LinkSpec` bypasses the corresponding
material construction:

```python
explicit_pcs = PCS.from_links([
    LinkSpec.circular(
        length=0.15,
        radius=0.01,
        density=1000.0,
        stiffness=jnp.diag(
            jnp.array([0.2, 0.8, 0.8, 100.0, 30.0, 30.0])
        ),
        damping=jnp.diag(
            jnp.array([0.01, 0.02, 0.02, 0.5, 0.1, 0.1])
        ),
        reference_strain=reference,
    )
])
```

## Immutable runtime updates

Use the narrowest update method that owns a parameter. PCS, PlanarPCS, and GVS
share `update_link_params`:

```python
stiffness = pcs.params.link.stiffness
stiffness = stiffness.at[0].set(1.1 * stiffness[0])

pcs = pcs.update_link_params(
    stiffness=stiffness,
    damping=0.9 * pcs.params.link.damping,
)
```

GVS additionally exposes joint updates:

```python
joint_damping = gvs.params.joint.damping
joint_damping = joint_damping.at[0, 0, 0].set(0.03)
gvs = gvs.update_joint_params(damping=joint_damping)
```

Use nested `.replace(...)` and `with_params(...)` when replacing multiple
components atomically:

```python
replacement = gvs.params.replace(
    link=gvs.params.link.replace(
        density=1.05 * gvs.params.link.density,
        stiffness=1.1 * gvs.params.link.stiffness,
    ),
    joint=gvs.params.joint.replace(
        stiffness=1.2 * gvs.params.joint.stiffness,
        damping=1.1 * gvs.params.joint.damping,
    ),
)
gvs = gvs.with_params(replacement)
```

Top-level values use `update_params`:

```python
pcs = pcs.update_params(gravity=jnp.zeros(3))
```

Updates may change numeric values but not static layout. Changing the number of
links, GVS basis orders, active strain selectors, joint types, matrix padding,
or quadrature padding requires reconstruction.

## Isotropic material updates

Generalized stiffness and damping matrices are the canonical runtime values.
Young's modulus, shear modulus, and material damping remain convenient
construction and optimization variables in a separate caller-owned PyTree:

```python
from soromox.systems import IsotropicMaterialParams

material = IsotropicMaterialParams(
    young_modulus=jnp.array([1.0e6]),
    shear_modulus=jnp.array([3.4e5]),
    material_damping_coefficient=jnp.array([1.0e4]),
)

material = material.replace(
    young_modulus=1.1 * material.young_modulus,
    shear_modulus=0.95 * material.shear_modulus,
    material_damping_coefficient=(
        1.2 * material.material_damping_coefficient
    ),
)
updated_pcs = pcs.with_isotropic_material(material)
```

The same call works for PlanarPCS and GVS. Scalar fields are broadcast; arrays
must contain one value per link. The robot stores only the generated canonical
matrices, not `material`.

Inspect the mapping without updating the robot:

```python
stiffness, damping = pcs.link_matrices_from_material(material)
updated = pcs.with_isotropic_material(material)

assert jnp.allclose(updated.params.link.stiffness, stiffness)
assert jnp.allclose(updated.params.link.damping, damping)
```

## Differentiating parameter updates

Typed params and robot updates are JAX PyTrees, so a loss can construct a
candidate robot without mutation. This example differentiates all three
isotropic material fields:

```python
import jax

robot = pcs.with_isotropic_material(material)
target_stiffness = 1.05 * robot.stiffness_matrix()
target_damping = 0.9 * robot.damping_matrix(jnp.zeros(robot.num_dofs))

def material_loss(candidate_material):
    candidate = robot.with_isotropic_material(candidate_material)
    stiffness_error = candidate.stiffness_matrix() - target_stiffness
    damping_error = (
        candidate.damping_matrix(jnp.zeros(candidate.num_dofs))
        - target_damping
    )
    return (
        jnp.mean(stiffness_error**2)
        + jnp.mean(damping_error**2)
    )

value, gradient = jax.value_and_grad(material_loss)(material)
print(value)
print(gradient.young_modulus)
print(gradient.shear_modulus)
print(gradient.material_damping_coefficient)
```

Cached unit-response operators reduce every material evaluation to batched
scalar-matrix combinations:

```text
K = E K_E + G K_G
D = eta D_eta
```

The same immutable pattern applies to other numeric fields. For example,
canonical stiffness can be differentiated directly:

```python
def matrix_loss(link_stiffness):
    candidate = robot.update_link_params(stiffness=link_stiffness)
    return jnp.mean(
        (candidate.stiffness_matrix() - target_stiffness) ** 2
    )

matrix_value, matrix_gradient = jax.value_and_grad(matrix_loss)(
    robot.params.link.stiffness
)
```

## Positive parameterization

Optimize material properties in log space when they must remain nonnegative.
This constrains the material scalars without imposing positive-definiteness on
the canonical generalized matrices:

```python
log_material = jax.tree.map(jnp.log, material)

def decode(log_values):
    return IsotropicMaterialParams(
        young_modulus=jnp.exp(log_values.young_modulus),
        shear_modulus=jnp.exp(log_values.shear_modulus),
        material_damping_coefficient=jnp.exp(
            log_values.material_damping_coefficient
        ),
    )
```

## Optax optimization loop

The following loop uses the project's examples dependency and keeps the
optimized variable as the log-material PyTree:

```python
import optax

optimizer = optax.adam(1.0e-2)
optimizer_state = optimizer.init(log_material)

def log_loss(log_values):
    candidate = robot.with_isotropic_material(decode(log_values))
    return jnp.mean(
        (candidate.stiffness_matrix() - target_stiffness) ** 2
    )

value_and_grad = jax.jit(jax.value_and_grad(log_loss))
for _ in range(100):
    value, gradients = value_and_grad(log_material)
    updates, optimizer_state = optimizer.update(
        gradients,
        optimizer_state,
        log_material,
    )
    log_material = optax.apply_updates(log_material, updates)

final_material = decode(log_material)
optimized_robot = robot.with_isotropic_material(final_material)
```

`robot` may be a PCS, PlanarPCS, or GVS instance.

## Geometry and material co-optimization

Length or cross-section updates refresh unit-response operators but do not
silently overwrite canonical matrices. Apply geometry first and material
second:

```python
new_section = robot.params.link.cross_section.replace(
    coefficients=(
        1.05 * robot.params.link.cross_section.coefficients
    )
)
geometry_robot = robot.update_link_params(cross_section=new_section)

assert jnp.allclose(
    geometry_robot.params.link.stiffness,
    robot.params.link.stiffness,
)
cooptimized_robot = geometry_robot.with_isotropic_material(material)
```

In a geometry/material optimization loop, construct `geometry_robot` from the
candidate geometry first, then apply the candidate material. This evaluates the
material mapping with the refreshed geometry operators while preserving the
explicit update order.

## Material variables or direct matrices?

Prefer isotropic material optimization when Young's modulus, shear modulus, and
material damping have physical meaning in the identification problem. Optimize
canonical `params.link.stiffness` and `params.link.damping` directly for
anisotropic, coupled, or learned constitutive models.

PCS matrices have shape `(N, 6, 6)`, PlanarPCS matrices `(N, 3, 3)`, and GVS
matrices `(N, max_dof, max_dof)` with zero padding beyond each link's active
coordinates. The update and optimization pattern is otherwise identical.
