# Constant Strain

The constant-strain package evaluates the Lie operators used to propagate
kinematics along rod segments with constant body strain. Planar and spatial
operators have separate namespaces, so their function names remain concise:

```python
from soromox.utils.lie_algebra.constant_strain import se2, se3

planar_tangent = se2.tangent(xi_planar, arc_length, eps)
spatial_operators = se3.operators(xi_spatial, arc_length, eps, xid_spatial)
```

Use a single-operator function when only one result is required. Use
`operators` when several related expressions are needed at the same strain and
arclength; it shares intermediate work and returns a fixed named bundle.

## Package Structure

- `constant_strain.se2` provides planar operators using direct rotation and
  translation blocks.
- `constant_strain.se3` provides spatial operators using exact reduced matrix
  polynomials.
- `ConstantStrainOperators` is the common named result returned by both
  modules' `operators` functions.

## Operator Bundle

::: soromox.utils.lie_algebra.constant_strain.ConstantStrainOperators

## SE(2)

::: soromox.utils.lie_algebra.constant_strain.se2

## SE(3)

::: soromox.utils.lie_algebra.constant_strain.se3
