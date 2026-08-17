# Lie Algebra

The Lie algebra utilities are organized by semantic responsibility:

- `so2` and `so3` contain pure rotational Lie group and Lie algebra operators.
- `se2` and `se3` contain rigid-body Lie group and Lie algebra operators,
  including left Jacobians and their analytic directional derivatives for
  accumulated algebra coordinates.
- `jacobian_coefficients` contains reusable stable scalar coefficients.
- [`constant_strain.se2` and `constant_strain.se3`](constant-strain.md) contain
  arclength operators for planar and spatial rod segment kinematics.

## SO(2)

::: soromox.utils.lie_algebra.so2

## SO(3)

::: soromox.utils.lie_algebra.so3

## SE(2)

::: soromox.utils.lie_algebra.se2

## SE(3)

::: soromox.utils.lie_algebra.se3

## Jacobian Coefficients

::: soromox.utils.lie_algebra.jacobian_coefficients

## Constant-Strain Operators

The rod-specific adjoint, tangent, tangent-derivative, and bundled operator
APIs have a dedicated [Constant Strain](constant-strain.md) reference.
