# Utilities

Utility modules and helper functions for SoRoMoX.

## Overview

This section contains utility modules that provide supporting functionality for
robot systems, including parameter management, numerical conventions, geometry,
and Lie algebra tools.

## Available Utilities

### [Parameters](parameters.md)

Parameter handling, validation, and default configurations for soft robot systems.

- Default parameter sets for rapid prototyping
- Validation functions for parameter dictionaries
- Conversion utilities between parameter representations
- Factory functions for common robot configurations

### [Geometry](geometry.md)

Pose encodings, rotation representation conversions, and geometric errors.

- Direct pose-coordinate transform helpers
- Quaternion, rotation-vector, rotation-matrix, and 6D conversions
- Geodesic orientation errors for control and tracking

### [Numerics](numerics.md)

Finite singular-point conventions for common JAX numerical operations.

- Safe square root, Euclidean norm, division, and normalization
- Defined values and derivatives at removable or model singularities
- Strict-singularity diagnostics for locating invalid configurations

### [Lie Algebra](lie-algebra.md)

Explicit namespaces for rigid-body Lie groups and algebras.

- Pure ``SO(2)`` / ``so(2)``, ``SO(3)`` / ``so(3)``, ``SE(2)`` / ``se(2)``,
  and ``SE(3)`` / ``se(3)`` maps
- Reusable stable scalar Jacobian coefficients

### [Constant Strain](constant-strain.md)

Rod-segment operators organized into planar and spatial namespaces.

- Constant-strain adjoint, inverse-adjoint, tangent, and tangent derivative
- Bundled evaluation when several related operators are needed
- Separate planar and spatial operator namespaces

## Quick Links

- [Parameters API Reference](parameters.md) - Parameter management utilities
- [Geometry API Reference](geometry.md) - Pose, rotation, and error utilities
- [Numerics API Reference](numerics.md) - Safe numerical primitives
- [Lie Algebra API Reference](lie-algebra.md) - Lie group and Jacobian utilities
- [Constant-Strain API Reference](constant-strain.md) - Rod-segment Lie operators
