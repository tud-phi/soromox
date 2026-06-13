# Utilities

Utility modules and helper functions for SoRoMoX.

## Overview

This section contains utility modules that provide supporting functionality for robot systems, including parameter management and symbolic derivation tools.

## Available Utilities

### [Parameters](parameters.md)

Parameter handling, validation, and default configurations for soft robot systems.

- Default parameter sets for rapid prototyping
- Validation functions for parameter dictionaries
- Conversion utilities between parameter representations
- Factory functions for common robot configurations

### [Symbolic Derivation](symbolic-derivation.md)

Symbolic mathematics and derivation utilities using SymPy for soft robot modeling.

- Symbolic derivation of kinematics and dynamics
- Code generation from symbolic expressions
- Verification of mathematical correctness
- Reference implementations for new robot types

### [Geometry](geometry.md)

Pose encodings, rotation representation conversions, and geometric errors.

- Direct pose-coordinate transform helpers
- Quaternion, rotation-vector, rotation-matrix, and 6D conversions
- Geodesic orientation errors for control and tracking

### [Lie Algebra](lie-algebra.md)

Explicit namespaces for rigid-body and constant-strain operators.

- Pure ``SO(2)`` / ``so(2)``, ``SO(3)`` / ``so(3)``, ``SE(2)`` / ``se(2)``,
  and ``SE(3)`` / ``se(3)`` maps
- Constant-strain adjoint and tangent operators

## Quick Links

- [Parameters API Reference](parameters.md) - Parameter management utilities
- [Symbolic Derivation API Reference](symbolic-derivation.md) - Symbolic math tools
- [Geometry API Reference](geometry.md) - Pose, rotation, and error utilities
- [Lie Algebra API Reference](lie-algebra.md) - Lie group and constant-strain utilities
