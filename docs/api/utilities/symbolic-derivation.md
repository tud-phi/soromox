# Symbolic Derivation

Symbolic mathematics and derivation utilities using SymPy for soft robot modeling.

## Overview

This module provides tools for symbolic derivation of robot kinematics and dynamics equations using [SymPy](https://www.sympy.org/). These symbolic expressions can then be used to:

- Verify mathematical correctness of implementations
- Generate optimized JAX code automatically
- Understand the structure of kinematic and dynamic equations
- Debug complex mathematical relationships

## Workflow

The typical workflow for symbolic derivation is:

1. **Define symbolic variables** - Configuration parameters, strains, physical properties
2. **Derive kinematics symbolically** - Forward kinematics, Jacobians using SymPy
3. **Derive dynamics symbolically** - Lagrangian mechanics, inertia/Coriolis matrices
4. **Simplify expressions** - Use SymPy simplification to reduce complexity
5. **Generate code** - Convert symbolic expressions to JAX-compatible code

## Current Implementation

The module currently contains:

- **`symbolic_utils.py`** - General utilities for symbolic derivation applicable to any robot type
- **`planar_hsa.py`** - Complete symbolic derivation for Planar HSA robots (reference example)

## Example: Using Symbolic Utilities

```python
import sympy as sp
from soromox.symbolic_derivation.symbolic_utils import (
    skew_symmetric,
    rotation_matrix_2d,
)

# Define symbolic variables
theta = sp.Symbol('theta')
x, y = sp.symbols('x y')

# Create 2D rotation matrix
R = rotation_matrix_2d(theta)

# Compute skew-symmetric matrix
omega = sp.Symbol('omega')
Omega_skew = skew_symmetric([0, 0, omega])
```

## Extending for New Robot Types

To add symbolic derivation for a new robot system:

1. Create a new file `src/soromox/symbolic_derivation/your_robot.py`
2. Define symbolic variables for configuration and parameters
3. Derive forward kinematics using SymPy
4. Compute Jacobian via symbolic differentiation
5. Derive dynamics using Lagrangian mechanics
6. Simplify and export expressions

See `planar_hsa.py` as a complete reference example.

## API Reference

::: soromox.symbolic_derivation
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source
