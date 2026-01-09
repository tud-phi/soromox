# GVS (Geometric Variable Strain)

3D continuum soft robots with generalized strain basis functions.

## Overview

`GVS` (Geometric Variable Strain) extends the PCS approach by allowing arbitrary strain basis functions instead of piecewise constant assumptions. This provides:

- **Flexible strain parametrization**: Multiple basis function types
- **Arbitrary joint types**: Various joint configurations at segment boundaries
- **Higher accuracy**: Better representation of complex deformation patterns

## Quick Start

```python
import jax.numpy as jnp
from soromox.systems import GVS

# Create a GVS robot with Legendre polynomial basis
robot = GVS(
    num_segments=2,
    max_dof=6,                         # Max DOFs per segment
    V_L=jnp.array([0.1, 0.1]),         # Segment lengths
    V_basistype_idx=jnp.array([3, 3]), # Legendre basis (index 3)
    # ... additional parameters
)

# Configuration
q = jnp.zeros(robot.dof_tot_system)

# Forward kinematics
chi = robot.forward_kinematics(q, s=robot.length)
```

## Basis Function Types

GVS supports multiple basis function types for strain representation:

| Index | Type | Description |
|-------|------|-------------|
| 0 | `B_Monomial` | Polynomial basis |
| 1 | `B_Fourier` | Fourier series |
| 2 | `B_Gaussian` | Gaussian radial basis |
| 3 | `B_Legendre` | Legendre polynomials |
| 4 | `B_Chebyshev` | Chebyshev polynomials |
| 5 | `B_IMQ` | Inverse multiquadric |

## Joint Types

At segment boundaries, GVS supports various joint types:

| Joint | DOF | Description |
|-------|-----|-------------|
| `B_Fixed` | 0 | Rigidly connected |
| `B_Free` | 6 | Fully free |
| `B_Revolute` | 1 | Single rotation |
| `B_Prismatic` | 1 | Single translation |
| `B_Spherical` | 3 | 3-axis rotation |
| `B_Planar` | 3 | Planar motion |
| `B_Cylindrical` | 2 | Rotation + translation |
| `B_Helical` | 1 | Coupled rotation-translation |

## Key Features

### Strain Basis Matrices

The GVS system uses basis matrices to represent strains:

- `V_B_joint`: Joint basis matrices at segment boundaries
- `V_B_Xs`: Strain basis matrices along each segment

### Integration Points

Physical properties are evaluated at Gauss-Legendre integration points:

- `V_Ms`: Mass matrices at integration points
- `V_Es`: Stiffness matrices at integration points
- `V_Gs`: Damping matrices at integration points

## When to Use GVS vs PCS

| Use GVS when... | Use PCS when... |
|-----------------|-----------------|
| Complex deformation patterns | Simple constant strain segments |
| Comparing basis function approaches | Standard continuum modeling |
| Research on strain parametrization | Real-time applications |
| Custom joint configurations | Straightforward serial segments |

## API Reference

::: soromox.systems.gvs.core.GVS
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source
