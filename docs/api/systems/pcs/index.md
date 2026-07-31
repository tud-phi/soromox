# PCS (Piecewise Constant Strain) Systems

This section covers continuum soft robots modeled using the Piecewise Constant Strain (PCS) approach based on discrete Cosserat rod theory.

## Overview

PCS systems model continuum soft robots by dividing them into segments, each with constant strain. This approach provides:

- Accurate continuum modeling with manageable complexity
- Support for both 2D (planar) and 3D (spatial) robots
- Various actuation methods (tendons, pressure chambers)
- Efficient numerical implementation using Gauss-Legendre quadrature

## Available Systems

### Base Implementations

#### [Spatial PCS](pcs.md)

Core 3D PCS implementation for spatial continuum robots.

- 6 strain components per segment: [kappa_x, kappa_y, kappa_z, sigma_x, sigma_y, sigma_z]
- SE(3) kinematics with 4x4 transformation matrices
- Foundation for 3D specialized variants

#### [Planar PCS](planar-pcs.md)

2D PCS continuum soft robots.

- 3 strain components per segment: [kappa_z, sigma_x, sigma_y]
- SE(2) kinematics with [theta, x, y] pose representation
- Suitable for real-time applications

### Composable actuation

`PCS` and `PlanarPCS` accept actuator models through `actuators=`. Tendons,
pushing rods, simplified muscles, and equivalent pressure chambers use the
shared [threadlike actuation model](../../actuation/threadlike.md). Passive
routed mechanics are installed separately with `ThreadlikeImpedance`.

### Pneumatic actuation

#### [I-Support](isupport.md)

3D pneumatic soft robot with specific chamber configuration.

- 3 chambers per segment
- Specialized for the I-Support robot platform

## When to Use

| System | Dimension | Actuation | Use Case |
|--------|-----------|-----------|----------|
| `PCS` | 3D | Identity or [composable](../../actuation/index.md) | General 3D continuum modeling |
| `PlanarPCS` | 2D | Identity or [composable](../../actuation/index.md) | General 2D continuum modeling |
| `ISupport` | 3D | [Threadlike pneumatic](../../actuation/threadlike.md#modality-presets) | I-Support robot platform |

## References

The PCS model is based on:

Renda, F., Boyer, F., Dias, J., & Seneviratne, L. (2018). Discrete cosserat approach for multisection soft manipulator dynamics. *IEEE Transactions on Robotics*, 34(6), 1518-1533.
