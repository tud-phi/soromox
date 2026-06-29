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

#### [PCS (Spatial)](pcs.md)

Core 3D PCS implementation for spatial continuum robots.

- 6 strain components per segment: [kappa_x, kappa_y, kappa_z, sigma_x, sigma_y, sigma_z]
- SE(3) kinematics with 4x4 transformation matrices
- Foundation for 3D specialized variants

#### [Planar PCS](planar-pcs.md)

2D PCS continuum soft robots.

- 3 strain components per segment: [kappa_z, sigma_x, sigma_y]
- SE(2) kinematics with [theta, x, y] pose representation
- Suitable for real-time applications

### Tendon Actuated

#### [Tendon Actuated PCS](tendon-actuated-pcs.md)

3D PCS robots with active and passive tendon actuation.

- Tendon routing with configurable paths
- Active tendon tension control
- Passive spring-damper tendons

#### [Tendon Actuated Planar PCS](tendon-actuated-planar-pcs.md)

2D PCS robots with tendon actuation.

- Cable-driven planar continuum robots
- Configurable tendon distances from backbone
- Segment-selective actuation

### Pressure Actuated

#### [Pressure Actuated Planar PCS](pressure-actuated-planar-pcs.md)

2D PCS robots with prescribed chamber-pressure actuation.

- Pressure chamber modeling
- Pressure-based control
- Configurable chamber geometry

#### [I-Support](isupport.md)

3D pneumatic soft robot with specific chamber configuration.

- 3 chambers per segment
- Specialized for the I-Support robot platform

## When to Use

| System | Dimension | Actuation | Use Case |
|--------|-----------|-----------|----------|
| `PCS` | 3D | None (strain input) | General 3D continuum modeling |
| `PlanarPCS` | 2D | None (strain input) | General 2D continuum modeling |
| `TendonActuatedPCS` | 3D | Tendons | 3D cable-driven robots |
| `TendonActuatedPlanarPCS` | 2D | Active/passive routed tendons | 2D cable-driven robots |
| `PressureActuatedPlanarPCS` | 2D | Pressure | Pressure-driven soft actuators |
| `ISupport` | 3D | Pneumatic | I-Support robot platform |

## References

The PCS model is based on:

Renda, F., Boyer, F., Dias, J., & Seneviratne, L. (2018). Discrete cosserat approach for multisection soft manipulator dynamics. *IEEE Transactions on Robotics*, 34(6), 1518-1533.
