# Systems API

This module contains the main system implementations for different robot types in SoRoMoX.

## Overview

SoRoMoX provides implementations for various robot systems, from classical rigid-body pendulums to advanced soft continuum robots. Each system includes:

- **Forward Kinematics**: Computing end-effector positions and orientations
- **Jacobians**: Computing velocity relationships and sensitivities  
- **Dynamics**: Mass matrices, Coriolis forces, and gravitational effects
- **Factory Functions**: Convenient initialization and configuration

## Available Systems

### Pendulum Systems

Multiple implementations of pendulum-based robot systems, from classical rigid-body chains to cable-driven mechanisms.

#### [Classical Pendulum](pendulum/pendulum.md)

Classical (optionally articulated) N-link pendulum implementation

- Rigid-body dynamics
- Revolute joints
- Configurable number of links
- Optional: turn into an articulated soft robot by adding elastic and dissipative forces at the pendulum joints.

#### [Tendon Actuated Pendulum](pendulum/tendon-actuated-pendulum.md)

N-link pendulum systems with active and passive tendon actuation

- Cable-driven actuation mechanisms
- Active and passive tendon support
- Spring-damper systems for compliance
- Hybrid rigid-soft robot characteristics

### PCS (Piecewise Constant Strain) Systems

Multiple implementations of continuum soft robots using piecewise constant strain modeling.

#### [General PCS](pcs.md)

Core PCS implementation providing the fundamental framework.

- Base PCS mathematical framework
- Foundation for specialized variants
- General continuum robot modeling

#### [Planar PCS](planar-pcs.md)

Planar PCS continuum soft robots in 2D.

- Cosserat rod theory based
- Piecewise constant strain assumptions
- Numerical implementation with Gauss-Legendre integration
- Suitable for real-time applications

#### [Pneumatic Planar PCS](pneumatic-actuated-planar-pcs.md)

Pneumatically actuated planar PCS robots.

- Pressure-based actuation
- Pneumatic chamber modeling
- Soft pneumatic robot applications
- Real-time pressure control

#### [Tendon Planar PCS](tendon-actuated-planar-pcs.md)

Tendon actuated planar PCS robots.

- Cable-driven actuation
- Tendon tension control
- Precise manipulation capabilities
- Cable-driven continuum robots

### [Planar HSA Systems](planar-hsa.md)

Handed Shearing Auxetics (HSA) soft robots with unique mechanical properties.

- Auxetic material properties
- Specialized deformation patterns
- Advanced control capabilities
- Research-oriented implementation
