# API Overview

This page provides an overview of the main components and modules in the SoRoMoX package.

## Overview

SoRoMoX is a comprehensive library for modeling, simulating, and controlling soft robots using JAX. The package is organized into several main components:

- **[Systems](systems/index.md)**: Robot system implementations (articulated soft robots, pendulums, PCS, GVS, HSA)
- **[Control](control/index.md)**: Model-based controllers for trajectory tracking and regulation
- **[Rendering](rendering/index.md)**: Visualization and animation utilities
- **[Actuation](actuation.md)**: Actuation modeling and utilities
- **[Utilities](utilities/index.md)**: Parameter management and symbolic derivation tools

---

## Systems

The [Systems](systems/index.md) module contains implementations for various robot systems, from classical rigid-body pendulums and spatial articulated soft robots to advanced soft continuum robots. All systems provide:

- Forward kinematics and Jacobians
- Dynamics (mass matrices, Coriolis forces, gravitational effects)
- Energy computation methods
- Support for various actuation methods (direct torque, tendons, pneumatics)

SoRoMoX includes four main system categories:
- **Articulated Systems**: Planar pendulums, tendon-actuated pendulums, and spatial articulated soft robots
- **PCS Systems**: Piecewise Constant Strain continuum robots (2D & 3D)
- **GVS Systems**: Geometric Variable Strain robots (3D)
- **HSA Systems**: Handed Shearing Auxetics robots (2D)

---

## Control

The [Control](control/index.md) module provides model-based controllers for soft robots, organized into three control spaces:

- **Configuration-Space Controllers**: Operating in the robot's generalized coordinates
- **Operational-Space Controllers**: Operating in task space (end-effector positions/orientations)
- **Actuation-Space Controllers**: Operating in a transformed coordinate system ideal for underactuated robots

All controllers follow a unified architecture that decomposes control action into model-based and feedback terms, enabling high-performance simulation and real-time control applications.

---

## Rendering

The [Rendering](rendering/index.md) module provides a class-based architecture for visualizing soft robots. Multiple renderer implementations are available:

- **MatplotlibRenderer**: Generic, works with any robot (2D/3D)
- **Open3DRenderer**: Interactive 3D visualization
- **ViserRenderer**: Web-based interactive 3D visualization
- **OpenCVPlanarRenderer**: Fast 2D rendering
- **OpenCVPlanarHSARenderer**: Specialized for PlanarHSA

All renderers inherit from a common base class and provide cached forward kinematics and a unified interface.

---

## Actuation

The [Actuation](actuation.md) module provides utilities for modeling and working with different actuation methods, including tendon routing, pneumatic chambers, and direct torque actuation.

---

## Utilities

The [Utilities](utilities/index.md) module includes:

- **[Parameters](utilities/parameters.md)**: Parameter handling, validation, and default configurations for soft robot systems
- **[Symbolic Derivation](utilities/symbolic-derivation.md)**: Symbolic mathematics and derivation utilities using SymPy for soft robot modeling

---

## Getting Started

To get started with SoRoMoX:

1. **Installation**: See the [Installation Guide](../installation.md) for setup instructions
2. **Quick Start**: Check out the [Quick Start Guide](../user-guide/quick-start.md) for basic usage examples
3. **Examples**: Explore the [Examples](../user-guide/examples.md) for comprehensive use cases

For detailed API documentation, navigate to the specific modules using the sidebar.
