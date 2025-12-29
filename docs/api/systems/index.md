# Systems

This section provides documentation for all robot system implementations in SoRoMoX.

## Overview

SoRoMoX includes implementations for various robot systems, from classical rigid-body pendulums to advanced soft continuum robots. Each system provides:

- **Forward Kinematics**: Computing end-effector positions and orientations
- **Jacobians**: Computing velocity relationships and sensitivities
- **Dynamics**: Mass matrices, Coriolis forces, and gravitational effects
- **Energy Methods**: Kinetic, potential, and total energy computation

## Base Classes

All robot systems in SoRoMoX inherit from a common base class hierarchy:

### DynamicalSystem

The `DynamicalSystem` class is the fundamental base class for all dynamical systems in SoRoMoX. It provides the interface for forward dynamics computation and time integration.

::: soromox.systems.DynamicalSystem
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source

### SoftRobot

The `SoftRobot` class extends `DynamicalSystem` with interfaces specific to soft robots, including methods for forward kinematics and Jacobians parameterized by arc-length or position along the robot backbone.

::: soromox.systems.SoftRobot
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source

## Common Components

### SystemState

The `SystemState` class is a container for the robot state used throughout SoRoMoX for simulation, control, and analysis.

::: soromox.systems.SystemState
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source

---

## System Categories

SoRoMoX organizes systems into the following categories:

| Category | Description | Dimension |
|----------|-------------|-----------|
| [Pendulum Systems](pendulum/index.md) | Articulated N-link chains | 2D |
| [PCS Systems](pcs/index.md) | Piecewise Constant Strain continuum robots | 2D & 3D |
| [GVS Systems](gvs/index.md) | Generalized Variable Strain robots | 3D |
| [HSA Systems](hsa/index.md) | Handed Shearing Auxetics robots | 2D |

## System Summary

### Pendulum Systems

Articulated robot systems modeled as N-link chains with revolute joints.

| System | Actuation | Use Case |
|--------|-----------|----------|
| [Pendulum](pendulum/pendulum.md) | Direct torque | Benchmarking, rigid-body dynamics |
| [Tendon Actuated Pendulum](pendulum/tendon-actuated-pendulum.md) | Tendons | Cable-driven articulated robots |

### PCS Systems

Continuum soft robots using piecewise constant strain modeling.

| System | Dimension | Actuation | Use Case |
|--------|-----------|-----------|----------|
| [PCS](pcs/pcs.md) | 3D | Strain input | General 3D continuum |
| [Planar PCS](pcs/planar-pcs.md) | 2D | Strain input | General 2D continuum |
| [Tendon Actuated PCS](pcs/tendon-actuated-pcs.md) | 3D | Tendons | 3D cable-driven |
| [Tendon Actuated Planar PCS](pcs/tendon-actuated-planar-pcs.md) | 2D | Tendons | 2D cable-driven |
| [Pneumatic Actuated Planar PCS](pcs/pneumatic-actuated-planar-pcs.md) | 2D | Pneumatic | Soft pneumatic |
| [I-Support](pcs/isupport.md) | 3D | Pneumatic | I-Support platform |

### GVS Systems

Continuum robots with generalized variable strain parametrization.

| System | Actuation | Use Case |
|--------|-----------|----------|
| [GVS](gvs/gvs.md) | Strain input | Flexible basis functions |
| [Tendon Actuated GVS](gvs/tendon-actuated-gvs.md) | Tendons | Complex cable-driven |

### HSA Systems

Soft robots based on handed shearing auxetics.

| System | Actuation | Use Case |
|--------|-----------|----------|
| [Planar HSA](hsa/planar-hsa.md) | Rod actuation | Auxetic soft robots |

