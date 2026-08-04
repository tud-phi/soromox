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

PCS and GVS share continuum-link, cross-section, isotropic-material, and joint
components. Their physical meaning, ownership, matrix shapes, and construction
specifications are described in
[Continuum Robot Components](continuum-components.md). Practical immutable
replacement and gradient-based identification examples are in
[Parameters, Updates, and Optimization](../../user-guide/parameters-and-optimization.md).

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
| [Articulated Systems](articulated/index.md) | Planar and spatial articulated chains, including compliant joints and tendon actuation | 2D & 3D |
| [PCS Systems](pcs/index.md) | Piecewise Constant Strain continuum robots, including PlanarHSA | 2D & 3D |
| [GVS Systems](gvs/index.md) | Geometric Variable Strain robots | 3D |

## System Summary

### Articulated Systems

Planar and spatial articulated robot systems modeled as serial rigid-link
chains. `Pendulum` provides the planar benchmark and `ArticulatedSoftRobot`
extends the interface to spatial screw-axis chains. Both hosts accept composable
actuators and passive elements.

| System | Actuation | Use Case |
|--------|-----------|----------|
| [Pendulum](pendulum/pendulum.md) | Identity or composable | Planar benchmark and cable-driven mechanisms |
| [Articulated Soft Robot](articulated/articulated-soft-robot.md) | Identity or composable | Spatial rigid-link chains with compliant joints |
| [McKibben UMArm](articulated/mckibben-umarm.md) | McKibben pressure actuation | Pneumatic rigid-soft hybrid arm |

### PCS Systems

Continuum soft robots using piecewise constant strain modeling.

| System | Dimension | Actuation | Use Case |
|--------|-----------|-----------|----------|
| [PCS](pcs/pcs.md) | 3D | Identity or [composable](../actuation/index.md) | General 3D continuum |
| [Planar PCS](pcs/planar-pcs.md) | 2D | Identity or [composable](../actuation/index.md) | General 2D continuum |
| [Planar HSA](pcs/planar-hsa.md) | 2D | Motor or direct effort | Handed shearing auxetics |
| [I-Support](pcs/isupport.md) | 3D | [Threadlike pneumatic](../actuation/threadlike.md#modality-presets) | I-Support platform |

### GVS Systems

Continuum robots with Geometric Variable Strain (GVS) parametrization.

| System | Actuation | Use Case |
|--------|-----------|----------|
| [GVS](gvs/gvs.md) | Identity or [composable](../actuation/index.md) | Flexible basis functions |

## Future Systems and Roadmap

The following system capabilities are on the roadmap or under consideration for
future development. They are not committed release targets, but they represent
areas where contributions would be especially valuable. See the
[Contributing Guide](../../development/contributing.md) for details on getting
started.

| Capability | Motivation |
|------------|------------|
| `PlanarGVS` | Provide a faster planar setting for prototyping controllers and algorithms that use higher-order geometric shape parametrizations before moving to full spatial GVS models. |
| Floating-base functionality across all systems | Support robots whose base pose is part of the system state, rather than assuming a fixed base. |
| Kinematic trees, particularly for GVS | Extend beyond serial kinematic chains to support branched soft robot architectures. |
| Closed-chain kinematics, particularly for GVS | Enable modeling of closed-loop mechanisms and parallel soft robots. |
