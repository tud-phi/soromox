# Articulated Systems

This section covers planar and spatial articulated robot systems, including
rigid-link chains, composable actuation, and joints with optional compliance.

## Overview

Articulated systems model robots as serial chains of rigid links connected by
generalized joints. They are useful for:

- planar rigid-body dynamics benchmarks
- cable-driven articulated mechanisms
- spatial rigid-body manipulator baselines
- soft-joint articulated robots
- comparison with continuum robot models
- controllers that require dense configuration-space dynamics

## Available Systems

### [Pendulum](../pendulum/pendulum.md)

Planar N-link rigid-body chain with revolute joints, optional joint stiffness
and damping, and lightweight benchmark dynamics.

### [Articulated Soft Robot](articulated-soft-robot.md)

Spatial serial-chain implementation with one screw-axis joint per link,
optional joint stiffness and damping, dense matrix dynamics, and a dedicated
forward-dynamics path.

### [McKibben UMArm](mckibben-umarm.md)

Spatial articulated UMArm with direct-pressure McKibben muscle actuation.

## System Comparison

| System | Dimension | Actuation | Use Case |
|--------|-----------|-----------|----------|
| `Pendulum` | 2D | Identity or composable | Planar revolute-chain benchmark and cable-driven mechanisms |
| `ArticulatedSoftRobot` | 3D | Identity or composable | Spatial rigid-link chains with optional compliant joints |
| `McKibbenActuatedUMArm` | 3D | McKibben pressure | Pneumatic rigid-soft hybrid arm |
