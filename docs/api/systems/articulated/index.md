# Articulated Systems

This section covers planar and spatial articulated robot systems, including
rigid-link chains, tendon-actuated chains, and joints with optional compliance.

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

### [Tendon Actuated Pendulum](../pendulum/tendon-actuated-pendulum.md)

Planar N-link pendulum with active and passive tendon actuation for
cable-driven articulated mechanisms.

### [Articulated Soft Robot](articulated-soft-robot.md)

Spatial serial-chain implementation with one screw-axis joint per link,
optional joint stiffness and damping, dense matrix dynamics, and a dedicated
forward-dynamics path.

## System Comparison

| System | Dimension | Use Case |
|--------|-----------|----------|
| `Pendulum` | 2D | Planar revolute-chain benchmark and planar special case |
| `TendonActuatedPendulum` | 2D | Cable-driven planar articulated chains |
| `ArticulatedSoftRobot` | 3D | Spatial rigid-link chains with optional compliant joints |
