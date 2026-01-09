# GVS Systems (Geometric Variable Strain)

This section covers continuum soft robots modeled using the Geometric Variable Strain (GVS) approach, which extends PCS with flexible strain basis functions.

## Overview

GVS systems generalize the PCS approach by allowing arbitrary strain basis functions instead of piecewise constant assumptions. This provides:

- Flexible strain parametrization with multiple basis function types
- Support for various joint types at segment boundaries
- Higher accuracy for complex deformation patterns
- Unified framework for different soft robot architectures

## Basis Functions

GVS supports multiple basis function types for strain representation:

| Type | Description |
|------|-------------|
| `B_Monomial` | Polynomial basis functions |
| `B_Fourier` | Fourier series basis |
| `B_Gaussian` | Gaussian radial basis functions |
| `B_Legendre` | Legendre polynomial basis |
| `B_Chebyshev` | Chebyshev polynomial basis |
| `B_IMQ` | Inverse multiquadric basis |

## Joint Types

GVS supports various joint types at segment connections:

| Joint | DOF | Description |
|-------|-----|-------------|
| `B_Fixed` | 0 | Rigidly connected |
| `B_Free` | 6 | Fully free (6 DOF) |
| `B_Revolute` | 1 | Single rotation axis |
| `B_Prismatic` | 1 | Single translation axis |
| `B_Spherical` | 3 | 3-axis rotation |
| `B_Planar` | 3 | Planar motion |
| `B_Cylindrical` | 2 | Rotation + translation along axis |
| `B_Helical` | 1 | Coupled rotation-translation |

## Available Systems

### [GVS](gvs.md)

Core 3D GVS implementation for generalized continuum robots.

- Arbitrary strain basis functions
- Multiple joint types supported
- SE(3) kinematics

### [Tendon Actuated GVS](tendon-actuated-gvs.md)

3D GVS robots with tendon actuation.

- Tendon routing with configurable paths
- Combined with flexible strain parametrization

## When to Use

| System | Use Case |
|--------|----------|
| `GVS` | Complex deformations requiring flexible basis, research applications |
| `TendonActuatedGVS` | Cable-driven robots with complex kinematics |

!!! tip "GVS vs PCS"
    Use **PCS** for standard applications where piecewise constant strain is sufficient. Use **GVS** when you need more flexible strain parametrization, such as for robots with complex deformation patterns or when comparing different basis function approaches.
