# GVS Systems (Geometric Variable Strain)

This section covers continuum soft robots modeled with the Geometric Variable Strain (GVS) approach.

## Overview

GVS systems generalize PCS by allowing arbitrary strain basis functions instead of piecewise constant assumptions. A model is assembled from `GVSSegment` entries, where each segment contains a `LinkSpec`, `JointSpec`, `StrainBasisSpec`, and `num_gauss_points`.

## Public Specs

| Spec | Purpose |
|------|---------|
| `GVSSegment` | Complete declaration of one GVS segment |
| `LinkSpec` | Link geometry, material properties, and length |
| `JointSpec` | Preceding joint type and optional joint parameters |
| `StrainBasisSpec` | Strain basis family, active components, orders, and reference strain |

## Basis Functions

| Name | Description |
|------|-------------|
| `monomial` | Polynomial basis functions |
| `legendre` | Legendre polynomial basis |
| `chebyshev` | Chebyshev polynomial basis |
| `fourier` | Fourier series basis |
| `gaussian` | Gaussian radial basis functions |
| `imq` | Inverse multiquadric basis |

## Joint Types

| Name | DOF | Description |
|------|-----|-------------|
| `fixed` | 0 | Rigidly connected |
| `revolute` | 1 | Single rotation axis |
| `prismatic` | 1 | Single translation axis |
| `helical` | 1 | Coupled rotation and translation |
| `cylindrical` | 2 | Rotation plus translation along one axis |
| `planar` | 3 | Planar motion |
| `spherical` | 3 | Three-axis rotation |
| `free` | 6 | Fully free spatial joint |

## Available Systems

### [GVS](gvs.md)

Core 3D GVS implementation for generalized continuum robots.

### [Tendon Actuated GVS](tendon-actuated-gvs.md)

3D GVS robots with cable-driven actuation.

!!! tip "GVS vs PCS"
    Use **PCS** when piecewise constant strain is sufficient. Use **GVS** when you need flexible basis functions, segment-specific joints, or hybrid soft-rigid parametrizations.
