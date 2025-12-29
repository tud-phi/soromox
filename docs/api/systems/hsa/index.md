# HSA Systems (Handed Shearing Auxetics)

This section covers soft robots based on Handed Shearing Auxetics (HSA), which exhibit unique mechanical properties through auxetic material structures.

## Overview

HSA systems model soft robots that use handed shearing auxetics - materials that exhibit negative Poisson's ratio. These robots have unique deformation characteristics:

- Auxetic material properties for specialized deformation
- Combined bending and twisting motions
- High load capacity relative to weight
- Novel actuation mechanisms

## Available Systems

### [Planar HSA](planar-hsa.md)

Planar HSA soft robots with symbolic computation.

- Piecewise constant strain with symbolic Cosserat rod model
- Support for hysteresis modeling via Bouc-Wen model
- Precompiled lambda functions for efficient evaluation
- Multiple physical rods per segment

## Key Features

- **Symbolic Derivation**: Uses SymPy for symbolic computation of kinematics and dynamics
- **Hysteresis Support**: Optional Bouc-Wen hysteresis model for accurate material behavior
- **Multi-Rod Segments**: Multiple physical rods per virtual backbone segment
- **Platform Modeling**: Includes platform kinematics between segments

## When to Use

| System | Use Case |
|--------|----------|
| `PlanarHSA` | HSA-based soft robots, auxetic material research, planar HSA platforms |

## References

The planar HSA model is part of the publication:

Stölzle, M., Rus, D., & Della Santina, C. (2023). An experimental study of model-based control for planar handed shearing auxetics robots. *International Symposium on Experimental Robotics*, 153-167.
