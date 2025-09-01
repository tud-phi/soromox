# Utils API

This module contains utility functions and helper modules used throughout SoRoMoX.

## Overview

SoRoMoX provides various utility modules that support the core robot system implementations. These utilities handle mathematical operations, numerical integration, parameter management, visualization, and symbolic computations.

## Available Utilities

### [Math Utils](math-utils.md)
Core mathematical operations and linear algebra utilities.

- Matrix operations and manipulations
- Specialized robotics mathematics
- Linear algebra helper functions
- Numerical computation utilities

### [Parameters](parameters.md)
Parameter handling and configuration management.

- Robot parameter validation
- Default parameter sets
- Configuration utilities
- Parameter conversion functions

### [Rendering](rendering.md)
Visualization and animation tools for robot systems.

- 2D robot visualization
- OpenCV-based rendering
- Animation generation
- Trajectory plotting

### [Symbolic Derivation](symbolic-derivation.md)
SymPy-based symbolic mathematics for robot modeling.

- Symbolic kinematics derivation
- Dynamic equation generation
- Expression optimization
- JAX code generation from symbolic expressions

## Common Patterns

All utility modules are designed to integrate seamlessly with the main robot systems, providing:

- **JAX Compatibility**: All functions work with JAX arrays and transformations
- **Vectorization**: Support for batch operations where applicable  
- **Type Safety**: Comprehensive type hints for all functions
- **Documentation**: Detailed docstrings with examples
