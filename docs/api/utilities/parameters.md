# Parameters

Parameter handling, validation, and default configurations for soft robot systems.

## Overview

This module provides utilities for managing robot parameters across different system types. It includes:

- **Default parameter sets** - Physically realistic baseline parameters for rapid prototyping
- **Validation functions** - Ensure parameter dictionaries have required keys and valid ranges
- **Conversion utilities** - Transform between different parameter representations
- **Factory functions** - Generate parameter dictionaries for common robot configurations

## Parameter Dictionary Structure

All SoRoMoX robot systems use parameter dictionaries with standardized keys:

### Common Parameters (All Systems)

| Parameter | Type | Description | Units |
|-----------|------|-------------|-------|
| `L` | Array (N,) | Segment lengths | meters |
| `r` | Array (N,) | Cross-section radii | meters |
| `rho` | Array (N,) | Material density | kg/m³ |
| `E` | Array (N,) | Young's modulus (elasticity) | Pa |
| `G` | Array (N,) | Shear modulus | Pa |
| `D` | Array (N,N) | Damping matrix | Pa·s |
| `g` | Array (2,) or (3,) | Gravity vector | m/s² |

### System-Specific Parameters

Different robot types may require additional parameters:

- **PCS Systems**: Strain selectors, reference strains
- **Tendon-Actuated**: Tendon routing parameters, stiffnesses
- **Pneumatic**: Chamber geometry, pressure limits
- **HSA Systems**: Auxetic material properties, handedness

## Example Usage

```python
import jax.numpy as jnp
from soromox.parameters import hsa_params

# Get default HSA parameters
params = hsa_params.get_default_hsa_params(num_segments=3)

# Customize parameters
params["L"] = jnp.array([0.1, 0.15, 0.1])  # Custom segment lengths
params["E"] = jnp.array([5e4, 5e4, 5e4])   # Custom stiffness

# Use with robot
from soromox.systems import PlanarHSA
robot = PlanarHSA(num_segments=3, params=params)
```

## Creating Custom Parameter Sets

For new robot types, create parameter dictionaries following the standard structure:

```python
def create_custom_robot_params(num_segments: int) -> dict:
    """Create parameters for custom robot."""
    return {
        "L": jnp.ones(num_segments) * 0.1,      # 10 cm segments
        "r": jnp.ones(num_segments) * 0.01,     # 1 cm radius
        "rho": jnp.ones(num_segments) * 1000.0, # Water density
        "E": jnp.ones(num_segments) * 1e5,      # 100 kPa stiffness
        "G": jnp.ones(num_segments) * 3e4,      # Typical for elastomers
        "D": jnp.zeros((num_segments, num_segments)),  # No damping
        "g": jnp.array([0.0, -9.81]),           # Standard gravity
        # Add system-specific parameters here
    }
```

## Current Implementations

- **`hsa_params.py`** - Parameter utilities for HSA (Handed Shearing Auxetics) systems

Additional parameter modules for other robot types can be added following the same pattern.

## Best Practices

1. **Use SI units consistently** - Meters, kilograms, seconds, Pascals
2. **Validate parameters** - Check for positive values where required (lengths, radii, etc.)
3. **Provide defaults** - Include reasonable default values for rapid prototyping
4. **Document ranges** - Note physically realistic parameter ranges in docstrings
5. **Use arrays** - JAX arrays for all numerical parameters to enable differentiation

## API Reference

::: soromox.parameters
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source
