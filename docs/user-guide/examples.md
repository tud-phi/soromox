# Examples

This page provides an overview of the example scripts included with SoRoMoX.

## Available Examples

### Pendulum Examples

#### Single Pendulum
```bash
python examples/simulate_pendulum.py
```

Demonstrates:
- Basic pendulum dynamics
- Numerical integration with JAX
- Visualization of results

#### Double Pendulum
```python
# Modify examples/simulate_pendulum.py
num_links = 2  # Change this line
```

### Soft Robot Examples

#### Planar PCS Robot
```bash
python examples/simulate_planar_pcs.py
```

Features:
- Piecewise Constant Strain kinematics
- Continuum robot dynamics
- Parameter sensitivity analysis

#### HSA Robot
```bash
python examples/simulate_planar_hsa.py
```

Demonstrates:
- Handed Shearing Auxetics mechanics
- Motor-to-end-effector Jacobians
- Control applications

### Analysis Examples

#### Symbolic Derivation
```bash
python examples/derive_planar_pcs.py
```

Shows how to:
- Generate symbolic expressions
- Export for numerical use
- Validate against numerical methods

#### Benchmarking
```bash
python examples/benchmark_planar_pcs.py
```

Compares:
- Different integration methods
- Symbolic vs numerical approaches
- Performance characteristics

## Running Examples

### Prerequisites

Install example dependencies:
```bash
pip install -e ".[examples]"
```

### Environment Setup

Always source environment variables first:
```bash
source 01-configure-env-vars.sh
```

### Basic Workflow

1. **Derive symbolic expressions** (if needed):
   ```bash
   python examples/derive_[robot_type].py
   ```

2. **Run simulation**:
   ```bash
   python examples/simulate_[robot_type].py
   ```

3. **Analyze results** in generated plots and videos

## Customizing Examples

### Parameter Modification

Most examples allow easy parameter modification:

```python
# In any example file
params = {
    "L": 0.2,  # Change length
    "r": 0.01,  # Change radius  
    "E": 1e6,  # Change Young's modulus
    # ... other parameters
}
```

### Output Customization

Control visualization and output:

```python
# Enable/disable video recording
create_video = True

# Adjust simulation time
t_span = (0.0, 10.0)  # 10 seconds

# Change integration parameters
dt = 0.001  # Smaller timestep for higher accuracy
```

## Advanced Examples

### GUI Testing Tools

Interactive parameter exploration:
```bash
python tests/gui_test_fwd_kine_eps_planar_pcs.py
```

Features:
- Real-time parameter adjustment
- Multiple Jacobian method comparison
- Strain visualization

### Segment Fusion Analysis
```bash
python examples/analyze_segment_fusion.py
```

Investigates:
- Multi-segment behavior
- Coupling effects
- Performance optimization

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure SoRoMoX is installed with `pip install -e .`
2. **Missing dependencies**: Install examples group with `pip install -e ".[examples]"`
3. **Environment variables**: Always run `source 01-configure-env-vars.sh`

### Performance Tips

- Use smaller `dt` for accuracy vs speed tradeoffs
- Enable JAX JIT compilation for repeated runs
- Consider using GPU acceleration for large-scale studies

## Creating New Examples

To create a new example:

1. **Choose a base example** similar to your use case
2. **Copy and modify** parameters and logic
3. **Test thoroughly** with different configurations
4. **Add documentation** explaining the new features

Example template:
```python
#!/usr/bin/env python3
"""
New Example: Description of what this example demonstrates
"""

import jax.numpy as jnp
from soromox.systems import YourSystem

# Parameters
params = {
    # Your parameters here
}

# Main simulation logic
def main():
    # Your code here
    pass

if __name__ == "__main__":
    main()
```
