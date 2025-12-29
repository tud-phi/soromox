# Contributing

Thank you for your interest in contributing to Soft Robot Models in jaX (SoRoMoX)! This guide will help you get started.

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/soromox.git
cd soromox
```

### 2. Install Development Dependencies

```bash
pip install -e ".[dev,docs,examples]"
```

### 3. Set Up Pre-commit Hooks

```bash
pre-commit install
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Follow our coding standards:
- Use type hints
- Write docstrings in Google format
- Follow PEP 8 style guidelines
- Add tests for new functionality

### 3. Run Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_planar_pcs_num.py

# Run with coverage
pytest --cov=soromox
```

### 4. Format and Lint Code

We use [Ruff](https://docs.astral.sh/ruff/) for both code formatting and linting. Ruff is configured in `pyproject.toml` and follows our project's style guidelines.

#### Command Line Usage

```bash
# Auto-format code
ruff format .

# Format specific directories
ruff format src tests examples

# Check formatting without making changes
ruff format --check .

# Run linting checks
ruff check .

# Auto-fix linting issues
ruff check --fix .

# Run both formatting and linting
ruff format . && ruff check .
```

You can also use the Makefile targets:

```bash
# Format code
make format

# Check formatting (for CI)
make format-check
```

#### VS Code Integration

For automated formatting on file save, install the [Ruff VS Code extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff). The project includes VS Code settings (see `.vscode/settings.json`) that configure:

- Ruff as the default formatter for Python files
- Format on save enabled
- Automatic import organization on save
- Automatic linting fixes on save

After installing the Ruff extension, your code will be automatically formatted and linted whenever you save a Python file.

#### Ruff Configuration

Ruff is configured in `pyproject.toml` with the following key settings:
- Line length: 88 characters
- Target Python version: 3.10+
- Enabled lint rules: pycodestyle, Pyflakes, isort, flake8-bugbear, and more
- Quote style: double quotes
- Import organization: first-party imports from `soromox`

For more details, see the `[tool.ruff]` section in `pyproject.toml`.

### 5. Build Documentation

```bash
# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build
```

### 6. Commit Changes

```bash
git add .
git commit -m "feat: add new robot system"
```

Use conventional commit messages:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions
- `refactor:` for code refactoring

## Extending SoRoMoX

For guidance on adding custom robot systems or renderers, see the [Extending SoRoMoX](extending.md) guide which covers:

- Implementing custom soft robot systems
- Adding new renderer backends
- Contributing new controllers
- Understanding the base class interfaces
- Best practices for extensibility

## Code Style

### Python Style

- Follow PEP 8
- Use type hints for all function signatures
- Maximum line length: 88 characters
- Use descriptive variable names

### Docstring Format

Use Google-style docstrings:

```python
def forward_kinematics(params: Dict, q: Array) -> Array:
    """
    Compute forward kinematics for the robot.
    
    Args:
        params: Dictionary of robot parameters including:
            - l: Segment lengths
            - r: Segment radii  
            - E: Young's moduli
        q: Configuration vector of shape (n_dof,)
        
    Returns:
        End-effector position of shape (2,) for planar robots
        
    Raises:
        ValueError: If configuration vector has wrong dimensions
        
    Example:
        >>> params = {"L": jnp.array([0.1]), "r": jnp.array([0.01])}
        >>> q = jnp.array([0.0, 0.0, -1.0])
        >>> pos = forward_kinematics(params, q)
    """
```

### JAX Best Practices

- Use `jax.numpy` instead of `numpy`
- Make functions JAX-transformable (pure, no side effects)
- Use `jit` for performance-critical functions
- Avoid Python loops in favor of JAX operations

## Testing Guidelines

### Test Structure

```python
import pytest
import jax.numpy as jnp
from soromox.systems import YourSystem

class TestYourSystem:
    def setup_method(self):
        """Set up test fixtures."""
        self.params = {
            # Test parameters
        }
    
    def test_forward_kinematics(self):
        """Test forward kinematics."""
        # Test implementation
        pass
    
    def test_jacobian_computation(self):
        """Test Jacobian computation."""
        # Test implementation
        pass
```

### Test Coverage

Aim for high test coverage:
- Unit tests for individual functions
- Integration tests for complete workflows
- Property-based tests for mathematical relationships
- Regression tests for bug fixes

## Documentation Guidelines

### API Documentation

- Document all public functions and classes
- Include examples in docstrings
- Use type hints consistently
- Explain mathematical concepts clearly

### User Documentation

- Write clear, step-by-step tutorials
- Include complete, runnable examples
- Explain the mathematical background
- Provide troubleshooting guides

## Submitting Changes

### 1. Push Changes

```bash
git push origin feature/your-feature-name
```

### 2. Create Pull Request

- Provide a clear description of changes
- Reference related issues
- Include screenshots for UI changes
- Ensure all tests pass

### 3. Code Review

- Address reviewer feedback
- Update tests and documentation as needed
- Maintain a clean commit history

## Getting Help

- Open an issue for bugs or feature requests
- Join discussions in pull requests
- Check existing documentation and examples
- Ask questions in the community

Thank you for contributing to SoRoMoX!
