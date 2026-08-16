# Numerics

The numerical helpers define finite behavior for common operations whose plain
JAX derivatives are undefined at singular inputs. They are useful when a valid
model configuration can contain a zero vector, zero distance, or removable
quotient.

The default conventions are:

- use the analytic limiting value where it exists, and zero otherwise;
- define the derivative to be zero exactly at the singular point; and
- match the corresponding JAX operation in value and derivative away from the
  singular point.

For example:

```python
import jax
import jax.numpy as jnp

from soromox.utils import safe_norm, safe_normalize

zero = jnp.zeros(3)
gradient = jax.grad(safe_norm)(zero)  # [0., 0., 0.]
direction = safe_normalize(zero)     # [0., 0., 0.]
```

If only the squared norm is needed, prefer `jnp.sum(x**2, axis=...)`; it is
already finite everywhere and avoids an unnecessary square root. Use
`safe_norm` when the norm itself is required.

## Diagnosing Singularities

The finite defaults can hide where a model reaches a singular configuration.
Use `soromox.autodiff.strict_singularities_mode()` while diagnosing a model to
restore the underlying JAX operations and expose non-finite results.

## API Reference

::: soromox.utils.numerics
