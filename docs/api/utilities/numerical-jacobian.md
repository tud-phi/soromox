# Numerical Jacobian

The numerical Jacobian utility estimates gradients and Jacobians with finite
differences. It supports first-order two-point differences, second-order
three-point differences, and one-sided schemes when variable bounds prevent a
central difference.

Use `approx_derivative` when an analytical or automatic Jacobian is unavailable
or when an independent derivative check is useful.

## API Reference

::: soromox.utils.numerical_jacobian
