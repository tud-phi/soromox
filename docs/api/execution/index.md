# Execution Backends

The `soromox.execution` package defines the backend names and optional
tuning parameters used by supported system classes. The user-facing workflow is
described in
[Execution Devices and Backends](../../user-guide/execution-devices-and-backends.md).

!!! note "Use the system methods in normal applications"
    Most users should not call classes or functions in this package directly.
    Construct `GVS`, `PCS`, or `PlanarPCS` with `backend="auto"`, `"jax"`, or
    `"warp"`, then call the public kinematics and dynamics methods. The
    lower-level interfaces below primarily support typed configuration and
    external integration packages.

## Public configuration

`ExecutionBackend` is the accepted backend-name type. `GVSBackendParams` and
`PCSBackendParams` group family-specific launch tuning rather than adding
backend-specific keywords to each system constructor.

```python
from soromox.execution import PCSBackendParams
from soromox.systems import PCS

robot = PCS.from_links(
    links,
    backend="auto",
    backend_params=PCSBackendParams(warp_block_dim=256),
)
```

::: soromox.execution
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2
      members:
        - ExecutionBackend
        - GVSBackendParams
        - PCSBackendParams
        - DEFAULT_GVS_BLOCK_DIM
        - DEFAULT_GVS_LARGE_BLOCK_DIM
        - DEFAULT_PLANAR_PCS_BLOCK_DIM
        - DEFAULT_PCS_BLOCK_DIM

## Child modules

- [`soromox.execution.config`](configuration.md) defines stable typed
  configuration.
- [`soromox.execution.warp`](warp/index.md) describes the optional Warp-native
  integration surface.
- [`soromox.execution.warp.gvs`](warp/gvs.md) exposes reusable GVS
  operands, kernels, and launch functions.
- [`soromox.execution.warp.pcs`](warp/pcs.md) exposes reusable PCS
  operands, kernels, and launch functions.

Backend dispatch, transformation rules, lazy loading, and family executors are
private implementation modules. Their names and call contracts may change
without notice and are intentionally excluded from the public API reference.
