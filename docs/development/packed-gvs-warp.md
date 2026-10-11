# Native packed GVS kernels

The Warp backend now exposes explicit-index `*_at` functions and typed `packed_*_kernel` entry points. Homogeneous entry points keep their existing signatures and call the same numerical functions. External integrators can evaluate distinct morphologies in one launch without editing Python kernel source or grouping models.

A descriptor contains the array views for one world. A work entry is `wp.vec3i(world, local_item, 0)`. Kernels select that descriptor and evaluate its singleton batch at environment index zero. Each view retains its actual dimensions; callers allocate the sum of capacities, keep backing arrays alive, and validate that every work item addresses the corresponding view. These are low-level kernels, so invalid descriptors are not checked on the device.

For example, import `CellTermsArrays` and `packed_cooperative_cell_terms_kernel` from `soromox.execution.warp.gvs.cell`, populate one descriptor per world with that world's existing cell operands and workspaces, and launch:

```python
wp.launch(
    packed_cooperative_cell_terms_kernel,
    dim=(work.size, 2),
    inputs=[descriptors, work],
    block_dim=128,
    device=device,
)
```

Use `(world, local_cell, 0)` work entries for cells, `(world, local_joint, 0)` for joints, `(world, local_sample, 0)` for sample kernels, and `(world, 0, 0)` for whole-chain kernels. Array field names and shapes match the original kernel arguments. The third work component is reserved and must be zero.

Sample kernels also provide `packed_*_warp_kernel` variants: use `(world, first_sample, 0)` entries at sample intervals of 32 and launch with `dim=(work.size, 32)`. These keep each warp within a single world and guard the final partial warp. They trade unused lanes for reduced divergence and are optional; the scalar work-list entry points remain available.

Packed entry points live beside their homogeneous counterparts:

- `common.joints` and `common.joints_cooperative`: all supported joint bases.
- `gvs.cell`: scalar and two-thread cell terms, including higher-order Magnus terms and every strain component.
- `gvs.chain`, `gvs.floating_chain`, and `gvs.packed`: fixed, floating, and mixed chain mechanics. Launch tiled chain kernels with the same block width used to allocate chain workspaces.
- `gvs.kinematics` and `gvs.directional_kinematics`: poses, dense Jacobians, JVPs and wrench VJPs.
- `common.floating_kinematics`: spatial floating-base composition.
- `gvs.actuation.threadlike`: existing threadlike strain preparation, matrices and generalized forces.

Mixed chain kernels branch once per world block. Model order stays unchanged. Coordinate/basis maps, quadrature, spatial primitives and numerical accumulation order are shared with the established backend. FP64 remains mandatory for mechanics. These forward kernels retain the existing differentiation boundary; public SoRoMoX model methods provide custom derivatives.

## Adopted and optional optimizations

`launch_cell_terms` uses two threads per cell on CUDA, dividing independent tangent columns and adjoint entries. CPU execution remains scalar. There is no DOF-dependent dispatch threshold.

`prepare_dynamics_terms(operands)` returns a batched term function for repeated forward evaluation with immutable operands. It computes fixed-joint terms once when every joint is fixed. Moving-joint models evaluate joint terms normally. Construct it outside `jax.jit`, then JIT the returned callable. Rebuild it whenever any model operand changes. This explicit prepared API has no custom derivative rule; use the public model API for differentiation.

`common.linear_solve` exposes an optional 32-thread FP64 Cholesky solve for Warp-native loops, including packed descriptors. It uses a pivot floor and reports a nonzero status for a floored pivot. Callers must inspect this diagnostic. It does not change the default JAX linear solver or its differentiation behavior. The kernel requires a tiled launch with exactly 32 threads.

See [the committed benchmark report](https://github.com/tud-phi/soromox/blob/main/tools/benchmarks/results/2026-10-11-native-gvs/README.md) for measured effects and limitations. `tools/benchmarks/native_gvs.py` compares full Warp-backed JAX terms and forward dynamics; `cooperative_solve.py` compares the optional native solver using those actual inertia matrices. Setup, compilation and warm-up are excluded. GPU benchmarks were run outside the sandbox on an RTX 5090, with at most 512 worlds.
