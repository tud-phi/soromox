# Execution Backends

For most applications, leave `backend="auto"` unchanged. A supported model then
uses the accelerated Warp kinematics and dynamics implementations on a GPU and
the JAX/XLA implementations on CPU, while retaining the same system methods and
numerical outputs. Set the option explicitly only when comparing implementations,
reproducing a benchmark, or requiring a particular execution path.

```python
from soromox.systems import LinkSpec, PlanarPCS

robot = PlanarPCS.from_links(
    [
        LinkSpec.circular(
            length=0.1,
            radius=0.01,
            density=1000.0,
            young_modulus=1e6,
            shear_modulus=1e5,
            material_damping_coefficient=318.0,
            reference_strain=[0.0, 1.0, 0.0],
        )
    ],
    backend="auto",
)
```

The setting affects selected continuum-system kinematics and dynamics operations.

## Choosing a backend

The `backend` constructor argument accepts three values:

| Value | Behavior |
|---|---|
| `"auto"` | Uses Warp for supported primal kinematics and dynamics on a GPU and JAX/XLA otherwise. This is the default. |
| `"jax"` | Always uses the reference JAX/XLA implementation. |
| `"warp"` | Requests the Warp implementation where the system, quadrature rule, and device support it. |

Automatic selection has no batch-size, model-order, or GPU-model crossover.
This keeps behavior predictable across devices. If a GPU environment does not
have the optional Warp dependency installed, either install it or construct the
model with `backend="jax"`.

Install SoRoMoX together with the CUDA 13 and Warp dependencies using:

```bash
pip install "soromox[cuda13]"
```

If JAX already has working GPU support, add only Warp with:

```bash
pip install "soromox[warp]"
```

!!! note "Backend configuration is static"
    Choose the model backend during construction. It is part of the model's
    compiled structure, so changing it creates a different compilation rather
    than a runtime branch inside every dynamics call.

## Supported systems and methods

| System | Warp support | Requirements |
|---|---|---|
| `PlanarPCS` | Kinematics and dynamics on GPU; Warp-native linear-threadlike integration; explicit Warp kinematics is also available on CPU | Dynamics requires exactly five Gauss points; Warp requires FP64 arrays |
| `PCS` | Kinematics and dynamics on GPU; Warp-native linear-threadlike integration; explicit Warp kinematics is also available on CPU | Dynamics requires exactly five Gauss points; Warp requires FP64 arrays |
| `GVS` | Kinematics and dynamics on GPU; Warp-native linear-threadlike integration; explicit Warp execution is also available on CPU | FP64 arrays |

Other systems, including `PlanarHSA`, continue to use JAX/XLA. For PCS models
with a quadrature rule other than five Gauss points, `"auto"` falls back to
JAX. Explicitly requesting `"warp"` for such a model raises an error instead of
silently changing its quadrature.

The selected dynamics backend is used by:

- `dynamics_terms(q, qd)`, which returns `(B, Cqd, G)`;
- `forward_dynamics(t, y, actuation_args)`;
- `rollout_to(...)` and `rollout_closed_loop_to(...)`, through their calls to
  `forward_dynamics`.

The selected kinematics backend is used by:

- `forward_kinematics(q, s)` and
  `forward_kinematics_abscissa_batched(q, s_ps)`;
- `jacobian_inertialframe(q, s)` and
  `jacobian_inertialframe_abscissa_batched(q, s_ps)`;
- `forward_kinematics_and_jacobian_inertialframe(...)` and its
  `*_abscissa_batched` counterpart, which compute both results in one Warp
  traversal.

The setting does **not** change direct calls to `inertia_matrix`,
`coriolis_matrix`, `gravitational_force`, body-frame Jacobians, Jacobian
derivatives, energy functions, constitutive forces, passive threadlike
impedance, actuator path coordinates, or rendering. Those methods remain JAX
operations. To evaluate backend-accelerated inertia,
Coriolis/centrifugal, and gravity terms together, call `dynamics_terms`.

### Threadlike actuation

Built-in linear threadlike layouts have public Warp-native operands, FP64
kernels, and allocation-free launchers for actuation matrices `(E, D, A)` and
fused direct-effort forces `(E, D)`. These low-level APIs are always available
from `soromox.systems.execution.warp.pcs` and `.gvs` for Newton and other
Warp-native integrations. They support multiple ordered threadlike actuator
groups and every built-in modality.

For spatial `PCS`, `actuation_matrix` uses the Warp implementation when the GPU
backend is selected. `PlanarPCS` and `GVS` actuation matrices, and all
system-level `actuation_force` methods, use JAX. Passing `backend="warp"` for an
unsupported system-level operation explains how to access the low-level
Warp-native API. Passive threadlike impedance and actuator path coordinates
also remain JAX operations.

When Warp dynamics is selected, `GVS`, `PCS`, and `PlanarPCS` forward dynamics
combine eligible built-in linear threadlike actuation with the dynamics
evaluation. Other actuation layouts continue to evaluate actuation with JAX.

## Model-level and per-call selection

The constructor setting controls normal use:

```python
robot = PlanarPCS.from_links(links, backend="auto")
B, Cqd, G = robot.dynamics_terms(q, qd)
yd = robot.forward_dynamics(t, y)
trajectory = robot.rollout_to(initial_state, t1=1.0)
```

The accelerated kinematics and dynamics methods additionally accept per-call
overrides. This is useful for validation and benchmarking without constructing
another model:

```python
B_jax, Cqd_jax, G_jax = robot.dynamics_terms(q, qd, backend="jax")
B_warp, Cqd_warp, G_warp = robot.dynamics_terms(q, qd, backend="warp")
yd_jax = robot.forward_dynamics(t, y, backend="jax")
yd_warp = robot.forward_dynamics(t, y, backend="warp")
pose_jax = robot.forward_kinematics(q, s, backend="jax")
pose_warp = robot.forward_kinematics(q, s, backend="warp")
```

The rollout helpers use the model-level setting so every evaluation in an
integration follows one consistent policy.

## Batches and `vmap`

`dynamics_terms` accepts either one state or a leading environment batch:

```python
B, Cqd, G = robot.dynamics_terms(q, qd)
B_batch, Cqd_batch, G_batch = robot.dynamics_terms(q_batch, qd_batch)
```

Normal JAX vectorization has the same batched Warp behavior:

```python
import jax

B_batch, Cqd_batch, G_batch = jax.vmap(robot.dynamics_terms)(q_batch, qd_batch)
```

The mapped call is combined into one batch-shaped Warp pipeline. It does not
launch an independent one-environment pipeline for every batch item.

Kinematics keeps scalar and abscissa-batched methods distinct. The scalar
methods accept one configuration and one abscissa. Use the explicit
`*_abscissa_batched` methods for one configuration and multiple abscissae:

```python
poses = robot.forward_kinematics_abscissa_batched(q, s_ps)
jacobians = robot.jacobian_inertialframe_abscissa_batched(q, s_ps)
```

Normal JAX transformations cover spatial, environment, pairwise, and combined
batches without broadening the single-point method's input contract:

```python
# Multiple abscissae for one configuration: N
poses_spatial = jax.vmap(
    robot.forward_kinematics,
    in_axes=(None, 0),
)(q, s_ps)

# Multiple configurations at one shared scalar abscissa: E
poses_environments = jax.vmap(
    robot.forward_kinematics,
    in_axes=(0, None),
)(q_batch, s)

# Pairwise configurations and scalar abscissae: E
poses_pairwise = jax.vmap(robot.forward_kinematics)(q_batch, s_batch)

# Cartesian configurations by shared abscissae: E x N
poses_cartesian = jax.vmap(
    robot.forward_kinematics_abscissa_batched,
    in_axes=(0, None),
)(q_batch, s_ps)

# A different abscissa batch for every configuration: E x N
poses_per_environment = jax.vmap(
    robot.forward_kinematics_abscissa_batched,
)(q_batch, s_per_environment)
```

Spatial JAX `vmap` calls use each system's specialized abscissa traversal. Warp
maps are combined into one canonical environment-by-abscissa executor call,
including nested scalar `vmap` expressions for Cartesian batches.

## Differentiation

JAX transformations always differentiate the JAX implementation, even when
the model uses Warp for ordinary primal GPU calls. This applies to accelerated
kinematics and Jacobians as well as `model.dynamics_terms` and complete
`forward_dynamics` evaluations.

```python
import jax
import jax.numpy as jnp

def objective(q_value):
    B, Cqd, G = robot.dynamics_terms(q_value, qd)
    return jnp.sum(B) + jnp.sum(Cqd**2) + jnp.sum(G**2)

gradient = jax.grad(objective)(q)
```

The same routing applies to `jax.jvp`, `jax.jacfwd`, `jax.jacrev`, and reverse
mode through a rollout. The returned derivatives therefore retain the existing
JAX semantics; gradients are not computed by differentiating the current
forward-only Warp kernels.

## Performance settings

The default block sizes are portable choices rather than GPU-specific tuning
heuristics:

- `PlanarPCS`: `PCSBackendParams(warp_block_dim=128)`;
- `PCS`: `PCSBackendParams(warp_block_dim=192)`;
- `GVS`: `GVSBackendParams(warp_block_dim=128)` for at most 64 active
  coordinates and `warp_block_dim=192` for larger models.

Advanced users can provide a multiple of 32 from 32 through 1024 when the model
is constructed:

```python
from soromox.systems import GVS, PCS, GVSBackendParams, PCSBackendParams

robot = PCS.from_links(
    links,
    backend="auto",
    backend_params=PCSBackendParams(warp_block_dim=256),
)

gvs_robot = GVS.from_segments(
    segments,
    backend="auto",
    backend_params=GVSBackendParams(warp_block_dim=192),
)
```

Changing this value affects compilation and may improve or reduce performance
depending on topology, batch size, and GPU architecture. Benchmark the complete
application rather than selecting a block size from occupancy alone.

For reliable GPU measurements:

1. enable JAX 64-bit mode before constructing state arrays;
2. warm up the exact state and batch shapes before timing;
3. keep environments in one explicit batch or one `vmap` call;
4. synchronize results when measuring elapsed wall-clock time.

```python
import jax

jax.config.update("jax_enable_x64", True)
```

The first call includes JAX and Warp compilation. Repeated calls reuse compiled
programs for the same static model and input shapes.

## Advanced Warp-native integrations

Most users should stay with the system methods described above. Integrators
that need caller-owned Warp buffers or CUDA-graph-capturable mechanics can use
the public family namespaces under `soromox.systems.execution.warp`. Those
lower-level interfaces have stricter buffer, dtype, and launch-order contracts
and are intended for integration packages rather than ordinary simulation
scripts.
