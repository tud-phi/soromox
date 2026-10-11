# Native GVS indexing and Warp backend trials

Baseline: SoRoMoX `79f8bf0f9cc78bc911b47bb98cc8fa1aef737839` (`origin/main` when the branch was created). Candidate: this PR. Hardware/software: RTX 5090, Warp 1.16, JAX 0.11.1, FP64 throughout, at most 512 worlds. Runs were serial, outside the sandbox, with JAX preallocation disabled. Timings exclude construction, compilation, and warm-up and are medians of five repeats. Each repeat dispatches ten complete calls before synchronizing. This includes the normal JAX dispatch path, rather than timing only an isolated cell kernel.

Models: `small` has two links and linear bases for two bending strains (8 DOFs); `moving` adds alternating revolute/prismatic joints (10 DOFs). `large8` and `large16` use 8/16 fixed-joint links, quadratic bases for all six strains on every link (144/288 DOFs). States and velocities are nonzero. Quadrature alternates between five and seven points. Full forward acceleration includes gravity, Coriolis, stiffness and damping.

## Full default-backend and prepared-executor timings

Microseconds per call. The prepared API caches invariant fixed-joint terms. Its moving-joint case evaluates joints normally. Neither API uses the optional native Cholesky experiment below for these measurements; all forward columns use the established JAX solve.

| Case | Worlds | Prior terms | Native terms | Prepared terms | Prior forward | Native forward | Prepared forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 32 | 185.29 | 174.46 | 156.89 | 215.78 | 202.68 | 188.88 |
| small | 128 | 185.94 | 176.66 | 157.73 | 214.34 | 188.50 | 187.27 |
| small | 512 | 272.76 | 272.11 | 235.03 | 307.18 | 302.17 | 274.00 |
| moving | 32 | 184.21 | 154.22 | 175.70 | 219.48 | 191.90 | 211.72 |
| moving | 128 | 182.50 | 176.96 | 175.82 | 224.71 | 213.99 | 212.08 |
| moving | 512 | 293.20 | 272.89 | 275.18 | 329.34 | 311.16 | 318.40 |
| large8 | 32 | 1461.79 | 1418.20 | 1397.21 | 2421.48 | 2369.40 | 2335.00 |
| large8 | 128 | 1521.70 | 1447.24 | 1398.04 | 2475.19 | 2423.74 | 2376.76 |
| large8 | 512 | 4195.79 | 4258.36 | 4123.01 | 6115.63 | 6148.38 | 5998.17 |
| large16 | 32 | 7792.82 | 7774.53 | 7734.42 | 12228.13 | 12174.27 | 12144.83 |
| large16 | 128 | 8569.22 | 8555.18 | 8535.84 | 13361.73 | 13344.25 | 13456.25 |
| large16 | 512 | 34071.59 | 34065.44 | 33787.33 | 44337.82 | 44318.67 | 44140.15 |

The default full forward call improved by approximately 2–13% for the 8/10-DOF cases in these runs. Large-model results were mostly neutral (roughly -0.5% to +2%). The optional prepared path improved the fixed 8-DOF reference by approximately 11–13% against the baseline; its moving-joint timings show why this is not a universal speedup.

The default change adopts two threads per CUDA cell to divide independent tangent columns and adjoint entries. Native per-world indexing also supports external packed integrators. The prepared executor is explicit because its cache is valid only for immutable model operands; it must be rebuilt after changes. Public differentiable model calls retain their established derivative rules. This forward-only prepared API does not introduce a new derivative rule.

## Optional native linear solve

These trials reuse actual GVS inertia matrices from the default native run. The native column times a 32-thread Warp Cholesky kernel inside a warmed CUDA graph (30 calls per replay); the JAX column includes 30 normal JIT dispatches of the existing LU-based solve. Both use FP64 and five repeats. The different dispatch boundaries make this a native-loop integration trial, not evidence that substituting this kernel will accelerate the full JAX backend.

| Model | Worlds | DOFs | Native Cholesky (us) | JAX solve (us) |
|---|---:|---:|---:|---:|
| small | 32 | 8 | 15.22 | 47.95 |
| small | 128 | 8 | 15.09 | 39.11 |
| small | 512 | 8 | 15.56 | 41.34 |
| moving | 32 | 10 | 19.74 | 48.54 |
| moving | 128 | 10 | 19.73 | 49.08 |
| moving | 512 | 10 | 20.24 | 41.65 |
| large8 | 32 | 144 | 2173.84 | 929.97 |
| large8 | 128 | 144 | 2246.45 | 968.09 |
| large8 | 512 | 144 | 3008.37 | 1828.66 |
| large16 | 32 | 288 | 15389.85 | 4387.76 |
| large16 | 128 | 288 | 15586.91 | 4785.91 |
| large16 | 512 | 288 | 25529.05 | 10216.10 |

A transposed factor workspace was also trialed (`soromox-column-solve-trial.json`). It did not provide a consistent gain; the 288-DOF solves were slower. The default factor layout therefore remains unchanged.

The native kernel remains an explicit low-level API for Warp-native loops. The default JAX solve is unchanged. Packed descriptors support different dimensions per world, and pivot-floor status is exposed. Newton retains its specialized fused passive-force/system assembly around the cooperative solve.

## Accuracy and coverage

`validate_native_gvs.py` compares every saved world across revisions. Inertia, Coriolis and gravity terms must be bitwise equal. For forward acceleration it checks normalized backward error below 2e-14 and a condition-aware FP64 forward-error bound. The large quadratic models are ill-conditioned, so absolute acceleration differences alone are misleading; each JSON entry records the inertia condition number, absolute/relative acceleration difference and normalized solve residuals. Across all measured worlds, M, Cqd and G were bitwise equal. The small and moving cases also had bitwise-equal acceleration. For the large cases, condition numbers reached approximately 8.2e8: the largest relative acceleration change was 1.23e-10 (absolute 3.09e-5), while normalized solve residuals remained below 1.2e-17. No FP32 mechanics or altered strain/basis formulation is used.

GPU regression tests compare the public Warp API with JAX, including its derivative fallback, higher-order basis families, moving joints, dense and directional kinematics, native packed cells, fixed/floating prepared execution, cooperative solves through 288 DOFs, mixed-size packed solves and pivot-floor diagnostics. Newton's dependent PR additionally validates packed poses/Jacobians/JVP/VJP, all seven moving-joint families, identity/threadlike actuation, prescribed bases, mapped bodies, four integration policies and CUDA graph replay.

Validated on this checkout: 18 packed GPU tests passed; 14 existing GVS/GVS-directional GPU tests passed; 78 shared PCS/actuation tests passed with one skip; six architecture tests passed. Ruff checks and formatting passed for the changed files.

## Reproduce

Use the same script with PYTHONPATH pointing to each checkout's `src` directory, a Python environment with the versions above, and GPU access:

```bash
python tools/benchmarks/native_gvs.py --output /tmp/baseline.json
python tools/benchmarks/native_gvs.py --output /tmp/native.json
python tools/benchmarks/native_gvs.py --cache-fixed-joints --output /tmp/prepared.json
python tools/benchmarks/cooperative_solve.py --inputs /tmp/native.json --output /tmp/solve.json
JAX_PLATFORMS=cpu python -m tools.benchmarks.validate_native_gvs --baseline /tmp/baseline.json --candidate /tmp/native.json
JAX_PLATFORMS=cpu python -m tools.benchmarks.validate_native_gvs --baseline /tmp/baseline.json --candidate /tmp/prepared.json
SOROMOX_TEST_CUDA=1 python -m pytest tests/execution/warp/test_packed.py tests/execution/warp/test_gvs.py tests/execution/warp/test_gvs_directional.py -q
```

The benchmark writes NPZ numerical outputs next to its JSON. Large NPZ arrays are regenerated by these commands and are not committed. Raw JSON timing samples and validation metrics are committed here. The initial `soromox-baseline.json` / `soromox-native.json` runs used an equivalent scalar-vmapped forward expression; the `final` files use one identical batched harness for the default and prepared comparisons.
