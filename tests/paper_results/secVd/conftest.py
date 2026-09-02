"""Run the Section Vd tests on the CPU.

The case deliberately holds no opinion about which backend is faster, so a run
defers to JAX, which selects the GPU whenever one is present. For these tests
that is the wrong default twice over.

It is slower. A Section Vd rollout is tens of thousands of strictly sequential
solver steps on matrices of a few dozen entries, so the GPU is bound by
kernel-launch latency rather than throughput. Pinning the CPU takes this package
from roughly 90 s to 25 s.

And it is non-deterministic. cuSolver and cuBLAS allocate their handles lazily,
and that allocation fails outright when anything else on the machine holds the
card -- another test session, or an optimization run, which the Warp loader
gives a CUDA context to even when its own work is on the CPU. The failure
surfaces as a fixture error nowhere near its cause.

``JAX_PLATFORMS`` is the mechanism rather than ``jax.default_device`` because
execution dispatch keys on :func:`jax.default_backend`, which the latter does
not change: pinning only the default *device* leaves dispatch selecting the
fused Warp CUDA path and then handing it CPU arrays, which fails.

JAX reads the variable once, at import. Running this package directly gets the
CPU; inside a larger session that already imported JAX the setting is inert and
the tests run wherever JAX landed, which is correct but slower.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
