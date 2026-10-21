"""Run all Section Vd tests on the CPU."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
