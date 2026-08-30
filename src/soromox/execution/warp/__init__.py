"""Warp-native building blocks for accelerated system execution.

The family subpackages separate reusable mechanics kernels from Soromox's
JAX-facing dispatch machinery. The :mod:`.gvs` and :mod:`.pcs` packages expose
runtime operands, workspace-shape contracts, raw kernels, and direct launch
functions that external Warp-native integrators can use with preallocated
buffers. Public kernel symbols are loaded lazily and require the optional
``warp-lang`` package only when requested.

Loader and JAX transformation modules are implementation details and are not
part of the stable integration API.
"""
