# Warp Integration

The `soromox.execution.warp` namespace contains allocation-free,
Warp-native building blocks for integrations that need to embed SoRoMoX
mechanics in a larger Warp or Newton solver.

!!! warning "Not needed for ordinary SoRoMoX simulation"
    Normal users should select `backend="warp"` or leave `backend="auto"` on a
    supported system. Direct kernel launches require caller-owned buffers,
    exact FP64 shapes, the documented launch ordering, and explicit CUDA-graph
    lifetime management.

The public child namespaces are:

- [GVS Warp API](gvs.md)
- [PCS Warp API](pcs.md)

::: soromox.execution.warp
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2
      members: false
