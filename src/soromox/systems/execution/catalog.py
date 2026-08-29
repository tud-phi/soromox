"""Execution capabilities of production continuum-system families."""

from soromox.systems.execution.types import (
    ActuationCapabilities,
    DynamicsCapabilities,
    KinematicsCapabilities,
)

# These flags are generated from the documented GPU production gate. The
# low-level Warp-native threadlike API is always available independently.
PLANAR_PCS_ACTUATION = ActuationCapabilities(
    family_name="PlanarPCS",
    warp_executor="pcs",
    matrix_enabled=False,
    force_enabled=False,
)
PCS_ACTUATION = ActuationCapabilities(
    family_name="PCS",
    warp_executor="pcs",
    matrix_enabled=True,
    force_enabled=False,
    warp_cpu_supported=True,
)
GVS_ACTUATION = ActuationCapabilities(
    family_name="GVS",
    warp_executor="gvs",
    matrix_enabled=False,
    force_enabled=False,
)

GVS_DYNAMICS = DynamicsCapabilities(
    family_name="GVS",
    warp_executor="gvs",
    warp_cpu_supported=True,
    fused_threadlike_force_enabled=True,
)
PCS_DYNAMICS = DynamicsCapabilities(
    family_name="PCS",
    warp_executor="pcs",
    fused_threadlike_force_enabled=True,
)
GVS_KINEMATICS = KinematicsCapabilities(
    family_name="GVS",
    warp_executor="gvs",
)
PCS_KINEMATICS = KinematicsCapabilities(
    family_name="PCS",
    warp_executor="pcs",
)

__all__ = [
    "GVS_ACTUATION",
    "GVS_DYNAMICS",
    "GVS_KINEMATICS",
    "PCS_ACTUATION",
    "PCS_DYNAMICS",
    "PCS_KINEMATICS",
    "PLANAR_PCS_ACTUATION",
]
