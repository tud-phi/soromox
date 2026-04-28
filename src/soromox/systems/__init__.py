from soromox.autodiff import (
    custom_jvp_enabled,
    custom_jvp_mode,
    set_custom_jvp_enabled,
)
from soromox.systems.dynamical_system import DynamicalSystem
from soromox.systems.soft_robot import (
    CrossSectionGeometry,
    SoftRobot,
)
from soromox.systems.system_state import EnvironmentState, SystemState

from .articulated import ArticulatedSoftRobot
from .gvs import (
    GVS,
    GVSSegment,
    JointSpec,
    LinkSpec,
    StrainBasisSpec,
    TendonActuatedGVS,
)
from .hsa import PlanarHSA
from .pcs import (
    PCS,
    ISupport,
    PlanarPCS,
    PneumaticActuatedPlanarPCS,
    TendonActuatedPCS,
    TendonActuatedPlanarPCS,
)
from .pendulum import Pendulum, TendonActuatedPendulum

__all__ = [
    # base classes
    "DynamicalSystem",
    "EnvironmentState",
    "SoftRobot",
    "CrossSectionGeometry",
    "SystemState",
    "custom_jvp_enabled",
    "custom_jvp_mode",
    "set_custom_jvp_enabled",
    # articulated systems
    "ArticulatedSoftRobot",
    # gvs systems
    "GVS",
    "GVSSegment",
    "LinkSpec",
    "JointSpec",
    "StrainBasisSpec",
    "TendonActuatedGVS",
    # hsa systems
    "PlanarHSA",
    # pendulum systems
    "Pendulum",
    "TendonActuatedPendulum",
    # pcs systems
    "PCS",
    "ISupport",
    "PlanarPCS",
    "PneumaticActuatedPlanarPCS",
    "TendonActuatedPCS",
    "TendonActuatedPlanarPCS",
]
