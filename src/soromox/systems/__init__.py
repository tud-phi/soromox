from soromox.autodiff import (
    custom_jvp_enabled,
    custom_jvp_mode,
    set_custom_jvp_enabled,
)
from soromox.systems.articulated.params import (
    ArticulatedSoftRobotParams,
    McKibbenActuatedUMArmParams,
)
from soromox.systems.dynamical_system import DynamicalSystem
from soromox.systems.gvs.params import GVSLinkParams, GVSParams
from soromox.systems.gvs.structures import (
    GVSJointStructure,
    GVSLinkStructure,
    GVSSegmentStructure,
    GVSStrainBasisStructure,
    GVSStructure,
)
from soromox.systems.hsa.params import PlanarHSAParams
from soromox.systems.hsa.structures import PlanarHSAStructure
from soromox.systems.params import (
    DEFAULT_GRAVITY_MAGNITUDE,
    BaseArticulatedSoftRobotParams,
    BaseContinuumSoftRobotParams,
    BaseSoftRobotParams,
    BaseSystemParams,
)
from soromox.systems.pcs.params import (
    ISupportParams,
    PCSParams,
    PlanarPCSParams,
)
from soromox.systems.pcs.structures import (
    ISupportStructure,
    PCSStructure,
    PlanarPCSStructure,
)
from soromox.systems.pendulum.params import PendulumParams
from soromox.systems.soft_robot import (
    CrossSectionGeometry,
    SoftRobot,
)
from soromox.systems.system_state import EnvironmentState, SystemState

from .articulated import ArticulatedSoftRobot, McKibbenActuatedUMArm
from .gvs import (
    GVS,
    GVSSegment,
    JointSpec,
    LinkSpec,
    StrainBasisSpec,
)
from .hsa import PlanarHSA
from .pcs import (
    PCS,
    ISupport,
    PlanarPCS,
)
from .pendulum import Pendulum

__all__ = [
    # base classes
    "DEFAULT_GRAVITY_MAGNITUDE",
    "DynamicalSystem",
    "EnvironmentState",
    "SoftRobot",
    "CrossSectionGeometry",
    "SystemState",
    "custom_jvp_enabled",
    "custom_jvp_mode",
    "set_custom_jvp_enabled",
    "BaseSystemParams",
    "BaseSoftRobotParams",
    "BaseContinuumSoftRobotParams",
    "BaseArticulatedSoftRobotParams",
    # articulated systems
    "ArticulatedSoftRobot",
    "ArticulatedSoftRobotParams",
    "McKibbenActuatedUMArm",
    "McKibbenActuatedUMArmParams",
    # gvs systems
    "GVS",
    "GVSParams",
    "GVSLinkParams",
    "GVSStructure",
    "GVSSegmentStructure",
    "GVSLinkStructure",
    "GVSJointStructure",
    "GVSStrainBasisStructure",
    "GVSSegment",
    "LinkSpec",
    "JointSpec",
    "StrainBasisSpec",
    # hsa systems
    "PlanarHSA",
    "PlanarHSAParams",
    "PlanarHSAStructure",
    # pendulum systems
    "Pendulum",
    "PendulumParams",
    # pcs systems
    "PCS",
    "PCSParams",
    "PCSStructure",
    "ISupportStructure",
    "ISupport",
    "ISupportParams",
    "PlanarPCS",
    "PlanarPCSParams",
    "PlanarPCSStructure",
]
