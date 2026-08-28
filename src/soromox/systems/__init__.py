from soromox.autodiff import (
    custom_jvp_enabled,
    custom_jvp_mode,
    set_custom_jvp_enabled,
)
from soromox.systems.articulated.params import (
    ArticulatedSoftRobotParams,
    McKibbenActuatedUMArmParams,
)
from soromox.systems.components import (
    ContinuumLinkParams,
    CrossSectionGeometry,
    CrossSectionParams,
    IsotropicMaterialParams,
    JointParams,
    JointSpec,
    LinearProfile,
    LinkSpec,
    shear_modulus_from_poisson_ratio,
)
from soromox.systems.dynamical_system import DynamicalSystem
from soromox.systems.execution import (
    ExecutionBackend,
    GVSBackendParams,
    PCSBackendParams,
)
from soromox.systems.gvs.params import GVSParams
from soromox.systems.gvs.structures import (
    GVSJointStructure,
    GVSLinkStructure,
    GVSSegmentStructure,
    GVSStrainBasisStructure,
    GVSStructure,
)
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
    PlanarHSAParams,
    PlanarPCSParams,
)
from soromox.systems.pcs.structures import (
    ISupportStructure,
    PCSStructure,
    PlanarHSAStructure,
    PlanarPCSStructure,
)
from soromox.systems.pendulum.params import PendulumParams
from soromox.systems.soft_robot import SoftRobot
from soromox.systems.system_state import EnvironmentState, SystemState

from .articulated import ArticulatedSoftRobot, McKibbenActuatedUMArm
from .gvs import (
    GVS,
    GVSSegment,
    StrainBasisSpec,
)
from .pcs import (
    PCS,
    ISupport,
    PlanarHSA,
    PlanarPCS,
)
from .pendulum import Pendulum

__all__ = [
    # base classes
    "DEFAULT_GRAVITY_MAGNITUDE",
    "DynamicalSystem",
    "EnvironmentState",
    "ExecutionBackend",
    "GVSBackendParams",
    "PCSBackendParams",
    "SoftRobot",
    "CrossSectionGeometry",
    "CrossSectionParams",
    "ContinuumLinkParams",
    "JointParams",
    "IsotropicMaterialParams",
    "LinearProfile",
    "LinkSpec",
    "JointSpec",
    "shear_modulus_from_poisson_ratio",
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
    "GVSStructure",
    "GVSSegmentStructure",
    "GVSLinkStructure",
    "GVSJointStructure",
    "GVSStrainBasisStructure",
    "GVSSegment",
    "StrainBasisSpec",
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
    "PlanarHSA",
    "PlanarHSAParams",
    "PlanarHSAStructure",
    "PlanarPCS",
    "PlanarPCSParams",
    "PlanarPCSStructure",
]
