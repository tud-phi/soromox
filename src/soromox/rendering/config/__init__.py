"""Public configuration types shared by all rendering backends."""

from soromox.rendering.config.camera import (
    CameraConfig,
)
from soromox.rendering.config.colors import (
    ActuatorStyleConfig,
    BackboneColorConfig,
    RendererColorConfig,
)
from soromox.rendering.config.output import (
    RenderOutputConfig,
    VideoEncodingConfig,
)
from soromox.rendering.config.renderer import (
    GeometryConfig,
    RendererConfig,
)
from soromox.rendering.config.scene import (
    AmbientLightConfig,
    BackdropConfig,
    DirectionalLightConfig,
    GroundPlaneConfig,
    MaterialConfig,
    PointLightConfig,
    SceneConfig,
)

__all__ = [
    "CameraConfig",
    "ActuatorStyleConfig",
    "BackboneColorConfig",
    "RendererColorConfig",
    "RenderOutputConfig",
    "VideoEncodingConfig",
    "GeometryConfig",
    "RendererConfig",
    "AmbientLightConfig",
    "BackdropConfig",
    "DirectionalLightConfig",
    "GroundPlaneConfig",
    "MaterialConfig",
    "PointLightConfig",
    "SceneConfig",
]
