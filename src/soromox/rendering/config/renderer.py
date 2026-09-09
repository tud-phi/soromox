"""Composed renderer defaults, geometry sampling and configuration validation."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any, Literal

import numpy as np

from soromox.rendering.config._validation import RGB, _nonnegative, _validate_vector
from soromox.rendering.config.camera import CameraConfig
from soromox.rendering.config.colors import (
    ActuatorStyleConfig,
    BackboneColorConfig,
    RendererColorConfig,
)
from soromox.rendering.config.output import RenderOutputConfig
from soromox.rendering.config.scene import (
    AmbientLightConfig,
    BackdropConfig,
    DirectionalLightConfig,
    GroundPlaneConfig,
    MaterialConfig,
    PointLightConfig,
    SceneConfig,
)


@dataclass
class GeometryConfig:
    """Shared geometry sampling and display dimensions.

    Attributes:
        num_points: Backbone samples, at least two.
        cross_section_resolution: Contour samples, at least three.
        backbone_style: Swept surface or discrete cross-section markers.
        base_plate_style: Circular mounting shape for Open3D and Viser. The
            default flared collar has a flange, tapered body and upper rim.
        base_plate_radius_scale: Base plate radius relative to cross-section size.
        base_plate_thickness: Plate thickness in metres.
        line_width: Backbone width in pixels for line-based renderers; None selects
            a cross-section-derived width where supported.
        actuator_line_width: Actuator line width in pixels.
        grid_spacing: Multi-robot layout spacing in metres.
    """

    num_points: int = 80
    cross_section_resolution: int = 48
    backbone_style: Literal["swept", "discrete"] = "swept"
    base_plate_style: Literal[
        "disk", "beveled_disk", "truncated_cone", "flared_collar"
    ] = "flared_collar"
    base_plate_radius_scale: float = 2.0
    base_plate_thickness: float = 0.06
    line_width: float | None = 4.0
    actuator_line_width: float = 2.0
    grid_spacing: tuple[float, float] = (0.5, 0.5)

    def __post_init__(self) -> None:
        """Validate sample counts, dimensions and backbone style.

        Returns:
            None.

        Raises:
            ValueError: A count, dimension or style is invalid.
        """
        for name, minimum in (("num_points", 2), ("cross_section_resolution", 3)):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.backbone_style not in ("swept", "discrete"):
            raise ValueError("backbone_style must be swept or discrete")
        if self.base_plate_style not in (
            "disk",
            "beveled_disk",
            "truncated_cone",
            "flared_collar",
        ):
            raise ValueError("Unknown base_plate_style")
        for name in (
            "base_plate_radius_scale",
            "base_plate_thickness",
            "actuator_line_width",
        ):
            _nonnegative(getattr(self, name), name)
        if self.line_width is not None:
            _nonnegative(self.line_width, "line_width")
        spacing = np.asarray(self.grid_spacing)
        if (
            spacing.shape != (2,)
            or not np.isfinite(spacing).all()
            or (spacing <= 0).any()
        ):
            raise ValueError("grid_spacing must contain two positive finite distances")


@dataclass
class RendererConfig:
    """Composed configuration accepted by every renderer.

    Attributes:
        scene: Scene appearance, lighting and ground plane.
        camera: Default camera and exposure.
        colors: Robot colors; helper colors are supplied to rendering operations.
        geometry: Sampling, mesh style and display dimensions.
        output: Image dimensions and video encoding.
    """

    scene: SceneConfig = field(default_factory=SceneConfig.technical)
    camera: CameraConfig = field(default_factory=CameraConfig)
    colors: RendererColorConfig = field(default_factory=RendererColorConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    output: RenderOutputConfig = field(default_factory=RenderOutputConfig)

    @classmethod
    def clay(
        cls,
        color: RGB = (0.72, 0.65, 0.56),
        *,
        scene_extent: float = 1.2,
        **overrides: Any,
    ) -> RendererConfig:
        """Create a studio with uniform matte clay robot surfaces.

        Args:
            color: Warm-grey sRGB clay color; helper object colors are unaffected.
            scene_extent: Reference scene extent in metres for preset lighting.
            **overrides: Complete configuration sections replacing preset defaults.

        Returns:
            Editable renderer configuration with monochrome robot colors.

        Raises:
            ValueError: Color or scene extent is invalid.
        """
        _validate_vector(color, "color", color=True)
        if not np.isfinite(scene_extent) or scene_extent <= 0:
            raise ValueError("scene_extent must be finite and positive")
        scale = scene_extent / 1.2
        # Clay owns its lighting and scenery independently of the neutral studio.
        scene = SceneConfig(
            background=(0.58, 0.58, 0.58),
            ground=GroundPlaneConfig(color=(0.58, 0.58, 0.58), grid=False),
            backdrop=BackdropConfig(enabled=True, radius=0.7, wall_offset=0.15),
            ambient=AmbientLightConfig(strength=0.8),
            material=MaterialConfig(roughness=0.9),
            shadows=True,
            ambient_occlusion=True,
            lights=(
                DirectionalLightConfig(illuminance_lux=70000, color=(1.0, 1.0, 1.0)),
                PointLightConfig.from_lumens(
                    250000 * scale**2,
                    position=tuple(np.array([0.0, -0.8, 0.8]) * scale),
                    range_m=6.0 * scale,
                ),
                PointLightConfig.from_lumens(
                    150000 * scale**2,
                    position=tuple(np.array([0.0, 0.4, 0.75]) * scale),
                    range_m=6.0 * scale,
                    color=(1.0, 1.0, 1.0),
                ),
                PointLightConfig.from_lumens(
                    100000 * scale**2,
                    position=tuple(np.array([0.7, 0.4, 0.6]) * scale),
                    range_m=6.0 * scale,
                    color=(0.85, 0.92, 1.0),
                ),
            ),
        )
        colors = RendererColorConfig(
            backbone=BackboneColorConfig(
                segment_palette=[color], robot_palette=[color]
            ),
            base_plate_color=color,
            actuators=ActuatorStyleConfig(default_color=color),
            robot_override=color,
        )
        return replace(cls(scene=scene, colors=colors), **overrides)


def validate_config(config: RendererConfig) -> None:
    """Validate an editable configuration immediately before renderer construction.

    Args:
        config: Complete renderer configuration, including edits made after a
            preset factory returned it.

    Returns:
        None. Validates nested dataclass fields without changing them.

    Raises:
        TypeError: A top-level section has an incorrect type.
        ValueError: A field violates its configuration constraints.
    """
    for name, kind in (
        ("scene", SceneConfig),
        ("camera", CameraConfig),
        ("colors", RendererColorConfig),
        ("geometry", GeometryConfig),
        ("output", RenderOutputConfig),
    ):
        if not isinstance(getattr(config, name), kind):
            raise TypeError(f"{name} must be {kind.__name__}")

    def visit(value):
        """Validate nested values before their containing dataclass.

        Args:
            value: A configuration value or collection.

        Returns:
            None.

        Raises:
            ValueError: A nested configuration fails validation.
        """
        if is_dataclass(value):
            for item in fields(value):
                visit(getattr(value, item.name))
            validator = getattr(value, "__post_init__", None)
            if validator is not None:
                validator()
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(config)
    if not np.isfinite(config.camera.exposure_ev100):
        raise ValueError("exposure_ev100 must be finite")
