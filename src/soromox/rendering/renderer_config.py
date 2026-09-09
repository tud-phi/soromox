"""Backend-independent renderer settings with photometric light units."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Literal

import numpy as np

from soromox.rendering.camera_config import CameraConfig
from soromox.rendering.color_config import (
    ActuatorStyleConfig,
    BackboneColorConfig,
    RendererColorConfig,
)
from soromox.rendering.video_encoding import VideoEncodingConfig

RGB = tuple[float, float, float]
Vector3 = tuple[float, float, float]


def _validate_vector(value, name, *, color=False, nonzero=False):
    """Validate a finite vector, optionally restricting its range or length.

    Args:
        value: Three numeric components.
        name: Field name used in errors.
        color: Require sRGB components in [0, 1].
        nonzero: Require a nonzero vector.

    Returns:
        None.

    Raises:
        ValueError: The vector violates a requested constraint.
    """
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite three-vector")
    if color and ((a < 0).any() or (a > 1).any()):
        raise ValueError(f"{name} must lie in [0, 1]")
    if nonzero and np.linalg.norm(a) == 0:
        raise ValueError(f"{name} must be nonzero")


def _nonnegative(value, name):
    """Validate a finite nonnegative scalar.

    Args:
        value: Numeric scalar.
        name: Field name used in errors.

    Returns:
        None.

    Raises:
        ValueError: The value is negative or nonfinite.
    """
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


@dataclass
class AmbientLightConfig:
    """Approximate environment illumination.

    Attributes:
        strength: Nonnegative relative strength; one is the reference environment.
        color: sRGB illumination tint in [0, 1]. Viser approximates environment
            lighting with a world-up hemisphere; Open3D uses its environment map.
    """

    strength: float = 0.6
    color: RGB = (1.0, 1.0, 1.0)

    def __post_init__(self):
        """Validate intensity and tint.

        Returns:
            None.

        Raises:
            ValueError: Strength or color is invalid.
        """
        _nonnegative(self.strength, "strength")
        _validate_vector(self.color, "color", color=True)


@dataclass
class DirectionalLightConfig:
    """Distant light with constant illuminance.

    Attributes:
        direction: World direction in which light travels; normalized by adapters.
        illuminance_lux: Nonnegative illuminance in lux.
        color: sRGB illumination tint.
        cast_shadow: Whether this light casts shadows when scene shadows are enabled.
    """

    direction: Vector3 = (-0.25, 0.15, -1.0)
    illuminance_lux: float = 60000.0
    color: RGB = (1.0, 0.97, 0.94)
    cast_shadow: bool = True

    def __post_init__(self):
        """Validate photometric intensity, color and direction.

        Returns:
            None.

        Raises:
            ValueError: A light field is invalid.
        """
        _validate_vector(self.direction, "direction", nonzero=True)
        _validate_vector(self.color, "color", color=True)
        _nonnegative(self.illuminance_lux, "illuminance_lux")


@dataclass
class PointLightConfig:
    """Isotropic point light with inverse-square attenuation.

    Attributes:
        position: World position in metres.
        intensity_candela: Nonnegative luminous intensity in candela.
        color: sRGB illumination tint.
        range_m: Positive finite influence distance in metres.
        cast_shadow: Whether this light casts shadows.
    """

    position: Vector3 = (-0.3, -0.4, 1.1)
    intensity_candela: float = 2785.21
    color: RGB = (1.0, 0.98, 0.96)
    range_m: float = 4.0
    cast_shadow: bool = False

    @classmethod
    def from_lumens(cls, lumens: float, **kwargs) -> PointLightConfig:
        """Construct an isotropic light from its total luminous flux.

        Args:
            lumens: Total flux in lumens, distributed over 4 pi steradians.
            **kwargs: Remaining point-light fields.

        Returns:
            Point light with intensity equal to ``lumens / (4*pi)``.

        Raises:
            ValueError: Flux or another field is invalid.
        """
        _nonnegative(lumens, "lumens")
        return cls(intensity_candela=lumens / (4 * np.pi), **kwargs)

    def __post_init__(self):
        """Validate photometric intensity, position, color and range.

        Returns:
            None.

        Raises:
            ValueError: A light field is invalid.
        """
        _validate_vector(self.position, "position")
        _validate_vector(self.color, "color", color=True)
        _nonnegative(self.intensity_candela, "intensity_candela")
        if not np.isfinite(self.range_m) or self.range_m <= 0:
            raise ValueError("range_m must be finite and positive")


@dataclass
class MaterialConfig:
    """Default surface response for robot geometry.

    Attributes:
        shading: Lit standard/toon shading or unlit color rendering.
        roughness: Roughness in [0, 1].
        metallic: Metallic fraction in [0, 1].
        reflectance: Dielectric reflectance in [0, 1].
        opacity: Multiplier for per-object alpha in [0, 1].
        flat_shading: Use face normals rather than smooth vertex normals.
        wireframe: Draw mesh edges instead of filled triangles when supported.
    """

    shading: Literal["standard", "unlit", "toon3", "toon5"] = "standard"
    roughness: float = 0.72
    metallic: float = 0.0
    reflectance: float = 0.35
    opacity: float = 1.0
    flat_shading: bool = False
    wireframe: bool = False

    def __post_init__(self):
        """Validate the shading mode and normalized material parameters.

        Returns:
            None.

        Raises:
            ValueError: Shading or a material parameter is invalid.
        """
        if self.shading not in ("standard", "unlit", "toon3", "toon5"):
            raise ValueError("Unknown material shading")
        for name in ("roughness", "metallic", "reflectance", "opacity"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")


@dataclass
class GroundPlaneConfig:
    """World floor or per-robot base reference plane.

    Attributes:
        visible: Display the ground reference, including its surface and grid.
        surface: Draw a filled surface; False displays only the optional grid.
        alignment: World alignment or alignment with each robot's base tangent.
        height: Signed displacement along the normal, in metres.
        normal: Explicit world normal; None selects the robot's world-up direction.
        size: Side length in metres; None fits the complete scene.
        color: Surface sRGB color.
        opacity: Surface opacity in [0, 1].
        grid: Display a grid on the floor.
        grid_spacing: Grid spacing in metres; None selects a nearby 1/2/5 step.
        grid_major_every: Number of minor intervals between major grid lines.
        grid_major_color: Major grid line sRGB color.
        grid_color: Grid sRGB color.
        receive_shadow: Receive cast shadows where supported.
    """

    visible: bool = True
    surface: bool = True
    alignment: Literal["world", "base"] = "world"
    height: float = 0.0
    normal: Vector3 | None = None
    size: float | None = None
    color: RGB = (0.94, 0.95, 0.96)
    opacity: float = 1.0
    grid: bool = True
    grid_spacing: float | None = None
    grid_color: RGB = (0.86, 0.86, 0.86)
    grid_major_every: int = 5
    grid_major_color: RGB = (0.72, 0.72, 0.72)
    receive_shadow: bool = True

    def __post_init__(self):
        """Validate alignment, dimensions and surface colors.

        Returns:
            None.

        Raises:
            ValueError: A ground field is invalid.
        """
        if self.alignment not in ("world", "base"):
            raise ValueError("alignment must be world or base")
        if not np.isfinite(self.height):
            raise ValueError("height must be finite")
        if self.normal is not None:
            _validate_vector(self.normal, "normal", nonzero=True)
        for name in ("size", "grid_spacing"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.opacity <= 1:
            raise ValueError("opacity must lie in [0, 1]")
        _validate_vector(self.color, "color", color=True)
        _validate_vector(self.grid_color, "grid_color", color=True)
        _validate_vector(self.grid_major_color, "grid_major_color", color=True)
        if (
            isinstance(self.grid_major_every, bool)
            or not isinstance(self.grid_major_every, (int, np.integer))
            or self.grid_major_every < 1
        ):
            raise ValueError("grid_major_every must be a positive integer")


@dataclass
class BackdropConfig:
    """Curved floor/wall dimensions relative to the fitted scene extent.

    Attributes:
        enabled: Replace the ground plane with a curved backdrop.
        width: Width as a multiple of scene extent.
        depth: Forward floor reach as a multiple of scene extent.
        height: Wall height as a multiple of scene extent.
        radius: Floor-to-wall bend radius as a multiple of scene extent.
        wall_offset: Distance behind scene center before the bend, in scene extents.
    """

    enabled: bool = False
    width: float = 5.0
    depth: float = 2.5
    height: float = 2.5
    radius: float = 0.5
    wall_offset: float = 0.42

    def __post_init__(self):
        """Validate positive backdrop dimensions.

        Returns:
            None.

        Raises:
            ValueError: A dimension is nonfinite or nonpositive.
        """
        for name in ("width", "depth", "height", "radius", "wall_offset"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass
class SceneConfig:
    """Shared scene appearance; unsupported features produce backend warnings.

    Attributes:
        background: sRGB background color.
        ground: Ground surface and reference grid.
        backdrop: Curved studio scenery.
        ambient: Relative environment illumination.
        lights: Directional and point lights in physical units.
        material: Default robot surface material.
        shadows: Enable cast shadows globally.
        ambient_occlusion: Enable local occlusion shading.
        tone_mapping: Backend default, linear, or ACES output transformation.
        backbone_cast_shadow: Backbone shadow casting.
        sphere_cast_shadow: Helper sphere shadow casting.
    """

    background: RGB = (1.0, 1.0, 1.0)
    ground: GroundPlaneConfig = field(
        default_factory=lambda: GroundPlaneConfig(surface=False)
    )
    backdrop: BackdropConfig = field(default_factory=BackdropConfig)
    ambient: AmbientLightConfig = field(
        default_factory=lambda: AmbientLightConfig(strength=0.8)
    )
    lights: tuple[DirectionalLightConfig | PointLightConfig, ...] = field(
        default_factory=lambda: (
            DirectionalLightConfig(illuminance_lux=80000, color=(1.0, 1.0, 1.0)),
            PointLightConfig.from_lumens(
                250000, position=(0.0, -0.8, 0.8), range_m=6.0
            ),
        )
    )
    material: MaterialConfig = field(default_factory=MaterialConfig)
    shadows: bool = False
    ambient_occlusion: bool = False
    tone_mapping: Literal["backend-default", "linear", "aces"] = "backend-default"
    backbone_cast_shadow: bool = True
    sphere_cast_shadow: bool = True

    def __post_init__(self):
        """Validate the background and output transformation.

        Returns:
            None.

        Raises:
            ValueError: The background, light type or tone mapping is invalid.
        """
        _validate_vector(self.background, "background", color=True)
        if self.tone_mapping not in ("backend-default", "linear", "aces"):
            raise ValueError("Unknown tone_mapping")
        if any(
            not isinstance(light, (DirectionalLightConfig, PointLightConfig))
            for light in self.lights
        ):
            raise ValueError(
                "lights must contain directional or point light configurations"
            )

    @classmethod
    def technical(cls, *, scene_extent: float = 1.2, **overrides) -> SceneConfig:
        """Create a neutral technical view with a grid and balanced illumination.

        Args:
            scene_extent: Reference extent in metres; scales preset point lighting.
            **overrides: Scene fields replacing the defaults.

        Returns:
            Independent, editable scene configuration.

        Raises:
            ValueError: Scene extent is not finite and positive.
        """
        if not np.isfinite(scene_extent) or scene_extent <= 0:
            raise ValueError("scene_extent must be finite and positive")
        scene = cls()
        scale = scene_extent / 1.2
        point = scene.lights[1]
        point.position = tuple(np.asarray(point.position) * scale)
        point.intensity_candela *= scale**2
        point.range_m *= scale
        return replace(scene, **overrides)

    @classmethod
    def flat(cls, **overrides) -> SceneConfig:
        """Create an unlit white scene preserving assigned object colors.

        Args:
            **overrides: Scene fields replacing the preset defaults.

        Returns:
            Editable scene without floor, lighting, shadows or ambient occlusion.
        """
        return replace(
            cls(
                ground=GroundPlaneConfig(visible=False),
                lights=(),
                ambient=AmbientLightConfig(strength=0),
                material=MaterialConfig(shading="unlit"),
                tone_mapping="linear",
            ),
            **overrides,
        )

    @classmethod
    def studio(
        cls,
        style: Literal["neutral", "bright", "dark"] = "neutral",
        *,
        scene_extent: float = 1.2,
        **overrides,
    ) -> SceneConfig:
        """Create a studio scene with concrete world-space lights.

        Args:
            style: Neutral grey, bright white, or dark charcoal studio. The dark
                studio uses a frontal key to illuminate upright robot surfaces.
            scene_extent: Reference object arrangement extent in metres. Positions
                scale linearly and point intensities quadratically with this value.
            **overrides: Scene fields replacing the preset defaults.

        Returns:
            Editable scene with a curved backdrop and matte materials.

        Raises:
            ValueError: Style or scene extent is invalid.
        """
        if style not in ("neutral", "bright", "dark"):
            raise ValueError("style must be neutral, bright or dark")
        if not np.isfinite(scene_extent) or scene_extent <= 0:
            raise ValueError("scene_extent must be finite and positive")
        scale = scene_extent / 1.2
        color = {
            "neutral": (0.58, 0.58, 0.58),
            "bright": (0.95, 0.95, 0.95),
            "dark": (0.045, 0.05, 0.065),
        }[style]
        key = {"neutral": 70000, "bright": 85000, "dark": 100000}[style]
        ambient = {"neutral": 0.8, "bright": 1.2, "dark": 0.4}[style]
        lights = [
            DirectionalLightConfig(
                illuminance_lux=key,
                color=(1.0, 1.0, 1.0),
                direction=(-0.25, 0.9, -0.55)
                if style == "dark"
                else (-0.25, 0.15, -1.0),
            ),
            PointLightConfig.from_lumens(
                {"neutral": 250000, "bright": 400000, "dark": 300000}[style] * scale**2,
                position=tuple(np.array([0.0, -0.8, 0.8]) * scale),
                range_m=6.0 * scale,
            ),
            PointLightConfig.from_lumens(
                {"neutral": 150000, "bright": 250000, "dark": 250000}[style] * scale**2,
                position=tuple(
                    np.array([0.0, 0.4, 0.75] if style != "dark" else [0.8, 0.35, 0.75])
                    * scale
                ),
                range_m=6.0 * scale,
                color=(1.0, 1.0, 1.0) if style != "dark" else (0.8, 0.88, 1.0),
            ),
        ]
        return replace(
            cls(
                background=color,
                ground=GroundPlaneConfig(color=color, grid=False, surface=True),
                backdrop=BackdropConfig(enabled=True, wall_offset=0.15, radius=0.7),
                ambient=AmbientLightConfig(strength=ambient),
                lights=tuple(lights),
                shadows=True,
                ambient_occlusion=True,
            ),
            **overrides,
        )


@dataclass
class GeometryConfig:
    """Shared geometry sampling and display dimensions.

    Attributes:
        num_points: Backbone samples, at least two.
        cross_section_resolution: Contour samples, at least three.
        backbone_style: Swept surface or discrete cross-section markers.
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
    base_plate_radius_scale: float = 2.0
    base_plate_thickness: float = 0.06
    line_width: float | None = 4.0
    actuator_line_width: float = 2.0
    grid_spacing: tuple[float, float] = (0.5, 0.5)

    def __post_init__(self):
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
class RenderOutputConfig:
    """Image dimensions and video encoding defaults.

    Attributes:
        width: Positive image width in pixels.
        height: Positive image height in pixels.
        video: Default video encoding configuration.
    """

    width: int = 800
    height: int = 600
    video: VideoEncodingConfig = field(default_factory=VideoEncodingConfig)

    def __post_init__(self):
        """Validate integer image dimensions.

        Returns:
            None.

        Raises:
            ValueError: A dimension is not a positive integer.
        """
        for name in ("width", "height"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")


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
        cls, color: RGB = (0.72, 0.65, 0.56), *, scene_extent: float = 1.2, **overrides
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
        scene = SceneConfig.studio(scene_extent=scene_extent)
        scene.material.roughness = 0.9
        scale = scene_extent / 1.2
        scene.lights += (
            PointLightConfig.from_lumens(
                100000 * scale**2,
                position=tuple(np.array([0.7, 0.4, 0.6]) * scale),
                range_m=6.0 * scale,
                color=(0.85, 0.92, 1.0),
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
