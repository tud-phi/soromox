"""Scene, lighting, material and ground settings with photometric units."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np

from soromox.rendering.config._validation import (
    RGB,
    Vector3,
    _nonnegative,
    _validate_vector,
)


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

    def __post_init__(self) -> None:
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

    def __post_init__(self) -> None:
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
    def from_lumens(cls, lumens: float, **kwargs: Any) -> PointLightConfig:
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

    def __post_init__(self) -> None:
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

    def __post_init__(self) -> None:
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

    def __post_init__(self) -> None:
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

    def __post_init__(self) -> None:
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

    def __post_init__(self) -> None:
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
    def technical(cls, *, scene_extent: float = 1.2, **overrides: Any) -> SceneConfig:
        """Create a white technical view with a grid and balanced illumination.

        Linear tone mapping preserves the white background in Open3D. Backends
        with fixed output transforms approximate the requested tone mapping.

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
        scene = cls(tone_mapping="linear")
        scale = scene_extent / 1.2
        point = scene.lights[1]
        point.position = tuple(np.asarray(point.position) * scale)
        point.intensity_candela *= scale**2
        point.range_m *= scale
        return replace(scene, **overrides)

    @classmethod
    def flat(cls, **overrides: Any) -> SceneConfig:
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
        **overrides: Any,
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
