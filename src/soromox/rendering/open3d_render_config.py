"""Appearance settings for Open3D exports and interactive previews."""

from dataclasses import dataclass, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Open3DRenderConfig:
    """Modern material and lighting settings.

    Animated interactive previews approximate these settings using legacy shading.
    Studio scenery is fixed in world coordinates with Z up. Its dimensions and
    light positions scale with the robot scene bounds; ``floor_z`` is in metres.
    CameraConfig sets the camera; RendererColorConfig sets robot colors.

    Attributes:
        roughness: Surface roughness in [0, 1]; larger values broaden highlights.
        metallic: Metallic material fraction in [0, 1].
        reflectance: Dielectric surface reflectance in [0, 1].
        shadows: Whether modern rendering computes cast shadows.
        ambient_occlusion: Whether to darken occluded surface regions.
        sun_direction: Nonzero world-space direction in which sunlight travels.
        sun_color: Sunlight RGB components in [0, 1].
        sun_intensity: Nonnegative directional-light intensity in Open3D units.
        indirect_light_intensity: Nonnegative environment-light intensity.
        studio_backdrop: Whether to add a curved grey floor and wall.
        backdrop_color: Backdrop and studio background RGB values in [0, 1].
        floor_z: World-space floor height in metres.
        fill_intensity: Nonnegative fill-light intensity at the reference scene
            scale; zero disables the light. Intensity scales with scene size
            squared, preserving illumination as geometry dimensions change.
    """

    roughness: float = 0.72
    metallic: float = 0.0
    reflectance: float = 0.35
    shadows: bool = True
    ambient_occlusion: bool = True
    sun_direction: tuple[float, float, float] = (-0.25, 0.15, -1.0)
    sun_color: tuple[float, float, float] = (1.0, 0.97, 0.94)
    sun_intensity: float = 60000.0
    indirect_light_intensity: float = 60000.0
    studio_backdrop: bool = False
    backdrop_color: tuple[float, float, float] = (0.48, 0.48, 0.48)
    floor_z: float = 0.0
    fill_intensity: float = 0.0

    @classmethod
    def studio(cls, **overrides: Any) -> "Open3DRenderConfig":
        """Create a studio preset with a curved backdrop and local fill light.

        Args:
            **overrides: Field values that replace the preset defaults.

        Returns:
            Validated immutable rendering configuration with the requested values.

        Raises:
            TypeError: An override names an unknown field.
            ValueError: An override violates a field's range or shape constraints.
        """
        return replace(cls(studio_backdrop=True, fill_intensity=35000.0), **overrides)

    def __post_init__(self) -> None:
        """Validate material ranges, light vectors, colors and floor height.

        Returns:
            None. Leaves the immutable configuration unchanged.

        Raises:
            ValueError: A material or color is outside [0, 1], a light intensity
                is negative or nonfinite, the sunlight direction is invalid, or
                the floor height is nonfinite.
        """
        for name in ("roughness", "metallic", "reflectance"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        for name in ("sun_intensity", "indirect_light_intensity", "fill_intensity"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        direction = np.asarray(self.sun_direction)
        if (
            direction.shape != (3,)
            or not np.all(np.isfinite(direction))
            or np.linalg.norm(direction) == 0
        ):
            raise ValueError("sun_direction must be a finite nonzero 3-vector")
        for name in ("sun_color", "backdrop_color"):
            color = np.asarray(getattr(self, name))
            if color.shape != (3,) or not np.all((color >= 0) & (color <= 1)):
                raise ValueError(
                    f"{name} must contain three values between zero and one"
                )
        if not np.isfinite(self.floor_z):
            raise ValueError("floor_z must be finite")
