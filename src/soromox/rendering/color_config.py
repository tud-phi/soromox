"""Shared color configuration and palette utilities for renderers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from jax import Array

ColorSpec = str | Array | np.ndarray | list | tuple | None

DEFAULT_SEGMENT_PALETTE = "coolwarm"
DEFAULT_ROBOT_PALETTE = "soromox:okabe-ito"


@dataclass(frozen=True)
class BackboneColorConfig:
    """Color configuration for the robot backbone.

    Hierarchy (low -> high specificity):
        robot_palette -> segment_palette -> point_palette
        robot_colors -> segment_colors -> point_colors
        robot_segment_colors -> robot_point_colors

    More specific inputs override less specific ones.
    """

    # Palette specs (strings or arrays); used when explicit colors are not provided.
    segment_palette: ColorSpec = DEFAULT_SEGMENT_PALETTE
    robot_palette: ColorSpec = DEFAULT_ROBOT_PALETTE
    point_palette: ColorSpec | None = None

    # Explicit colors (arrays or lists). These override palettes at the same level.
    robot_colors: ColorSpec | None = None
    segment_colors: ColorSpec | None = None
    robot_segment_colors: ColorSpec | None = None
    point_colors: ColorSpec | None = None
    robot_point_colors: ColorSpec | None = None


@dataclass(frozen=True)
class ActuatorStyleConfig:
    """Default style configuration for rendered actuator visual layers."""

    default_color: tuple[float, float, float] = (0.9, 0.15, 0.15)
    default_radius: float | None = None
    scalar_colormap: str = "viridis"
    kind_colors: Mapping[str, tuple[float, float, float]] = field(default_factory=dict)
    kind_radii: Mapping[str, float] = field(default_factory=dict)

    def color_for_kind(self, kind: str) -> tuple[float, float, float]:
        """Return the configured color for an actuator semantic kind."""
        return self.kind_colors.get(kind, self.default_color)

    def radius_for_kind(self, kind: str) -> float | None:
        """Return the configured tube radius for an actuator semantic kind."""
        return self.kind_radii.get(kind, self.default_radius)


@dataclass(frozen=True)
class RendererColorConfig:
    """Shared renderer colors with sensible defaults."""

    backbone: BackboneColorConfig = field(default_factory=BackboneColorConfig)
    base_plate_color: tuple[float, float, float] = (0.2, 0.2, 0.2)
    ground_plane_color: tuple[float, float, float] = (0.94, 0.95, 0.96)
    ground_plane_grid_color: tuple[float, float, float] = (0.72, 0.75, 0.78)
    actuators: ActuatorStyleConfig = field(default_factory=ActuatorStyleConfig)


@dataclass(frozen=True)
class BuiltinPalette:
    colors: np.ndarray
    kind: Literal["categorical", "gradient"] = "categorical"
    description: str | None = None


@dataclass(frozen=True)
class ResolvedBackboneColors:
    """Resolved backbone colors for rendering."""

    per_robot_point_rgba: np.ndarray  # (N, P, 4)
    per_robot_segment_rgba: np.ndarray  # (N, S, 4)
    robot_colors_rgba: np.ndarray  # (N, 4)
    segment_colors_rgba: np.ndarray  # (S, 4) for legend
    point_colors_rgba: np.ndarray  # (P, 4) for legend
    legend_level: Literal["robot", "segment", "point"]


@dataclass(frozen=True)
class ColorLegend:
    """Lightweight color legend for UI or plots."""

    level: Literal["robot", "segment", "point"]
    colors_rgba: np.ndarray
    labels: list[str]


def _hex_to_rgb(hex_color: str) -> np.ndarray:
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color!r}")
    return np.array(
        [int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float64
    )


def _palette_from_hex_list(
    colors: list[str],
    *,
    kind: Literal["categorical", "gradient"] = "categorical",
    description: str | None = None,
) -> BuiltinPalette:
    return BuiltinPalette(
        colors=np.array([_hex_to_rgb(c) for c in colors], dtype=np.float64),
        kind=kind,
        description=description,
    )


BUILTIN_PALETTES: dict[str, BuiltinPalette] = {
    "soromox:okabe-ito": _palette_from_hex_list(
        [
            "#E69F00",
            "#56B4E9",
            "#009E73",
            "#F0E442",
            "#0072B2",
            "#D55E00",
            "#CC79A7",
            "#000000",
        ],
        description="Okabe-Ito colorblind-safe categorical palette.",
    ),
    "soromox:tol-bright": _palette_from_hex_list(
        [
            "#4477AA",
            "#EE6677",
            "#228833",
            "#CCBB44",
            "#66CCEE",
            "#AA3377",
            "#BBBBBB",
        ],
        description="Paul Tol bright categorical palette.",
    ),
    "soromox:tol-muted": _palette_from_hex_list(
        [
            "#332288",
            "#88CCEE",
            "#44AA99",
            "#117733",
            "#999933",
            "#DDCC77",
            "#CC6677",
            "#882255",
            "#AA4499",
        ],
        description="Paul Tol muted categorical palette.",
    ),
    "soromox:ember": _palette_from_hex_list(
        ["#1B2A41", "#E76F51", "#F4A261"],
        kind="gradient",
        description="Warm publication-friendly gradient.",
    ),
    "soromox:glacier": _palette_from_hex_list(
        ["#0B3C5D", "#328CC1", "#D9E4EC"],
        kind="gradient",
        description="Cool publication-friendly gradient.",
    ),
    "soromox:slate": _palette_from_hex_list(
        ["#1F2933", "#CBD2D9"],
        kind="gradient",
        description="Neutral monochrome gradient.",
    ),
}


BUILTIN_THEMES: dict[str, RendererColorConfig] = {
    "soromox:paper": RendererColorConfig(
        backbone=BackboneColorConfig(
            segment_palette="soromox:ember",
            robot_palette="soromox:okabe-ito",
        ),
        base_plate_color=(0.2, 0.2, 0.2),
        actuators=ActuatorStyleConfig(default_color=(0.8, 0.2, 0.2)),
    ),
    "soromox:cool": RendererColorConfig(
        backbone=BackboneColorConfig(
            segment_palette="soromox:glacier",
            robot_palette="soromox:tol-muted",
        ),
        base_plate_color=(0.15, 0.2, 0.25),
        actuators=ActuatorStyleConfig(default_color=(0.2, 0.45, 0.7)),
    ),
    "soromox:mono": RendererColorConfig(
        backbone=BackboneColorConfig(
            segment_palette="soromox:slate",
            robot_palette="soromox:slate",
        ),
        base_plate_color=(0.2, 0.2, 0.2),
        actuators=ActuatorStyleConfig(default_color=(0.35, 0.35, 0.35)),
    ),
}


def list_builtin_palettes() -> dict[str, BuiltinPalette]:
    """Return built-in palettes for publications and quick styling."""
    return dict(BUILTIN_PALETTES)


def list_builtin_themes() -> dict[str, RendererColorConfig]:
    """Return built-in themes as RendererColorConfig presets."""
    return dict(BUILTIN_THEMES)


def get_color_theme(name: str) -> RendererColorConfig:
    """Fetch a built-in theme by name."""
    if name not in BUILTIN_THEMES:
        raise KeyError(f"Unknown theme {name!r}. Available: {sorted(BUILTIN_THEMES)}")
    return BUILTIN_THEMES[name]


def validate_rgb(color: tuple[float, float, float], *, name: str = "color") -> None:
    """Validate a normalized RGB tuple used by renderer configuration objects."""
    try:
        color_array = np.asarray(color, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an RGB tuple with values in [0, 1].") from exc
    if (
        color_array.shape != (3,)
        or not np.all(np.isfinite(color_array))
        or np.any(color_array < 0.0)
        or np.any(color_array > 1.0)
    ):
        raise ValueError(f"{name} must be an RGB tuple with values in [0, 1].")


def ensure_rgba(colors: np.ndarray) -> np.ndarray:
    """Ensure array has an RGBA channel."""
    arr = np.asarray(colors, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[-1] not in (3, 4):
        raise ValueError(
            f"Color array must have 3 or 4 channels; got shape {arr.shape}"
        )
    if arr.shape[-1] == 3:
        alpha = np.ones((*arr.shape[:-1], 1), dtype=np.float64)
        arr = np.concatenate([arr, alpha], axis=-1)
    return np.clip(arr, 0.0, 1.0)


def _interpolate_palette(colors: np.ndarray, count: int) -> np.ndarray:
    if colors.shape[0] == 1:
        return np.tile(colors, (count, 1))
    xs = np.linspace(0.0, 1.0, colors.shape[0])
    xi = np.linspace(0.0, 1.0, count)
    out = np.zeros((count, colors.shape[1]), dtype=np.float64)
    for c in range(colors.shape[1]):
        out[:, c] = np.interp(xi, xs, colors[:, c])
    return out


def _resolve_named_palette(name: str, count: int) -> np.ndarray:
    if name in BUILTIN_PALETTES:
        palette = BUILTIN_PALETTES[name]
        if palette.kind == "gradient":
            return _interpolate_palette(palette.colors, count)
        colors = palette.colors
        if colors.shape[0] < count:
            reps = int(np.ceil(count / colors.shape[0]))
            colors = np.tile(colors, (reps, 1))
        return colors[:count]

    from matplotlib import colormaps

    cmap = colormaps.get_cmap(name)
    if count == 1:
        colors = np.array([cmap(0.5)[:3]], dtype=np.float64)
    else:
        colors = np.array([cmap(i / (count - 1))[:3] for i in range(count)])
    return colors


def normalize_palette(
    color_spec: ColorSpec,
    count: int,
    *,
    default_palette: ColorSpec,
    name: str,
) -> tuple[np.ndarray, bool]:
    """Resolve a color palette (string or array) to RGBA."""
    spec = default_palette if color_spec is None else color_spec
    if isinstance(spec, str):
        palette = _resolve_named_palette(spec, count)
        rgba = ensure_rgba(palette)
        return rgba[:count], False

    palette = np.asarray(spec, dtype=np.float64)
    if palette.ndim == 1:
        palette = palette[None, :]
    if palette.shape[-1] not in (3, 4):
        raise ValueError(
            f"{name} palette must have 3 or 4 channels; got shape {palette.shape}"
        )
    has_alpha = palette.shape[-1] == 4
    rgba = ensure_rgba(palette)
    if rgba.shape[0] < count:
        reps = int(np.ceil(count / rgba.shape[0]))
        rgba = np.tile(rgba, (reps, 1))
    return rgba[:count], has_alpha


def normalize_color_array(
    color_spec: ColorSpec | None,
    target_shape: tuple[int, ...],
    *,
    name: str,
) -> tuple[np.ndarray | None, bool]:
    """Normalize explicit color arrays and broadcast to target shape."""
    if color_spec is None:
        return None, False
    colors = np.asarray(color_spec, dtype=np.float64)
    if colors.ndim == 1:
        colors = colors[None, :]
    if colors.shape[-1] not in (3, 4):
        raise ValueError(f"{name} must have 3 or 4 channels; got shape {colors.shape}")
    has_alpha = colors.shape[-1] == 4
    colors = ensure_rgba(colors)
    try:
        colors = np.broadcast_to(colors, target_shape + (4,))
    except ValueError as exc:
        raise ValueError(
            f"{name} must be broadcastable to shape {target_shape + (4,)}; "
            f"got {colors.shape}"
        ) from exc
    return np.array(colors, copy=True), has_alpha
