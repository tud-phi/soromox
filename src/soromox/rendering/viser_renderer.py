"""Viser-based web renderer for soft robots.

Provides interactive 3D visualization accessible via web browser with:
- Real-time animation playback with GUI controls
- Live mode for streaming robot states
- Multiple robot overlay
- Dynamic spheres for setpoints/obstacles
- Embedded plot panels
- Video export via FFmpeg
"""

from __future__ import annotations

__all__ = ["ViserRenderer"]

import atexit
import threading
import time
import webbrowser
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from plotly.graph_objects import Figure as PlotlyFigure
else:
    PlotlyFigure = Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

try:
    import plotly.graph_objects as go
    import trimesh
    import viser

    VISER_AVAILABLE = True
except ImportError:
    VISER_AVAILABLE = False
    go = None

from soromox.rendering.base import BaseSoftRobotRenderer
from soromox.rendering.camera_config import CameraConfig
from soromox.rendering.color_config import RendererColorConfig, ensure_rgba
from soromox.rendering.video_encoding import FFmpegVideoWriter, VideoEncodingConfig
from soromox.systems.soft_robot import SoftRobot

# =============================================================================
# Data structures
# =============================================================================


@dataclass
class AnimationState:
    """State for animation playback."""

    frame_idx: int = 0
    playing: bool = False
    loop: bool = False
    playback_speed: float = 1.0
    last_tick: float = 0.0
    num_frames: int = 1
    ts: np.ndarray = field(default_factory=lambda: np.array([0.0]))

    @property
    def dt_sequence(self) -> np.ndarray:
        """Compute time deltas between frames."""
        if len(self.ts) < 2:
            return np.array([1.0 / 30.0])
        return np.diff(self.ts)


@dataclass
class SceneHandles:
    """Handles for Viser scene objects."""

    # Robot geometry handles - indexed [robot_idx][point_idx]
    base_plates: list = field(default_factory=list)
    backbone_points: list[list] = field(default_factory=list)
    tendon_lines: list = field(default_factory=list)

    # Track if geometry has been initially built (for efficient updates)
    geometry_initialized: bool = False
    # Store number of robots and points for geometry validation
    num_robots: int = 0
    num_backbone_points: int = 0

    # Lighting
    lights: list = field(default_factory=list)

    # Auxiliary geometry
    static_spheres: list = field(default_factory=list)
    dynamic_spheres: list = field(default_factory=list)
    dynamic_trajectories: list[np.ndarray] = field(default_factory=list)

    # Custom primitives registry
    custom_primitives: dict = field(default_factory=dict)


# =============================================================================
# Color utilities
# =============================================================================
def _rgb_to_viser_color(rgb: np.ndarray) -> tuple[int, int, int]:
    """Convert RGB floats [0,1] to Viser color tuple (0-255)."""
    rgb = np.clip(rgb[:3], 0.0, 1.0)
    return (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


def _rgba_to_viser_color_and_opacity(
    rgba: np.ndarray, *, opaque_threshold: float = 0.999
) -> tuple[tuple[int, int, int], float | None]:
    """Convert RGBA floats [0,1] to Viser color tuple and optional opacity."""
    rgba = ensure_rgba(np.asarray(rgba, dtype=np.float64))[0]
    opacity = float(np.clip(rgba[3], 0.0, 1.0))
    if opacity >= opaque_threshold:
        opacity = None
    return _rgb_to_viser_color(rgba), opacity


def _direction_to_quaternion(
    direction: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compute quaternion (wxyz) to rotate Z-axis to align with direction.

    Args:
        direction: Target direction vector (3,), will be normalized

    Returns:
        Quaternion as (w, x, y, z) tuple
    """
    direction = np.asarray(direction, dtype=np.float64)
    dir_norm = np.linalg.norm(direction)
    if dir_norm < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)  # Identity quaternion
    direction = direction / dir_norm

    z_axis = np.array([0.0, 0.0, 1.0])

    # Check for parallel/anti-parallel cases
    dot = np.dot(z_axis, direction)
    if dot > 0.9999:
        # Already aligned with Z
        return (1.0, 0.0, 0.0, 0.0)
    elif dot < -0.9999:
        # Opposite to Z - rotate 180° around X axis
        return (0.0, 1.0, 0.0, 0.0)

    # General case: axis-angle to quaternion
    axis = np.cross(z_axis, direction)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))

    # Quaternion from axis-angle
    half_angle = angle / 2.0
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)

    return (float(w), float(xyz[0]), float(xyz[1]), float(xyz[2]))


# =============================================================================
# ViserRenderer
# =============================================================================


class ViserRenderer(BaseSoftRobotRenderer):
    """Viser-based web visualization for soft robots.

    Provides interactive 3D visualization accessible via web browser with:
    - Real-time animation playback with GUI controls
    - Live mode for streaming robot states
    - Multiple robot overlay
    - Dynamic spheres for setpoints/obstacles
    - Embedded plot panels
    - Video export via FFmpeg

    Example:
        >>> renderer = ViserRenderer(robot, port=8080)
        >>> renderer.show(q)  # Opens browser at localhost:8080

        >>> # Animation replay
        >>> renderer.render_sequence(ts, q_ts, playback_speed=1.0, loop=True)

        >>> # Live mode
        >>> ctrl = renderer.start_live_mode(callback=lambda t: compute_q(t))
    """

    def __init__(
        self,
        robot: SoftRobot,
        width: int = 1920,
        height: int = 1200,
        num_points: int = 80,
        background_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        color_config: RendererColorConfig | None = None,
        host: str = "0.0.0.0",
        port: int = 8080,
        backbone_style: Literal["discrete", "swept"] = "discrete",
        sphere_resolution: int = 3,
        cylinder_sections: int = 48,
        grid_spacing: tuple[float, float] = (0.5, 0.5),
        base_offsets: Array | None = None,
        base_plate_radius_scale: float = 2.0,
        base_plate_thickness: float = 0.06,
        tendon_line_width: float = 3.0,
        camera_fov: float = 75.0,
        # Lighting parameters
        enable_default_lights: bool = True,
        add_directional_light: bool = True,
        directional_light_intensity: float = 0.8,
        directional_light_direction: tuple[float, float, float] = (-0.5, -1.0, -0.5),
        directional_light_color: tuple[int, int, int] = (255, 255, 255),
        add_ambient_light: bool = True,
        ambient_light_intensity: float = 0.4,
        ambient_light_color: tuple[int, int, int] = (255, 255, 255),
        cast_shadows: bool = True,
        # Material & shading parameters
        material: Literal["standard", "toon3", "toon5"] = "standard",
        flat_shading: bool = False,
        wireframe: bool = False,
        # Shadow configuration (per-geometry type)
        backbone_cast_shadow: bool = True,
        sphere_cast_shadow: bool = True,
        auto_start: bool = True,
        open_browser: bool = True,
    ):
        """Initialize Viser renderer.

        Args:
            robot: SoftRobot system with forward_kinematics method
            width: Default render width in pixels
            height: Default render height in pixels
            num_points: Number of points for backbone curve discretization
            background_color: RGB background color (0-1 range)
            color_config: Shared renderer color configuration
            host: Server bind address (0.0.0.0 for all interfaces)
            port: Server port number
            backbone_style: "discrete" (spheres) or "swept" (cylinders)
            sphere_resolution: Icosphere subdivision level (1=low, 2=medium, 3=good, 4=high)
            cylinder_sections: Number of cylinder cross-section segments (higher=smoother)
            grid_spacing: (x, y) spacing for multi-robot grid layout
            base_offsets: Explicit base position offsets (N, 3)
            base_plate_radius_scale: Base plate radius relative to robot radius
            base_plate_thickness: Base plate thickness in meters
            tendon_line_width: Line width for tendon visualization
            camera_fov: Camera field of view in degrees
            enable_default_lights: Enable Viser's default lighting
            add_directional_light: Add custom directional light
            directional_light_intensity: Directional light intensity (0-1+)
            directional_light_direction: Directional light direction vector (x, y, z)
            directional_light_color: Directional light RGB color (0-255)
            add_ambient_light: Add custom ambient light
            ambient_light_intensity: Ambient light intensity (0-1+)
            ambient_light_color: Ambient light RGB color (0-255)
            cast_shadows: Enable shadow casting for default lights
            material: Material type ("standard", "toon3", "toon5")
            flat_shading: Use flat shading instead of smooth
            wireframe: Render geometry as wireframe
            backbone_cast_shadow: Enable shadow casting for backbone geometry
            sphere_cast_shadow: Enable shadow casting for sphere geometry
            auto_start: Start server immediately
            open_browser: Open browser automatically when show() is called
        """
        if not VISER_AVAILABLE:
            raise ImportError(
                "viser is required for ViserRenderer. "
                "Install it with: pip install viser"
            )

        super().__init__(
            robot,
            width,
            height,
            num_points,
            background_color,
            color_config=color_config,
        )

        self._host = host
        self._port = port
        self._backbone_style = backbone_style
        self._sphere_resolution = sphere_resolution
        self._cylinder_sections = cylinder_sections
        self._grid_spacing = grid_spacing
        self._base_offsets = base_offsets
        self._base_plate_radius_scale = base_plate_radius_scale
        self._base_plate_thickness = base_plate_thickness
        self._tendon_line_width = tendon_line_width
        self._camera_fov = camera_fov
        self._open_browser = open_browser

        # Lighting parameters
        self._enable_default_lights = enable_default_lights
        self._add_directional_light = add_directional_light
        self._directional_light_intensity = directional_light_intensity
        self._directional_light_direction = directional_light_direction
        self._directional_light_color = directional_light_color
        self._add_ambient_light = add_ambient_light
        self._ambient_light_intensity = ambient_light_intensity
        self._ambient_light_color = ambient_light_color
        self._cast_shadows = cast_shadows

        # Material & shading parameters
        self._material = material
        self._flat_shading = flat_shading
        self._wireframe = wireframe

        # Shadow configuration
        self._backbone_cast_shadow = backbone_cast_shadow
        self._sphere_cast_shadow = sphere_cast_shadow

        # Get robot radius for sizing
        self._robot_radius = self._get_robot_radius()

        # Server and scene state
        self._server: viser.ViserServer | None = None
        self._scene_handles: SceneHandles | None = None
        self._animation_state: AnimationState | None = None
        self._gui_handles: dict = {}
        self._lock = threading.Lock()
        self._running = False

        # Live mode state
        self._live_mode_active = False
        self._live_callback: Callable[[float], np.ndarray] | None = None
        self._live_thread: threading.Thread | None = None

        if auto_start:
            self.start()

    def _get_robot_radius(self) -> float:
        """Get representative robot radius for visualization."""
        if hasattr(self.robot, "r"):
            r = self.robot.r
            if hasattr(r, "__iter__"):
                return float(np.mean(np.asarray(r)))
            return float(r)
        return 0.02  # Default radius

    @property
    def is_3d(self) -> bool:
        """Always True for Viser (3D only)."""
        return True

    @property
    def server(self) -> viser.ViserServer:
        """Access underlying Viser server."""
        if self._server is None:
            raise RuntimeError("Server not started. Call start() first.")
        return self._server

    @property
    def url(self) -> str:
        """URL to access the visualization."""
        if self._server is not None:
            # Use actual port (may differ from requested port if it was already in use)
            actual_port = self._server.get_port()
            return f"http://localhost:{actual_port}"
        return f"http://localhost:{self._port}"

    def _extract_positions(self, poses: Array) -> Array:
        """Extract 3D positions from SE(2) or SE(3) poses.

        Supports both planar (SE(2)) and 3D (SE(3)) robots by converting
        SE(2) poses [theta, x, y] to 3D coordinates [x, y, 0].

        Args:
            poses: Forward kinematics output - either:
                - (N, 3) for SE(2) poses [theta, x, y]
                - (N, 4, 4) for SE(3) transformation matrices

        Returns:
            Position array of shape (N, 3) with xyz coordinates.
        """
        return self._extract_positions_3d(poses)

    def start(self) -> None:
        """Start the Viser server."""
        if self._server is not None:
            return

        self._server = viser.ViserServer(host=self._host, port=self._port)

        # Note: Viser doesn't support setting background color via API
        # Background is controlled by the client/browser theme

        # Set Z-up coordinate convention for the scene
        self._server.scene.set_up_direction("-z")

        # Initialize empty scene handles
        self._scene_handles = SceneHandles()

        self._running = True

        # Register cleanup
        atexit.register(self.stop)

        print(f"[ViserRenderer] Server started at {self.url}")

    def stop(self) -> None:
        """Stop the Viser server."""
        if self._server is None:
            return

        self._running = False

        # Stop live mode if active
        if self._live_mode_active:
            self._stop_live_mode_internal()

        # Close server - use stop() method instead of close()
        try:
            self._server.stop()
        except Exception:
            pass  # Server may already be closed
        self._server = None
        self._scene_handles = None

        print("[ViserRenderer] Server stopped")

    def _clear_scene(self) -> None:
        """Clear all geometry from the scene."""
        if self._server is None:
            return

        # Remove all scene children
        # Viser doesn't have a clear method, so we remove by setting visibility
        # or rely on garbage collection when we recreate handles
        self._scene_handles = SceneHandles()

    def _build_robot_geometry(
        self,
        curves: np.ndarray,
        point_colors: np.ndarray,
        *,
        base_plate_color: tuple[float, float, float],
    ) -> None:
        """Build robot backbone geometry in the scene.

        Args:
            curves: Backbone curves of shape (num_robots, num_points, 3)
            point_colors: Per-robot colors of shape (num_robots, num_points, 4)
        """
        if self._server is None or self._scene_handles is None:
            return

        num_robots = curves.shape[0]
        num_points = curves.shape[1]

        # Clear existing robot geometry
        self._scene_handles.backbone_points = []
        self._scene_handles.base_plates = []

        for robot_idx in range(num_robots):
            curve = curves[robot_idx]  # (num_points, 3)

            # Create backbone geometry
            robot_points = []

            if self._backbone_style == "discrete":
                # Render as spheres at each backbone point
                for pt_idx in range(num_points):
                    pos = curve[pt_idx]
                    color_rgba = point_colors[robot_idx, pt_idx]
                    color, opacity = _rgba_to_viser_color_and_opacity(color_rgba)

                    handle = self._server.scene.add_icosphere(
                        name=f"/robots/robot_{robot_idx}/backbone/pt_{pt_idx}",
                        radius=self._robot_radius * 0.8,
                        color=color,
                        opacity=opacity,
                        subdivisions=self._sphere_resolution,
                        position=tuple(pos),
                        material=self._material,
                        flat_shading=self._flat_shading,
                        wireframe=self._wireframe,
                        cast_shadow=self._backbone_cast_shadow,
                    )
                    robot_points.append(handle)

            else:  # "swept" style - cylinders between points
                for pt_idx in range(num_points - 1):
                    p0 = curve[pt_idx]
                    p1 = curve[pt_idx + 1]
                    color_rgba = point_colors[robot_idx, pt_idx]
                    color, opacity = _rgba_to_viser_color_and_opacity(color_rgba)

                    # Compute cylinder parameters
                    direction = p1 - p0
                    length = float(np.linalg.norm(direction))
                    if length < 1e-9:
                        continue

                    center = (p0 + p1) / 2.0
                    # Compute quaternion for orientation (Z-aligned cylinder rotated to direction)
                    wxyz = _direction_to_quaternion(direction)

                    # Create Z-aligned cylinder mesh, apply rotation via handle
                    cylinder_mesh = self._make_cylinder_trimesh(
                        length=length,
                        radius=self._robot_radius,
                        color=color_rgba[:3],
                        direction=None,  # Keep Z-aligned for efficient updates
                    )
                    handle = self._server.scene.add_mesh_simple(
                        name=f"/robots/robot_{robot_idx}/backbone/seg_{pt_idx}",
                        vertices=cylinder_mesh.vertices,
                        faces=cylinder_mesh.faces,
                        color=color,
                        opacity=opacity,
                        wireframe=self._wireframe,
                        material=self._material,
                        flat_shading=self._flat_shading,
                        cast_shadow=self._backbone_cast_shadow,
                        position=tuple(center),
                        wxyz=wxyz,
                    )
                    robot_points.append(handle)

                # Add sphere at tip
                tip_pos = curve[-1]
                color_rgba = point_colors[robot_idx, -1]
                color, opacity = _rgba_to_viser_color_and_opacity(color_rgba)
                handle = self._server.scene.add_icosphere(
                    name=f"/robots/robot_{robot_idx}/backbone/tip",
                    radius=self._robot_radius,
                    color=color,
                    opacity=opacity,
                    subdivisions=self._sphere_resolution,
                    position=tuple(tip_pos),
                    material=self._material,
                    flat_shading=self._flat_shading,
                    wireframe=self._wireframe,
                    cast_shadow=self._backbone_cast_shadow,
                )
                robot_points.append(handle)

            self._scene_handles.backbone_points.append(robot_points)

            # Create base plate (Z-aligned, no rotation needed)
            base_pos = curve[0].copy()
            base_pos[2] -= self._base_plate_thickness / 2
            base_handle = self._server.scene.add_mesh_trimesh(
                name=f"/robots/robot_{robot_idx}/base_plate",
                mesh=self._make_cylinder_trimesh(
                    length=self._base_plate_thickness,
                    radius=self._robot_radius * self._base_plate_radius_scale,
                    color=base_plate_color,
                    direction=None,  # Already Z-aligned
                ),
                position=tuple(base_pos),
            )
            self._scene_handles.base_plates.append(base_handle)

        # Mark geometry as initialized for efficient updates
        self._scene_handles.geometry_initialized = True
        self._scene_handles.num_robots = num_robots
        self._scene_handles.num_backbone_points = num_points

    def _build_tendon_geometry(
        self,
        q: Array,
        base_offsets: np.ndarray,
        num_robots: int,
        *,
        tendon_color: tuple[float, float, float] | None = None,
    ) -> None:
        """Build tendon line geometry in the scene.

        Args:
            q: Robot configurations (num_robots, DOF)
            base_offsets: Base position offsets (num_robots, 3)
            num_robots: Number of robots
        """
        if not self._has_tendons or self._server is None or self._scene_handles is None:
            return

        # Clear existing tendon geometry
        self._scene_handles.tendon_lines = []

        # Compute tendon curves for all robots at once using batched method
        # Returns shape (num_robots, num_tendons, num_points, 3) or None
        tendon_curves = self.compute_tendon_curves_batched(q, jnp.asarray(base_offsets))

        if tendon_curves is None:
            return

        tendon_curves = np.asarray(
            tendon_curves
        )  # (num_robots, num_tendons, num_points, 3)

        color_arr = (
            np.array(tendon_color)
            if tendon_color is not None
            else np.array(self.color_config.tendon_color)
        )
        color = _rgb_to_viser_color(color_arr)

        # Process all robots and tendons
        for robot_idx in range(num_robots):
            robot_tendon_curves = tendon_curves[
                robot_idx
            ]  # (num_tendons, num_points, 3)
            num_tendons = robot_tendon_curves.shape[0]

            for tendon_idx in range(num_tendons):
                tendon_curve = robot_tendon_curves[tendon_idx]  # (num_points, 3)

                # Create line segments for each tendon
                # Viser expects shape (N, 2, 3) for N line segments
                num_segments = len(tendon_curve) - 1
                if num_segments > 0:
                    # Reshape to (N, 2, 3) where each segment has start and end points
                    points = np.stack(
                        [tendon_curve[:-1], tendon_curve[1:]], axis=1
                    )  # (num_segments, 2, 3)

                    handle = self._server.scene.add_line_segments(
                        name=f"/robots/robot_{robot_idx}/tendons/tendon_{tendon_idx}",
                        points=points,
                        colors=np.tile(color, (num_segments, 2, 1)).astype(np.uint8),
                        line_width=self._tendon_line_width,
                    )
                    self._scene_handles.tendon_lines.append(handle)

    def _make_cylinder_trimesh(
        self,
        length: float,
        radius: float,
        color: tuple | np.ndarray,
        direction: np.ndarray | None = None,
    ):
        """Create a trimesh cylinder, optionally aligned with a direction.

        Args:
            length: Cylinder height
            radius: Cylinder radius
            color: RGB color (0-1 range)
            direction: If provided, rotate cylinder to align with this direction.
                      If None, cylinder remains Z-aligned (for use with handle rotation).
        """
        # Create cylinder along Z axis
        cylinder = trimesh.creation.cylinder(
            radius=radius,
            height=length,
            sections=self._cylinder_sections,
        )

        # Apply rotation only if direction is provided
        if direction is not None:
            direction = np.asarray(direction, dtype=np.float64)
            dir_norm = np.linalg.norm(direction)
            if dir_norm > 1e-9:
                direction = direction / dir_norm

            z_axis = np.array([0.0, 0.0, 1.0])
            if np.allclose(direction, z_axis):
                R = np.eye(3)
            elif np.allclose(direction, -z_axis):
                R = np.diag([1.0, -1.0, -1.0])
            else:
                v = np.cross(z_axis, direction)
                s = np.linalg.norm(v)
                c = np.dot(z_axis, direction)
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2 + 1e-12))

            # Apply rotation
            transform = np.eye(4)
            transform[:3, :3] = R
            cylinder.apply_transform(transform)

        # Set vertex colors
        color_arr = np.asarray(color)[:3]
        color_uint8 = (np.clip(color_arr, 0, 1) * 255).astype(np.uint8)
        colors = np.tile(
            np.append(color_uint8, 255), (len(cylinder.vertices), 1)
        ).astype(np.uint8)
        cylinder.visual.vertex_colors = colors

        return cylinder

    def _update_robot_geometry(
        self,
        curves: np.ndarray,
    ) -> None:
        """Update existing robot geometry positions (and orientations for swept mode).

        This is more efficient than rebuilding geometry each frame:
        - Discrete mode: Only updates sphere positions
        - Swept mode: Updates cylinder positions and orientations via handle properties

        Args:
            curves: Backbone curves of shape (num_robots, num_points, 3)
        """
        if self._scene_handles is None:
            return

        num_robots = min(len(curves), len(self._scene_handles.backbone_points))
        num_points = curves.shape[1]

        for robot_idx in range(num_robots):
            curve = curves[robot_idx]  # (num_points, 3)
            robot_points = self._scene_handles.backbone_points[robot_idx]

            if self._backbone_style == "discrete":
                # Update sphere positions only
                for pt_idx, handle in enumerate(robot_points):
                    if pt_idx < len(curve):
                        handle.position = tuple(curve[pt_idx])
            else:
                # Swept style: update cylinder positions and orientations
                # robot_points contains (num_points - 1) cylinders + 1 tip sphere
                num_segments = num_points - 1
                for seg_idx in range(min(len(robot_points) - 1, num_segments)):
                    handle = robot_points[seg_idx]
                    p0 = curve[seg_idx]
                    p1 = curve[seg_idx + 1]

                    # Compute new center and orientation
                    direction = p1 - p0
                    length = float(np.linalg.norm(direction))
                    if length < 1e-9:
                        continue

                    center = (p0 + p1) / 2.0
                    wxyz = _direction_to_quaternion(direction)

                    # Update handle properties (no mesh recreation!)
                    handle.position = tuple(center)
                    handle.wxyz = wxyz

                # Update tip sphere position (last element in robot_points)
                if len(robot_points) > 0:
                    tip_handle = robot_points[-1]
                    tip_handle.position = tuple(curve[-1])

    def _build_static_spheres(
        self,
        positions: np.ndarray,
        radii: np.ndarray,
        colors: np.ndarray,
    ) -> None:
        """Build static sphere geometry.

        Args:
            positions: Sphere centers (N, 3)
            radii: Sphere radii (N,)
            colors: Sphere colors (N, 3) or (N, 4)
        """
        if self._server is None or self._scene_handles is None:
            return

        colors = ensure_rgba(colors)

        for i, (pos, radius, color) in enumerate(zip(positions, radii, colors)):
            viser_color, opacity = _rgba_to_viser_color_and_opacity(color)
            handle = self._server.scene.add_icosphere(
                name=f"/spheres/static/sphere_{i}",
                radius=float(radius),
                color=viser_color,
                opacity=opacity,
                subdivisions=self._sphere_resolution,
                position=tuple(pos),
                material=self._material,
                flat_shading=self._flat_shading,
                wireframe=self._wireframe,
                cast_shadow=self._sphere_cast_shadow,
            )
            self._scene_handles.static_spheres.append(handle)

    def _build_dynamic_spheres(
        self,
        trajectories: np.ndarray,
        radii: np.ndarray,
        colors: np.ndarray,
        frame_idx: int = 0,
    ) -> None:
        """Build dynamic (time-varying) sphere geometry.

        Args:
            trajectories: Sphere trajectories (N, T, 3)
            radii: Sphere radii (N,)
            colors: Sphere colors (N, 3) or (N, 4)
            frame_idx: Initial frame index
        """
        if self._server is None or self._scene_handles is None:
            return

        colors = ensure_rgba(colors)
        self._scene_handles.dynamic_trajectories = list(trajectories)

        for i, (traj, radius, color) in enumerate(zip(trajectories, radii, colors)):
            pos = traj[min(frame_idx, len(traj) - 1)]
            viser_color, opacity = _rgba_to_viser_color_and_opacity(color)
            handle = self._server.scene.add_icosphere(
                name=f"/spheres/dynamic/sphere_{i}",
                radius=float(radius),
                color=viser_color,
                opacity=opacity,
                subdivisions=self._sphere_resolution,
                position=tuple(pos),
                material=self._material,
                flat_shading=self._flat_shading,
                wireframe=self._wireframe,
                cast_shadow=self._sphere_cast_shadow,
            )
            self._scene_handles.dynamic_spheres.append(handle)

    def _update_dynamic_spheres(self, frame_idx: int) -> None:
        """Update dynamic sphere positions for given frame."""
        if self._scene_handles is None:
            return

        for handle, traj in zip(
            self._scene_handles.dynamic_spheres,
            self._scene_handles.dynamic_trajectories,
        ):
            idx = min(frame_idx, len(traj) - 1)
            handle.position = tuple(traj[idx])

    def _setup_camera(
        self,
        curves: np.ndarray,
        camera_config: CameraConfig | None = None,
    ) -> None:
        """Configure camera to view the robot(s).

        Args:
            curves: Backbone curves array of shape (N, num_points, 3)
            camera_config: Optional camera configuration. If None, uses
                default settings from renderer initialization.
        """
        if self._server is None:
            return

        # Use provided config or create default
        config = camera_config or CameraConfig(fov=self._camera_fov)

        # Compute bounding box of all curves
        all_points = curves.reshape(-1, 3)
        center = np.mean(all_points, axis=0)
        extent = np.max(all_points, axis=0) - np.min(all_points, axis=0)
        max_extent = float(np.max(extent))

        # Compute camera position and look_at from config
        camera_pos, look_at = config.compute_auto_position(center, max_extent)

        # Helper function to configure a client's camera
        def configure_camera(client: viser.ClientHandle) -> None:
            client.camera.position = tuple(camera_pos)
            client.camera.look_at = tuple(look_at)
            client.camera.fov = config.fov

        # Update already-connected clients
        for client in self._server.get_clients().values():
            configure_camera(client)

        # Set camera for new clients
        @self._server.on_client_connect
        def on_connect(client: viser.ClientHandle) -> None:
            configure_camera(client)

    def _setup_lighting(self) -> None:
        """Configure scene lighting for enhanced visual quality."""
        if self._server is None or self._scene_handles is None:
            return

        # Clear existing lights
        for light in self._scene_handles.lights:
            light.remove()
        self._scene_handles.lights = []

        # Configure default lights
        self._server.scene.configure_default_lights(
            enabled=self._enable_default_lights,
            cast_shadow=self._cast_shadows,
        )

        # Add directional light (key light)
        if self._add_directional_light:
            direction = np.array(self._directional_light_direction)
            direction = direction / np.linalg.norm(direction)  # Normalize

            directional_light = self._server.scene.add_light_directional(
                name="/lights/directional",
                color=self._directional_light_color,
                intensity=self._directional_light_intensity,
                wxyz=(1.0, 0.0, 0.0, 0.0),  # Identity quaternion
                position=(0.0, 0.0, 0.0),
                cast_shadow=self._cast_shadows,
            )
            self._scene_handles.lights.append(directional_light)

            # Orient the light (Viser uses wxyz quaternion, position defines light origin)
            # For directional lights, we set a far position in the opposite direction
            light_distance = 10.0
            light_pos = tuple(-direction * light_distance)
            directional_light.position = light_pos

        # Add ambient light (fill light)
        if self._add_ambient_light:
            ambient_light = self._server.scene.add_light_ambient(
                name="/lights/ambient",
                color=self._ambient_light_color,
                intensity=self._ambient_light_intensity,
            )
            self._scene_handles.lights.append(ambient_light)

    def render_frame(
        self,
        q: Array,
        *,
        base_offsets: Array | None = None,
        color_config: RendererColorConfig | None = None,
        camera_config: CameraConfig | None = None,
        static_spheres_positions: Array | None = None,
        static_spheres_radii: Array | None = None,
        static_spheres_colors: Array | None = None,
        capture_client_idx: int = 0,
    ) -> np.ndarray:
        """Render single configuration and capture as image.

        Note: Requires at least one connected client to capture.

        Args:
            q: Robot configuration (DOF,) or batched (N, DOF)
            base_offsets: Base position offsets (N, 3)
            color_config: Shared renderer color configuration
            camera_config: Camera configuration (fov, position, look_at, etc.)
            static_spheres_positions: Static sphere positions (M, 3)
            static_spheres_radii: Static sphere radii (M,)
            static_spheres_colors: Static sphere colors (M, 3)
            capture_client_idx: Index of client to capture from

        Returns:
            RGB image as numpy array (height, width, 3), dtype uint8
        """
        if self._server is None:
            self.start()

        # Ensure q is 2D (N, DOF)
        q = jnp.asarray(q)
        if q.ndim == 1:
            q = q[None, :]  # (1, DOF)

        num_robots = q.shape[0]

        # Compute base offsets
        if base_offsets is not None:
            base_offsets = jnp.asarray(base_offsets)
        elif self._base_offsets is not None:
            base_offsets = jnp.asarray(self._base_offsets)[:num_robots]
        else:
            base_offsets = self._compute_grid_offsets(num_robots, self._grid_spacing)

        # Pad to 3D if needed
        if base_offsets.shape[1] == 2:
            base_offsets = jnp.concatenate(
                [base_offsets, jnp.zeros((num_robots, 1))], axis=1
            )

        # Compute backbone curves
        curves = np.asarray(self.compute_backbone_curves_batched(q, base_offsets))

        cfg = color_config or self.color_config
        resolved_colors = self.resolve_backbone_colors(num_robots, color_config=cfg)

        # Build scene
        self._clear_scene()
        self._build_robot_geometry(
            curves,
            resolved_colors.per_robot_point_rgba,
            base_plate_color=cfg.base_plate_color,
        )

        # Add static spheres if provided
        if static_spheres_positions is not None:
            self._build_static_spheres(
                np.asarray(static_spheres_positions),
                np.asarray(
                    static_spheres_radii
                    or np.ones(len(static_spheres_positions)) * 0.02
                ),
                np.asarray(
                    static_spheres_colors
                    or np.ones((len(static_spheres_positions), 3)) * 0.5
                ),
            )

        self._setup_camera(curves, camera_config)

        # Wait for client and capture
        time.sleep(0.1)  # Give time for scene to render

        clients = list(self._server.get_clients().values())
        if len(clients) > capture_client_idx:
            client = clients[capture_client_idx]
            try:
                render = client.camera.get_render(height=self.height, width=self.width)
                return np.array(render)
            except Exception as e:
                print(f"[ViserRenderer] Failed to capture frame: {e}")

        # Return blank image if no client
        return np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

    def show(
        self,
        q: Array,
        *,
        base_offsets: Array | None = None,
        color_config: RendererColorConfig | None = None,
        camera_config: CameraConfig | None = None,
        render_tendons: bool = True,
        static_spheres_positions: Array | None = None,
        static_spheres_radii: Array | None = None,
        static_spheres_colors: Array | None = None,
        blocking: bool = True,
    ) -> None:
        """Display single frame interactively.

        Opens browser automatically and optionally blocks.

        Args:
            q: Robot configuration (DOF,) or batched (N, DOF)
            base_offsets: Base position offsets (N, 3)
            color_config: Shared renderer color configuration
            camera_config: Camera configuration (fov, position, look_at, etc.)
            render_tendons: If True, render tendons (if available)
            static_spheres_positions: Static sphere positions (M, 3)
            static_spheres_radii: Static sphere radii (M,)
            static_spheres_colors: Static sphere colors (M, 3)
            blocking: If True, block until user closes browser
        """
        if self._server is None:
            self.start()

        # Ensure q is 2D (N, DOF)
        q = jnp.asarray(q)
        if q.ndim == 1:
            q = q[None, :]  # (1, DOF)

        num_robots = q.shape[0]

        # Compute base offsets
        if base_offsets is not None:
            base_offsets = jnp.asarray(base_offsets)
        elif self._base_offsets is not None:
            base_offsets = jnp.asarray(self._base_offsets)[:num_robots]
        else:
            base_offsets = self._compute_grid_offsets(num_robots, self._grid_spacing)

        # Pad to 3D if needed
        if base_offsets.shape[1] == 2:
            base_offsets = jnp.concatenate(
                [base_offsets, jnp.zeros((num_robots, 1))], axis=1
            )

        # Compute backbone curves
        curves = np.asarray(self.compute_backbone_curves_batched(q, base_offsets))

        cfg = color_config or self.color_config
        resolved_colors = self.resolve_backbone_colors(num_robots, color_config=cfg)

        # Build scene
        self._clear_scene()
        self._build_robot_geometry(
            curves,
            resolved_colors.per_robot_point_rgba,
            base_plate_color=cfg.base_plate_color,
        )

        # Add tendon visualization if available and requested
        if render_tendons and self._has_tendons:
            self._build_tendon_geometry(
                q, np.asarray(base_offsets), num_robots, tendon_color=cfg.tendon_color
            )

        # Add static spheres if provided
        if static_spheres_positions is not None:
            self._build_static_spheres(
                np.asarray(static_spheres_positions),
                np.asarray(
                    static_spheres_radii
                    or np.ones(len(static_spheres_positions)) * 0.02
                ),
                np.asarray(
                    static_spheres_colors
                    or np.ones((len(static_spheres_positions), 3)) * 0.5
                ),
            )

        self._setup_camera(curves, camera_config)
        self._setup_lighting()

        # Open browser
        if self._open_browser:
            webbrowser.open(self.url)

        print(f"[ViserRenderer] Visualization at {self.url}")

        if blocking:
            print("[ViserRenderer] Press Ctrl+C to exit")
            try:
                while self._running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass

    def render_sequence(
        self,
        ts: Array,
        q_ts: Array,
        *,
        playback_speed: float = 1.0,
        autoplay: bool = True,
        loop: bool = False,
        record_path: str | None = None,
        record_every_n: int = 1,
        video_config: VideoEncodingConfig | None = None,
        camera_config: CameraConfig | None = None,
        base_offsets: Array | None = None,
        color_config: RendererColorConfig | None = None,
        render_tendons: bool = True,
        multi_robot_layout: Literal["grid", "overlay"] = "grid",
        static_spheres_positions: Array | None = None,
        static_spheres_radii: Array | None = None,
        static_spheres_colors: Array | None = None,
        dynamic_spheres_positions: Array | None = None,
        dynamic_spheres_radii: Array | None = None,
        dynamic_spheres_colors: Array | None = None,
        blocking: bool = True,
        plot_configurations: bool = False,
        plot_tendon_positions: bool = False,
        custom_plots: dict[str, tuple] | None = None,
        robot_name: str = "Robot",
    ) -> None:
        """Render animated trajectory with full visualization options.

        Args:
            ts: Time stamps (T,)
            q_ts: Configurations (T, DOF) or batched (N, T, DOF)
            playback_speed: Speed multiplier
            autoplay: Start playing immediately
            loop: Loop animation
            record_path: Path to save video (mp4, mov)
            record_every_n: Record every N frames
            video_config: FFmpeg encoding settings
            camera_config: Camera configuration (fov, position, look_at, etc.)
            base_offsets: Base position offsets (N, 2/3)
            color_config: Shared renderer color configuration
            render_tendons: If True, render tendons (if available)
            multi_robot_layout: "grid" for side-by-side, "overlay" for same position
            static_spheres_positions: Static sphere positions
            static_spheres_radii: Static sphere radii
            static_spheres_colors: Static sphere colors
            dynamic_spheres_positions: Time-varying sphere positions
            dynamic_spheres_radii: Time-varying sphere radii
            dynamic_spheres_colors: Time-varying sphere colors
            blocking: If True, block until viewer closes
            plot_configurations: If True, add configuration vs time plot to GUI
            plot_tendon_positions: If True, add tendon position plot to GUI
            custom_plots: Dictionary mapping plot names to (figure, aspect) tuples
                         for custom plotly figures to add to the GUI
            robot_name: Name for plot titles
        """
        if self._server is None:
            self.start()

        ts = np.asarray(ts)
        q_ts = jnp.asarray(q_ts)

        if q_ts.ndim == 3 and (plot_configurations or plot_tendon_positions):
            raise ValueError(
                "Plotting configurations or tendon positions requires q_ts with shape "
                "(T, DOF); got batched (N, T, DOF)."
            )

        # Save original 2D q_ts for plots (before reshaping to 3D)
        q_ts_for_plots = q_ts if q_ts.ndim == 2 else q_ts[0]

        # Handle shape: (T, DOF) -> (1, T, DOF)
        if q_ts.ndim == 2:
            q_ts = q_ts[None, :, :]  # (1, T, DOF)

        num_robots, num_frames, num_dofs = q_ts.shape

        # Compute base offsets
        if multi_robot_layout == "overlay":
            # All robots at same position
            base_offsets = jnp.zeros((num_robots, 3))
        elif base_offsets is not None:
            base_offsets = jnp.asarray(base_offsets)
        elif self._base_offsets is not None:
            base_offsets = jnp.asarray(self._base_offsets)[:num_robots]
        else:
            base_offsets = self._compute_grid_offsets(num_robots, self._grid_spacing)

        # Pad to 3D if needed
        if base_offsets.shape[1] == 2:
            base_offsets = jnp.concatenate(
                [base_offsets, jnp.zeros((num_robots, 1))], axis=1
            )

        cfg = color_config or self.color_config
        resolved_colors = self.resolve_backbone_colors(num_robots, color_config=cfg)

        # Precompute all backbone curves for first frame
        curves_0 = np.asarray(
            self.compute_backbone_curves_batched(q_ts[:, 0, :], base_offsets)
        )

        # Build initial scene
        self._clear_scene()
        self._build_robot_geometry(
            curves_0,
            resolved_colors.per_robot_point_rgba,
            base_plate_color=cfg.base_plate_color,
        )

        # Add tendon visualization if available and requested
        if render_tendons and self._has_tendons:
            self._build_tendon_geometry(
                q_ts[:, 0, :],
                np.asarray(base_offsets),
                num_robots,
                tendon_color=cfg.tendon_color,
            )

        # Add static spheres
        if static_spheres_positions is not None:
            self._build_static_spheres(
                np.asarray(static_spheres_positions),
                np.asarray(
                    static_spheres_radii
                    or np.ones(len(static_spheres_positions)) * 0.02
                ),
                np.asarray(
                    static_spheres_colors
                    or np.ones((len(static_spheres_positions), 3)) * 0.5
                ),
            )

        # Add dynamic spheres
        if dynamic_spheres_positions is not None:
            self._build_dynamic_spheres(
                np.asarray(dynamic_spheres_positions),
                np.asarray(
                    dynamic_spheres_radii
                    or np.ones(len(dynamic_spheres_positions)) * 0.02
                ),
                np.asarray(
                    dynamic_spheres_colors
                    or np.ones((len(dynamic_spheres_positions), 3)) * 0.2
                ),
            )

        self._setup_camera(curves_0, camera_config)
        self._setup_lighting()

        # Setup animation state
        self._animation_state = AnimationState(
            frame_idx=0,
            playing=autoplay,
            loop=loop,
            playback_speed=playback_speed,
            num_frames=num_frames,
            ts=ts,
        )

        # Setup GUI controls
        self._setup_playback_gui()

        # Add plots after playback controls (so they appear at the end)
        if plot_configurations or plot_tendon_positions or custom_plots:
            with self._server.gui.add_folder("Plots"):
                if plot_configurations:
                    config_fig = self.create_configuration_plot(
                        ts, q_ts_for_plots, robot_name=robot_name
                    )
                    self.add_gui_plotly("Configurations", config_fig, aspect=2.0)

                if plot_tendon_positions and hasattr(self.robot, "tendon_length"):
                    tendon_fig = self.create_tendon_position_plot(
                        ts, q_ts_for_plots, self.robot, robot_name=robot_name
                    )
                    self.add_gui_plotly("Tendon Positions", tendon_fig, aspect=2.0)

                # Add custom plots
                if custom_plots:
                    for plot_name, (figure, aspect) in custom_plots.items():
                        self.add_gui_plotly(plot_name, figure, aspect=aspect)

        # Open browser
        if self._open_browser:
            webbrowser.open(self.url)

        print(f"[ViserRenderer] Animation at {self.url}")
        print(f"[ViserRenderer] {num_frames} frames, {num_robots} robot(s)")

        # Video recording setup
        video_writer = None
        if record_path is not None:
            fps = 1.0 / np.mean(np.diff(ts)) if len(ts) > 1 else 30.0
            video_writer = FFmpegVideoWriter(
                record_path,
                self.width,
                self.height,
                fps=fps,
                input_pix_fmt="rgb24",
                video_config=video_config,
            )
            print(f"[ViserRenderer] Recording to {record_path}")

        # Animation loop
        try:
            while self._running:
                with self._lock:
                    state = self._animation_state
                    if state is None:
                        break

                    if state.playing:
                        now = time.time()
                        dt = state.dt_sequence
                        frame_dt = dt[min(state.frame_idx, len(dt) - 1)] / max(
                            state.playback_speed, 0.01
                        )

                        if now - state.last_tick >= frame_dt:
                            state.last_tick = now
                            next_idx = state.frame_idx + 1

                            if next_idx >= state.num_frames:
                                if state.loop:
                                    next_idx = 0
                                else:
                                    state.playing = False
                                    next_idx = state.num_frames - 1

                            if next_idx != state.frame_idx:
                                state.frame_idx = next_idx
                                self._update_frame(
                                    next_idx,
                                    q_ts,
                                    base_offsets,
                                    resolved_colors.per_robot_point_rgba,
                                    base_plate_color=cfg.base_plate_color,
                                    tendon_color=cfg.tendon_color,
                                    render_tendons=render_tendons,
                                )

                                # Record frame
                                if (
                                    video_writer is not None
                                    and next_idx % record_every_n == 0
                                ):
                                    clients = list(self._server.get_clients().values())
                                    if clients:
                                        try:
                                            frame = clients[0].camera.get_render(
                                                height=self.height, width=self.width
                                            )
                                            video_writer.write(np.array(frame))
                                        except Exception:
                                            pass

                    # Update GUI
                    self._update_playback_gui(state)

                time.sleep(0.01)  # ~100 Hz update rate

                if not blocking:
                    break

        except KeyboardInterrupt:
            pass
        finally:
            if video_writer is not None:
                video_writer.close()
                print(f"[ViserRenderer] Video saved: {record_path}")

    def _update_frame(
        self,
        frame_idx: int,
        q_ts: Array,
        base_offsets: Array,
        point_colors: np.ndarray,
        *,
        base_plate_color: tuple[float, float, float],
        tendon_color: tuple[float, float, float],
        render_tendons: bool = True,
    ) -> None:
        """Update scene for given frame index.

        Uses efficient position/orientation updates when geometry is already
        initialized, avoiding expensive mesh recreation each frame.
        """
        # Get configurations for this frame
        q_frame = q_ts[:, frame_idx, :]  # (num_robots, DOF)
        num_robots = q_frame.shape[0]

        # Compute new curves
        curves = np.asarray(self.compute_backbone_curves_batched(q_frame, base_offsets))

        # Check if we can use efficient updates (geometry already built with same config)
        can_update = (
            self._scene_handles is not None
            and self._scene_handles.geometry_initialized
            and self._scene_handles.num_robots == num_robots
            and self._scene_handles.num_backbone_points == curves.shape[1]
        )

        if can_update:
            # Efficient update: only modify position/orientation properties
            self._update_robot_geometry(curves)
        else:
            # Full rebuild needed (first frame or configuration changed)
            self._build_robot_geometry(
                curves, point_colors, base_plate_color=base_plate_color
            )

        # Update tendon geometry if available and requested
        if render_tendons and self._has_tendons:
            self._build_tendon_geometry(
                q_frame,
                np.asarray(base_offsets),
                num_robots,
                tendon_color=tendon_color,
            )

        # Update dynamic spheres
        self._update_dynamic_spheres(frame_idx)

    def _setup_playback_gui(self) -> None:
        """Setup playback control GUI elements."""
        if self._server is None:
            return

        with self._server.gui.add_folder("Playback"):
            self._gui_handles["play_button"] = self._server.gui.add_button(
                "Play/Pause",
                icon=viser.Icon.PLAYER_PLAY,
            )
            self._gui_handles["frame_slider"] = self._server.gui.add_slider(
                "Frame",
                min=0,
                max=max(1, self._animation_state.num_frames - 1),
                step=1,
                initial_value=0,
            )
            self._gui_handles["speed_slider"] = self._server.gui.add_slider(
                "Speed",
                min=0.1,
                max=5.0,
                step=0.1,
                initial_value=1.0,
            )
            self._gui_handles["loop_checkbox"] = self._server.gui.add_checkbox(
                "Loop",
                initial_value=self._animation_state.loop,
            )
            self._gui_handles["time_text"] = self._server.gui.add_text(
                "Time",
                initial_value="t = 0.00 s",
                disabled=True,
            )

        # Wire up callbacks
        @self._gui_handles["play_button"].on_click
        def _(_):
            if self._animation_state is not None:
                self._animation_state.playing = not self._animation_state.playing
                self._animation_state.last_tick = time.time()

        @self._gui_handles["frame_slider"].on_update
        def _(event):
            if self._animation_state is not None and not self._animation_state.playing:
                self._animation_state.frame_idx = int(event.target.value)

        @self._gui_handles["speed_slider"].on_update
        def _(event):
            if self._animation_state is not None:
                self._animation_state.playback_speed = event.target.value

        @self._gui_handles["loop_checkbox"].on_update
        def _(event):
            if self._animation_state is not None:
                self._animation_state.loop = event.target.value

    def _update_playback_gui(self, state: AnimationState) -> None:
        """Update GUI to reflect current animation state."""
        if "frame_slider" in self._gui_handles:
            self._gui_handles["frame_slider"].value = state.frame_idx

        if "time_text" in self._gui_handles:
            t = state.ts[state.frame_idx] if state.frame_idx < len(state.ts) else 0.0
            self._gui_handles["time_text"].value = f"t = {t:.2f} s"

        if "play_button" in self._gui_handles:
            self._gui_handles["play_button"].icon = (
                viser.Icon.PLAYER_PAUSE if state.playing else viser.Icon.PLAYER_PLAY
            )

    # =========================================================================
    # Plot panels
    # =========================================================================

    def add_gui_plotly(
        self, name: str, figure: PlotlyFigure, aspect: float = 1.0
    ) -> Any:
        """Add a plotly figure to the GUI.

        Args:
            name: Unique name for the plot
            figure: Plotly figure object
            aspect: Aspect ratio (width/height)

        Returns:
            Viser plotly handle
        """
        if self._server is None:
            self.start()

        if go is None:
            raise ImportError(
                "plotly is required for plot panels. "
                "Install it with: pip install plotly"
            )

        handle = self._server.gui.add_plotly(figure=figure, aspect=aspect)
        return handle

    def create_configuration_plot(
        self,
        ts: Array | np.ndarray,
        q_ts: Array | np.ndarray,
        robot_name: str = "Robot",
    ) -> PlotlyFigure:
        """Create a plotly figure showing configurations over time.

        Args:
            ts: Time array (T,)
            q_ts: Configuration array (T, DOF)
            robot_name: Name for the plot title

        Returns:
            Plotly figure
        """
        if go is None:
            raise ImportError("plotly is required. Install with: pip install plotly")

        ts = np.asarray(ts)
        q_ts = np.asarray(q_ts)
        fig = go.Figure()

        # Plot each configuration variable
        num_dofs = q_ts.shape[1]
        for i in range(num_dofs):
            fig.add_trace(
                go.Scatter(
                    x=ts,
                    y=q_ts[:, i],
                    mode="lines",
                    name=f"q[{i}]",
                )
            )

        fig.update_layout(
            title=f"{robot_name} - Configurations vs Time",
            xaxis_title="Time [s]",
            yaxis_title="Configuration",
            height=400,
            margin={"l": 50, "r": 50, "t": 50, "b": 50},
        )

        return fig

    def create_tendon_position_plot(
        self,
        ts: Array | np.ndarray,
        q_ts: Array | np.ndarray,
        robot: SoftRobot,
        robot_name: str = "Robot",
    ) -> PlotlyFigure:
        """Create a plotly figure showing tendon positions over time.

        Args:
            ts: Time array (T,)
            q_ts: Configuration array (T, DOF)
            robot: Robot instance with tendon_length method
            robot_name: Name for the plot title

        Returns:
            Plotly figure
        """
        if go is None:
            raise ImportError("plotly is required. Install with: pip install plotly")

        if not hasattr(robot, "tendon_length"):
            raise AttributeError("Robot does not have tendon_length method")

        ts = np.asarray(ts)
        q_ts = jnp.asarray(q_ts)
        fig = go.Figure()

        # Compute tendon lengths for all timesteps using vmap
        tendon_lengths_ts = jax.vmap(robot.tendon_length)(q_ts)
        tendon_lengths_ts = np.asarray(tendon_lengths_ts)  # (T, num_tendons)
        num_tendons = tendon_lengths_ts.shape[1]

        # Plot each tendon
        for i in range(num_tendons):
            fig.add_trace(
                go.Scatter(
                    x=ts,
                    y=tendon_lengths_ts[:, i],
                    mode="lines",
                    name=f"Tendon {i}",
                )
            )

        fig.update_layout(
            title=f"{robot_name} - Tendon Positions vs Time",
            xaxis_title="Time [s]",
            yaxis_title="Tendon Length [m]",
            height=400,
            margin={"l": 50, "r": 50, "t": 50, "b": 50},
        )

        return fig

    # =========================================================================
    # Live mode
    # =========================================================================

    def start_live_mode(
        self,
        callback: Callable[[float], np.ndarray] | None = None,
        dt: float = 0.033,
    ) -> LiveModeController:
        """Start live visualization mode.

        Two usage patterns:

        1. Callback mode (renderer pulls):
            >>> def get_state(t):
            ...     return compute_robot_state(t)
            >>> ctrl = renderer.start_live_mode(callback=get_state)

        2. Stream mode (user pushes):
            >>> ctrl = renderer.start_live_mode()
            >>> for q in simulation_loop():
            ...     ctrl.push_state(q)

        Args:
            callback: Optional state provider function (time) -> q
            dt: Time step for callback mode in seconds

        Returns:
            LiveModeController for managing the live session
        """
        if self._server is None:
            self.start()

        controller = LiveModeController(self, callback, dt)
        controller.start()
        return controller

    def _stop_live_mode_internal(self) -> None:
        """Internal method to stop live mode."""
        self._live_mode_active = False
        if self._live_thread is not None:
            self._live_thread.join(timeout=1.0)
            self._live_thread = None

    # =========================================================================
    # Dynamic primitives API
    # =========================================================================

    def add_dynamic_sphere(
        self,
        name: str,
        position: np.ndarray,
        radius: float,
        color: tuple[float, float, float] = (0.2, 0.2, 0.8),
        opacity: float = 1.0,
    ) -> Any:
        """Add a dynamic sphere to the scene.

        Args:
            name: Unique identifier for the sphere
            position: Initial position (3,)
            radius: Sphere radius
            color: RGB color (0-1)
            opacity: Opacity (0-1)

        Returns:
            Viser sphere handle
        """
        if self._server is None:
            self.start()

        handle = self._server.scene.add_icosphere(
            name=f"/custom/{name}",
            radius=radius,
            color=_rgb_to_viser_color(np.array(color)),
            subdivisions=self._sphere_resolution,
            position=tuple(position),
        )
        self._scene_handles.custom_primitives[name] = handle
        return handle

    def update_dynamic_sphere(
        self,
        name: str,
        position: np.ndarray | None = None,
        radius: float | None = None,
        color: tuple[float, float, float] | None = None,
    ) -> None:
        """Update properties of a dynamic sphere.

        Args:
            name: Sphere identifier
            position: New position (3,)
            radius: New radius
            color: New RGB color
        """
        if name not in self._scene_handles.custom_primitives:
            return

        handle = self._scene_handles.custom_primitives[name]
        if position is not None:
            handle.position = tuple(position)

    def remove_dynamic_sphere(self, name: str) -> None:
        """Remove a dynamic sphere from the scene.

        Args:
            name: Sphere identifier
        """
        if name in self._scene_handles.custom_primitives:
            handle = self._scene_handles.custom_primitives.pop(name)
            handle.remove()

    def add_custom_primitive(
        self,
        name: str,
        primitive_type: Literal["sphere", "box", "cylinder", "mesh"],
        **kwargs: Any,
    ) -> Any:
        """Add a custom primitive to the scene for extensibility.

        Args:
            name: Unique identifier
            primitive_type: Type of primitive
            **kwargs: Primitive-specific parameters

        Returns:
            Viser scene handle
        """
        if self._server is None:
            self.start()

        if primitive_type == "sphere":
            handle = self._server.scene.add_icosphere(
                name=f"/custom/{name}",
                **kwargs,
            )
        elif primitive_type == "box":
            handle = self._server.scene.add_box(
                name=f"/custom/{name}",
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported primitive type: {primitive_type}")

        self._scene_handles.custom_primitives[name] = handle
        return handle

    # =========================================================================
    # Video export
    # =========================================================================

    def render_to_video(
        self,
        ts: Array,
        q_ts: Array,
        output_path: str,
        *,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        video_config: VideoEncodingConfig | None = None,
        camera_position: tuple[float, float, float] | None = None,
        camera_target: tuple[float, float, float] | None = None,
        **render_kwargs: Any,
    ) -> None:
        """Render sequence directly to video file.

        This method renders each frame and writes to video using FFmpeg.
        Requires at least one connected client for capture.

        Args:
            ts: Time stamps
            q_ts: Configurations
            output_path: Output video path (.mp4, .mov)
            width: Frame width (default: self.width)
            height: Frame height (default: self.height)
            fps: Output FPS (if None, derived from ts)
            video_config: FFmpeg settings
            camera_position: Fixed camera position
            camera_target: Camera look-at target
            **render_kwargs: Additional render_sequence arguments
        """
        print("[ViserRenderer] render_to_video requires an interactive session.")
        print(
            f"[ViserRenderer] Use render_sequence with record_path='{output_path}' instead."
        )

        # Call render_sequence with recording enabled
        self.render_sequence(
            ts,
            q_ts,
            record_path=output_path,
            video_config=video_config,
            blocking=True,
            **render_kwargs,
        )


# =============================================================================
# LiveModeController
# =============================================================================


class LiveModeController:
    """Controller for live robot state updates.

    Supports two patterns:
    1. Callback mode: Renderer calls user function at each frame
    2. Stream mode: User pushes states via push_state()
    """

    def __init__(
        self,
        renderer: ViserRenderer,
        callback: Callable[[float], np.ndarray] | None = None,
        dt: float = 0.033,
    ):
        self._renderer = renderer
        self._callback = callback
        self._dt = dt
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._state_queue: list[np.ndarray] = []
        self._start_time = 0.0
        self._resolved_colors = None
        self._resolved_colors_num_robots = None

    def start(self) -> None:
        """Start the live mode."""
        if self._running:
            return

        self._running = True
        self._renderer._live_mode_active = True
        self._start_time = time.time()

        if self._callback is not None:
            # Callback mode - run in thread
            self._thread = threading.Thread(target=self._callback_loop, daemon=True)
            self._thread.start()

        print("[ViserRenderer] Live mode started")

    def stop(self) -> None:
        """Stop the live mode."""
        self._running = False
        self._renderer._live_mode_active = False

        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        print("[ViserRenderer] Live mode stopped")

    def push_state(self, q: np.ndarray) -> None:
        """Push a new configuration to the renderer.

        Args:
            q: Configuration array (DOF,) or batched (N, DOF)
        """
        if not self._running:
            return

        with self._lock:
            self._state_queue.append(np.asarray(q))

        # Process immediately
        self._process_state(q)

    def push_state_with_extras(
        self,
        q: np.ndarray,
        dynamic_spheres: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Push state with auxiliary geometry updates.

        Args:
            q: Configuration array
            dynamic_spheres: Dict mapping sphere names to positions
        """
        self.push_state(q)

        if dynamic_spheres is not None:
            for name, pos in dynamic_spheres.items():
                self._renderer.update_dynamic_sphere(name, position=pos)

    def from_generator(
        self,
        state_generator: Generator[np.ndarray, None, None],
        fps: float = 30.0,
    ) -> None:
        """Consume states from a generator at specified FPS.

        Args:
            state_generator: Generator yielding configurations
            fps: Target frame rate
        """
        dt = 1.0 / fps

        for q in state_generator:
            if not self._running:
                break

            self.push_state(q)
            time.sleep(dt)

    def _callback_loop(self) -> None:
        """Main loop for callback mode."""
        while self._running:
            t = time.time() - self._start_time

            if self._callback is not None:
                try:
                    q = self._callback(t)
                    self._process_state(q)
                except Exception as e:
                    print(f"[ViserRenderer] Callback error: {e}")

            time.sleep(self._dt)

    def _process_state(self, q: np.ndarray) -> None:
        """Process and render a state.

        Uses efficient position/orientation updates when geometry is already
        initialized, avoiding expensive mesh recreation each frame.
        """
        q = jnp.asarray(q)
        if q.ndim == 1:
            q = q[None, :]

        num_robots = q.shape[0]

        # Compute base offsets
        base_offsets = self._renderer._compute_grid_offsets(
            num_robots, self._renderer._grid_spacing
        )
        if base_offsets.shape[1] == 2:
            base_offsets = jnp.concatenate(
                [base_offsets, jnp.zeros((num_robots, 1))], axis=1
            )

        # Compute curves
        curves = np.asarray(
            self._renderer.compute_backbone_curves_batched(q, base_offsets)
        )

        if (
            self._resolved_colors is None
            or self._resolved_colors_num_robots != num_robots
        ):
            self._resolved_colors = self._renderer.resolve_backbone_colors(num_robots)
            self._resolved_colors_num_robots = num_robots

        # Check if we can use efficient updates
        scene_handles = self._renderer._scene_handles
        can_update = (
            scene_handles is not None
            and scene_handles.geometry_initialized
            and scene_handles.num_robots == num_robots
            and scene_handles.num_backbone_points == curves.shape[1]
        )

        if can_update:
            # Efficient update: only modify position/orientation properties
            self._renderer._update_robot_geometry(curves)
        else:
            # Full rebuild needed (first frame or configuration changed)
            self._renderer._build_robot_geometry(
                curves,
                self._resolved_colors.per_robot_point_rgba,
                base_plate_color=self._renderer.color_config.base_plate_color,
            )
