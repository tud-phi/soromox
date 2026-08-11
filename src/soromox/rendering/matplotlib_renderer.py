"""Matplotlib-based renderer for any continuum soft robot."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Line3DCollection

if TYPE_CHECKING:
    from IPython.display import HTML

    AnimateReturn = HTML | FuncAnimation | None
else:
    AnimateReturn = Any | FuncAnimation | None

from soromox.rendering.actuators import (
    TrajectoryActuatorVisualLayer,
    resolve_actuator_rgba,
)
from soromox.rendering.base import BaseSoftRobotRenderer
from soromox.rendering.camera_config import CameraConfig
from soromox.rendering.color_config import RendererColorConfig
from soromox.systems.soft_robot import SoftRobot


class MatplotlibRenderer(BaseSoftRobotRenderer):
    """Matplotlib visualization for any continuum soft robot.

    Supports both 2D (planar) and 3D robots, with FuncAnimation and slider modes.
    The dimensionality is auto-detected from the robot's forward kinematics output.

    Example:
        ```python
        renderer = MatplotlibRenderer(robot)
        renderer.show(q)
        renderer.animate(ts, q_ts, mode="slider")
        ```
    """

    def __init__(
        self,
        robot: SoftRobot,
        width: int = 800,
        height: int = 600,
        num_points: int = 50,
        background_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        color_config: RendererColorConfig | None = None,
        show_ground_plane: bool = True,
        ground_plane_size: float | None = None,
        line_width: float = 4.0,
        grid_spacing: tuple[float, float] = (0.3, 0.3),
        base_offsets: Array | None = None,
        actuator_line_width: float = 2.0,
    ):
        """Initialize Matplotlib renderer.

        Args:
            robot: Robot system with forward_kinematics method
            width: Figure width in pixels
            height: Figure height in pixels
            num_points: Number of points for backbone discretization
            background_color: RGB background color (0-1 range)
            color_config: Shared renderer color configuration
            show_ground_plane: Whether to draw a reference plane through the base
            ground_plane_size: Optional side length of the reference plane in meters
            line_width: Width of backbone line
            grid_spacing: (x, y) spacing for batched layouts
            base_offsets: Optional explicit base offsets for batched layouts
            actuator_line_width: Line width for actuator polylines
        """
        super().__init__(
            robot,
            width,
            height,
            num_points,
            background_color,
            color_config=color_config,
            show_ground_plane=show_ground_plane,
            ground_plane_size=ground_plane_size,
        )
        self.line_width = line_width
        self.grid_spacing = grid_spacing
        self._base_offsets = base_offsets
        self.actuator_line_width = actuator_line_width

    def render_frame(
        self,
        q: Array,
        *,
        base_offsets: Array | None = None,
        color_config: RendererColorConfig | None = None,
        camera_config: CameraConfig | None = None,
        render_actuators: bool = True,
        actuator_inputs: Array | None = None,
    ) -> np.ndarray:
        """Render configuration(s) to an RGB image array.

        Args:
            q: Robot configuration array. Shape (DOF,) for a single robot or (N, DOF)
                for batched rendering.
            base_offsets: Optional explicit base offsets of shape (N, 2) or (N, 3) for
                batched layouts. When provided for a single robot, accepts (dim,) or
                (1, dim).
            color_config: Shared renderer color configuration.
            camera_config: Camera configuration. Matplotlib uses its field of view
                and the direction from position to look-at for 3D plots.
            render_actuators: Whether to render actuator visual layers if available.
            actuator_inputs: Optional actuator inputs for scalar-colored layers.

        Returns:
            img (np.ndarray): RGB image of shape (height, width, 3), dtype uint8.
        """
        q_arr = jnp.asarray(q)
        batched = q_arr.ndim == 2
        cfg = color_config or self.color_config

        if not batched and q_arr.ndim != 1:
            raise ValueError(
                f"q must have shape (DOF,) or (N, DOF); got shape {q_arr.shape}"
            )

        spacing = self.grid_spacing
        if batched:
            num_robots = int(q_arr.shape[0])
            uses_configured_offsets = (
                base_offsets is None and self._base_offsets is not None
            )
            offset_source = (
                self._compute_grid_offsets(num_robots, spacing)
                if (base_offsets is None and self._base_offsets is None)
                else base_offsets
                if base_offsets is not None
                else self._base_offsets
            )
            base_offsets_arr = self._normalize_base_offsets(
                offset_source,
                num_robots=num_robots,
                target_dim=3 if self.is_3d else 2,
                allow_extra_rows=uses_configured_offsets,
            )

            curves = self.compute_backbone_curves_batched(q_arr, base_offsets_arr)
            resolved_colors = self.resolve_backbone_colors(num_robots, color_config=cfg)
            max_extent = float(np.max(np.abs(curves))) if curves.size else 0.0
            width_m = max(self.L_max * 3, 2.0 * max_extent * 1.1)
        else:
            num_robots = 1
            curves = self.compute_backbone_curve(q_arr)[None, ...]
            offset_source = (
                base_offsets if base_offsets is not None else self._base_offsets
            )
            single_offset = self._single_base_offset(
                offset_source,
                target_dim=int(curves.shape[-1]),
                allow_extra_rows=base_offsets is None
                and self._base_offsets is not None,
            )
            if single_offset is not None:
                curves = curves + single_offset
            resolved_colors = self.resolve_backbone_colors(1, color_config=cfg)
            width_m = self.L_max * 3

        curves_np = np.array(curves)
        actuator_layers = ()
        if render_actuators and self._has_actuator_visuals:
            if batched:
                actuator_layers = self.compute_actuator_visual_layers_batched(
                    q_arr,
                    base_offsets_arr,
                    actuator_inputs=actuator_inputs,
                )
            else:
                actuator_offsets = (
                    jnp.zeros((1, int(curves.shape[-1])), dtype=q_arr.dtype)
                    if single_offset is None
                    else jnp.asarray(single_offset).reshape(1, -1)
                )
                actuator_layers = self.compute_actuator_visual_layers_batched(
                    q_arr[None, :],
                    actuator_offsets,
                    actuator_inputs=actuator_inputs,
                )

        fig = plt.figure(figsize=(self.width / 100, self.height / 100), dpi=100)
        fig.patch.set_facecolor(self.background_color)

        if self.is_3d:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)

        scene_center = None
        scene_extent = None
        if curves_np.size:
            flat = curves_np.reshape(-1, curves_np.shape[-1])
            scene_center = np.mean(flat, axis=0)
            scene_extent = float(np.max(np.ptp(flat, axis=0)))

        self._setup_axes(
            ax,
            width_m,
            center_origin=batched,
            camera_config=camera_config,
            scene_center=scene_center,
            scene_extent=scene_extent,
        )

        self._plot_ground_plane(ax, curves_np, cfg)
        self._plot_backbone(ax, curves_np, resolved_colors.per_robot_point_rgba)
        self._plot_base_markers(ax, curves_np, cfg.base_plate_color)

        self._plot_actuator_layers(ax, actuator_layers, cfg)

        ax.set_facecolor(self.background_color)
        plt.tight_layout()

        # Render to numpy array
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        if hasattr(fig.canvas, "tostring_rgb"):
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape((height, width, 3))
        elif hasattr(fig.canvas, "buffer_rgba"):
            img = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
        else:
            argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            argb = argb.reshape((height, width, 4))
            img = argb[..., 1:4]
        plt.close(fig)

        return img

    def show(
        self,
        q: Array,
        *,
        base_offsets: Array | None = None,
        color_config: RendererColorConfig | None = None,
        camera_config: CameraConfig | None = None,
        render_actuators: bool = True,
        actuator_inputs: Array | None = None,
    ) -> None:  # type: ignore[override]
        """Display a single frame interactively (supports batched inputs).

        Args:
            q: Robot configuration array. Shape (DOF,) or (N, DOF).
            base_offsets: Optional explicit base offsets for batched layouts
                (N, 2/3) or (1, dim) for single robot.
            color_config: Shared renderer color configuration.
            camera_config: Camera configuration. Matplotlib uses its field of view
                and the direction from position to look-at for 3D plots.
            render_actuators: Whether to render actuator visual layers if available.
            actuator_inputs: Optional actuator inputs for scalar-colored layers.

        Returns:
            None
        """
        img = self.render_frame(
            q,
            base_offsets=base_offsets,
            color_config=color_config,
            camera_config=camera_config,
            render_actuators=render_actuators,
            actuator_inputs=actuator_inputs,
        )
        plt.figure(figsize=(self.width / 100, self.height / 100))
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def animate(
        self,
        ts: Array,
        q_ts: Array,
        interval: int = 50,
        mode: str = "slider",
        show: bool = True,
        playback_speed: float = 1.0,
        record_path: str | None = None,
        base_offsets: Array | None = None,
        color_config: RendererColorConfig | None = None,
        camera_config: CameraConfig | None = None,
        render_actuators: bool = True,
        actuator_inputs: Array | None = None,
    ) -> AnimateReturn:
        """Interactive matplotlib animation with slider or auto-play.

        Args:
            ts: Time stamps array of shape (T,)
            q_ts: Configurations array of shape (T, DOF) or (N, T, DOF)
            interval: Frame interval in ms (for animation mode)
            mode: "slider" for manual scrubbing, "animation" for auto-play
            show: Whether to call plt.show()
            playback_speed: Multiplier for playback speed (>1 = faster)
            record_path: Optional path to save animation (mp4/gif depending on writer)
            base_offsets: Optional explicit base offsets for batched layouts
            color_config: Shared renderer color configuration.
            camera_config: Camera configuration. Matplotlib uses its field of view
                and the direction from position to look-at for 3D plots.
            render_actuators: Whether to render actuator visual layers if available
            actuator_inputs: Optional actuator inputs for scalar-colored layers.

        Returns:
            HTML object for Jupyter display (animation mode only)
        """
        if mode not in ("slider", "animation"):
            raise ValueError("mode must be 'slider' or 'animation'")
        if record_path is not None and mode != "animation":
            raise ValueError("record_path is only supported in animation mode")
        if playback_speed <= 0:
            raise ValueError("playback_speed must be positive")

        ts_arr = jnp.asarray(ts).reshape((-1,))
        q_ts_arr = jnp.asarray(q_ts)

        if q_ts_arr.ndim == 3:
            return self._animate_batched(
                ts_arr,
                q_ts_arr,
                interval=interval,
                mode=mode,
                show=show,
                base_offsets=base_offsets,
                color_config=color_config,
                render_actuators=render_actuators,
                actuator_inputs=actuator_inputs,
                record_path=record_path,
                playback_speed=playback_speed,
                camera_config=camera_config,
            )
        if q_ts_arr.ndim == 2:
            return self._animate_single(
                ts_arr,
                q_ts_arr,
                interval=interval,
                mode=mode,
                show=show,
                base_offsets=base_offsets,
                color_config=color_config,
                render_actuators=render_actuators,
                actuator_inputs=actuator_inputs,
                record_path=record_path,
                playback_speed=playback_speed,
                camera_config=camera_config,
            )

        raise ValueError(
            f"q_ts must have shape (T, DOF) or (N, T, DOF); got shape {q_ts_arr.shape}"
        )

    def _animate_single(
        self,
        ts: Array,
        q_ts: Array,
        interval: int,
        mode: str,
        show: bool,
        base_offsets: Array | None,
        color_config: RendererColorConfig | None,
        render_actuators: bool,
        actuator_inputs: Array | None,
        record_path: str | None,
        playback_speed: float,
        camera_config: CameraConfig | None,
    ):
        """Animate a single robot (legacy behavior).

        Args:
            ts: Time stamps array of shape (T,).
            q_ts: Configurations array of shape (T, DOF).
            interval: Frame interval in ms (for animation mode).
            mode: "slider" or "animation".
            show: Whether to display the plot.
            base_offsets: Optional base offset of shape (dim,) or (1, dim) applied to the robot.
            color_config: Shared renderer color configuration.
            render_actuators: Whether to render actuator visual layers if available.
            actuator_inputs: Optional actuator inputs for scalar-colored layers.
            record_path: Optional path to save animation (mp4/gif depending on writer).
            playback_speed: Playback speed multiplier (>0).

        Returns:
            FuncAnimation or None depending on mode.
        """
        ts_np = np.asarray(ts)
        # Precompute backbone curves once for single robot
        curves = np.array(
            jax.vmap(self.compute_backbone_curve)(q_ts), dtype=np.float64
        )  # (T, P, dim)
        offset_source = base_offsets if base_offsets is not None else self._base_offsets
        single_offset = self._single_base_offset(
            offset_source,
            target_dim=int(curves.shape[-1]),
            allow_extra_rows=base_offsets is None and self._base_offsets is not None,
        )
        if single_offset is not None:
            curves = curves + single_offset
        all_curves = curves[None, ...]  # (1, T, P, dim)
        actuator_layers = ()
        if render_actuators and self._has_actuator_visuals:
            actuator_offsets = (
                jnp.zeros((1, int(curves.shape[-1])), dtype=q_ts.dtype)
                if single_offset is None
                else jnp.asarray(single_offset).reshape(1, -1)
            )
            actuator_layers = self.compute_actuator_visual_layers_trajectory(
                q_ts[None, ...],
                actuator_offsets,
                actuator_inputs=actuator_inputs,
            )

        cfg = color_config or self.color_config
        resolved_colors = self.resolve_backbone_colors(1, color_config=cfg)

        return self._animate_backbone_curves(
            ts_np,
            all_curves,
            point_colors=resolved_colors.per_robot_point_rgba,
            interval=interval,
            mode=mode,
            show=show,
            center_origin=False,
            actuator_layers=actuator_layers,
            record_path=record_path,
            base_plate_color=cfg.base_plate_color,
            ground_plane_color_config=cfg,
            playback_speed=playback_speed,
            camera_config=camera_config,
        )

    def _animate_batched(
        self,
        ts: Array,
        q_ts: Array,
        interval: int,
        mode: str,
        show: bool,
        base_offsets: Array | None,
        color_config: RendererColorConfig | None,
        render_actuators: bool,
        actuator_inputs: Array | None,
        record_path: str | None,
        playback_speed: float,
        camera_config: CameraConfig | None,
    ):
        """Animate multiple robots arranged in a grid layout.

        Args:
            ts: Time stamps array of shape (T,).
            q_ts: Configurations array of shape (N, T, DOF) or (T, DOF) promoted to batch.
            interval: Frame interval in ms (for animation mode).
            mode: "slider" or "animation".
            show: Whether to display the plot.
            base_offsets: Optional base offsets of shape (N, 2/3) for batched layouts.
            color_config: Shared renderer color configuration.
            render_actuators: Whether to render actuator visual layers if available.
            actuator_inputs: Optional actuator inputs for scalar-colored layers.
            record_path: Optional path to save animation (mp4/gif depending on writer).
            playback_speed: Playback speed multiplier (>0).

        Returns:
            FuncAnimation or None depending on mode.
        """
        ts = jnp.asarray(ts).reshape((-1,))
        q_ts = jnp.asarray(q_ts)

        if q_ts.ndim == 2:
            q_ts = q_ts[None, ...]
        if q_ts.ndim != 3:
            raise ValueError(
                f"Batched animation expects q_ts with ndim 2 or 3, got {q_ts.ndim}"
            )

        num_robots, num_steps, _ = q_ts.shape
        if ts.shape[0] != num_steps:
            raise ValueError(
                f"ts length ({ts.shape[0]}) must match q_ts time dimension ({num_steps})"
            )

        spacing = self.grid_spacing
        uses_configured_offsets = (
            base_offsets is None and self._base_offsets is not None
        )
        offset_source = (
            self._compute_grid_offsets(int(num_robots), spacing)
            if (base_offsets is None and self._base_offsets is None)
            else base_offsets
            if base_offsets is not None
            else self._base_offsets
        )
        base_offsets_arr = self._normalize_base_offsets(
            offset_source,
            num_robots=int(num_robots),
            target_dim=3 if self.is_3d else 2,
            allow_extra_rows=uses_configured_offsets,
        )

        # Precompute all backbone curves with offsets applied
        def curves_for_timestep(q_batch: Array) -> Array:
            return self.compute_backbone_curves_batched(q_batch, base_offsets_arr)

        q_ts_time_first = q_ts.transpose(1, 0, 2)  # (T, N, DOF)
        all_curves_time_first = jax.vmap(curves_for_timestep)(q_ts_time_first)
        all_curves = np.asarray(all_curves_time_first.transpose(1, 0, 2, 3))

        cfg = color_config or self.color_config
        resolved_colors = self.resolve_backbone_colors(
            int(num_robots), color_config=cfg
        )

        actuator_layers = ()
        if render_actuators and self._has_actuator_visuals:
            actuator_layers = self.compute_actuator_visual_layers_trajectory(
                q_ts,
                base_offsets_arr,
                actuator_inputs=actuator_inputs,
            )

        return self._animate_backbone_curves(
            ts=np.asarray(ts),
            all_curves=all_curves,
            point_colors=resolved_colors.per_robot_point_rgba,
            interval=interval,
            mode=mode,
            show=show,
            center_origin=True,
            actuator_layers=actuator_layers,
            record_path=record_path,
            base_plate_color=cfg.base_plate_color,
            ground_plane_color_config=cfg,
            playback_speed=playback_speed,
            camera_config=camera_config,
        )

    def _animate_backbone_curves(
        self,
        ts: np.ndarray,
        all_curves: np.ndarray,
        point_colors: np.ndarray,
        interval: int,
        mode: str,
        show: bool,
        center_origin: bool,
        actuator_layers: tuple[TrajectoryActuatorVisualLayer, ...] = (),
        record_path: str | None = None,
        base_plate_color: tuple[float, float, float] | None = None,
        ground_plane_color_config: RendererColorConfig | None = None,
        playback_speed: float = 1.0,
        camera_config: CameraConfig | None = None,
    ):
        """Shared Matplotlib animation for one or more robots."""
        num_robots, num_steps, _, _ = all_curves.shape
        base_plate_color = base_plate_color or self.color_config.base_plate_color

        max_extent = float(np.max(np.abs(all_curves))) if all_curves.size else 0.0
        width_m = max(self.L_max * 3, 2.0 * max_extent * 1.1)

        fig = plt.figure(figsize=(self.width / 100, self.height / 100))
        fig.patch.set_facecolor(self.background_color)

        if self.is_3d:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)

        scene_center = None
        scene_extent = None
        if all_curves.size:
            flat = all_curves.reshape(-1, all_curves.shape[-1])
            scene_center = np.mean(flat, axis=0)
            scene_extent = float(np.max(np.ptp(flat, axis=0)))

        self._setup_axes(
            ax,
            width_m,
            center_origin=center_origin,
            camera_config=camera_config,
            scene_center=scene_center,
            scene_extent=scene_extent,
        )
        title_text = ax.set_title("t = 0.00 s")

        segment_colors = [
            self._segment_colors_from_point_pairs(point_colors[idx])
            for idx in range(num_robots)
        ]
        lines = []
        for idx in range(num_robots):
            segments = self._curve_to_segments(all_curves[idx, 0])
            if self.is_3d:
                lc = Line3DCollection(
                    segments,
                    colors=segment_colors[idx],
                    linewidths=self.line_width,
                )
                ax.add_collection3d(lc)
            else:
                lc = LineCollection(
                    segments,
                    colors=segment_colors[idx],
                    linewidths=self.line_width,
                )
                ax.add_collection(lc)
            lines.append(lc)

        self._plot_ground_plane(
            ax,
            all_curves[:, 0],
            ground_plane_color_config or self.color_config,
        )
        base_artists = self._plot_base_markers(ax, all_curves[:, 0], base_plate_color)

        actuator_artists = []
        actuator_colors = [
            resolve_actuator_rgba(
                layer,
                default_color=self.color_config.actuators.color_for_kind(layer.kind),
                scalar_colormap=self.color_config.actuators.scalar_colormap,
            )
            for layer in actuator_layers
        ]
        for layer_idx, layer in enumerate(actuator_layers):
            points = np.asarray(layer.points)
            line_width = layer.line_width or self.actuator_line_width
            for robot_idx in range(num_robots):
                for actuator_idx in range(points.shape[2]):
                    if self.is_3d:
                        (artist,) = ax.plot([], [], [], lw=line_width)
                    else:
                        (artist,) = ax.plot([], [], lw=line_width)
                    actuator_artists.append(
                        (layer_idx, robot_idx, actuator_idx, artist)
                    )

        def _update_lines(frame_idx: int):
            curves_frame = all_curves[:, frame_idx]  # (N, P, dim)
            for idx, line in enumerate(lines):
                segments = self._curve_to_segments(curves_frame[idx])
                line.set_segments(segments)
            self._update_base_markers(base_artists, curves_frame)
            for layer_idx, robot_idx, actuator_idx, artist in actuator_artists:
                layer = actuator_layers[layer_idx]
                pts = np.asarray(layer.points)[robot_idx, frame_idx, actuator_idx]
                artist.set_data(pts[:, 0], pts[:, 1])
                if self.is_3d:
                    artist.set_3d_properties(pts[:, 2])
                artist.set_color(
                    actuator_colors[layer_idx][robot_idx, frame_idx, actuator_idx]
                )

        if mode == "animation":

            def init():
                for line in lines:
                    if self.is_3d:
                        line.set_segments(np.zeros((0, 2, 3)))
                    else:
                        line.set_segments(np.zeros((0, 2, 2)))
                title_text.set_text("t = 0.00 s")
                for _, _, _, artist in actuator_artists:
                    artist.set_data([], [])
                    if self.is_3d:
                        artist.set_3d_properties([])
                flat_actuators = [artist for _, _, _, artist in actuator_artists]
                return lines + flat_actuators + base_artists + [title_text]

            def update(frame_idx):
                frame_idx_int = int(frame_idx)
                _update_lines(frame_idx_int)
                title_text.set_text(f"t = {float(ts[frame_idx_int]):.2f} s")
                flat_actuators = [artist for _, _, _, artist in actuator_artists]
                return lines + flat_actuators + base_artists + [title_text]

            actual_interval = max(1, int(round(interval / playback_speed)))
            ani = FuncAnimation(
                fig,
                update,
                frames=num_steps,
                init_func=init,
                blit=False,
                interval=actual_interval,
            )

            if record_path is not None:
                fps_eff = 1000.0 / max(1.0, float(actual_interval))
                try:
                    ani.save(record_path, writer="ffmpeg", fps=fps_eff)
                    print(
                        f"[Matplotlib] Animation saved to {record_path} (fps={fps_eff})"
                    )
                except Exception as exc:
                    print(
                        f"[Matplotlib] Failed to save animation to {record_path}: {exc}"
                    )

            if show:
                plt.show()

            plt.close(fig)

            try:
                from IPython.display import HTML

                return HTML(ani.to_jshtml())
            except ImportError:
                return None

        # Slider mode
        ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])

        def update_plot(frame_idx):
            frame_idx = int(frame_idx)
            _update_lines(frame_idx)
            title_text.set_text(f"t = {float(ts[frame_idx]):.2f} s")
            fig.canvas.draw_idle()

        slider = Slider(
            ax=ax_slider,
            label="Frame",
            valmin=0,
            valmax=num_steps - 1,
            valinit=0,
            valstep=1,
        )
        slider.on_changed(update_plot)
        update_plot(0)

        if show:
            plt.show()

        plt.close(fig)
        return None

    def render_sequence(
        self,
        ts: Array,
        q_ts: Array,
        *,
        interval: int = 50,
        record_path: str | None = None,
        playback_speed: float = 1.0,
        camera_config: CameraConfig | None = None,
        color_config: RendererColorConfig | None = None,
        render_actuators: bool = True,
        actuator_inputs: Array | None = None,
        show: bool = False,
    ) -> None:
        """Render an animated sequence (optionally saving to disk).

        This reuses the animation code path with mode='animation'.

        Args:
            ts: Time stamps array of shape (T,)
            q_ts: Configurations array of shape (T, DOF) or (N, T, DOF)
            interval: Frame interval in ms
            record_path: Optional path to save animation
            playback_speed: Multiplier for playback speed
            camera_config: Camera configuration. Matplotlib uses its field of view
                and the direction from position to look-at for 3D plots.
            color_config: Shared renderer color configuration.
            render_actuators: Whether to render actuator visual layers if available
            actuator_inputs: Optional actuator inputs for scalar-colored layers.
            show: Whether to display the animation
        """
        self.animate(
            ts=ts,
            q_ts=q_ts,
            interval=interval,
            mode="animation",
            show=show,
            record_path=record_path,
            playback_speed=playback_speed,
            camera_config=camera_config,
            color_config=color_config,
            render_actuators=render_actuators,
            actuator_inputs=actuator_inputs,
        )

    @staticmethod
    def _curve_to_segments(curve: np.ndarray) -> np.ndarray:
        if curve.shape[0] < 2:
            return np.zeros((0, 2, curve.shape[1]), dtype=curve.dtype)
        return np.stack([curve[:-1], curve[1:]], axis=1)

    @staticmethod
    def _base_plate_marker_points(
        base: np.ndarray,
        axis: np.ndarray,
        marker_diameter: float,
        dim: int,
    ) -> np.ndarray:
        """Return points that represent the base plate face in Matplotlib."""
        base = np.asarray(base, dtype=np.float64)
        axis = np.asarray(axis, dtype=np.float64)
        axis_norm = np.linalg.norm(axis)
        if axis_norm <= 1e-12:
            axis = np.zeros(dim, dtype=np.float64)
            axis[0] = 1.0
        else:
            axis = axis / axis_norm

        radius = 0.5 * marker_diameter
        if dim == 2:
            normal = np.array([-axis[1], axis[0]], dtype=np.float64)
            return np.stack(
                [
                    base - radius * normal,
                    base + radius * normal,
                ],
                axis=0,
            )

        reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(axis, reference))) > 0.95:
            reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = np.cross(axis, reference)
        u = u / np.linalg.norm(u)
        v = np.cross(axis, u)
        angles = np.linspace(0.0, 2.0 * np.pi, 65)
        return base[None, :] + radius * (
            np.cos(angles)[:, None] * u[None, :] + np.sin(angles)[:, None] * v
        )

    def _plot_actuator_layers(
        self,
        ax,
        actuator_layers,
        color_config: RendererColorConfig,
    ) -> None:
        """Plot batched actuator visual layers on a Matplotlib axis."""
        for layer in actuator_layers:
            points = np.asarray(layer.points, dtype=np.float64)
            colors = resolve_actuator_rgba(
                layer,
                default_color=color_config.actuators.color_for_kind(layer.kind),
                scalar_colormap=color_config.actuators.scalar_colormap,
            )
            line_width = layer.line_width or self.actuator_line_width
            for robot_idx in range(points.shape[0]):
                for actuator_idx in range(points.shape[1]):
                    pts = points[robot_idx, actuator_idx]
                    color = colors[robot_idx, actuator_idx]
                    if self.is_3d:
                        ax.plot(
                            pts[:, 0],
                            pts[:, 1],
                            pts[:, 2],
                            lw=line_width,
                            color=color,
                        )
                    else:
                        ax.plot(
                            pts[:, 0],
                            pts[:, 1],
                            lw=line_width,
                            color=color,
                        )

    def _plot_base_markers(
        self,
        ax,
        curves: np.ndarray,
        base_plate_color: tuple[float, float, float],
    ) -> list:
        """Draw compact base markers at the configured base positions."""
        artists = []
        if curves.size == 0:
            return artists
        dim = int(curves.shape[-1])
        axis = self._base_tangent_axis(dim=dim)
        marker_len = max(0.04 * self.L_max, 1e-3)
        for curve in curves:
            base = np.asarray(curve[0], dtype=np.float64)
            marker = self._base_plate_marker_points(base, axis, marker_len, dim)
            if dim == 3:
                (artist,) = ax.plot(
                    marker[:, 0],
                    marker[:, 1],
                    marker[:, 2],
                    color=base_plate_color,
                    linewidth=max(self.line_width, 2.0),
                )
            else:
                (artist,) = ax.plot(
                    marker[:, 0],
                    marker[:, 1],
                    color=base_plate_color,
                    linewidth=max(self.line_width, 2.0),
                )
            artists.append(artist)
        return artists

    def _plot_ground_plane(
        self,
        ax,
        curves: np.ndarray,
        color_config: RendererColorConfig,
    ) -> list:
        """Draw a base-aligned ground reference without affecting robot geometry."""
        if not self.show_ground_plane or curves.size == 0:
            return []

        dim = int(curves.shape[-1])
        bases = np.asarray(curves[:, 0], dtype=np.float64)
        center = np.mean(bases, axis=0)
        normal = self._base_tangent_axis(dim=dim)
        size = self._resolve_ground_plane_size()
        center = center - 0.0125 * max(self.L_max, 1e-3) * normal

        if dim == 2:
            tangent = np.array([-normal[1], normal[0]], dtype=np.float64)
            endpoints = (
                center[None, :] + 0.5 * size * np.array([-1.0, 1.0])[:, None] * tangent
            )
            return ax.plot(
                endpoints[:, 0],
                endpoints[:, 1],
                color=color_config.ground_plane_grid_color,
                linewidth=1.25,
                alpha=0.75,
                zorder=0,
            )

        reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(normal, reference))) > 0.95:
            reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = np.cross(normal, reference)
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        coords = np.linspace(-0.5 * size, 0.5 * size, 9)
        uu, vv = np.meshgrid(coords, coords)
        points = center + uu[..., None] * u + vv[..., None] * v
        surface = ax.plot_surface(
            points[..., 0],
            points[..., 1],
            points[..., 2],
            color=color_config.ground_plane_color,
            alpha=0.42,
            linewidth=0.0,
            shade=False,
            zorder=0,
        )
        wireframe = ax.plot_wireframe(
            points[..., 0],
            points[..., 1],
            points[..., 2],
            color=color_config.ground_plane_grid_color,
            alpha=0.38,
            linewidth=0.45,
            zorder=0,
        )
        return [surface, wireframe]

    def _update_base_markers(self, artists: list, curves: np.ndarray) -> None:
        """Update animated Matplotlib base markers."""
        if not artists:
            return
        dim = int(curves.shape[-1])
        axis = self._base_tangent_axis(dim=dim)
        marker_len = max(0.04 * self.L_max, 1e-3)
        for artist, curve in zip(artists, curves):
            base = np.asarray(curve[0], dtype=np.float64)
            marker = self._base_plate_marker_points(base, axis, marker_len, dim)
            if dim == 3:
                artist.set_data(
                    marker[:, 0],
                    marker[:, 1],
                )
                artist.set_3d_properties(marker[:, 2])
            else:
                artist.set_data(
                    marker[:, 0],
                    marker[:, 1],
                )

    @staticmethod
    def _segment_colors_from_point_pairs(point_colors: np.ndarray) -> np.ndarray:
        if point_colors.shape[0] < 2:
            return np.zeros((0, point_colors.shape[1]), dtype=point_colors.dtype)
        return 0.5 * (point_colors[:-1] + point_colors[1:])

    def _plot_backbone(
        self,
        ax,
        curves: np.ndarray,
        point_colors: np.ndarray,
    ) -> list:
        collections = []
        for idx, curve in enumerate(curves):
            segments = self._curve_to_segments(curve)
            seg_colors = self._segment_colors_from_point_pairs(point_colors[idx])
            if self.is_3d:
                lc = Line3DCollection(
                    segments,
                    colors=seg_colors,
                    linewidths=self.line_width,
                )
                ax.add_collection3d(lc)
            else:
                lc = LineCollection(
                    segments,
                    colors=seg_colors,
                    linewidths=self.line_width,
                )
                ax.add_collection(lc)
            collections.append(lc)
        return collections

    def _setup_axes(
        self,
        ax,
        width_m: float,
        center_origin: bool = False,
        camera_config: CameraConfig | None = None,
        scene_center: np.ndarray | None = None,
        scene_extent: float | None = None,
    ) -> None:
        """Set up axis limits and labels.

        Args:
            ax: Matplotlib axes object
            width_m: Width in meters for limits
        """
        if self.is_3d:
            center = (
                np.array(scene_center, dtype=np.float64)
                if scene_center is not None
                else np.array([0.0, 0.0, width_m * 0.5], dtype=np.float64)
            )
            content_extent = (
                float(scene_extent) if scene_extent is not None else width_m
            )
            plot_extent = max(1.15 * self.L_max, 1.1 * content_extent)
            if self.show_ground_plane:
                plot_extent = max(plot_extent, self._resolve_ground_plane_size())
            half_extent = 0.5 * plot_extent
            ax.set_xlim(center[0] - half_extent, center[0] + half_extent)
            ax.set_ylim(center[1] - half_extent, center[1] + half_extent)
            ax.set_zlim(center[2] - half_extent, center[2] + half_extent)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_zlabel("Z [m]")
            config = camera_config or CameraConfig()
            fov = float(np.clip(config.fov, 1.0, 179.0))
            focal_length = 1.0 / np.tan(np.deg2rad(fov) / 2.0)
            if hasattr(ax, "set_proj_type"):
                try:
                    ax.set_proj_type("persp", focal_length=focal_length)
                except TypeError:
                    ax.set_proj_type("persp")
            if hasattr(ax, "set_box_aspect"):
                default_half_fov = np.deg2rad(CameraConfig().fov) / 2.0
                zoom = np.clip(
                    np.tan(default_half_fov) / np.tan(np.deg2rad(fov) / 2.0),
                    0.85,
                    1.2,
                )
                try:
                    ax.set_box_aspect((1.0, 1.0, 1.0), zoom=float(zoom))
                except TypeError:
                    ax.set_box_aspect((1.0, 1.0, 1.0))
            camera_pos, look_at = config.compute_auto_position(
                center,
                content_extent,
                reference_transform=np.asarray(self.base_transform),
            )
            view_vec = np.asarray(camera_pos, dtype=np.float64) - np.asarray(
                look_at, dtype=np.float64
            )
            r_xy = float(np.hypot(view_vec[0], view_vec[1]))
            azim = np.degrees(np.arctan2(view_vec[1], view_vec[0]))
            elev = np.degrees(np.arctan2(view_vec[2], r_xy))
            ax.view_init(elev=elev, azim=azim)
        else:
            ax.set_xlim(-width_m / 2, width_m / 2)
            if center_origin:
                ax.set_ylim(-width_m / 2, width_m / 2)
            else:
                ax.set_ylim(0, width_m)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_aspect("equal")
