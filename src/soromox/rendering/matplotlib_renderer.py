"""Matplotlib-based renderer for any continuum soft robot."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

from soromox.rendering.base import BaseSoftRobotRenderer
from soromox.systems.soft_robot import SoftRobot


class MatplotlibRenderer(BaseSoftRobotRenderer):
    """Matplotlib visualization for any continuum soft robot.

    Supports both 2D (planar) and 3D robots, with FuncAnimation and slider modes.
    The dimensionality is auto-detected from the robot's forward kinematics output.

    Example:
        >>> renderer = MatplotlibRenderer(robot)
        >>> renderer.show(q)  # Single frame
        >>> renderer.animate(ts, q_ts, mode="slider")  # Interactive slider
        >>> renderer.animate(ts, q_ts, mode="animation")  # Auto-play
    """

    def __init__(
        self,
        robot: SoftRobot,
        width: int = 800,
        height: int = 600,
        num_points: int = 50,
        background_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        line_color: str = "blue",
        line_width: float = 4.0,
        robot_colors: str | list | None = None,
        grid_spacing: tuple[float, float] = (0.3, 0.3),
        base_offsets: Array | None = None,
        tendon_color: tuple[float, float, float] = (0.9, 0.1, 0.1),
        tendon_line_width: float = 2.0,
    ):
        """Initialize Matplotlib renderer.

        Args:
            robot: Robot system with forward_kinematics method
            width: Figure width in pixels
            height: Figure height in pixels
            num_points: Number of points for backbone discretization
            background_color: RGB background color (0-1 range)
            line_color: Color for backbone line
            line_width: Width of backbone line
            robot_colors: Optional color configuration for batched layouts
                (None/"same", colormap name, or explicit list cycled across robots)
            grid_spacing: (x, y) spacing for batched layouts
            base_offsets: Optional explicit base offsets for batched layouts
            tendon_color: Color for tendon polylines (if available)
            tendon_line_width: Line width for tendon polylines
        """
        super().__init__(robot, width, height, num_points, background_color)
        self.line_color = line_color
        self.line_width = line_width
        self.grid_spacing = grid_spacing
        self._base_offsets = base_offsets
        self._robot_colors_config = robot_colors
        self.tendon_color = tendon_color
        self.tendon_line_width = tendon_line_width

        self._has_tendons = hasattr(robot, "forward_kinematics_tendons")
        if self._has_tendons:
            self._batched_fk_tendons = jax.vmap(
                robot.forward_kinematics_tendons, in_axes=(None, 0), out_axes=1
            )

        # Auto-detect 3D vs 2D from FK output
        self._is_3d = self._detect_dimensionality()

    def _get_n_q(self) -> int:
        """Get the number of configuration variables from the robot.

        Different robot classes use different attribute names for DOF count.
        """
        # Try common attribute names in order of preference
        for attr in ["num_active_strains", "dof_tot_system", "num_dofs", "n_q"]:
            if hasattr(self.robot, attr):
                val = getattr(self.robot, attr)
                return int(val)
        raise AttributeError(
            f"Robot {type(self.robot).__name__} has no recognized DOF attribute. "
            "Expected one of: num_active_strains, dof_tot_system, num_dofs, n_q"
        )

    def _detect_dimensionality(self) -> bool:
        """Detect if robot is 3D based on FK output shape.

        Returns:
            True if 3D (SE(3) matrices), False if 2D (SE(2) poses)
        """
        # Create a test configuration
        n_q = self._get_n_q()
        q_test = jnp.zeros((n_q,))
        pose = self.robot.forward_kinematics(q_test, 0.0)

        # SE(3) returns 4x4 matrix, SE(2) returns 3-vector
        if pose.ndim == 2 and pose.shape == (4, 4):
            return True
        return False

    @property
    def is_3d(self) -> bool:
        """Return True for 3D robots, False for 2D/planar robots."""
        return self._is_3d

    def _extract_positions(self, poses: Array) -> Array:
        """Extract xyz or xy positions from FK poses.

        Args:
            poses: FK output - either (N, 4, 4) for SE(3) or (N, 3) for SE(2)

        Returns:
            Positions array (N, 3) for 3D or (N, 2) for 2D
        """
        if self.is_3d:
            # SE(3): extract translation from 4x4 matrices
            return poses[:, :3, 3]
        else:
            # SE(2): extract x, y from [theta, x, y] poses
            return poses[:, 1:3]

    def compute_tendon_curves(self, q: Array) -> Array | None:
        """Compute tendon paths if robot supports tendons."""
        if not self._has_tendons:
            return None
        s_ps = jnp.linspace(0.0, self.L_max, self.num_points)
        return self._batched_fk_tendons(q, s_ps)

    def compute_tendon_curves_batched(
        self, q_batch: Array, base_offsets: Array
    ) -> Array | None:
        """Compute tendon paths for multiple robots with base offsets."""
        if not self._has_tendons:
            return None
        s_ps = jnp.linspace(0.0, self.L_max, self.num_points)
        tendon_curves = jax.vmap(lambda q: self._batched_fk_tendons(q, s_ps))(q_batch)
        # tendon_curves shape: (N, n_tendon, P, dim)
        offsets = base_offsets
        if offsets.shape[1] + 1 == tendon_curves.shape[-1]:
            pad = jnp.zeros((offsets.shape[0], 1), dtype=offsets.dtype)
            offsets = jnp.concatenate([offsets, pad], axis=1)
        return tendon_curves + offsets[:, None, None, :]

    def render_frame(
        self,
        q: Array,
        *,
        base_offsets: Array | None = None,
        robot_colors: str | list | None = None,
        show_tendons: bool = True,
    ) -> np.ndarray:
        """Render configuration(s) to an RGB image array.

        Args:
            q: Robot configuration array. Shape (DOF,) for a single robot or (N, DOF)
                for batched rendering.
            base_offsets: Optional explicit base offsets of shape (N, 2) or (N, 3) for
                batched layouts. When provided for a single robot, accepts (dim,) or
                (1, dim).
            robot_colors: Optional color configuration for batched layouts
                (None/"same", colormap name, or explicit list cycled across robots).
            show_tendons: Whether to render tendons if available.

        Returns:
            img (np.ndarray): RGB image of shape (height, width, 3), dtype uint8.
        """
        q_arr = jnp.asarray(q)
        batched = q_arr.ndim == 2

        if not batched and q_arr.ndim != 1:
            raise ValueError(
                f"q must have shape (DOF,) or (N, DOF); got shape {q_arr.shape}"
            )

        spacing = self.grid_spacing
        colors_cfg = self._robot_colors_config if robot_colors is None else robot_colors

        if batched:
            num_robots = int(q_arr.shape[0])
            base_offsets_arr = (
                self._compute_grid_offsets(num_robots, spacing)
                if (base_offsets is None and self._base_offsets is None)
                else jnp.asarray(
                    base_offsets if base_offsets is not None else self._base_offsets
                )
            )

            curves = self.compute_backbone_curves_batched(q_arr, base_offsets_arr)
            colors = self._get_robot_colors(num_robots, colors_cfg)
            max_extent = float(np.max(np.abs(curves))) if curves.size else 0.0
            width_m = max(self.L_max * 3, 2.0 * max_extent * 1.1)
        else:
            curves = self.compute_backbone_curve(q_arr)[None, ...]
            if base_offsets is not None:
                offs = np.asarray(base_offsets, dtype=np.float64)
                if offs.ndim == 1:
                    offs = offs.reshape(1, -1)
                if offs.shape[0] != 1:
                    raise ValueError(
                        f"base_offsets must have shape (dim,) or (1, dim) for single robot; got {offs.shape}"
                    )
                if offs.shape[1] + 1 == curves.shape[-1]:
                    offs = np.concatenate([offs, np.zeros((offs.shape[0], 1))], axis=1)
                if offs.shape[1] != curves.shape[-1]:
                    raise ValueError(
                        f"base_offsets second dimension ({offs.shape[1]}) must match curve dimension ({curves.shape[-1]})"
                    )
                curves = curves + offs[0]
            colors = self._get_robot_colors(1, colors_cfg)
            width_m = self.L_max * 3

        curves_np = np.array(curves)
        tendon_curves_np = None
        if show_tendons and self._has_tendons:
            if batched:
                tendon_curves_np = np.array(
                    self.compute_tendon_curves_batched(q_arr, base_offsets_arr)
                )
            else:
                tendon_curves_np = np.array(self.compute_tendon_curves(q_arr))[
                    None, ...
                ]
                if base_offsets is not None:
                    tendon_curves_np = tendon_curves_np + offs[0]

        fig = plt.figure(figsize=(self.width / 100, self.height / 100), dpi=100)
        fig.patch.set_facecolor(self.background_color)

        if self.is_3d:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)

        self._setup_axes(ax, width_m, center_origin=batched)

        for idx, (curve, color) in enumerate(zip(curves_np, colors)):
            if self.is_3d:
                ax.plot(
                    curve[:, 0],
                    curve[:, 1],
                    curve[:, 2],
                    lw=self.line_width,
                    color=color,
                )
            else:
                ax.plot(
                    curve[:, 0],
                    curve[:, 1],
                    lw=self.line_width,
                    color=color,
                )

            if tendon_curves_np is not None:
                tc = tendon_curves_np[idx]
                for k in range(tc.shape[0]):
                    if self.is_3d:
                        ax.plot(
                            tc[k, :, 0],
                            tc[k, :, 1],
                            tc[k, :, 2],
                            lw=self.tendon_line_width,
                            color=self.tendon_color,
                        )
                    else:
                        ax.plot(
                            tc[k, :, 0],
                            tc[k, :, 1],
                            lw=self.tendon_line_width,
                            color=self.tendon_color,
                        )

        ax.set_facecolor(self.background_color)
        plt.tight_layout()

        # Render to numpy array
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return img

    def show(
        self,
        q: Array,
        *,
        base_offsets: Array | None = None,
        robot_colors: str | list | None = None,
        render_tendons: bool = True,
    ) -> None:  # type: ignore[override]
        """Display a single frame interactively (supports batched inputs).

        Args:
            q: Robot configuration array. Shape (DOF,) or (N, DOF).
            base_offsets: Optional explicit base offsets for batched layouts
                (N, 2/3) or (1, dim) for single robot.
            robot_colors: Optional color configuration for batched layouts
                (None/"same", colormap name, or explicit list cycled across robots).
            render_tendons: Whether to render tendons if available.

        Returns:
            None
        """
        img = self.render_frame(
            q,
            base_offsets=base_offsets,
            robot_colors=robot_colors,
            show_tendons=render_tendons,
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
        robot_colors: str | list | None = None,
        render_tendons: bool = True,
    ):
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
            robot_colors: Optional color configuration for batched layouts
                (None/"same", colormap name, or explicit list cycled across robots).
            render_tendons: Whether to render tendons if available

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
                robot_colors=robot_colors,
                render_tendons=render_tendons,
                record_path=record_path,
                playback_speed=playback_speed,
            )
        if q_ts_arr.ndim == 2:
            return self._animate_single(
                ts_arr,
                q_ts_arr,
                interval=interval,
                mode=mode,
                show=show,
                base_offsets=base_offsets,
                robot_colors=robot_colors,
                render_tendons=render_tendons,
                record_path=record_path,
                playback_speed=playback_speed,
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
        robot_colors: str | list | None,
        render_tendons: bool,
        record_path: str | None,
        playback_speed: float,
    ):
        """Animate a single robot (legacy behavior).

        Args:
            ts: Time stamps array of shape (T,).
            q_ts: Configurations array of shape (T, DOF).
            interval: Frame interval in ms (for animation mode).
            mode: "slider" or "animation".
            show: Whether to display the plot.
            base_offsets: Optional base offset of shape (dim,) or (1, dim) applied to the robot.
            robot_colors: Optional color configuration for the robot.
            render_tendons: Whether to render tendons if available.
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
        if base_offsets is not None:
            offs = np.asarray(base_offsets, dtype=np.float64)
            if offs.ndim == 1:
                offs = offs.reshape(1, -1)
            if offs.shape[0] not in (1,):
                raise ValueError(
                    f"base_offsets must have shape (dim,) or (1, dim) for single robot; got {offs.shape}"
                )
            if offs.shape[1] + 1 == curves.shape[-1]:
                offs = np.concatenate([offs, np.zeros((offs.shape[0], 1))], axis=1)
            if offs.shape[1] != curves.shape[-1]:
                raise ValueError(
                    f"base_offsets second dimension ({offs.shape[1]}) must match curve dimension ({curves.shape[-1]})"
                )
            curves = curves + offs[0]
        all_curves = curves[None, ...]  # (1, T, P, dim)
        tendon_curves = None
        if render_tendons and self._has_tendons:
            tendon_curves = np.array(
                jax.vmap(self.compute_tendon_curves)(q_ts), dtype=np.float64
            )[None, ...]  # (1, T, n_tendon, P, dim)
            if base_offsets is not None:
                tendon_curves = tendon_curves + offs[0]

        colors_cfg = self._robot_colors_config if robot_colors is None else robot_colors
        colors = self._get_robot_colors(1, colors_cfg)

        return self._animate_backbone_curves(
            ts_np,
            all_curves,
            colors=colors,
            interval=interval,
            mode=mode,
            show=show,
            center_origin=False,
            tendon_curves=tendon_curves,
            record_path=record_path,
            playback_speed=playback_speed,
        )

    def _animate_batched(
        self,
        ts: Array,
        q_ts: Array,
        interval: int,
        mode: str,
        show: bool,
        base_offsets: Array | None,
        robot_colors: str | list | None,
        render_tendons: bool,
        record_path: str | None,
        playback_speed: float,
    ):
        """Animate multiple robots arranged in a grid layout.

        Args:
            ts: Time stamps array of shape (T,).
            q_ts: Configurations array of shape (N, T, DOF) or (T, DOF) promoted to batch.
            interval: Frame interval in ms (for animation mode).
            mode: "slider" or "animation".
            show: Whether to display the plot.
            base_offsets: Optional base offsets of shape (N, 2/3) for batched layouts.
            robot_colors: Optional color configuration for batched layouts.
            render_tendons: Whether to render tendons if available.
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
        colors_cfg = self._robot_colors_config if robot_colors is None else robot_colors
        base_offsets_arr = (
            self._compute_grid_offsets(int(num_robots), spacing)
            if (base_offsets is None and self._base_offsets is None)
            else jnp.asarray(
                base_offsets if base_offsets is not None else self._base_offsets
            )
        )

        # Precompute all backbone curves with offsets applied
        def curves_for_timestep(q_batch: Array) -> Array:
            return self.compute_backbone_curves_batched(q_batch, base_offsets_arr)

        q_ts_time_first = q_ts.transpose(1, 0, 2)  # (T, N, DOF)
        all_curves_time_first = jax.vmap(curves_for_timestep)(q_ts_time_first)
        all_curves = np.asarray(all_curves_time_first.transpose(1, 0, 2, 3))

        # Axis sizing based on geometry spread
        max_extent = float(np.max(np.abs(all_curves))) if all_curves.size else 0.0
        width_m = max(self.L_max * 3, 2.0 * max_extent * 1.1)

        colors = self._get_robot_colors(int(num_robots), colors_cfg)

        tendon_curves = None
        if render_tendons and self._has_tendons:
            tendons = np.array(
                jax.vmap(
                    lambda q_batch: self.compute_tendon_curves_batched(
                        q_batch, base_offsets_arr
                    )
                )(q_ts_time_first)
            )  # (T, N, n_t, P, dim)
            tendon_curves = tendons.transpose(1, 0, 2, 3, 4)  # (N, T, n_t, P, dim)

        return self._animate_backbone_curves(
            ts=np.asarray(ts),
            all_curves=all_curves,
            colors=colors,
            interval=interval,
            mode=mode,
            show=show,
            center_origin=True,
            tendon_curves=tendon_curves,
            record_path=record_path,
            playback_speed=playback_speed,
        )

    def _animate_backbone_curves(
        self,
        ts: np.ndarray,
        all_curves: np.ndarray,
        colors: list,
        interval: int,
        mode: str,
        show: bool,
        center_origin: bool,
        tendon_curves: np.ndarray | None = None,
        record_path: str | None = None,
        playback_speed: float = 1.0,
    ):
        """Shared Matplotlib animation for one or more robots."""
        num_robots, num_steps, _, _ = all_curves.shape

        max_extent = float(np.max(np.abs(all_curves))) if all_curves.size else 0.0
        width_m = max(self.L_max * 3, 2.0 * max_extent * 1.1)

        fig = plt.figure(figsize=(self.width / 100, self.height / 100))
        fig.patch.set_facecolor(self.background_color)

        if self.is_3d:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)

        self._setup_axes(ax, width_m, center_origin=center_origin)
        title_text = ax.set_title("t = 0.00 s")

        if self.is_3d:
            lines = [
                ax.plot([], [], [], lw=self.line_width, color=color)[0]
                for color in colors
            ]
        else:
            lines = [
                ax.plot([], [], lw=self.line_width, color=color)[0] for color in colors
            ]

        tendon_lines: list[list] = []
        if tendon_curves is not None:
            for idx in range(num_robots):
                robot_lines = []
                n_tend = tendon_curves.shape[2]
                for _ in range(n_tend):
                    if self.is_3d:
                        (tl,) = ax.plot(
                            [],
                            [],
                            [],
                            lw=self.tendon_line_width,
                            color=self.tendon_color,
                        )
                    else:
                        (tl,) = ax.plot(
                            [], [], lw=self.tendon_line_width, color=self.tendon_color
                        )
                    robot_lines.append(tl)
                tendon_lines.append(robot_lines)

        def _update_lines(frame_idx: int):
            curves_frame = all_curves[:, frame_idx]  # (N, P, dim)
            for idx, line in enumerate(lines):
                curve = curves_frame[idx]
                line.set_data(curve[:, 0], curve[:, 1])
                if self.is_3d:
                    line.set_3d_properties(curve[:, 2])
            if tendon_curves is not None:
                for idx, tc in enumerate(tendon_curves):
                    tend_frame = tc[frame_idx]
                    for k, tl in enumerate(tendon_lines[idx]):
                        tl.set_data(tend_frame[k, :, 0], tend_frame[k, :, 1])
                        if self.is_3d:
                            tl.set_3d_properties(tend_frame[k, :, 2])

        if mode == "animation":

            def init():
                for line in lines:
                    line.set_data([], [])
                    if self.is_3d:
                        line.set_3d_properties([])
                title_text.set_text("t = 0.00 s")
                for robot_lines in tendon_lines:
                    for tl in robot_lines:
                        tl.set_data([], [])
                        if self.is_3d:
                            tl.set_3d_properties([])
                flat_tendon = [tl for sub in tendon_lines for tl in sub]
                return lines + flat_tendon + [title_text]

            def update(frame_idx):
                frame_idx_int = int(frame_idx)
                _update_lines(frame_idx_int)
                title_text.set_text(f"t = {float(ts[frame_idx_int]):.2f} s")
                flat_tendon = [tl for sub in tendon_lines for tl in sub]
                return lines + flat_tendon + [title_text]

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
        render_tendons: bool = True,
        record_path: str | None = None,
        playback_speed: float = 1.0,
        show: bool = False,
    ) -> None:
        """Render an animated sequence (optionally saving to disk).

        This reuses the animation code path with mode='animation'.
        """
        self.animate(
            ts=ts,
            q_ts=q_ts,
            interval=interval,
            mode="animation",
            show=show,
            render_tendons=render_tendons,
            record_path=record_path,
            playback_speed=playback_speed,
        )

    def _setup_axes(self, ax, width_m: float, center_origin: bool = False) -> None:
        """Set up axis limits and labels.

        Args:
            ax: Matplotlib axes object
            width_m: Width in meters for limits
        """
        if self.is_3d:
            ax.set_xlim(-width_m / 2, width_m / 2)
            ax.set_ylim(-width_m / 2, width_m / 2)
            ax.set_zlim(0, width_m)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_zlabel("Z [m]")
        else:
            ax.set_xlim(-width_m / 2, width_m / 2)
            if center_origin:
                ax.set_ylim(-width_m / 2, width_m / 2)
            else:
                ax.set_ylim(0, width_m)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_aspect("equal")

    def _get_robot_colors(
        self, num_robots: int, robot_colors: str | list | None
    ) -> list:
        """Resolve colors for each robot in a batch."""
        if robot_colors is None or robot_colors == "same":
            return [self.line_color] * num_robots
        if isinstance(robot_colors, str):
            import matplotlib.cm as cm

            cmap = cm.get_cmap(robot_colors)
            return [
                tuple(cmap(i / max(1, num_robots - 1))[:3]) for i in range(num_robots)
            ]
        return [robot_colors[i % len(robot_colors)] for i in range(num_robots)]
