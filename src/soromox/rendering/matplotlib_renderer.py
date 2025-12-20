"""Matplotlib-based renderer for any continuum soft robot."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import numpy as np
from jax import Array

from soromox.rendering.base import BaseContinuumSoftRobotRenderer


class MatplotlibRenderer(BaseContinuumSoftRobotRenderer):
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
        robot,
        width: int = 800,
        height: int = 600,
        num_points: int = 50,
        background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        line_color: str = "blue",
        line_width: float = 4.0,
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
        """
        super().__init__(robot, width, height, num_points, background_color)
        self.line_color = line_color
        self.line_width = line_width

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

    def render_frame(self, q: Array) -> np.ndarray:
        """Render single configuration to RGB image array.

        Args:
            q: Robot configuration array

        Returns:
            RGB image as numpy array of shape (height, width, 3), dtype uint8
        """
        curve = np.array(self.compute_backbone_curve(q))
        width_m = self.L_max * 3

        fig = plt.figure(figsize=(self.width / 100, self.height / 100), dpi=100)
        fig.patch.set_facecolor(self.background_color)

        if self.is_3d:
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(
                curve[:, 0],
                curve[:, 1],
                curve[:, 2],
                lw=self.line_width,
                color=self.line_color,
            )
            ax.set_xlim(-width_m / 2, width_m / 2)
            ax.set_ylim(-width_m / 2, width_m / 2)
            ax.set_zlim(0, width_m)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_zlabel("Z [m]")
        else:
            ax = fig.add_subplot(111)
            ax.plot(curve[:, 0], curve[:, 1], lw=self.line_width, color=self.line_color)
            ax.set_xlim(-width_m / 2, width_m / 2)
            ax.set_ylim(0, width_m)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_aspect("equal")

        ax.set_facecolor(self.background_color)
        plt.tight_layout()

        # Render to numpy array
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return img

    def animate(
        self,
        ts: Array,
        q_ts: Array,
        interval: int = 50,
        mode: str = "slider",
        show: bool = True,
    ):
        """Interactive matplotlib animation with slider or auto-play.

        Args:
            ts: Time stamps array of shape (T,)
            q_ts: Configurations array of shape (T, DOF)
            interval: Frame interval in ms (for animation mode)
            mode: "slider" for manual scrubbing, "animation" for auto-play
            show: Whether to call plt.show()

        Returns:
            HTML object for Jupyter display (animation mode only)
        """
        if mode not in ("slider", "animation"):
            raise ValueError("mode must be 'slider' or 'animation'")

        ts = np.array(ts)
        q_ts = np.array(q_ts)
        width_m = self.L_max * 3

        fig = plt.figure(figsize=(self.width / 100, self.height / 100))

        if self.is_3d:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)

        ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])

        if mode == "animation":
            if self.is_3d:
                (line,) = ax.plot([], [], [], lw=self.line_width, color=self.line_color)
            else:
                (line,) = ax.plot([], [], lw=self.line_width, color=self.line_color)

            self._setup_axes(ax, width_m)
            title_text = ax.set_title("t = 0.00 s")

            def init():
                if self.is_3d:
                    line.set_data([], [])
                    line.set_3d_properties([])
                else:
                    line.set_data([], [])
                title_text.set_text("t = 0.00 s")
                return line, title_text

            def update(frame_idx):
                q = q_ts[frame_idx]
                t = ts[frame_idx]
                curve = np.array(self.compute_backbone_curve(q))

                if self.is_3d:
                    line.set_data(curve[:, 0], curve[:, 1])
                    line.set_3d_properties(curve[:, 2])
                else:
                    line.set_data(curve[:, 0], curve[:, 1])

                title_text.set_text(f"t = {t:.2f} s")
                return line, title_text

            ani = FuncAnimation(
                fig,
                update,
                frames=len(q_ts),
                init_func=init,
                blit=False,
                interval=interval,
            )

            if show:
                plt.show()

            plt.close(fig)

            # Return HTML for Jupyter
            try:
                from IPython.display import HTML

                return HTML(ani.to_jshtml())
            except ImportError:
                return None

        else:  # slider mode

            def update_plot(frame_idx):
                frame_idx = int(frame_idx)
                ax.cla()
                self._setup_axes(ax, width_m)
                ax.set_title(f"t = {ts[frame_idx]:.2f} s")

                q = q_ts[frame_idx]
                curve = np.array(self.compute_backbone_curve(q))

                if self.is_3d:
                    ax.plot(
                        curve[:, 0],
                        curve[:, 1],
                        curve[:, 2],
                        lw=self.line_width,
                        color=self.line_color,
                    )
                else:
                    ax.plot(
                        curve[:, 0], curve[:, 1], lw=self.line_width, color=self.line_color
                    )

                fig.canvas.draw_idle()

            slider = Slider(
                ax=ax_slider,
                label="Frame",
                valmin=0,
                valmax=len(ts) - 1,
                valinit=0,
                valstep=1,
            )
            slider.on_changed(update_plot)

            update_plot(0)

            if show:
                plt.show()

            plt.close(fig)
            return None

    def _setup_axes(self, ax, width_m: float) -> None:
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
            ax.set_ylim(0, width_m)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_aspect("equal")
