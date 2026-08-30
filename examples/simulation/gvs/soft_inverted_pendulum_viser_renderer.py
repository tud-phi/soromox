"""Viser motion rendering for the soft inverted-pendulum examples.

References:
    Della Santina, C. (2020). The soft inverted pendulum with affine
    curvature. 2020 59th IEEE Conference on Decision and Control (CDC),
    4135-4142. https://doi.org/10.1109/CDC42340.2020.9303976

    Ajithkumar, A. (2023). Control and stabilization of soft inverted
    pendulum on a cart. Master's thesis, University of Maryland, College Park.
    https://doi.org/10.13016/dspace/7jir-df1p
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from jax import Array

from soromox.rendering import CameraConfig, ViserRenderer
from soromox.systems import GVS, SystemState

if __package__:
    from .soft_inverted_pendulum import split_pendulum_state
else:
    from soft_inverted_pendulum import split_pendulum_state


class SoftCartPendulumViserRenderer(ViserRenderer):
    """Viser renderer adding a moving cart, wheels, and rail to GVS motion.

    References:
        Ajithkumar, A. (2023). Control and stabilization of soft inverted
        pendulum on a cart. Master's thesis, University of Maryland, College
        Park. https://doi.org/10.13016/dspace/7jir-df1p
    """

    _cart_positions: np.ndarray
    _cart_handle: Any | None
    _wheel_handles: list[Any]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the renderer with empty sequence-specific cart state."""
        super().__init__(*args, **kwargs)
        self._cart_positions = np.zeros((1,))
        self._cart_handle = None
        self._wheel_handles = []

    def render_sequence(self, ts: Array, q_ts: Array, **kwargs: Any) -> None:
        """Render the GVS motion and synchronize cart geometry with each frame.

        Args:
            ts: Trajectory time stamps with shape ``(num_frames,)``.
            q_ts: Configurations with shape ``(num_frames, 3)`` and ordering
                ``[x_cart, theta_0, theta_1]``.
            **kwargs: Additional arguments forwarded to
                :meth:`ViserRenderer.render_sequence`.

        Raises:
            ValueError: If ``q_ts`` does not contain one scalar cart trajectory.
        """
        configurations = np.asarray(q_ts)
        if configurations.ndim != 2 or configurations.shape[1] != 3:
            raise ValueError("q_ts must have shape (num_frames, 3).")
        self._cart_positions = configurations[:, 0]
        super().render_sequence(ts, q_ts, **kwargs)

    def _after_sequence_scene_built(self) -> None:
        """Add sequence-only cart geometry after the robot scene is rebuilt."""
        if self._server is None or self._scene_handles is None:
            return
        cart_min = float(np.min(self._cart_positions))
        cart_max = float(np.max(self._cart_positions))
        rail_length = max(1.2, cart_max - cart_min + 0.8)
        rail_center = 0.5 * (cart_min + cart_max)
        rail = self._server.scene.add_box(
            name="/soft_cart_pendulum/rail",
            dimensions=(0.035, rail_length, 0.025),
            position=(0.0, rail_center, -0.175),
            color=(90, 90, 90),
        )
        self._cart_handle = self._server.scene.add_box(
            name="/soft_cart_pendulum/cart",
            dimensions=(0.18, 0.30, 0.12),
            position=(0.0, float(self._cart_positions[0]), -0.06),
            color=(55, 118, 171),
        )
        self._wheel_handles = [
            self._server.scene.add_icosphere(
                name=f"/soft_cart_pendulum/wheel_{index}",
                radius=0.055,
                position=(0.0, float(self._cart_positions[0] + offset), -0.135),
                color=(35, 35, 35),
            )
            for index, offset in enumerate((-0.095, 0.095))
        ]
        self._scene_handles.custom_primitives.update(
            {
                "soft_cart_rail": rail,
                "soft_cart_body": self._cart_handle,
                **{
                    f"soft_cart_wheel_{index}": handle
                    for index, handle in enumerate(self._wheel_handles)
                },
            }
        )

    def _after_sequence_frame_updated(self, frame_idx: int) -> None:
        """Move the cart primitives to the current prismatic-joint coordinate."""
        if self._server is None or self._cart_handle is None:
            return
        cart_position = float(self._cart_positions[frame_idx])
        with self._server.atomic():
            self._cart_handle.position = (0.0, cart_position, -0.06)
            for handle, offset in zip(self._wheel_handles, (-0.095, 0.095)):
                handle.position = (0.0, cart_position + offset, -0.135)


def render_motion(
    robot: GVS,
    trajectory: SystemState,
    *,
    cart: bool,
    port: int,
    record_path: Path | None = None,
) -> None:
    """Render an interactive Viser animation and optionally record it.

    Args:
        robot: Fixed or cart soft inverted-pendulum model.
        trajectory: Simulated trajectory.
        cart: Whether to render the moving cart geometry.
        port: Local Viser server port.
        record_path: Optional MP4 output. Recording requires a connected Viser
            browser client.
    """
    q_ts, _ = split_pendulum_state(trajectory.y, cart=cart)
    renderer_type = SoftCartPendulumViserRenderer if cart else ViserRenderer
    renderer = renderer_type(
        robot,
        num_points=100,
        ground_plane_size=2.5,
        port=port,
        open_browser=True,
    )
    camera = CameraConfig(
        position=(2.2, 0.0, 0.55),
        look_at=(0.0, 0.0, 0.45),
        up=(0.0, 0.0, 1.0),
        fov=45.0,
    )
    renderer.render_sequence(
        trajectory.t,
        q_ts,
        playback_speed=1.0,
        autoplay=True,
        loop=record_path is None,
        record_path=None if record_path is None else str(record_path),
        stop_when_recording_done=record_path is not None,
        camera_config=camera,
        plot_configurations=True,
        robot_name="Soft cart pendulum" if cart else "Soft inverted pendulum",
    )


__all__ = ["SoftCartPendulumViserRenderer", "render_motion"]
