"""Render Section Vd optimized robot trajectories with Viser."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

# Geometry rendering should allocate only what it needs instead of reserving
# most accelerator memory on startup.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

CODE_DIR = Path(__file__).resolve().parent
CASE_DIR = CODE_DIR.parent
PAPER_RESULTS_DIR = CASE_DIR.parent
for path in (CODE_DIR, PAPER_RESULTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paper_style import PAPER_COLORS  # noqa: E402
from secvd_case import (  # noqa: E402
    build_sec_vd_robot,
    end_effector_pose_trajectory,
)
from secvd_results import METHODS, load_results  # noqa: E402

from soromox.rendering import BackboneColorConfig, RendererColorConfig, ViserRenderer

DEFAULT_DATA_DIR = CASE_DIR / "data"
DEFAULT_OUTPUT_DIR = CASE_DIR / "outputs"


def _hex_rgb(value: str) -> np.ndarray:
    value = value.removeprefix("#")
    return np.array([int(value[idx : idx + 2], 16) for idx in (0, 2, 4)]) / 255.0


ROBOT_COLOR = _hex_rgb(PAPER_COLORS["post_opt_1"])
CURRENT_POSITION_COLOR = _hex_rgb(PAPER_COLORS["post_opt_2"])
TARGET_COLOR = _hex_rgb(PAPER_COLORS["target"])
TARGET_TRAIL_COLOR = 0.35 * np.ones(3) + 0.65 * TARGET_COLOR
DESIRED_SHAPE_COLOR = np.array([150.0, 165.0, 181.0]) / 255.0
DESIRED_SHAPE_OPACITY = 0.45
SYNERGISTIC_ROBOT_OPACITY = 0.45
TARGET_WIREFRAME_RADIUS_SCALE = 1.03
MARKER_RADIUS = 0.004
TARGET_MARKER_RADIUS = 0.008
TRAIL_RADIUS = 0.0012


def _normalized(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return fallback if norm < 1e-9 else vector / norm


def make_tube_ring_vertices(
    curve: np.ndarray, *, radius: float, radial_segments: int = 24
) -> np.ndarray:
    """Create the material rings used by the shared desired-shape wireframe."""
    curve = np.asarray(curve, dtype=np.float64)
    if curve.ndim != 2 or curve.shape[0] < 2 or curve.shape[1] != 3:
        raise ValueError(f"Expected curve with shape (N, 3), got {curve.shape}")
    tangents = np.empty_like(curve)
    tangents[0] = curve[1] - curve[0]
    tangents[-1] = curve[-1] - curve[-2]
    if len(curve) > 2:
        tangents[1:-1] = curve[2:] - curve[:-2]
    fallback_tangent = np.array([0.0, 0.0, 1.0])
    tangents = np.asarray(
        [_normalized(tangent, fallback_tangent) for tangent in tangents]
    )
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(tangents[0], reference))) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    normal = _normalized(np.cross(tangents[0], reference), np.array([1.0, 0.0, 0.0]))
    angles = np.linspace(0.0, 2.0 * np.pi, radial_segments, endpoint=False)
    vertices = np.empty((len(curve), radial_segments, 3), dtype=np.float32)
    for point_idx, (point, tangent) in enumerate(zip(curve, tangents, strict=True)):
        if point_idx:
            normal = _normalized(normal - np.dot(normal, tangent) * tangent, normal)
        binormal = _normalized(np.cross(tangent, normal), reference)
        vertices[point_idx] = point[None, :] + radius * (
            np.cos(angles)[:, None] * normal[None, :]
            + np.sin(angles)[:, None] * binormal[None, :]
        )
    return vertices


def make_tube_wire_segments(
    curve: np.ndarray,
    *,
    radius: float,
    radial_segments: int = 24,
    ring_stride: int = 3,
    axial_count: int = 8,
) -> np.ndarray:
    """Create the desired-shape wireframe used in Section Vc and Section Vd."""
    rings = make_tube_ring_vertices(
        curve, radius=radius, radial_segments=radial_segments
    )
    ring_indices = list(range(0, len(rings), max(1, ring_stride)))
    if ring_indices[-1] != len(rings) - 1:
        ring_indices.append(len(rings) - 1)
    segments: list[np.ndarray] = []
    for point_idx in ring_indices:
        ring = rings[point_idx]
        segments.extend(
            np.stack([ring[idx], ring[(idx + 1) % radial_segments]])
            for idx in range(radial_segments)
        )
    axial_indices = np.unique(
        np.linspace(
            0, radial_segments - 1, min(axial_count, radial_segments), dtype=int
        )
    )
    for segment_idx in axial_indices:
        segments.extend(
            np.stack([rings[idx, segment_idx], rings[idx + 1, segment_idx]])
            for idx in range(len(rings) - 1)
        )
    return np.asarray(segments, dtype=np.float32)


def _desired_wire_color() -> tuple[int, int, int]:
    rgb = DESIRED_SHAPE_OPACITY * DESIRED_SHAPE_COLOR + (
        1.0 - DESIRED_SHAPE_OPACITY
    ) * np.ones(3)
    return tuple(np.rint(255.0 * rgb).astype(int))


class SecVdTrackingRenderer(ViserRenderer):
    """Viser renderer with the collocated desired-shape overlay."""

    def __init__(
        self,
        *args: Any,
        desired_q_ts: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        self._desired_q_ts = (
            None if desired_q_ts is None else np.asarray(desired_q_ts, dtype=np.float64)
        )
        self._target_wire_handle: Any | None = None
        self._target_curves: np.ndarray | None = None
        self._target_is_static = False
        super().__init__(*args, **kwargs)
        if self._desired_q_ts is not None:
            self._target_is_static = np.allclose(
                self._desired_q_ts,
                self._desired_q_ts[:1],
                rtol=1e-8,
                atol=1e-10,
            )
            if self._target_is_static:
                self._target_curves = np.asarray(
                    self.compute_backbone_curves_batched(
                        self._desired_q_ts[:1],
                        np.zeros((1, 3), dtype=np.float64),
                    )
                )
            else:
                # Avoid vectorizing many full robot geometries into one large
                # accelerator allocation. The one-configuration function is
                # compiled once and then reused for each reference frame.
                self._target_curves = np.stack(
                    [
                        np.asarray(
                            self.compute_backbone_curves_batched(
                                q_des[None, :], np.zeros((1, 3), dtype=np.float64)
                            )
                        )[0]
                        for q_des in self._desired_q_ts
                    ],
                    axis=0,
                )

    def _update_target_wireframe(self, frame_idx: int) -> None:
        if self._server is None or self._target_curves is None:
            return
        if self._target_is_static and self._target_wire_handle is not None:
            return
        idx = int(np.clip(frame_idx, 0, len(self._target_curves) - 1))
        curve = self._target_curves[idx]
        points = make_tube_wire_segments(
            curve,
            radius=(
                float(np.mean(self._backbone_marker_scales))
                * TARGET_WIREFRAME_RADIUS_SCALE
            ),
        )
        if self._target_wire_handle is None:
            self._target_wire_handle = self._server.scene.add_line_segments(
                "/tracking/desired_configuration_wireframe",
                points=points,
                colors=_desired_wire_color(),
                line_width=1.2,
            )
        else:
            self._target_wire_handle.points = points

    def _after_sequence_scene_built(self) -> None:
        if self._server is None:
            return
        self._server.scene.set_background_image(
            np.full((2, 2, 3), 255, dtype=np.uint8), format="png"
        )
        self._update_target_wireframe(0)

    def _after_sequence_frame_updated(self, frame_idx: int) -> None:
        self._update_target_wireframe(frame_idx)


def resample_trajectory(
    t: np.ndarray, q: np.ndarray, pose: np.ndarray, fps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearly resample a dense rollout onto an exact video frame grid."""
    if fps < 1:
        raise ValueError("--fps must be at least 1")
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    pose = np.asarray(pose, dtype=float)
    if t.ndim != 1 or q.shape[0] != len(t) or pose.shape != (len(t), 6):
        raise ValueError("Time, configuration, and pose trajectories are misaligned")
    frame_count = max(2, int(np.ceil((t[-1] - t[0]) * fps)) + 1)
    frame_t = np.linspace(t[0], t[-1], frame_count)
    frame_q = np.column_stack(
        [np.interp(frame_t, t, q[:, idx]) for idx in range(q.shape[1])]
    )
    frame_pose = np.column_stack(
        [np.interp(frame_t, t, pose[:, idx]) for idx in range(pose.shape[1])]
    )
    return frame_t, frame_q, frame_pose


def validate_pose_consistency(
    robot, total_length: float, data: dict[str, np.ndarray]
) -> None:
    """Ensure stored poses match FK from the authoritative configurations."""
    stored = np.asarray(data["x_ts_best"])[0]
    reconstructed = np.asarray(
        end_effector_pose_trajectory(robot, data["q_ts_best"][0], total_length)
    )
    if not np.allclose(reconstructed, stored, rtol=1e-6, atol=1e-8):
        maximum = float(np.max(np.abs(reconstructed - stored)))
        raise ValueError(
            f"Stored pose does not agree with q_ts_best FK (max error {maximum:.3e})"
        )


def _color_config(num_points: int, *, opacity: float = 1.0) -> RendererColorConfig:
    rgba = np.r_[ROBOT_COLOR, opacity]
    return RendererColorConfig(
        backbone=BackboneColorConfig(point_colors=np.tile(rgba, (num_points, 1))),
        base_plate_color=(0.2, 0.2, 0.2),
    )


def _convert_gif(mp4_path: Path, gif_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for --gif")
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-i",
            str(mp4_path),
            "-vf",
            "fps=12,scale=960:-2:flags=lanczos,split[s0][s1];"
            "[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
            str(gif_path),
        ],
        check=True,
    )


def render_method(
    *,
    name: str,
    data_dir: Path,
    output_dir: Path,
    fps: int,
    make_gif: bool,
    force: bool,
    port: int,
    open_browser: bool,
    record_client_timeout: float,
    record_frame_timeout: float,
    renderer_factory: Callable = SecVdTrackingRenderer,
) -> list[Path]:
    mp4_path = output_dir / f"{name}_tracking.mp4"
    gif_path = output_dir / f"{name}_tracking.gif"
    requested = [mp4_path] + ([gif_path] if make_gif else [])
    existing = [path for path in requested if path.exists()]
    if existing and not force:
        raise FileExistsError(f"Refusing to overwrite {existing}; pass --force")

    data = load_results(data_dir / name, expected_method=name)
    robot, _routing, total_length = build_sec_vd_robot()
    validate_pose_consistency(robot, total_length, data)
    t, q, pose = resample_trajectory(
        data["t_ts"][0], data["q_ts_best"][0], data["x_ts_best"][0], fps
    )
    dense_t = np.asarray(data["t_ts"])[0]
    dense_reference_pose = np.asarray(data["x_des_ts"])[0]
    reference_pose = np.column_stack(
        [np.interp(t, dense_t, dense_reference_pose[:, idx]) for idx in range(6)]
    )
    actual_positions = pose[:, 3:6]
    desired_positions = reference_pose[:, 3:6] if name == "synergistic" else None
    desired_q_ts = None
    if name == "collocated":
        dense_q_des = np.asarray(data["q_des_ts"])[0]
        desired_q_ts = np.column_stack(
            [np.interp(t, dense_t, dense_q_des[:, idx]) for idx in range(q.shape[1])]
        )

    static_positions = None
    static_radii = None
    static_colors = None
    dynamic_positions = None
    dynamic_radii = None
    dynamic_colors = None
    if desired_positions is not None:
        trail_stride = max(1, len(pose) // 50)
        if not np.allclose(
            desired_positions, desired_positions[:1], rtol=1e-8, atol=1e-10
        ):
            desired_trail = desired_positions[::trail_stride]
            static_positions = desired_trail
            static_radii = np.full(len(desired_trail), TRAIL_RADIUS)
            static_colors = np.tile(TARGET_TRAIL_COLOR, (len(desired_trail), 1))
        dynamic_positions = np.stack([actual_positions, desired_positions], axis=0)
        dynamic_radii = np.array([MARKER_RADIUS, TARGET_MARKER_RADIUS])
        dynamic_colors = np.stack([CURRENT_POSITION_COLOR, TARGET_COLOR], axis=0)

    renderer = renderer_factory(
        robot,
        desired_q_ts=desired_q_ts,
        width=1920,
        height=1080,
        num_points=80,
        port=port,
        open_browser=open_browser,
        color_config=_color_config(
            80,
            opacity=SYNERGISTIC_ROBOT_OPACITY if name == "synergistic" else 1.0,
        ),
        backbone_style="swept",
        cross_section_resolution=64,
        background_color=(1.0, 1.0, 1.0),
        material="standard",
        flat_shading=False,
        wireframe=False,
        cast_shadows=False,
        backbone_cast_shadow=False,
        sphere_cast_shadow=False,
    )
    renderer.render_sequence(
        ts=t,
        q_ts=q,
        record_path=str(mp4_path),
        stop_when_recording_done=True,
        record_client_timeout=record_client_timeout,
        record_frame_timeout=record_frame_timeout,
        render_actuators=False,
        static_spheres_positions=static_positions,
        static_spheres_radii=static_radii,
        static_spheres_colors=static_colors,
        dynamic_spheres_positions=dynamic_positions,
        dynamic_spheres_radii=dynamic_radii,
        dynamic_spheres_colors=dynamic_colors,
        blocking=True,
        robot_name=f"Section Vd {name}",
    )
    outputs = [mp4_path]
    if make_gif:
        _convert_gif(mp4_path, gif_path)
        outputs.append(gif_path)
    return outputs


def render_saved_results(
    *,
    data_dir: Path,
    output_dir: Path,
    method: str = "both",
    fps: int = 24,
    make_gif: bool = False,
    force: bool = False,
    port: int = 8080,
    open_browser: bool = True,
    record_client_timeout: float = 30.0,
    record_frame_timeout: float = 10.0,
    renderer_factory: Callable = SecVdTrackingRenderer,
) -> list[Path]:
    if renderer_factory is None:
        raise RuntimeError("Viser is unavailable; install the rendering dependencies")
    selected = METHODS if method == "both" else (method,)
    if any(name not in METHODS for name in selected):
        raise ValueError(f"Unknown method {method!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = [output_dir / f"{name}_tracking.mp4" for name in selected]
    if make_gif:
        requested.extend(output_dir / f"{name}_tracking.gif" for name in selected)
    existing = [path for path in requested if path.exists()]
    if existing and not force:
        raise FileExistsError(f"Refusing to overwrite {existing}; pass --force")
    outputs: list[Path] = []
    for offset, name in enumerate(selected):
        outputs.extend(
            render_method(
                name=name,
                data_dir=data_dir,
                output_dir=output_dir,
                fps=fps,
                make_gif=make_gif,
                force=force,
                port=port + offset,
                open_browser=open_browser,
                record_client_timeout=record_client_timeout,
                record_frame_timeout=record_frame_timeout,
                renderer_factory=renderer_factory,
            )
        )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--method", choices=(*METHODS, "both"), default="both")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--gif", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--open-browser", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--record-client-timeout", type=float, default=30.0)
    parser.add_argument("--record-frame-timeout", type=float, default=10.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = render_saved_results(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        method=args.method,
        fps=args.fps,
        make_gif=args.gif,
        force=args.force,
        port=args.port,
        open_browser=args.open_browser,
        record_client_timeout=args.record_client_timeout,
        record_frame_timeout=args.record_frame_timeout,
    )
    print("Generated:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
