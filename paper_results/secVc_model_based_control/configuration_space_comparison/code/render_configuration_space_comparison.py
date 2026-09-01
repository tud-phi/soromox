"""Render saved configuration-space control comparisons with Viser."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from configuration_space_comparison_simulation import (
    OUTPUTS_DIR,
    ComparisonRun,
    RenderTarget,
    create_regulation_tracking_robot,
    create_robot,
    load_comparison_npz,
)

from soromox.rendering import BackboneColorConfig, RendererColorConfig, ViserRenderer

ACTUAL_RENDER_COLORS = np.array(
    [
        [0.00, 0.45, 0.70],
        [0.84, 0.37, 0.00],
        [0.00, 0.62, 0.45],
        [0.80, 0.47, 0.65],
        [0.90, 0.62, 0.00],
        [0.34, 0.71, 0.91],
        [0.58, 0.40, 0.74],
        [0.35, 0.35, 0.35],
    ],
    dtype=np.float64,
)
DESIRED_RENDER_COLOR = np.array([150.0, 165.0, 181.0], dtype=np.float64) / 255.0
TARGET_WIREFRAME_RADIUS_SCALE = 1.03


def _normalized(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return fallback
    return vector / norm


def make_tube_ring_vertices(
    curve: np.ndarray,
    *,
    radius: float,
    radial_segments: int = 24,
) -> np.ndarray:
    """Create ring vertices around a backbone curve."""
    curve = np.asarray(curve, dtype=np.float64)
    if curve.ndim != 2 or curve.shape[0] < 2 or curve.shape[1] != 3:
        raise ValueError(f"Expected curve with shape (N, 3), got {curve.shape}.")

    tangents = np.empty_like(curve)
    tangents[0] = curve[1] - curve[0]
    tangents[-1] = curve[-1] - curve[-2]
    if curve.shape[0] > 2:
        tangents[1:-1] = curve[2:] - curve[:-2]

    fallback_tangent = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    tangents = np.array(
        [_normalized(tangent, fallback_tangent) for tangent in tangents],
        dtype=np.float64,
    )

    reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(tangents[0], reference))) > 0.9:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    normal = _normalized(
        np.cross(tangents[0], reference),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )
    angles = np.linspace(0.0, 2.0 * np.pi, radial_segments, endpoint=False)
    vertices = np.empty((curve.shape[0], radial_segments, 3), dtype=np.float32)

    for point_idx, (point, tangent) in enumerate(zip(curve, tangents)):
        if point_idx:
            projected_normal = normal - np.dot(normal, tangent) * tangent
            normal = _normalized(projected_normal, normal)
        binormal = _normalized(np.cross(tangent, normal), reference)

        ring = point[None, :] + radius * (
            np.cos(angles)[:, None] * normal[None, :]
            + np.sin(angles)[:, None] * binormal[None, :]
        )
        vertices[point_idx] = ring.astype(np.float32)

    return vertices


def make_tube_mesh(
    curve: np.ndarray,
    *,
    radius: float,
    radial_segments: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a capped tube mesh around a backbone curve."""
    ring_vertices = make_tube_ring_vertices(
        curve,
        radius=radius,
        radial_segments=radial_segments,
    )
    curve = np.asarray(curve, dtype=np.float64)
    vertices = np.concatenate(
        [
            ring_vertices.reshape(-1, 3),
            curve[[0, -1]].astype(np.float32),
        ],
        axis=0,
    )

    start_center_idx = curve.shape[0] * radial_segments
    end_center_idx = start_center_idx + 1

    faces: list[tuple[int, int, int]] = []
    for point_idx in range(curve.shape[0] - 1):
        ring_start = point_idx * radial_segments
        next_ring_start = (point_idx + 1) * radial_segments
        for segment_idx in range(radial_segments):
            next_segment_idx = (segment_idx + 1) % radial_segments
            a = ring_start + segment_idx
            b = ring_start + next_segment_idx
            c = next_ring_start + segment_idx
            d = next_ring_start + next_segment_idx
            faces.append((a, c, b))
            faces.append((b, c, d))

    last_ring_start = (curve.shape[0] - 1) * radial_segments
    for segment_idx in range(radial_segments):
        next_segment_idx = (segment_idx + 1) % radial_segments
        faces.append((start_center_idx, segment_idx, next_segment_idx))
        faces.append(
            (
                end_center_idx,
                last_ring_start + next_segment_idx,
                last_ring_start + segment_idx,
            )
        )

    return vertices, np.asarray(faces, dtype=np.uint32)


def make_tube_wire_segments(
    curve: np.ndarray,
    *,
    radius: float,
    radial_segments: int = 24,
    ring_stride: int = 3,
    axial_count: int = 8,
) -> np.ndarray:
    """Create line segments that draw a stable wireframe tube."""
    ring_vertices = make_tube_ring_vertices(
        curve,
        radius=radius,
        radial_segments=radial_segments,
    )
    num_points = ring_vertices.shape[0]
    ring_stride = max(1, int(ring_stride))
    axial_count = max(1, min(int(axial_count), radial_segments))
    axial_indices = np.unique(
        np.linspace(0, radial_segments - 1, axial_count, dtype=int)
    )

    segments: list[np.ndarray] = []
    ring_indices = list(range(0, num_points, ring_stride))
    if ring_indices[-1] != num_points - 1:
        ring_indices.append(num_points - 1)

    for point_idx in ring_indices:
        ring = ring_vertices[point_idx]
        segments.extend(
            np.stack([ring[segment_idx], ring[(segment_idx + 1) % radial_segments]])
            for segment_idx in range(radial_segments)
        )

    for segment_idx in axial_indices:
        segments.extend(
            np.stack(
                [
                    ring_vertices[point_idx, segment_idx],
                    ring_vertices[point_idx + 1, segment_idx],
                ]
            )
            for point_idx in range(num_points - 1)
        )

    return np.asarray(segments, dtype=np.float32)


def target_wire_color(opacity: float) -> tuple[int, int, int]:
    """Blend the target wire color toward white to mimic low opacity."""
    opacity = float(np.clip(opacity, 0.0, 1.0))
    rgb = opacity * DESIRED_RENDER_COLOR + (1.0 - opacity) * np.ones(3)
    return tuple(int(channel * 255) for channel in rgb)


def select_render_target(run: ComparisonRun, key: str | None) -> RenderTarget:
    """Select a render target from saved metadata."""
    if key is None:
        return run.render_targets[0]
    for target in run.render_targets:
        if target.key == key or target.scenario_key == key:
            return target
    available = ", ".join(target.key for target in run.render_targets)
    raise ValueError(f"Unknown render target {key!r}. Available: {available}")


def controller_names_for_target(
    run: ComparisonRun,
    target: RenderTarget,
) -> tuple[str, ...]:
    """Return all controllers available for a target."""
    return tuple(run.results[target.scenario_key])


def select_initial_controller(
    run: ComparisonRun,
    target: RenderTarget,
    controller_name: str | None,
) -> str:
    """Select the initially visible controller for a target."""
    scenario_results = run.results[target.scenario_key]
    selected = controller_name or target.default_controller
    if selected in scenario_results:
        return selected

    available = ", ".join(scenario_results)
    raise ValueError(f"Unknown controller {selected!r}. Available: {available}")


def add_controller_visibility_selector(
    renderer: ViserRenderer,
    controller_names: tuple[str, ...],
    *,
    initial_controller: str,
    target_curves: np.ndarray,
    target_opacity: float,
    target_radial_segments: int,
    target_line_width: float,
) -> None:
    """Add a Viser dropdown that shows one controller and the desired overlay."""
    selected_idx = {"value": controller_names.index(initial_controller)}
    target_wire_handle: dict[str, Any] = {"handle": None}

    def apply_visibility() -> None:
        scene_handles = renderer._scene_handles
        if scene_handles is None:
            return

        for robot_idx, point_handles in enumerate(scene_handles.backbone_points):
            visible = robot_idx == selected_idx["value"]
            for handle in point_handles:
                if hasattr(handle, "visible"):
                    handle.visible = visible

        for robot_idx, handle in enumerate(scene_handles.base_plates):
            handle.visible = robot_idx == selected_idx["value"]

    def update_target_wireframe(frame_idx: int) -> None:
        if renderer._server is None:
            return
        points = make_tube_wire_segments(
            target_curves[frame_idx],
            radius=(
                float(np.mean(renderer._backbone_marker_scales))
                * TARGET_WIREFRAME_RADIUS_SCALE
            ),
            radial_segments=target_radial_segments,
        )
        color = target_wire_color(target_opacity)
        if target_wire_handle["handle"] is None:
            target_wire_handle["handle"] = renderer.server.scene.add_line_segments(
                "/desired_shape_wireframe",
                points=points,
                colors=color,
                line_width=target_line_width,
            )
        else:
            target_wire_handle["handle"].points = points

    original_setup_playback_gui = renderer._setup_playback_gui
    original_update_frame = renderer._update_frame

    def setup_playback_gui_with_controller_selector(
        on_frame_seek: Callable[[int], None] | None = None,
    ) -> None:
        original_setup_playback_gui(on_frame_seek=on_frame_seek)
        with renderer.server.gui.add_folder("Controller"):
            dropdown = renderer.server.gui.add_dropdown(
                "Visible controller",
                options=controller_names,
                initial_value=initial_controller,
            )
            renderer._gui_handles["controller_dropdown"] = dropdown

            @dropdown.on_update
            def _(event: Any) -> None:
                selected_idx["value"] = controller_names.index(event.target.value)
                apply_visibility()

        apply_visibility()
        update_target_wireframe(0)

    def update_frame_with_controller_visibility(*args: Any, **kwargs: Any) -> None:
        original_update_frame(*args, **kwargs)
        if args:
            update_target_wireframe(int(args[0]))
        apply_visibility()

    renderer._setup_playback_gui = setup_playback_gui_with_controller_selector
    renderer._update_frame = update_frame_with_controller_visibility


def render_run(
    run: ComparisonRun,
    *,
    target_key: str | None = None,
    initial_controller_name: str | None = None,
    viser_port: int = 8080,
    open_browser: bool = True,
    playback_speed: float = 1.0,
    autoplay: bool = True,
    loop: bool = False,
    num_points: int = 80,
    actual_opacity: float = 1.0,
    desired_opacity: float = 0.25,
    target_radial_segments: int = 24,
    target_line_width: float = 1.0,
    video_output: str | None = None,
    record_every_n: int = 1,
    record_client_timeout: float = 10.0,
    record_frame_timeout: float = 10.0,
    blocking: bool = True,
) -> None:
    """Render actual and desired robot shapes overlaid in Viser."""
    if ViserRenderer is None:
        raise ImportError(
            "Viser is required for rendering. Install the optional Viser extras "
            "or use the plotting script instead."
        )

    target = select_render_target(run, target_key)
    controllers = controller_names_for_target(run, target)
    initial_controller = select_initial_controller(run, target, initial_controller_name)
    scenario_results = run.results[target.scenario_key]
    first_result = scenario_results[initial_controller]

    ts = np.asarray(first_result["t"])
    q_actuals = [np.asarray(scenario_results[name]["q"]) for name in controllers]
    q_desired = np.asarray(first_result["metrics"]["q_des"])
    q_overlay = np.stack(q_actuals, axis=0)

    actual_colors = np.vstack(
        [
            np.append(
                ACTUAL_RENDER_COLORS[idx % len(ACTUAL_RENDER_COLORS)],
                np.clip(actual_opacity, 0.0, 1.0),
            )
            for idx in range(len(controllers))
        ]
    )
    color_config = RendererColorConfig(
        backbone=BackboneColorConfig(
            robot_colors=actual_colors,
        ),
        base_plate_color=(0.15, 0.15, 0.15),
    )

    if run.example_key == "regulation_tracking":
        robot, _, _ = create_regulation_tracking_robot()
    else:
        robot, _, _ = create_robot()
    renderer = ViserRenderer(
        robot,
        num_points=num_points,
        port=viser_port,
        backbone_style="discrete",
        open_browser=open_browser,
        color_config=color_config,
    )
    target_curves = np.asarray(
        renderer.compute_backbone_curves_batched(
            q_desired,
            np.zeros((len(q_desired), 3), dtype=np.float64),
        )
    )
    add_controller_visibility_selector(
        renderer,
        controllers,
        initial_controller=initial_controller,
        target_curves=target_curves,
        target_opacity=desired_opacity,
        target_radial_segments=target_radial_segments,
        target_line_width=target_line_width,
    )
    renderer.render_sequence(
        ts=ts,
        q_ts=q_overlay,
        playback_speed=playback_speed,
        autoplay=autoplay,
        loop=loop,
        record_path=video_output,
        record_every_n=record_every_n,
        stop_when_recording_done=video_output is not None,
        record_client_timeout=record_client_timeout,
        record_frame_timeout=record_frame_timeout,
        multi_robot_layout="overlay",
        color_config=color_config,
        render_actuators=False,
        blocking=blocking,
        robot_name=f"{target.title} ({len(controllers)} controller(s))",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", help="Saved comparison .npz file.")
    parser.add_argument(
        "--target",
        default=None,
        help="Render target key. For trajectory tracking, use 'slow' or 'fast'.",
    )
    parser.add_argument(
        "--controller",
        default=None,
        help=(
            "Initial controller selected in the Viser GUI. Defaults to the target's "
            "recommended controller."
        ),
    )
    parser.add_argument("--viser-port", type=int, default=8080, help="Viser port.")
    parser.add_argument(
        "--no-open-browser",
        action="store_true",
        help="Start Viser without opening a browser tab.",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier.",
    )
    parser.add_argument("--no-autoplay", action="store_true", help="Start paused.")
    parser.add_argument("--loop", action="store_true", help="Loop playback.")
    parser.add_argument(
        "--num-points",
        type=int,
        default=80,
        help="Number of backbone points used for rendering.",
    )
    parser.add_argument(
        "--actual-opacity",
        type=float,
        default=1.0,
        help="Opacity of the simulated robot shape.",
    )
    parser.add_argument(
        "--desired-opacity",
        type=float,
        default=0.25,
        help="Opacity of the desired robot shape.",
    )
    parser.add_argument(
        "--target-radial-segments",
        type=int,
        default=24,
        help="Number of radial segments in the desired-shape wireframe tube.",
    )
    parser.add_argument(
        "--target-line-width",
        type=float,
        default=1.0,
        help="Line width for the desired-shape wireframe.",
    )
    parser.add_argument(
        "--video-output",
        nargs="?",
        const="",
        default=None,
        help=(
            "Save one Viser rendering pass as an MP4. If no path is provided, "
            "defaults to '<trajectory-stem>-<target>-render.mp4'."
        ),
    )
    parser.add_argument(
        "--record-every-n",
        type=int,
        default=1,
        help="Record every Nth frame when --video-output is used.",
    )
    parser.add_argument(
        "--record-client-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for a Viser browser client when recording video.",
    )
    parser.add_argument(
        "--record-frame-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for each browser-rendered video frame.",
    )
    parser.add_argument(
        "--non-blocking",
        action="store_true",
        help="Return after initializing one Viser update loop iteration.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace an existing video file."
    )
    args = parser.parse_args()
    if args.video_output is not None and (args.no_autoplay or args.loop):
        parser.error("--video-output cannot be combined with --no-autoplay or --loop.")
    return args


def main() -> None:
    args = parse_args()
    run = load_comparison_npz(args.trajectory)
    target = select_render_target(run, args.target)

    video_output = args.video_output
    if video_output is not None:
        if video_output == "":
            stem = Path(args.trajectory).stem
            video_output = OUTPUTS_DIR / f"{stem}-{target.key}-render.mp4"
        video_path = Path(video_output)
        if video_path.exists() and not args.force:
            raise FileExistsError(
                f"Output already exists: {video_path}. Pass --force to replace it."
            )
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_output = str(video_path)
        print(f"Video will be saved to: {video_output}")

    controllers = controller_names_for_target(run, target)
    initial_controller = select_initial_controller(run, target, args.controller)
    print(
        f"Rendering {target.title}. Use the Viser Controller dropdown to choose "
        "the visible controller."
    )
    print(f"Initial controller: {initial_controller}")
    for idx, controller in enumerate(controllers):
        rgb = ACTUAL_RENDER_COLORS[idx % len(ACTUAL_RENDER_COLORS)]
        rgb_255 = tuple(int(channel * 255) for channel in rgb)
        print(f"  - {controller}: rgb{rgb_255}")
    target_style = "stable wireframe tube"
    print(f"  - Desired: {target_style}")
    render_run(
        run,
        target_key=target.key,
        initial_controller_name=initial_controller,
        viser_port=args.viser_port,
        open_browser=not args.no_open_browser,
        playback_speed=args.playback_speed,
        autoplay=not args.no_autoplay,
        loop=args.loop,
        num_points=args.num_points,
        actual_opacity=args.actual_opacity,
        desired_opacity=args.desired_opacity,
        target_radial_segments=args.target_radial_segments,
        target_line_width=args.target_line_width,
        video_output=video_output,
        record_every_n=args.record_every_n,
        record_client_timeout=args.record_client_timeout,
        record_frame_timeout=args.record_frame_timeout,
        blocking=not args.non_blocking,
    )


if __name__ == "__main__":
    main()
