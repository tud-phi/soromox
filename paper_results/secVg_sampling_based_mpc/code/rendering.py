"""Render a saved run_experiment.py run with Newton's ViewerRTX path tracer,
producing an MP4 (and optionally a GIF).

Reads the saved X_final trajectory directly -- no re-simulation. Writing q/qd
and the object pose into a state, then calling SolverGVS.sync_kinematics(), is
pure forward kinematics, so it needs only the saved trajectory itself.

Needs a real GPU with RTX path-tracing support; will not run on CPU alone.

Usage:
    python rendering.py --run-path ../data/square_icem_cylinder.npz --object-shape cylinder \
        --output ../outputs/square_icem_cylinder.mp4 --gif
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import newton
import newton.viewer
import numpy as np
import warp as wp

import run_experiment
from four_tendon_soft_arm import FourTendonSoftArm
from newton_cosserat.gvs import gvs_state_views

CODE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = CODE_DIR.parent / "outputs"
PAPER_RESULTS_DIR = CODE_DIR.parent.parent
if str(PAPER_RESULTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_RESULTS_DIR))

from paper_style import PAPER_COLORS  # noqa: E402


def paper_rgb(key: str, stop: float = 1.0) -> tuple[float, float, float]:
    """Return one PAPER_COLORS entry as a linear 0-1 RGB triple.

    PAPER_COLORS is sRGB; ViewerRTX's diffuseColor is linear, so unconverted values render washed out.

    Args:
        key: Key into paper_style.PAPER_COLORS.
        stop: Position on the white-to-colour ramp (blended in sRGB); 1.0 is the full colour.

    Returns:
        Linear-space red, green and blue in [0, 1].
    """
    value = PAPER_COLORS[key].removeprefix("#")
    channels = (int(value[idx : idx + 2], 16) / 255.0 for idx in (0, 2, 4))
    blended = (1.0 + stop * (c - 1.0) for c in channels)
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in blended)


DEVICE = "cuda"
ROD_COLOR = paper_rgb("post_opt_1", stop=0.875)
OBJECT_COLOR = paper_rgb("obstacle")
TABLE_COLOR = (0.34, 0.36, 0.40)
GOAL_COLOR = paper_rgb("target")
ROD_COLLISION_BODIES = 150  # render-only: many short capsules for a smooth rod (the experiment uses (1, 3, 12, 12))

CAMERA_EYE = (0.55, -0.45, 0.55)
CAMERA_TARGET = (0.04, 0.0, 0.15)

GOAL_TOLERANCE_OPACITY = 0.5
GOAL_MARKER_Z_OFFSET = 5.0e-4
GOAL_TOLERANCE_Z_OFFSET = 2.0e-4
GOAL_LINE_Z_OFFSET = 9.0e-4
GOAL_LINE_WIDTH = 0.0015

FRANKA_Q = np.array([0.0, -0.4, 0.0, -2.0, 0.0, 2.4, np.pi / 4.0])
FRANKA_IK_ITERATIONS = 64
FRANKA_END_EFFECTOR_LINK = "fr3_link8"
FRANKA_BASE_XY = (0.45, 0.45)
FRANKA_FLANGE_ALIGNMENT_ANGLE = -0.5 * np.pi

STUDIO_COLOR = (0.18, 0.175, 0.165)
STUDIO_AMBIENT = 60.0
STUDIO_KEY = 1000.0
STUDIO_FILL = 160.0
STUDIO_RIM = 160.0
STUDIO_KEY_ANGLE = 28.0
STUDIO_BACKDROP_START = 80.0
STUDIO_BACKDROP_RADIUS = (8.0, 8.0)
STUDIO_BACKDROP_EMISSION = 0.22
STUDIO_ROD_ROUGHNESS = 0.72
STUDIO_FLOOR_HEIGHT = -0.05


def _find_label(labels: list[str], suffix: str) -> int:
    """Return the index of the single label ending in `suffix`.

    Args:
        labels: Builder label list to search.
        suffix: Label suffix to match.

    Returns:
        Index of the matching label.
    """
    matches = [index for index, label in enumerate(labels) if label.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one label ending in {suffix!r}, found {len(matches)}")
    return matches[0]


def add_franka(builder: newton.ModelBuilder) -> tuple[int, int, int]:
    """Add a fixed-base Franka FR3 to `builder` for display only.

    The arm is left unowned by every solver entry, so it never takes part in
    the contact solve; `FrankaDisplay` supplies its body transforms.

    Args:
        builder: Newton builder receiving the arm.

    Returns:
        First body index, one-past-last body index, and end-effector body index.
    """
    body_start = builder.body_count
    builder.add_urdf(
        newton.utils.download_asset("franka_emika_panda") / "urdf/fr3.urdf",
        floating=False,
        parent_body=-1,
        enable_self_collisions=False,
        hide_visuals=False,
        parse_visuals_as_colliders=False,
        force_show_colliders=False,
    )
    end_effector = _find_label(builder.body_label, FRANKA_END_EFFECTOR_LINK)
    return body_start, builder.body_count, end_effector


class FrankaDisplay:
    """Pose a decorative Franka so its end effector tracks the rod's base.

    IK and FK run on a standalone arm model so eval_fk never overwrites the push scene's body poses.
    """

    def __init__(self, base_xy: tuple[float, float], device: str):
        """Build the standalone arm, stand it on the table, and set up IK.

        Args:
            base_xy: World (x, y) of the arm's base; it is yawed to face the
                world origin.
            device: Warp device used for the standalone model.
        """
        import newton.ik as ik

        builder = newton.ModelBuilder()
        self.body_start, self.body_end, self.end_effector = add_franka(builder)
        self.model = builder.finalize(device=device)
        self.state = self.model.state()
        self.joint_q = self.model.joint_q.reshape((1, self.model.joint_coord_count))
        self.joint_seed = wp.array(FRANKA_Q[None, :], dtype=wp.float32, device=device)
        self.joint_q.assign(self.joint_seed)

        yaw = float(np.arctan2(-base_xy[1], -base_xy[0]))
        self.placement = wp.transform(
            wp.vec3(base_xy[0], base_xy[1], 0.0),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw),
        )
        self.inverse_placement = wp.transform_inverse(self.placement)

        # Placeholder targets; body_poses() overwrites them every frame.
        local_target = self.inverse_placement
        self.position_objective = ik.IKObjectivePosition(
            link_index=self.end_effector,
            link_offset=wp.vec3(0.0),
            target_positions=wp.array([wp.vec3(*local_target.p)], dtype=wp.vec3, device=device),
            weight=2.0,
        )
        self.rotation_objective = ik.IKObjectiveRotation(
            link_index=self.end_effector,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.vec4(*local_target.q)], dtype=wp.vec4, device=device),
            weight=0.2,
        )
        self.solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[self.position_objective, self.rotation_objective],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    @staticmethod
    def _target_transform(base_pose: np.ndarray) -> wp.transform:
        """Convert one GVS base-pose row into an end-effector target transform.

        The rod's base frame is the flange frame turned by FRANKA_FLANGE_ALIGNMENT_ANGLE about y;
        this undoes that turn.

        Args:
            base_pose: Base pose, shape (7,) as (qx, qy, qz, qw, x, y, z).

        Returns:
            World transform the end effector should reach.
        """
        alignment = wp.transform(
            wp.vec3(0.0),
            wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), FRANKA_FLANGE_ALIGNMENT_ANGLE),
        )
        return wp.transform_multiply(
            wp.transform(wp.vec3(*base_pose[4:7]), wp.quat(*base_pose[:4])),
            wp.transform_inverse(alignment),
        )

    def body_poses(self, base_pose: np.ndarray) -> np.ndarray:
        """Solve one frame's joint angles and return the arm's body transforms.

        Args:
            base_pose: Rod base pose for this frame, shape (7,) as
                (qx, qy, qz, qw, x, y, z).

        Returns:
            World body transforms, shape (arm_bodies, 7) as
            (x, y, z, qx, qy, qz, qw).
        """
        local_target = wp.transform_multiply(self.inverse_placement, self._target_transform(base_pose))
        self.position_objective.set_target_position(0, wp.vec3(*local_target.p))
        self.rotation_objective.set_target_rotation(0, wp.vec4(*local_target.q))
        self.solver.step(self.joint_seed, self.joint_q, iterations=FRANKA_IK_ITERATIONS)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        body_q = self.state.body_q.numpy()
        poses = np.empty((self.body_end - self.body_start, 7), dtype=np.float32)
        for row, body in enumerate(range(self.body_start, self.body_end)):
            placed = wp.transform_multiply(
                self.placement, wp.transform(wp.vec3(*body_q[body][:3]), wp.quat(*body_q[body][3:7]))
            )
            poses[row, :3] = np.asarray(placed.p)
            poses[row, 3:7] = np.asarray(placed.q)
        return poses


def look_at(viewer, eye, target) -> None:
    """Aim a Z-up viewer camera at a world-space point."""
    eye, target = np.asarray(eye, dtype=float), np.asarray(target, dtype=float)
    direction = target - eye
    direction /= np.linalg.norm(direction)
    viewer.set_camera(
        wp.vec3(*eye),
        pitch=float(np.rad2deg(np.arcsin(direction[2]))),
        yaw=float(np.rad2deg(np.arctan2(direction[1], direction[0]))),
    )


class ExampleViewerRTX(newton.viewer.ViewerRTX):
    """RTX viewer with a neutral studio backdrop and lights, and opt-in shadow-link exclusion."""

    shadow_excluded_mesh_names: tuple[str, ...] = ()

    def __init__(self, *args, samples_per_pixel: int, **kwargs):
        """Configure the path-tracer sample budget.

        Args:
            samples_per_pixel: Path-tracer sample count per frame.
        """
        if samples_per_pixel < 1:
            raise ValueError("samples_per_pixel must be positive")
        self.samples_per_pixel = samples_per_pixel
        super().__init__(*args, environment="none", async_rendering=False, **kwargs)
        self.show_ui = False
        self.show_joints = False
        self.show_contacts = False
        self.show_com = False
        self.show_inertia_boxes = False
        self.show_collision = False

    def _add_camera_lights_and_render_product(self) -> None:
        """Configure the path-tracer render product, then add the studio."""
        from pxr import Sdf

        super()._add_camera_lights_and_render_product()
        product = self.stage.GetPrimAtPath(self._render_product_path)
        product.GetAttribute("omni:rtx:rendermode").Set("PathTracing")
        product.GetAttribute("omni:rtx:quality").Set(1)
        product.GetAttribute("omni:rtx:pt:samplesPerPixel").Set(self.samples_per_pixel)
        product.CreateAttribute("omni:rtx:pt:samplesPerIteration", Sdf.ValueTypeNames.Int).Set(
            min(32, self.samples_per_pixel)
        )
        product.GetAttribute("omni:rtx:pt:denoising:enabled").Set(self.samples_per_pixel > 1)
        product.GetAttribute("omni:rtx:dlss:frameGeneration").Set(False)
        self._add_studio()

    def _render_and_display(self) -> None:
        """Render until the path tracer converges to samples_per_pixel; save_screenshot() reads the result."""
        if self._rtx is None or self._should_close:
            return
        for attempt in range(self.samples_per_pixel + 2):
            self._render_products = self._rtx.step(
                render_products={self._render_product_path},
                delta_time=1.0 / self.fps if attempt == 0 else 0.0,
            )
            frames = [frame for product in self._render_products.values() for frame in product.frames]
            if frames and all(frame.converged for frame in frames):
                break
        else:
            raise RuntimeError(
                f"RTX did not converge to the requested sample count: {[frame.progression for frame in frames]}"
            )

    def _capture_screenshot_pixels(self) -> np.ndarray:
        """Read LdrColor from the last render; ovrtx may prefix the render-var key with its path."""
        from ovrtx import Device

        for product in self._render_products.values():
            for frame in product.frames:
                for key, render_var in frame.render_vars.items():
                    if key == "LdrColor" or key.endswith("/LdrColor"):
                        with render_var.map(device=Device.CPU) as mapping:
                            return np.array(np.from_dlpack(mapping), copy=True)
        raise RuntimeError("save_screenshot() could not find the LdrColor render output")

    def _apply_ground_material(self) -> None:
        """Skip Newton's ground material; _add_studio binds the backdrop material to the ground."""

    def _add_studio(self) -> None:
        """Author the studio backdrop mesh and key/fill/rim/ambient lights, excluding shadow_excluded_mesh_names from their shadows."""
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

        material = UsdShade.Material.Define(self.stage, "/root/Studio/Matte")
        shader = UsdShade.Shader.Define(self.stage, "/root/Studio/Matte/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*STUDIO_COLOR))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
        shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(STUDIO_BACKDROP_EMISSION))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        bend = STUDIO_BACKDROP_START
        radius_y, radius_z = STUDIO_BACKDROP_RADIUS
        span = bend + 2.0 * radius_y
        profile = [(-span, 0.0), (bend, 0.0)]
        normals = [(0.0, 0.0, 1.0)] * 2
        for angle in np.linspace(0.0, np.pi / 2, 33)[1:]:
            profile.append((bend + radius_y * np.sin(angle), radius_z * (1.0 - np.cos(angle))))
            normal = np.asarray((0.0, -radius_z * np.sin(angle), radius_y * np.cos(angle)))
            normals.append(tuple(normal / np.linalg.norm(normal)))
        profile.append((bend + radius_y, radius_z + 1.0))
        normals.append((0.0, -1.0, 0.0))
        vertices = [(float(x), float(y), float(STUDIO_FLOOR_HEIGHT + z)) for y, z in profile for x in (-span, span)]
        mesh = UsdGeom.Mesh.Define(self.stage, "/root/Studio/Backdrop")
        mesh.CreatePointsAttr(vertices)
        mesh.CreateFaceVertexCountsAttr([4] * (len(profile) - 1))
        mesh.CreateFaceVertexIndicesAttr(
            [index for row in range(len(profile) - 1) for index in (2 * row, 2 * row + 1, 2 * row + 3, 2 * row + 2)]
        )
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateNormalsAttr([Gf.Vec3f(*normal) for normal in normals for _ in range(2)])
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.CreateDoubleSidedAttr(True)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        for name, ground in self._meshes.items():
            if "plane" in name.lower():
                UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(material)

        rod_material = UsdShade.Material.Define(self.stage, "/root/Studio/Rod")
        rod_surface = UsdShade.Shader.Define(self.stage, "/root/Studio/Rod/Surface")
        rod_surface.CreateIdAttr("UsdPreviewSurface")
        color = UsdShade.Shader.Define(self.stage, "/root/Studio/Rod/Color")
        color.CreateIdAttr("UsdPrimvarReader_float3")
        color.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
        color.CreateOutput("result", Sdf.ValueTypeNames.Float3)
        rod_surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            color.ConnectableAPI(), "result"
        )
        rod_surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(STUDIO_ROD_ROUGHNESS)
        rod_surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        rod_material.CreateSurfaceOutput().ConnectToSource(rod_surface.ConnectableAPI(), "surface")
        for name, prototype in self._meshes.items():
            if "capsule" in name.lower():
                UsdShade.MaterialBindingAPI.Apply(prototype.GetPrim()).Bind(rod_material)

        dome = UsdLux.DomeLight.Define(self.stage, "/root/Studio/Ambient")
        dome.CreateColorAttr(Gf.Vec3f(1.0))
        dome.CreateIntensityAttr(STUDIO_AMBIENT)
        key = UsdLux.DistantLight.Define(self.stage, "/root/Studio/Key")
        key.CreateColorAttr(Gf.Vec3f(1.0))
        key.CreateIntensityAttr(STUDIO_KEY)
        key.CreateAngleAttr(STUDIO_KEY_ANGLE)
        UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(29.0, 14.0, 0.0))
        for name, position, intensity in (
            ("Fill", (0.0, -0.8, 0.8), STUDIO_FILL),
            ("Rim", (0.0, 0.4, 0.75), STUDIO_RIM),
        ):
            light = UsdLux.SphereLight.Define(self.stage, f"/root/Studio/{name}")
            light.CreateColorAttr(Gf.Vec3f(1.0))
            light.CreateIntensityAttr(intensity)
            light.CreateRadiusAttr(0.3)
            UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*position))

        mesh_paths = [
            self._mesh_prim_paths[self._qualify(name)]
            for name in self.shadow_excluded_mesh_names
            if self._qualify(name) in self._mesh_prim_paths
        ]
        for light_path in ("/root/Studio/Ambient", "/root/Studio/Key", "/root/Studio/Fill", "/root/Studio/Rim"):
            excludes = UsdLux.LightAPI(self.stage.GetPrimAtPath(light_path)).GetShadowLinkCollectionAPI().CreateExcludesRel()
            for mesh_path in mesh_paths:
                excludes.AddTarget(Sdf.Path(mesh_path))


def single_surface_opacity(desired: float) -> float:
    """Return the opacity to request so one flat surface renders at `desired`.

    ViewerRTX lowers authored opacity assuming a ray crosses _PREVIEW_SURFACE_OPACITY_LAYERS surfaces;
    a flat patch is crossed once, so this inverts that mapping.

    Args:
        desired: Intended display opacity of a single surface in [0, 1].

    Returns:
        Opacity to pass to log_mesh, below the threshold that reads as opaque.
    """
    layers = ExampleViewerRTX._PREVIEW_SURFACE_OPACITY_LAYERS
    return float(min(1.0 - (1.0 - desired) ** layers, 0.998))


class RTXFrameWriter:
    """Render video frames to PNG files with a fixed initial camera."""

    def __init__(self, model: newton.Model, output_dir: Path, *, width: int, height: int, fps: int, samples_per_pixel: int):
        """Create the frames directory and an RTX viewer for `model`.

        Args:
            model: Finalized Newton model to render.
            output_dir: Directory whose `frames/` subdirectory receives the PNGs.
            width: Frame width in pixels.
            height: Frame height in pixels.
            fps: Playback frame rate.
            samples_per_pixel: Path-tracer sample count per frame.
        """
        self.frames_dir = output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.frame_index = 0
        self.viewer = ExampleViewerRTX(
            width=width, height=height, headless=True, fps=fps, samples_per_pixel=samples_per_pixel,
        )
        self.viewer.set_model(model)
        look_at(self.viewer, CAMERA_EYE, CAMERA_TARGET)

    def write(self, time_value: float, state: newton.State, extra_log_fn) -> None:
        """Render one frame to a numbered PNG in `frames_dir`.

        Args:
            time_value: Simulation time of this frame.
            state: Newton state to render.
            extra_log_fn: Callback run between begin_frame() and
                end_frame(); animated-overlay log_lines() calls must run here,
                since begin_frame() clears any line batch queued earlier.
        """
        self.viewer.begin_frame(time_value)
        self.viewer.log_state(state)
        extra_log_fn()
        self.viewer.end_frame()
        self.viewer.save_screenshot(str(self.frames_dir / f"frame_{self.frame_index:04d}.png"))
        self.frame_index += 1

    def close(self) -> None:
        """Close the underlying viewer."""
        self.viewer.close()


def find_ffmpeg() -> str:
    """Return the ffmpeg executable path; the PNG frames are kept if it is missing."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode the video; the PNG sequence was retained")
    return ffmpeg


def encode_mp4(frames_dir: Path, output_path: Path, fps: int) -> None:
    """Encode the frame PNGs into an MP4 with ffmpeg.

    Args:
        frames_dir: Directory of `frame_%04d.png` files.
        output_path: MP4 path to write.
        fps: Playback frame rate.
    """
    subprocess.run(
        [find_ffmpeg(), "-y", "-framerate", str(fps), "-i", str(frames_dir / "frame_%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(output_path)],
        check=True,
    )


def encode_gif(frames_dir: Path, output_path: Path, fps: int) -> None:
    """Encode the frame PNGs into a palette-quantized GIF with ffmpeg.

    Args:
        frames_dir: Directory of `frame_%04d.png` files.
        output_path: GIF path to write.
        fps: Playback frame rate.
    """
    palette = frames_dir / "palette.png"
    vf = f"fps={fps},scale=640:-1:flags=lanczos"
    subprocess.run(
        [find_ffmpeg(), "-y", "-framerate", str(fps), "-i", str(frames_dir / "frame_%04d.png"),
         "-vf", vf + ",palettegen", str(palette)],
        check=True,
    )
    subprocess.run(
        [find_ffmpeg(), "-y", "-framerate", str(fps), "-i", str(frames_dir / "frame_%04d.png"), "-i", str(palette),
         "-lavfi", f"{vf} [x]; [x][1:v] paletteuse", str(output_path)],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed renderer configuration.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-path", type=Path, required=True, help="Saved .npz with an X_final trajectory.")
    parser.add_argument("--control-dt", type=float, default=0.05, help="Must match the original run's --control-dt.")
    parser.add_argument("--frames-per-step", type=int, default=10, help="Interpolated render frames per control step.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--samples-per-pixel", type=int, default=8)
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "push_rtx.mp4")
    parser.add_argument("--gif", action="store_true", help="Also encode a GIF alongside the MP4.")
    parser.add_argument(
        "--franka-base", type=float, nargs=2, metavar=("X", "Y"), default=FRANKA_BASE_XY,
        help="World (x, y) the decorative arm stands at; it is yawed to face the world origin.",
    )
    parser.add_argument(
        "--object-shape", choices=("box", "cylinder"), default="box",
        help="Must match the original run's --object-shape.",
    )
    return parser.parse_args()


def _slerp_quat(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical interpolation between two (x, y, z, w) quaternions."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        return q / np.linalg.norm(q)
    theta_0 = np.arccos(dot)
    theta = theta_0 * alpha
    s0 = np.cos(theta) - dot * np.sin(theta) / np.sin(theta_0)
    s1 = np.sin(theta) / np.sin(theta_0)
    return s0 * q0 + s1 * q1


def ordered_unique_goals(goal_xy: np.ndarray) -> list[tuple[float, float]]:
    """Distinct consecutive goal positions from the saved per-step goal_xy log, in visit order."""
    corners = [tuple(goal_xy[0])]
    for xy in goal_xy[1:]:
        if not np.allclose(xy, corners[-1]):
            corners.append(tuple(xy))
    return corners


def _flat_square_mesh(center_x: float, center_y: float, half_width: float, z: float):
    """Flat square mesh centered at (center_x, center_y, z).

    Args:
        center_x: Center world x.
        center_y: Center world y.
        half_width: Half the square's side length.
        z: World z of the flat mesh.

    Returns:
        (points, indices, normals) for a single quad.
    """
    points = np.array(
        [
            [center_x - half_width, center_y - half_width, z],
            [center_x + half_width, center_y - half_width, z],
            [center_x + half_width, center_y + half_width, z],
            [center_x - half_width, center_y + half_width, z],
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
    normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (4, 1))
    return points, indices, normals


def _flat_disc_mesh(center_x: float, center_y: float, radius: float, z: float, sides: int = 32):
    """Filled flat circle (center fan) centered at (center_x, center_y, z).

    Args:
        center_x: Center world x.
        center_y: Center world y.
        radius: Disc radius.
        z: World z of the flat mesh.
        sides: Rim polygon resolution.

    Returns:
        (points, indices, normals), same shape convention as `_flat_square_mesh`.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    rim = np.stack([center_x + radius * np.cos(angles), center_y + radius * np.sin(angles)], axis=-1)
    points_xy = np.concatenate([np.array([[center_x, center_y]], dtype=np.float32), rim.astype(np.float32)], axis=0)
    points = np.concatenate([points_xy, np.full((points_xy.shape[0], 1), z, dtype=np.float32)], axis=1)
    indices = np.array(
        [idx for i in range(sides) for idx in (0, 1 + i, 1 + (i + 1) % sides)], dtype=np.int32
    )
    normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (points.shape[0], 1))
    return points, indices, normals


def _flat_goal_shape_mesh(object_shape: str, center_x: float, center_y: float, half_extent: float, z: float):
    """Flat goal-marker/tolerance geometry matching the object's shape.

    Args:
        object_shape: "box" or "cylinder".
        center_x: Center world x.
        center_y: Center world y.
        half_extent: Half side length ("box") or radius ("cylinder").
        z: World z of the flat mesh.

    Returns:
        (points, indices, normals): a square for "box", a disc for "cylinder".
    """
    if object_shape == "cylinder":
        return _flat_disc_mesh(center_x, center_y, half_extent, z)
    return _flat_square_mesh(center_x, center_y, half_extent, z)


def build_goal_overlay(
    viewer,
    object_pose: np.ndarray,
    goal_xy: np.ndarray,
    reach_tolerance: np.ndarray,
    object_footprint: float,
    object_shape: str,
) -> tuple[list[tuple[float, float]], list[tuple]]:
    """Log the planned-path line and one marker/tolerance mesh pair per distinct goal corner.

    Must run before the first begin_frame()/end_frame() cycle. Per-step visibility is then
    toggled by `update_goal_overlay`, which re-logs the same geometry with `hidden` set.

    Args:
        viewer: RTX viewer to log into.
        object_pose: Per-step object pose, shape (num_steps, 7); row 0 starts the path line.
        goal_xy: Per-step active goal (x, y), shape (num_steps, 2).
        reach_tolerance: Per-step reach-tolerance radius, shape (num_steps,).
        object_footprint: Box side length or cylinder diameter [m].
        object_shape: "box" or "cylinder".

    Returns:
        Distinct (x, y) goal corners in visit order, and each corner's
        (tolerance_mesh, marker_mesh) as (points, indices, normals) arrays.
    """
    corners = ordered_unique_goals(goal_xy)
    waypoints = [tuple(object_pose[0, :2]), *corners, corners[0]]
    line_points = np.asarray([(x, y, GOAL_LINE_Z_OFFSET) for x, y in waypoints], dtype=np.float32)
    viewer.log_lines(
        "goal_full_path_line",
        wp.array(line_points[:-1], dtype=wp.vec3),
        wp.array(line_points[1:], dtype=wp.vec3),
        wp.array(np.tile(np.asarray(GOAL_COLOR, dtype=np.float32), (len(line_points) - 1, 1)), dtype=wp.vec3),
        width=GOAL_LINE_WIDTH,
    )

    half_width = 0.5 * object_footprint
    meshes = []
    for i, (goal_x, goal_y) in enumerate(corners):
        first_idx = next(j for j in range(len(goal_xy)) if np.allclose(goal_xy[j], (goal_x, goal_y)))
        tolerance_mesh = _flat_goal_shape_mesh(
            object_shape,
            goal_x, goal_y, half_width + reach_tolerance[first_idx], GOAL_TOLERANCE_Z_OFFSET
        )
        marker_mesh = _flat_goal_shape_mesh(object_shape, goal_x, goal_y, half_width, GOAL_MARKER_Z_OFFSET)
        meshes.append((tolerance_mesh, marker_mesh))
        for name, (points, indices, normals), opacity in (
            (f"goal_tolerance_{i}", tolerance_mesh, single_surface_opacity(GOAL_TOLERANCE_OPACITY)),
            (f"goal_marker_{i}", marker_mesh, None),
        ):
            viewer.log_mesh(
                name,
                wp.array(points, dtype=wp.vec3),
                wp.array(indices, dtype=wp.int32),
                normals=wp.array(normals, dtype=wp.vec3),
                color=GOAL_COLOR,
                roughness=1.0,
                metallic=0.0,
                opacity=opacity,
                hidden=(i != 0),
            )
    return corners, meshes


def update_goal_overlay(
    viewer, goal_xy: np.ndarray, corners: list[tuple[float, float]], meshes: list[tuple], timestep: int
) -> None:
    """Show the active goal corner's marker/tolerance pair and hide every other corner's.

    Material arguments are deliberately absent: ViewerRTX.log_mesh applies them only when it
    first registers a name and ignores them on every later call.

    Args:
        viewer: RTX viewer to log into.
        goal_xy: Per-step active goal (x, y), shape (num_steps, 2).
        corners: Distinct goal corners, from `build_goal_overlay`.
        meshes: Per-corner (tolerance_mesh, marker_mesh), from `build_goal_overlay`.
        timestep: Real step index to show.
    """
    active_xy = goal_xy[min(timestep, len(goal_xy) - 1)]
    for i, (corner, (tolerance_mesh, marker_mesh)) in enumerate(zip(corners, meshes, strict=True)):
        hidden = not np.allclose(active_xy, corner)
        for name, (points, indices, normals) in (
            (f"goal_tolerance_{i}", tolerance_mesh),
            (f"goal_marker_{i}", marker_mesh),
        ):
            viewer.log_mesh(
                name,
                wp.array(points, dtype=wp.vec3),
                wp.array(indices, dtype=wp.int32),
                normals=wp.array(normals, dtype=wp.vec3),
                hidden=hidden,
            )


def main() -> None:
    """Render a saved run_experiment.py trajectory to an MP4 (and optionally a GIF)."""
    args = parse_args()

    data = np.load(args.run_path)
    for required in ("X_final", "object_pose", "goal_xy", "reach_tolerance"):
        if required not in data.files:
            raise ValueError(f"{args.run_path} has no {required} array (files: {data.files})")
    X_final = data["X_final"]
    object_pose = data["object_pose"]
    if object_pose.shape != (X_final.shape[0], 7):
        raise ValueError(f"object_pose has shape {object_pose.shape}, expected {(X_final.shape[0], 7)}")
    goal_xy = data["goal_xy"]
    reach_tolerance = data["reach_tolerance"]

    robot = FourTendonSoftArm().model
    ndof = robot.num_internal_dofs
    num_base_coordinates = robot.num_base_coordinates
    state_dim = 2 * ndof + 5 + num_base_coordinates
    if X_final.shape[1] != state_dim:
        raise ValueError(f"X_final has state_dim={X_final.shape[1]}, expected {state_dim} for this robot")

    franka_bodies: list[int] = []

    def add_franka_geometry(builder: newton.ModelBuilder) -> None:
        """Append the decorative arm and record its bodies.

        The arm adds the model's first articulation, after which every body needs a joint, so the
        object gets a free joint and the table a fixed one; the model is never stepped.
        """
        body_start, body_end, _ = add_franka(builder)
        franka_bodies.extend(range(body_start, body_end))
        builder.add_joint_free(child=_find_label(builder.body_label, "rigid_object"), parent=-1)
        builder.add_joint_fixed(child=_find_label(builder.body_label, "table_surface"), parent=-1)

    model, box_bodies, coupled_solver, _, _ = run_experiment.build_scene(
        robot,
        DEVICE,
        object_shape=args.object_shape,
        object_mass=0.05,  # render-only scene, never stepped, so the mass is irrelevant
        world_count=1,
        num_collision_bodies=ROD_COLLISION_BODIES,
        tentacle_color=ROD_COLOR,
        object_color=OBJECT_COLOR,
        table_color=TABLE_COLOR,
        extra_geometry_fn=add_franka_geometry,
    )
    franka = FrankaDisplay(tuple(args.franka_base), DEVICE)

    state = model.state()
    q_view, qd_view = gvs_state_views(
        state, world_count=1, coordinate_count=robot.num_coordinates, velocity_count=robot.num_velocities
    )
    box_body = int(box_bodies[0])

    writer = RTXFrameWriter(
        model, args.output.parent / (args.output.stem + "_frames"),
        width=args.width, height=args.height, fps=args.fps, samples_per_pixel=args.samples_per_pixel,
    )

    goal_corners, goal_meshes = build_goal_overlay(
        writer.viewer, object_pose, goal_xy, reach_tolerance, run_experiment.OBJECT_FOOTPRINT, args.object_shape
    )
    writer.viewer.shadow_excluded_mesh_names = tuple(f"goal_tolerance_{i}" for i in range(len(goal_corners)))

    body_q = state.body_q.numpy()
    body_qd = state.body_qd.numpy()
    q_full = q_view.numpy()
    qd_full = qd_view.numpy()
    num_base_velocities = robot.num_base_velocities

    def write_frame(row: np.ndarray, t: float, box_pose: np.ndarray, log_overlay: bool, row_idx: int) -> None:
        """Write one interpolated frame: sync kinematics to `row`/`box_pose`, then render."""
        q_internal = row[:ndof]
        qd_internal = row[ndof : 2 * ndof]
        cube_x, cube_vx, cube_y, cube_vy = row[2 * ndof : 2 * ndof + 4]
        base_pose = row[-num_base_coordinates:]

        q_full[:, :num_base_coordinates] = base_pose[None, :]
        q_full[:, num_base_coordinates:] = q_internal[None, :]
        qd_full[:, :num_base_velocities] = 0.0
        qd_full[:, num_base_velocities:] = qd_internal[None, :]
        q_view.assign(q_full)
        qd_view.assign(qd_full)

        body_q[box_body, 0] = cube_x
        body_q[box_body, 1] = cube_y
        body_q[box_body, 2:7] = box_pose[2:7]
        body_qd[box_body, 0] = cube_vx
        body_qd[box_body, 1] = cube_vy
        body_q[franka_bodies] = franka.body_poses(base_pose)
        state.body_q.assign(body_q)
        state.body_qd.assign(body_qd)

        coupled_solver.solver("gvs").sync_kinematics(state)

        def extra_log_fn() -> None:
            """Log this frame's goal overlay."""
            if log_overlay:
                update_goal_overlay(writer.viewer, goal_xy, goal_corners, goal_meshes, row_idx)

        writer.write(t, state, extra_log_fn)

    num_rows = X_final.shape[0]
    try:
        for row_idx in range(num_rows - 1):
            row_start = X_final[row_idx]
            row_end = X_final[row_idx + 1]
            box_pose_start = object_pose[row_idx]
            box_pose_end = object_pose[row_idx + 1]
            for sub in range(args.frames_per_step):
                alpha = sub / args.frames_per_step
                row = row_start * (1.0 - alpha) + row_end * alpha
                t = (row_idx + alpha) * args.control_dt
                box_pose = np.concatenate(
                    [
                        box_pose_start[:3] * (1.0 - alpha) + box_pose_end[:3] * alpha,
                        _slerp_quat(box_pose_start[3:7], box_pose_end[3:7], alpha),
                    ]
                )
                write_frame(row, t, box_pose, log_overlay=(sub == 0), row_idx=row_idx)
        write_frame(
            X_final[num_rows - 1], (num_rows - 1) * args.control_dt, object_pose[num_rows - 1],
            log_overlay=True, row_idx=num_rows - 1,
        )
    finally:
        writer.close()

    encode_mp4(writer.frames_dir, args.output, args.fps)

    if args.gif:
        encode_gif(writer.frames_dir, args.output.with_suffix(".gif"), args.fps)


if __name__ == "__main__":
    main()
