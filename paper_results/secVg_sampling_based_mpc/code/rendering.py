"""Render a saved run_experiment.py run with Newton's ViewerRTX path tracer,
producing an MP4 (and optionally a GIF).

Reads the saved X_final trajectory directly -- no re-simulation. Writing q/qd
and the object pose into a state, then calling SolverGVS.sync_kinematics(), is
pure forward kinematics, so it needs only the saved trajectory itself.

Needs a real GPU with RTX path-tracing support; will not run on CPU alone.

Usage:
    python rendering.py --run-path ../data/run4.npz \
        --device cuda --output ../outputs/run4_push.mp4 --gif
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
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

    PAPER_COLORS holds sRGB hex, which the paper's Open3D and Viser renderers
    consume as-is. ViewerRTX instead feeds colours to UsdPreviewSurface's
    linear diffuseColor and tonemaps back to sRGB on output, so handing it the
    sRGB numbers unconverted lifts and desaturates every surface -- the coral
    reads as tan and the violet as pale lavender next to the other sections.

    Args:
        key: Key into paper_style.PAPER_COLORS.
        stop: Position along the house white-to-colour ramp, blended in sRGB
            before linearizing; 1.0 is the full colour and lower values mix in
            white, as the other sections' gradients do.

    Returns:
        Linear-space red, green and blue in [0, 1].
    """
    value = PAPER_COLORS[key].removeprefix("#")
    channels = (int(value[idx : idx + 2], 16) / 255.0 for idx in (0, 2, 4))
    blended = (1.0 + stop * (c - 1.0) for c in channels)
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in blended)


ROD_COLOR = paper_rgb("post_opt_1", stop=0.875)
OBJECT_COLOR = paper_rgb("obstacle")

FRANKA_Q = np.array([0.0, -0.4, 0.0, -2.0, 0.0, 2.4, np.pi / 4.0])
FRANKA_IK_ITERATIONS = 64
FRANKA_END_EFFECTOR_LINK = "fr3_link8"
FRANKA_BASE_XY = (0.45, 0.45)
FRANKA_FLANGE_ALIGNMENT_ANGLE = -0.5 * np.pi


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

    A standalone Franka model carries the IK and forward kinematics, keeping
    them off the push scene's coupled model, whose body poses the renderer
    writes itself and must not let eval_fk overwrite. The arm stands on the
    table at `base_xy`, yawed to face the world origin, and IK reaches the rod
    base from there.
    """

    def __init__(self, base_xy: tuple[float, float], device: str):
        """Build the standalone arm, stand it on the table, and set up IK.

        Args:
            base_xy: World (x, y) of the arm's base; it is yawed to face the
                world origin.
            device: Warp device used for the standalone model.
        """
        import newton.ik as ik

        builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, -9.81))
        self.body_start, self.body_end, self.end_effector = add_franka(builder)
        self.model = builder.finalize(device=device)
        self.state = self.model.state()
        self.joint_q = self.model.joint_q.reshape((1, self.model.joint_coord_count))
        self.joint_seed = wp.array(FRANKA_Q[None, :], dtype=wp.float32, device=device)
        self.joint_q.assign(np.asarray(FRANKA_Q[None, :], dtype=np.float32))

        yaw = float(np.arctan2(-base_xy[1], -base_xy[0]))
        self.placement = wp.transform(
            wp.vec3(float(base_xy[0]), float(base_xy[1]), 0.0),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw),
        )
        self.inverse_placement = wp.transform_inverse(self.placement)

        local_target = wp.transform_multiply(self.inverse_placement, wp.transform_identity())
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
        self.maximum_error = 0.0

    @staticmethod
    def _target_transform(base_pose: np.ndarray) -> wp.transform:
        """Convert one GVS base-pose row into an end-effector target transform.

        The rod's base frame is the flange frame turned by
        FRANKA_FLANGE_ALIGNMENT_ANGLE about y, matching how
        examples/gvs/franka_tentacle.py derives one from the other, so the
        alignment has to be undone to get back to a flange pose.

        Args:
            base_pose: Base pose, shape (7,) as (qx, qy, qz, qw, x, y, z).

        Returns:
            World transform the end effector should reach.
        """
        alignment = wp.transform(
            wp.vec3(0.0),
            wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(FRANKA_FLANGE_ALIGNMENT_ANGLE)),
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
        error = float(np.linalg.norm(body_q[self.end_effector][:3] - np.asarray(local_target.p)))
        self.maximum_error = max(self.maximum_error, error)
        poses = np.empty((self.body_end - self.body_start, 7), dtype=np.float32)
        for row, body in enumerate(range(self.body_start, self.body_end)):
            placed = wp.transform_multiply(
                self.placement, wp.transform(wp.vec3(*body_q[body][:3]), wp.quat(*body_q[body][3:7]))
            )
            poses[row, :3] = np.asarray(placed.p)
            poses[row, 3:7] = np.asarray(placed.q)
        return poses

MPC_ROLLOUT_GRADIENT_WORST_COLOR = (0.25, 0.25, 0.25)
MPC_ROLLOUT_GRADIENT_BEST_COLOR = (0.1, 0.85, 0.95)
MPC_ROLLOUT_LINE_WIDTH = 0.0015
BEST_ROLLOUT_GHOST_COLOR = (0.55, 0.85, 0.9)
OBJECT_GHOST_COLOR = BEST_ROLLOUT_GHOST_COLOR

STUDIO_COLOR = (0.18, 0.175, 0.165)


@dataclass(frozen=True)
class StudioPreset:
    """Lighting and backdrop configuration for ExampleViewerRTX's neutral studio.

    The scenery is authored in the viewer's USD stage and never enters the physics model.
    """

    color: tuple[float, float, float] = STUDIO_COLOR
    ambient: float = 60.0
    key: float = 1000.0
    fill: float = 160.0
    rim: float = 160.0
    key_angle: float = 28.0
    backdrop_start: float = 80.0
    backdrop_radius: tuple[float, float] = (8.0, 8.0)
    camera_distance: float = 1.8
    capsule_roughness: float | None = 0.72
    diffuse_backdrop: bool = True
    backdrop_emission: float = 0.22


NEUTRAL_STUDIO = StudioPreset()
STUDIO_PRESETS = {"neutral": NEUTRAL_STUDIO}


def look_at(viewer, eye, target) -> None:
    """Aim a Z-up viewer camera at a world-space point."""
    eye, target = np.asarray(eye, dtype=float), np.asarray(target, dtype=float)
    direction = target - eye
    distance = np.linalg.norm(direction)
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("Camera eye and target must be finite and distinct")
    direction /= distance
    viewer.set_camera(
        wp.vec3(*eye),
        pitch=float(np.rad2deg(np.arcsin(direction[2]))),
        yaw=float(np.rad2deg(np.arctan2(direction[1], direction[0]))),
    )


class ExampleViewerRTX(newton.viewer.ViewerRTX):
    """RTX viewer with a neutral studio or the original kitchen lighting."""

    def __init__(
        self, *args, samples_per_pixel: int = 256, studio: bool = True, studio_preset: str = "neutral", **kwargs
    ):
        """Configure the path-tracer sample budget and studio lighting.

        Args:
            samples_per_pixel: Path-tracer sample count per frame.
            studio: Use the neutral studio instead of the original kitchen scene.
            studio_preset: Key into STUDIO_PRESETS.
        """
        if samples_per_pixel < 1:
            raise ValueError("samples_per_pixel must be positive")
        self.samples_per_pixel = samples_per_pixel
        self.studio = studio
        self.lighting = STUDIO_PRESETS[studio_preset]
        self.scene_center = np.zeros(3)
        self.scene_extent = 1.0
        self.floor_height = -0.05
        super().__init__(*args, environment="none" if studio else "studio", async_rendering=False, **kwargs)
        self.show_ui = False
        self.show_joints = False
        self.show_contacts = False
        self.show_com = False
        self.show_inertia_boxes = False
        self.show_collision = False

    def frame_scene(
        self, positions: np.ndarray, *, floor_height: float | None = None, camera_distance: float | None = None
    ) -> None:
        """Fit the complete motion, optionally grounding a robot or work surface.

        Args:
            positions: World-space points the camera should frame, shape (N, 3).
            floor_height: Fixed floor world z, or None to infer from `positions`.
            camera_distance: Camera distance as a multiple of the scene extent.
        """
        points = np.asarray(positions).reshape(-1, 3)
        if not len(points) or not np.all(np.isfinite(points)):
            raise ValueError("Scene positions must be finite and nonempty")
        lower, upper = points.min(axis=0), points.max(axis=0)
        self.scene_center = (lower + upper) * 0.5
        self.scene_extent = max(0.25, float(np.max(upper - lower)))
        clearance = 0.12 * self.scene_extent
        if floor_height is None and self.model is not None:
            capsules = self.model.shape_type.numpy() == newton.GeoType.CAPSULE
            if np.any(capsules):
                sizes = self.model.shape_scale.numpy()[capsules]
                offsets = self.model.shape_transform.numpy()[capsules, :3]
                radius = np.max(np.linalg.norm(offsets, axis=1) + sizes[:, 0] + sizes[:, 1])
                clearance = max(clearance, float(radius) + 0.02 * self.scene_extent)
        self.floor_height = float(lower[2] - clearance)
        if floor_height is not None:
            if not np.isfinite(floor_height):
                raise ValueError("Studio floor height must be finite")
            self.floor_height = float(floor_height)
        if camera_distance is None:
            camera_distance = self.lighting.camera_distance
        if not np.isfinite(camera_distance) or camera_distance <= 0:
            raise ValueError("Camera distance must be finite and positive")
        direction = np.asarray((1.3, -1.8, 0.85))
        eye = self.scene_center + direction / np.linalg.norm(direction) * self.scene_extent * camera_distance
        self.camera.fov = 40.0
        self.camera.near = max(0.001, self.scene_extent * 0.001)
        look_at(self, eye, self.scene_center)

    def _add_camera_lights_and_render_product(self) -> None:
        """Configure the path-tracer render product, then add the studio if enabled."""
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
        if self.studio:
            self._add_studio()

    def _render_and_display(self) -> None:
        """Render until the path tracer converges to samples_per_pixel, then present the frame.

        Only the convergence loop is overridden; the converged products are handed to Newton's own
        `_accept_render`, which sets the state `save_screenshot()` and `get_frame()` expect.
        """
        if self._rtx is None or self._should_close:
            return
        step_kwargs = {"render_products": {self._render_product_path}}
        if self._use_ovstage:
            step_kwargs["ordinal"] = self._ovstage_ordinal
        for attempt in range(self.samples_per_pixel + 2):
            products = self._rtx.step(
                **step_kwargs,
                delta_time=1.0 / self.fps if attempt == 0 else 0.0,
            )
            frames = [frame for product in products.values() for frame in product.frames]
            if frames and all(frame.converged for frame in frames):
                break
        else:
            raise RuntimeError(
                f"RTX did not converge to the requested sample count: {[frame.progression for frame in frames]}"
            )
        self._accept_render(products)

    def _apply_ground_material(self) -> None:
        """Apply the kitchen scene's ground material, skipped under the studio."""
        if not self.studio:
            super()._apply_ground_material()

    def _add_studio(self) -> None:
        """Author the studio backdrop mesh and key/fill/rim/ambient lights into the USD stage."""
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

        center, extent = self.scene_center, self.scene_extent
        material = UsdShade.Material.Define(self.stage, "/root/Studio/Matte")
        shader = UsdShade.Shader.Define(self.stage, "/root/Studio/Matte/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*self.lighting.color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        if self.lighting.diffuse_backdrop:
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
            shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
            shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0))
        if self.lighting.backdrop_emission:
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(self.lighting.backdrop_emission)
            )
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        bend = self.lighting.backdrop_start
        radius_y, radius_z = self.lighting.backdrop_radius
        span = max(12.0, bend + 2.0 * radius_y)
        profile = [(-span, 0.0), (bend, 0.0)]
        normals = [(0.0, 0.0, 1.0)] * 2
        for angle in np.linspace(0.0, np.pi / 2, 33)[1:]:
            profile.append((bend + radius_y * np.sin(angle), radius_z * (1.0 - np.cos(angle))))
            normal = np.asarray((0.0, -radius_z * np.sin(angle), radius_y * np.cos(angle)))
            normals.append(tuple(normal / np.linalg.norm(normal)))
        profile.append((bend + radius_y, max(8.0, radius_z + 1.0)))
        normals.append((0.0, -1.0, 0.0))
        vertices = [
            (float(center[0] + x * extent), float(center[1] + y * extent), float(self.floor_height + z * extent))
            for y, z in profile
            for x in (-span, span)
        ]
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

        if self.lighting.capsule_roughness is not None:
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
            rod_surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(self.lighting.capsule_roughness)
            rod_surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            rod_material.CreateSurfaceOutput().ConnectToSource(rod_surface.ConnectableAPI(), "surface")
            for name, prototype in self._meshes.items():
                if "capsule" in name.lower():
                    UsdShade.MaterialBindingAPI.Apply(prototype.GetPrim()).Bind(rod_material)

        dome = UsdLux.DomeLight.Define(self.stage, "/root/Studio/Ambient")
        dome.CreateColorAttr(Gf.Vec3f(1.0))
        dome.CreateIntensityAttr(self.lighting.ambient)
        key = UsdLux.DistantLight.Define(self.stage, "/root/Studio/Key")
        key.CreateColorAttr(Gf.Vec3f(1.0))
        key.CreateIntensityAttr(self.lighting.key)
        key.CreateAngleAttr(self.lighting.key_angle)
        UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(29.0, 14.0, 0.0))
        for name, position, intensity in (
            ("Fill", (0.0, -0.8, 0.8), self.lighting.fill),
            ("Rim", (0.0, 0.4, 0.75), self.lighting.rim),
        ):
            light = UsdLux.SphereLight.Define(self.stage, f"/root/Studio/{name}")
            light.CreateColorAttr(Gf.Vec3f(1.0))
            light.CreateIntensityAttr(intensity)
            light.CreateRadiusAttr(0.3 * extent)
            UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*(center + np.asarray(position) * extent)))


def _ghost_step_opacity(step_index: int, base_opacity: float, ghost_count: int, min_opacity: float) -> float:
    """Dim a ghost config's opacity the further ahead it is in the horizon.

    Args:
        step_index: Horizon steps ahead of the current real state, 0-based.
        base_opacity: Opacity of the nearest (step_index=0) ghost config.
        ghost_count: Total number of ghost configs the ramp spans.
        min_opacity: Opacity of the farthest (step_index=ghost_count-1) ghost config.

    Returns:
        Opacity, linearly interpolated between base_opacity and min_opacity.
    """
    if ghost_count <= 1:
        return base_opacity
    t = step_index / (ghost_count - 1)
    return max(min_opacity, base_opacity + (min_opacity - base_opacity) * t)


GOAL_MARKER_COLOR = paper_rgb("target")
GOAL_MARKER_Z_OFFSET = 5.0e-4
GOAL_TOLERANCE_Z_OFFSET = 2.0e-4


def single_surface_opacity(desired: float) -> float:
    """Return the opacity to request so one flat surface renders at `desired`.

    ViewerRTX authors PreviewSurface opacity per ray hit and assumes a ray
    crosses _PREVIEW_SURFACE_OPACITY_LAYERS surfaces of a closed mesh, so it
    lowers what it authors accordingly. A flat patch is crossed once, which
    leaves it far more transparent than asked -- a requested 0.2 lands near
    0.05. Inverting that mapping restores the intended alpha.

    Args:
        desired: Intended display opacity of a single surface in [0, 1].

    Returns:
        Opacity to pass to log_mesh, below the threshold that reads as opaque.
    """
    layers = ExampleViewerRTX._PREVIEW_SURFACE_OPACITY_LAYERS
    return float(min(1.0 - (1.0 - desired) ** layers, 0.998))
GOAL_LINE_Z_OFFSET = 9.0e-4
GOAL_PATH_COLOR = paper_rgb("target")


class _QualityViewerRTX(ExampleViewerRTX):
    """ExampleViewerRTX plus opt-in shadow-link exclusion."""

    shadow_excluded_mesh_names: tuple[str, ...] = ()

    def _add_studio(self) -> None:
        """Build the base studio, then exclude shadow_excluded_mesh_names from every studio light."""
        super()._add_studio()
        if not self.shadow_excluded_mesh_names:
            return
        from pxr import Sdf, UsdLux

        mesh_paths = [
            self._mesh_prim_paths[self._qualify(name)]
            for name in self.shadow_excluded_mesh_names
            if self._qualify(name) in self._mesh_prim_paths
        ]
        if not mesh_paths:
            return
        for light_path in ("/root/Studio/Ambient", "/root/Studio/Key", "/root/Studio/Fill", "/root/Studio/Rim"):
            light_prim = self.stage.GetPrimAtPath(light_path)
            if not light_prim:
                continue
            excludes = UsdLux.LightAPI(light_prim).GetShadowLinkCollectionAPI().CreateExcludesRel()
            for mesh_path in mesh_paths:
                excludes.AddTarget(Sdf.Path(mesh_path))


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
        self.viewer = _QualityViewerRTX(
            width=width, height=height, headless=True, fps=fps, samples_per_pixel=samples_per_pixel,
        )
        self.viewer.set_model(model)
        self.set_camera_look_at(np.asarray([0.55, -0.45, 0.55]), np.asarray([0.04, 0.0, 0.15]))

    def set_camera_look_at(self, camera_position: np.ndarray, camera_target: np.ndarray) -> None:
        """Point the viewer's camera at `camera_target` from `camera_position`."""
        look_at(self.viewer, camera_position, camera_target)

    def write(self, time_value: float, state: newton.State, extra_log_fn=None) -> None:
        """Render one frame to a numbered PNG in `frames_dir`.

        Args:
            time_value: Simulation time of this frame.
            state: Newton state to render.
            extra_log_fn: Optional callback run between begin_frame() and
                end_frame(); animated-overlay log_lines() calls must run here,
                since begin_frame() clears any line batch queued earlier.
        """
        self.viewer.begin_frame(time_value)
        self.viewer.log_state(state)
        if extra_log_fn is not None:
            extra_log_fn()
        self.viewer.end_frame()
        self.viewer.save_screenshot(str(self.frames_dir / f"frame_{self.frame_index:04d}.png"))
        self.frame_index += 1

    def close(self) -> None:
        """Close the underlying viewer."""
        self.viewer.close()


def encode_mp4(frames_dir: Path, output_path: Path, fps: int) -> None:
    """Encode the frame PNGs into an MP4 with ffmpeg.

    Args:
        frames_dir: Directory of `frame_%04d.png` files.
        output_path: MP4 path to write.
        fps: Playback frame rate.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode the MP4; the PNG sequence was retained")
    subprocess.run(
        [ffmpeg, "-y", "-framerate", str(fps), "-i", str(frames_dir / "frame_%04d.png"),
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
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode the GIF")
    palette = frames_dir / "palette.png"
    vf = f"fps={fps},scale=640:-1:flags=lanczos"
    subprocess.run(
        [ffmpeg, "-y", "-framerate", str(fps), "-i", str(frames_dir / "frame_%04d.png"),
         "-vf", vf + ",palettegen", str(palette)],
        check=True,
    )
    subprocess.run(
        [ffmpeg, "-y", "-framerate", str(fps), "-i", str(frames_dir / "frame_%04d.png"), "-i", str(palette),
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
    parser.add_argument("--device", default="cuda", help="Requires a real GPU; cpu will not work.")
    parser.add_argument("--control-dt", type=float, default=0.1, help="Must match the original run's --control-dt.")
    parser.add_argument("--frames-per-step", type=int, default=10, help="Interpolated render frames per control step.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--samples-per-pixel", type=int, default=8)
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "push_rtx.mp4")
    parser.add_argument("--gif", action="store_true", help="Also encode a GIF alongside the MP4.")
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Render only the first N MPC steps and skip MP4/GIF encoding.",
    )
    parser.add_argument(
        "--show-rollouts", action="store_true",
        help="Overlay CEM candidates (trajectory lines, ghosts, cloud). Requires rollout_X_samples.",
    )
    parser.add_argument(
        "--mpc-line-candidates", type=int, default=None,
        help="Candidates to show in line overlays, sourced from diagnostics_iter_X_samples.",
    )
    parser.add_argument(
        "--backbone-points", type=int, default=25,
        help="Points sampled along the backbone per rollout-cloud ghost.",
    )
    parser.add_argument(
        "--no-tip-trajectories", dest="show_tip_trajectories", action="store_false", default=True,
        help="Disable the per-candidate rod-tip trajectory line overlay.",
    )
    parser.add_argument(
        "--show-backbone-lines", action="store_true", default=False,
        help="Show every candidate's full-rod shape at its final horizon step.",
    )
    parser.add_argument(
        "--show-box-trajectories", action="store_true", default=False,
        help="Show every candidate's predicted box-center path.",
    )
    parser.add_argument(
        "--show-cloud", action="store_true", default=False,
        help="Overlay a translucent mesh cloud of the last CEM iteration's candidates.",
    )
    parser.add_argument(
        "--cloud-opacity", type=float, default=0.1, help="--show-cloud: opacity of the best-of-rest sample.",
    )
    parser.add_argument(
        "--cloud-max-samples", type=int, default=30, help="--show-cloud: candidates drawn per step.",
    )
    parser.add_argument(
        "--cloud-horizon-step", type=int, default=-1,
        help="--show-cloud: horizon step to draw (-1 = final step).",
    )
    parser.add_argument(
        "--ghost-config-count", type=int, default=0,
        help="Translucent full-rod ghosts along the best candidate's horizon; 0 to disable.",
    )
    parser.add_argument(
        "--ghost-opacity", type=float, default=0.9, help="Opacity of the nearest ghost config.",
    )
    parser.add_argument(
        "--franka", action="store_true",
        help="Draw a decorative Franka FR3 whose end effector tracks the rod's base.",
    )
    parser.add_argument(
        "--franka-base", type=float, nargs=2, metavar=("X", "Y"), default=FRANKA_BASE_XY,
        help="World (x, y) the decorative arm stands at; it is yawed to face the world origin.",
    )
    parser.add_argument(
        "--ghost-min-opacity", type=float, default=0.08,
        help="Opacity of the farthest ghost config, the low end of the --ghost-opacity ramp.",
    )
    parser.add_argument(
        "--ghost-ring-sides", type=int, default=8, help="Cross-section polygon resolution for ghost tubes.",
    )
    parser.add_argument(
        "--goal-tolerance-opacity", type=float, default=0.5,
        help="Opacity of the goal tolerance halo; it composites over the light table, so low "
        "values wash its colour out.",
    )
    parser.add_argument(
        "--no-goal-marker", dest="show_goal_marker", action="store_false", default=True,
        help="Disable the table marker for the object's goal position.",
    )
    parser.add_argument(
        "--object-shape", choices=("box", "cylinder"), default="box",
        help="Must match the original run's --object-shape.",
    )
    return parser.parse_args()


def _rollout_box_path(X: np.ndarray, ndof: int, box_z: float) -> np.ndarray:
    """Box-center (x, y) path for one rollout, lifted to 3D at a fixed table-top z.

    Args:
        X: One rollout's trajectory, shape (horizon+1, state_dim).
        ndof: Robot internal DOF count.
        box_z: Fixed world z to draw the path at (the box only moves in x/y).

    Returns:
        Path vertices, shape (horizon+1, 3).
    """
    xy = np.stack([X[:, 2 * ndof], X[:, 2 * ndof + 2]], axis=-1)
    z = np.full((xy.shape[0], 1), box_z)
    return np.concatenate([xy, z], axis=-1)


def _rollout_backbone_poses(
    robot, X_row: np.ndarray, ndof: int, num_base_coordinates: int, s_ps: np.ndarray
) -> np.ndarray:
    """Arm backbone SE(3) poses (position + material frame) at one trajectory row.

    Args:
        robot: SoRoMoX GVS model.
        X_row: One state-trajectory row, shape (state_dim,).
        ndof: Robot internal DOF count.
        num_base_coordinates: Floating-base coordinate count (7: quat + xyz).
        s_ps: Arc-length sample points along [0, robot.length], shape (N,).

    Returns:
        World-frame poses, shape (N, 4, 4). Column [:3, 0] is the backbone
        tangent (material-frame +X, see newton_cosserat.gvs.add_gvs's
        docstring); columns [:3, 1]/[:3, 2] span the cross-section plane.
    """
    q_internal = X_row[:ndof]
    base_pose = X_row[-num_base_coordinates:]
    q_fk = jnp.concatenate([jnp.asarray(base_pose), jnp.asarray(q_internal)])
    poses = robot.forward_kinematics_abscissa_batched(q_fk, jnp.asarray(s_ps))
    return np.asarray(poses)


def _rollout_backbone(
    robot, X_row: np.ndarray, ndof: int, num_base_coordinates: int, s_ps: np.ndarray
) -> np.ndarray:
    """Arm backbone curve (position only) at one trajectory row (e.g. the final step).

    Returns:
        Backbone vertices in world frame, shape (N, 3).
    """
    poses = _rollout_backbone_poses(robot, X_row, ndof, num_base_coordinates, s_ps)
    return poses[:, :3, 3]


def cross_section_radii(robot, s_ps: np.ndarray, ndof: int, num_base_coordinates: int) -> np.ndarray:
    """Circular cross-section radius at each of `s_ps`'s arc-length samples.

    Args:
        robot: SoRoMoX GVS model; every link must be circular.
        s_ps: Arc-length sample points along [0, robot.length], shape (N,).
        ndof: Robot internal DOF count.
        num_base_coordinates: Floating-base coordinate count.

    Returns:
        Radius at each sample, shape (N,).
    """
    q_ref = jnp.zeros(ndof + num_base_coordinates, dtype=jnp.float64)
    radii = []
    for s in s_ps:
        geometry, outer_dimensions, _inner_dimensions = robot.cross_section_geometry(
            q_ref, jnp.asarray(float(s), dtype=jnp.float64)
        )
        if int(np.asarray(geometry)) != 0:
            raise ValueError(f"cross_section_radii assumes CIRCULAR cross-sections, got geometry={geometry} at s={s}")
        radii.append(float(np.asarray(outer_dimensions).reshape(-1)[0]))
    return np.asarray(radii, dtype=np.float64)


CAP_LATITUDE_RINGS = 4
"""Number of latitude bands approximating each tube end's hemisphere, between
the tube's own last cross-section ring (the hemisphere's equator) and its
pole vertex. More rings -> smoother dome, more triangles (`2 * ring_sides`
triangles per extra ring, plus one `ring_sides`-triangle fan at the pole)."""


def _ring_band_tris(ring_sides: int, lo_base: int, hi_base: int, *, reverse: bool) -> np.ndarray:
    """Quad-strip triangle indices between two same-size rings.

    Args:
        ring_sides: Vertices per ring.
        lo_base: First vertex index of the lower ring.
        hi_base: First vertex index of the upper ring.
        reverse: Flip triangle winding.

    Returns:
        Flat triangle-index array, length 6 * ring_sides.
    """
    j = np.arange(ring_sides)
    j_next = (j + 1) % ring_sides
    a, b = lo_base + j, lo_base + j_next
    c, d = hi_base + j, hi_base + j_next
    tris = np.stack([a, b, d, a, d, c] if not reverse else [a, d, b, a, c, d], axis=-1)
    return tris.reshape(-1)


def _pole_fan_tris(ring_sides: int, ring_base: int, pole_idx: int, *, reverse: bool) -> np.ndarray:
    """Triangle-fan indices connecting one ring to a single pole vertex.

    Args:
        ring_sides: Vertices in the ring.
        ring_base: First vertex index of the ring.
        pole_idx: Pole vertex index, relative to the ring.
        reverse: Flip triangle winding.

    Returns:
        Flat triangle-index array, length 3 * ring_sides.
    """
    j = np.arange(ring_sides)
    j_next = (j + 1) % ring_sides
    tris = np.stack([np.full(ring_sides, pole_idx), j_next, j] if reverse else [np.full(ring_sides, pole_idx), j, j_next], axis=-1)
    return (tris + np.array([0, ring_base, ring_base])).reshape(-1)


def tube_mesh_topology(
    num_strands: int, rings_per_strand: int, ring_sides: int, cap_rings: int = CAP_LATITUDE_RINGS
) -> np.ndarray:
    """Triangle indices for `num_strands` independent swept-circle tubes, matching `tube_mesh_points`'s vertex order.

    Args:
        num_strands: Number of independent tubes concatenated into one mesh.
        rings_per_strand: Body rings per tube.
        ring_sides: Cross-section polygon resolution (>=3).
        cap_rings: Latitude bands per hemispherical end cap.

    Returns:
        Flat int32 triangle-index array.
    """
    if ring_sides < 3:
        raise ValueError("ring_sides must be >= 3")

    body_base = 0
    start_cap_base = rings_per_strand * ring_sides
    start_pole = start_cap_base + cap_rings * ring_sides
    end_cap_base = start_pole + 1
    end_pole = end_cap_base + cap_rings * ring_sides
    vertices_per_strand = end_pole + 1

    tri_parts = [
        _ring_band_tris(ring_sides, body_base + i * ring_sides, body_base + (i + 1) * ring_sides, reverse=False)
        for i in range(rings_per_strand - 1)
    ]

    start_chain = [body_base] + [start_cap_base + k * ring_sides for k in range(cap_rings)]
    tri_parts += [
        _ring_band_tris(ring_sides, start_chain[k], start_chain[k + 1], reverse=True) for k in range(len(start_chain) - 1)
    ]
    tri_parts.append(_pole_fan_tris(ring_sides, start_chain[-1], start_pole, reverse=True))

    end_chain = [body_base + (rings_per_strand - 1) * ring_sides] + [
        end_cap_base + k * ring_sides for k in range(cap_rings)
    ]
    tri_parts += [
        _ring_band_tris(ring_sides, end_chain[k], end_chain[k + 1], reverse=False) for k in range(len(end_chain) - 1)
    ]
    tri_parts.append(_pole_fan_tris(ring_sides, end_chain[-1], end_pole, reverse=False))

    strand_tris = np.concatenate(tri_parts)
    strand_offset = np.arange(num_strands) * vertices_per_strand
    all_tris = (strand_tris[None, :] + strand_offset[:, None]).reshape(-1)
    return all_tris.astype(np.int32)


def _hemisphere_cap_points_normals(
    center: np.ndarray,
    radius: float,
    axis_dir: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    cos_t: np.ndarray,
    sin_t: np.ndarray,
    cap_rings: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Latitude-ring and pole points/normals for one hemispherical tube cap.

    Args:
        center: Cap sphere center (the tube's equator ring), shape (3,).
        radius: Cap/tube radius.
        axis_dir: Unit vector from equator toward the pole, shape (3,).
        y_axis: Unit radial direction at angle 0, shape (3,).
        z_axis: Unit radial direction at angle pi/2, shape (3,).
        cos_t: Cosines of the ring angle samples, shape (ring_sides,).
        sin_t: Sines of the ring angle samples, shape (ring_sides,).
        cap_rings: Latitude bands between equator and pole.

    Returns:
        (ring_points, ring_normals, pole_point, pole_normal); ring_points/
        normals have shape (cap_rings, ring_sides, 3), ordered equator-to-pole.
    """
    phi = (np.arange(cap_rings) + 1) / (cap_rings + 1) * (np.pi / 2.0)
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    radial_dirs = cos_t[:, None] * y_axis[None, :] + sin_t[:, None] * z_axis[None, :]

    ring_points = (
        center[None, None, :]
        + sin_phi[:, None, None] * radius * axis_dir[None, None, :]
        + cos_phi[:, None, None] * radius * radial_dirs[None, :, :]
    )
    ring_normals = cos_phi[:, None, None] * radial_dirs[None, :, :] + sin_phi[:, None, None] * axis_dir[None, None, :]

    pole_point = center + radius * axis_dir
    pole_normal = axis_dir
    return ring_points, ring_normals, pole_point, pole_normal


def tube_mesh_points(
    strand_poses: list[np.ndarray], radii: np.ndarray, ring_sides: int, cap_rings: int = CAP_LATITUDE_RINGS
) -> tuple[np.ndarray, np.ndarray]:
    """Vertex positions + normals for a batch of backbone-pose strands, vertex layout matching `tube_mesh_topology`.

    Args:
        strand_poses: One (N, 4, 4) SE(3)-pose array per strand (see
            `_rollout_backbone_poses`), all sharing the same N and `radii`
            (same `s_ps` sampling).
        radii: Per-sample cross-section radius, shape (N,) (see
            `cross_section_radii`).
        ring_sides: Cross-section polygon resolution (>=3).
        cap_rings: Latitude bands per hemispherical end cap (see
            `tube_mesh_topology`/`_hemisphere_cap_points_normals`).

    Returns:
        (points, normals): concatenated, each shape (total_vertices, 3) float32.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, ring_sides, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    all_points, all_normals = [], []
    for poses in strand_poses:
        positions = poses[:, :3, 3]
        x_axes = poses[:, :3, 0]
        y_axes = poses[:, :3, 1]
        z_axes = poses[:, :3, 2]
        radial = cos_t[None, :, None] * y_axes[:, None, :] + sin_t[None, :, None] * z_axes[:, None, :]
        ring_points = positions[:, None, :] + radii[:, None, None] * radial
        all_points.append(ring_points.reshape(-1, 3))
        all_normals.append(radial.reshape(-1, 3))

        start_rings, start_normals, start_pole_pt, start_pole_n = _hemisphere_cap_points_normals(
            positions[0], radii[0], -x_axes[0], y_axes[0], z_axes[0], cos_t, sin_t, cap_rings
        )
        end_rings, end_normals, end_pole_pt, end_pole_n = _hemisphere_cap_points_normals(
            positions[-1], radii[-1], x_axes[-1], y_axes[-1], z_axes[-1], cos_t, sin_t, cap_rings
        )
        all_points += [start_rings.reshape(-1, 3), start_pole_pt[None, :]]
        all_normals += [start_normals.reshape(-1, 3), start_pole_n[None, :]]
        all_points += [end_rings.reshape(-1, 3), end_pole_pt[None, :]]
        all_normals += [end_normals.reshape(-1, 3), end_pole_n[None, :]]

    points = np.concatenate(all_points, axis=0).astype(np.float32)
    normals = np.concatenate(all_normals, axis=0).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    return points, normals


def _polyline_segments(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Consecutive-vertex line segments for log_lines: (starts, ends), each (N-1, 3)."""
    return vertices[:-1], vertices[1:]


def _lerp_color(t: float, color_a: tuple[float, float, float], color_b: tuple[float, float, float]) -> tuple[float, float, float]:
    """Linearly interpolate between two RGB colors.

    Args:
        t: Interpolation fraction in [0, 1].
        color_a: Color at t=0.
        color_b: Color at t=1.

    Returns:
        Interpolated RGB color.
    """
    return tuple(a + (b - a) * t for a, b in zip(color_a, color_b))


def _slerp_quat(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical interpolation between two (x, y, z, w) quaternions."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        return q / np.linalg.norm(q)
    theta_0 = np.arccos(dot)
    theta = theta_0 * alpha
    s0 = np.cos(theta) - dot * np.sin(theta) / np.sin(theta_0)
    s1 = np.sin(theta) / np.sin(theta_0)
    return s0 * q0 + s1 * q1


def ordered_unique_goals(goal_xy: np.ndarray) -> list[tuple[float, float]]:
    """Distinct consecutive goal positions from the saved per-step goal_xy log, in visit order.

    Args:
        goal_xy: Per-step active goal (x, y), shape (num_steps, 2).

    Returns:
        Distinct goal positions, in the order they were visited.
    """
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


def _flat_goal_shape_mesh(center_x: float, center_y: float, half_extent: float, z: float):
    """Flat goal-marker/tolerance geometry matching run_experiment.OBJECT_SHAPE.

    Args:
        center_x: Center world x.
        center_y: Center world y.
        half_extent: Half side length ("box") or radius ("cylinder").
        z: World z of the flat mesh.

    Returns:
        (points, indices, normals): a square for "box", a disc for "cylinder".
    """
    if run_experiment.OBJECT_SHAPE == "cylinder":
        return _flat_disc_mesh(center_x, center_y, half_extent, z)
    return _flat_square_mesh(center_x, center_y, half_extent, z)


def _solid_box_mesh(center_x: float, center_y: float, center_z: float, half_extent: float):
    """Axis-aligned solid cube with flat (unsmoothed) face normals.

    Args:
        center_x: Center world x.
        center_y: Center world y.
        center_z: Center world z.
        half_extent: Half the cube's side length.

    Returns:
        (points, indices, normals).
    """
    h = half_extent
    corners = np.array(
        [[sx * h, sy * h, sz * h] for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)], dtype=np.float32
    )
    points = corners + np.array([center_x, center_y, center_z], dtype=np.float32)
    faces = [
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    ]
    face_normals = [
        (0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0),
    ]
    tri_points, tri_normals, indices = [], [], []
    for (a, b, c, d), normal in zip(faces, face_normals):
        base = len(tri_points)
        tri_points.extend([points[a], points[b], points[c], points[d]])
        tri_normals.extend([normal] * 4)
        indices.extend([base, base + 2, base + 1, base, base + 3, base + 2])
    return (
        np.asarray(tri_points, dtype=np.float32),
        np.asarray(indices, dtype=np.int32),
        np.asarray(tri_normals, dtype=np.float32),
    )


def _solid_cylinder_mesh(center_x: float, center_y: float, center_z: float, radius: float, half_height: float, sides: int = 24):
    """Solid upright cylinder: two rim rings plus two filled end caps.

    Args:
        center_x: Center world x.
        center_y: Center world y.
        center_z: Center world z.
        radius: Cylinder radius.
        half_height: Half the cylinder's height.
        sides: Rim polygon resolution.

    Returns:
        (points, indices, normals).
    """
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    rim_xy = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=-1)
    side_normals_xy = np.concatenate([rim_xy / radius, np.zeros((sides, 1), dtype=np.float32)], axis=1)

    bottom_rim = np.concatenate([rim_xy, np.full((sides, 1), -half_height, dtype=np.float32)], axis=1)
    top_rim = np.concatenate([rim_xy, np.full((sides, 1), half_height, dtype=np.float32)], axis=1)

    side_bottom_rim, side_top_rim = bottom_rim, top_rim
    cap_bottom_rim, cap_top_rim = bottom_rim, top_rim
    bottom_center = np.array([[0.0, 0.0, -half_height]], dtype=np.float32)
    top_center = np.array([[0.0, 0.0, half_height]], dtype=np.float32)

    points = np.concatenate(
        [side_bottom_rim, side_top_rim, cap_bottom_rim, cap_top_rim, bottom_center, top_center], axis=0
    )
    normals = np.concatenate(
        [
            side_normals_xy, side_normals_xy,
            np.tile([[0.0, 0.0, -1.0]], (sides, 1)), np.tile([[0.0, 0.0, 1.0]], (sides, 1)),
            [[0.0, 0.0, -1.0]], [[0.0, 0.0, 1.0]],
        ],
        axis=0,
    )
    side_bottom_base, side_top_base = 0, sides
    cap_bottom_base, cap_top_base = 2 * sides, 3 * sides
    bottom_center_idx, top_center_idx = 4 * sides, 4 * sides + 1

    indices = []
    for i in range(sides):
        j = (i + 1) % sides
        sbi, sbj, sti, stj = side_bottom_base + i, side_bottom_base + j, side_top_base + i, side_top_base + j
        indices.extend([sbi, sbj, stj, sbi, stj, sti])
        cbi, cbj, cti, ctj = cap_bottom_base + i, cap_bottom_base + j, cap_top_base + i, cap_top_base + j
        indices.extend([bottom_center_idx, cbj, cbi])
        indices.extend([top_center_idx, cti, ctj])

    points = points + np.array([center_x, center_y, center_z], dtype=np.float32)
    return points.astype(np.float32), np.asarray(indices, dtype=np.int32), normals.astype(np.float32)


def _object_ghost_mesh(center_x: float, center_y: float, center_z: float, half_extent: float):
    """Solid predicted-position geometry matching run_experiment.OBJECT_SHAPE.

    Args:
        center_x: Center world x.
        center_y: Center world y.
        center_z: Center world z.
        half_extent: Half side length ("box") or radius ("cylinder").

    Returns:
        (points, indices, normals): a cube for "box", a cylinder for "cylinder".
    """
    if run_experiment.OBJECT_SHAPE == "cylinder":
        return _solid_cylinder_mesh(center_x, center_y, center_z, half_extent, half_extent)
    return _solid_box_mesh(center_x, center_y, center_z, half_extent)


def _full_square_path_segments(
    object_pose: np.ndarray, goal_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One solid segment per leg of the planned goal path, closing back to the first corner.

    Args:
        object_pose: Per-step object pose, shape (num_steps, 7); only row 0 is used.
        goal_xy: Per-step active goal (x, y), shape (num_steps, 2).

    Returns:
        (starts, ends) segment endpoint arrays.
    """
    initial_x, initial_y = float(object_pose[0, 0]), float(object_pose[0, 1])
    corners = ordered_unique_goals(goal_xy)
    waypoints = [(initial_x, initial_y), *corners, corners[0]]
    starts = np.asarray(
        [(x, y, GOAL_LINE_Z_OFFSET) for x, y in waypoints[:-1]], dtype=np.float32
    )
    ends = np.asarray([(x, y, GOAL_LINE_Z_OFFSET) for x, y in waypoints[1:]], dtype=np.float32)
    return starts, ends


def log_full_square_path_line(viewer, object_pose: np.ndarray, goal_xy: np.ndarray) -> None:
    """Log the full planned-path line once; call before the first begin_frame()/end_frame() cycle.

    Args:
        viewer: RTX viewer to log into.
        object_pose: Per-step object pose, shape (num_steps, 7).
        goal_xy: Per-step active goal (x, y), shape (num_steps, 2).
    """
    line_starts, line_ends = _full_square_path_segments(object_pose, goal_xy)
    line_colors = np.tile(np.asarray(GOAL_PATH_COLOR, dtype=np.float32), (line_starts.shape[0], 1))
    viewer.log_lines(
        "goal_full_path_line",
        wp.array(line_starts, dtype=wp.vec3),
        wp.array(line_ends, dtype=wp.vec3),
        wp.array(line_colors, dtype=wp.vec3),
        width=0.0015,
    )


def build_dynamic_goal_overlay_static(
    viewer, object_pose: np.ndarray, goal_xy: np.ndarray, reach_tolerance: np.ndarray, box_side: float,
    tolerance_opacity: float,
) -> list[tuple[float, float]]:
    """Register one marker/tolerance mesh pair per distinct goal corner, plus the planned-path line.

    Must run before the first begin_frame()/end_frame() cycle. Per-step
    visibility is then toggled by `update_dynamic_goal_overlay_frame` instead
    of re-authoring geometry.

    Args:
        viewer: RTX viewer to log into.
        object_pose: Per-step object pose, shape (num_steps, 7).
        goal_xy: Per-step active goal (x, y), shape (num_steps, 2).
        reach_tolerance: Per-step reach-tolerance radius, shape (num_steps,).
        box_side: Object footprint size (box side length or cylinder diameter).
        tolerance_opacity: Display opacity of the tolerance halo in [0, 1].

    Returns:
        Distinct (x, y) goal corners, in visit order.
    """
    log_full_square_path_line(viewer, object_pose, goal_xy)

    corners = ordered_unique_goals(goal_xy)
    goal_half_width = 0.5 * box_side
    for i, (goal_x, goal_y) in enumerate(corners):
        first_idx = next(j for j in range(len(goal_xy)) if np.allclose(goal_xy[j], (goal_x, goal_y)))
        tolerance = float(reach_tolerance[first_idx])

        tolerance_points, tolerance_indices, tolerance_normals = _flat_goal_shape_mesh(
            goal_x, goal_y, goal_half_width + tolerance, GOAL_TOLERANCE_Z_OFFSET
        )
        viewer.log_mesh(
            f"goal_tolerance_{i}",
            wp.array(tolerance_points, dtype=wp.vec3),
            wp.array(tolerance_indices, dtype=wp.int32),
            normals=wp.array(tolerance_normals, dtype=wp.vec3),
            color=GOAL_MARKER_COLOR,
            roughness=1.0,
            metallic=0.0,
            opacity=single_surface_opacity(tolerance_opacity),
            hidden=(i != 0),
        )

        marker_points, marker_indices, marker_normals = _flat_goal_shape_mesh(
            goal_x, goal_y, goal_half_width, GOAL_MARKER_Z_OFFSET
        )
        viewer.log_mesh(
            f"goal_marker_{i}",
            wp.array(marker_points, dtype=wp.vec3),
            wp.array(marker_indices, dtype=wp.int32),
            normals=wp.array(marker_normals, dtype=wp.vec3),
            color=GOAL_MARKER_COLOR,
            roughness=1.0,
            metallic=0.0,
            hidden=(i != 0),
        )
    return corners


def update_dynamic_goal_overlay_frame(
    viewer, goal_xy: np.ndarray, reach_tolerance: np.ndarray, corners: list[tuple[float, float]], box_side: float,
    timestep: int,
) -> None:
    """Show the active goal corner's marker/tolerance mesh pair, hide every other corner's.

    Material arguments are deliberately absent: ViewerRTX.log_mesh applies them
    only when it first registers a name and ignores them on every later call,
    so the appearance comes from `build_dynamic_goal_overlay_static`.

    Args:
        viewer: RTX viewer to log into.
        goal_xy: Per-step active goal (x, y), shape (num_steps, 2).
        reach_tolerance: Per-step reach-tolerance radius, shape (num_steps,).
        corners: Distinct goal corners, from `build_dynamic_goal_overlay_static`.
        box_side: Object footprint size (box side length or cylinder diameter).
        timestep: Real step index to show.
    """
    idx = min(timestep, len(goal_xy) - 1)
    active = next(i for i, corner in enumerate(corners) if np.allclose(goal_xy[idx], corner))
    goal_half_width = 0.5 * box_side
    for i, (goal_x, goal_y) in enumerate(corners):
        visible = i == active
        first_idx = next(j for j in range(len(goal_xy)) if np.allclose(goal_xy[j], (goal_x, goal_y)))
        tolerance = float(reach_tolerance[first_idx])

        tolerance_points, tolerance_indices, tolerance_normals = _flat_goal_shape_mesh(
            goal_x, goal_y, goal_half_width + tolerance, GOAL_TOLERANCE_Z_OFFSET
        )
        viewer.log_mesh(
            f"goal_tolerance_{i}",
            wp.array(tolerance_points, dtype=wp.vec3),
            wp.array(tolerance_indices, dtype=wp.int32),
            normals=wp.array(tolerance_normals, dtype=wp.vec3),
            hidden=not visible,
        )

        marker_points, marker_indices, marker_normals = _flat_goal_shape_mesh(
            goal_x, goal_y, goal_half_width, GOAL_MARKER_Z_OFFSET
        )
        viewer.log_mesh(
            f"goal_marker_{i}",
            wp.array(marker_points, dtype=wp.vec3),
            wp.array(marker_indices, dtype=wp.int32),
            normals=wp.array(marker_normals, dtype=wp.vec3),
            hidden=not visible,
        )


def _tip_trajectory_segments(
    robot, step_X_samples: np.ndarray, ndof: int, num_base_coordinates: int, tip_s: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Every candidate's rod-tip polyline over its predicted horizon, best-cost-first.

    Args:
        robot: SoRoMoX GVS model.
        step_X_samples: Cost-sorted candidate trajectories, shape (N, horizon+1, state_dim).
        ndof: Robot internal DOF count.
        num_base_coordinates: Floating-base coordinate count.
        tip_s: Arc-length sample at the rod tip, shape (1,).

    Returns:
        Best-cost-first list of (starts, ends) segment pairs, one per candidate.
    """
    n_candidates, horizon_plus_1 = step_X_samples.shape[:2]
    segments = []
    for k in range(n_candidates):
        tip = np.empty((horizon_plus_1, 3), dtype=np.float32)
        for h in range(horizon_plus_1):
            pose = _rollout_backbone_poses(robot, step_X_samples[k, h], ndof, num_base_coordinates, tip_s)
            tip[h] = pose[0, :3, 3]
        segments.append(_polyline_segments(tip))
    return segments


def _box_trajectory_segments(
    step_X_samples: np.ndarray, ndof: int, box_z: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Every candidate's predicted box-center (x, y) path over its horizon, best-cost-first.

    Args:
        step_X_samples: Cost-sorted candidate trajectories, shape (N, horizon+1, state_dim).
        ndof: Robot internal DOF count.
        box_z: Fixed world z to draw the path at.

    Returns:
        Best-cost-first list of (starts, ends) segment pairs, one per candidate.
    """
    n_candidates = step_X_samples.shape[0]
    return [_polyline_segments(_rollout_box_path(step_X_samples[k], ndof, box_z)) for k in range(n_candidates)]


def _candidate_final_backbone_segments(
    robot, step_X_samples: np.ndarray, ndof: int, num_base_coordinates: int, s_ps: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Every candidate's full-rod backbone curve at its final horizon step, best-cost-first.

    Args:
        robot: SoRoMoX GVS model.
        step_X_samples: Cost-sorted candidate trajectories, shape (N, horizon+1, state_dim).
        ndof: Robot internal DOF count.
        num_base_coordinates: Floating-base coordinate count.
        s_ps: Arc-length sample points along the backbone, shape (M,).

    Returns:
        Best-cost-first list of (starts, ends) segment pairs, one per candidate.
    """
    n_candidates = step_X_samples.shape[0]
    return [
        _polyline_segments(_rollout_backbone(robot, step_X_samples[k, -1], ndof, num_base_coordinates, s_ps))
        for k in range(n_candidates)
    ]


def _rank_graded_line_batch(
    segments_best_first: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one color-graded (starts, ends, colors) line batch from per-candidate segments.

    Args:
        segments_best_first: Per-candidate (starts, ends) segment pairs, best-cost-first.

    Returns:
        (starts, ends, colors), each concatenated across every candidate.
    """
    n = len(segments_best_first)
    all_starts, all_ends, all_colors = [], [], []
    for k, (starts, ends) in enumerate(segments_best_first):
        t = k / max(1, n - 1)
        color = _lerp_color(t, MPC_ROLLOUT_GRADIENT_BEST_COLOR, MPC_ROLLOUT_GRADIENT_WORST_COLOR)
        all_starts.append(starts)
        all_ends.append(ends)
        all_colors.append(np.tile(color, (starts.shape[0], 1)))
    return np.concatenate(all_starts), np.concatenate(all_ends), np.concatenate(all_colors)


def _ghost_step_indices(horizon_len: int, ghost_config_count: int) -> np.ndarray:
    """Evenly-spaced horizon-step indices for ghost configs, excluding the current real state.

    Args:
        horizon_len: Predicted horizon length.
        ghost_config_count: Number of ghost configs requested.

    Returns:
        Step indices in [1, horizon_len], shape (min(ghost_config_count, horizon_len),).
    """
    count = max(0, min(ghost_config_count, horizon_len))
    if count == 0:
        return np.zeros(0, dtype=int)
    return np.linspace(1, horizon_len, count).round().astype(int)


def _best_rollout_ghost_strand_poses(
    robot, step_X_samples: np.ndarray, ndof: int, num_base_coordinates: int, s_ps: np.ndarray, ghost_config_count: int
) -> list[np.ndarray]:
    """Backbone poses for ghost configs along the lowest-cost candidate's predicted horizon.

    Args:
        robot: SoRoMoX GVS model.
        step_X_samples: Cost-sorted candidate trajectories, shape (N, horizon+1, state_dim).
        ndof: Robot internal DOF count.
        num_base_coordinates: Floating-base coordinate count.
        s_ps: Arc-length sample points along the backbone, shape (M,).
        ghost_config_count: Number of ghost configs requested.

    Returns:
        One (M, 4, 4) pose array per ghost config.
    """
    horizon_len = step_X_samples.shape[1] - 1
    step_indices = _ghost_step_indices(horizon_len, ghost_config_count)
    return [_rollout_backbone_poses(robot, step_X_samples[0, h], ndof, num_base_coordinates, s_ps) for h in step_indices]


def _best_rollout_object_ghost_centers(step_X_samples: np.ndarray, ndof: int, ghost_config_count: int) -> np.ndarray:
    """(x, y) object-center positions at the same horizon steps as the rod ghosts.

    Args:
        step_X_samples: Cost-sorted candidate trajectories, shape (N, horizon+1, state_dim).
        ndof: Robot internal DOF count.
        ghost_config_count: Number of ghost configs requested.

    Returns:
        Object centers, shape (len(_ghost_step_indices(...)), 2).
    """
    horizon_len = step_X_samples.shape[1] - 1
    step_indices = _ghost_step_indices(horizon_len, ghost_config_count)
    cube_x, cube_y = step_X_samples[0, step_indices, 2 * ndof], step_X_samples[0, step_indices, 2 * ndof + 2]
    return np.stack([cube_x, cube_y], axis=-1)


def _select_best_first_candidates(step_X: np.ndarray, step_cost: np.ndarray, max_count: int) -> np.ndarray:
    """Up to `max_count` candidates, best-cost-first, evenly spaced by cost rank if the population is larger.

    Args:
        step_X: Candidate trajectories, shape (N, horizon+1, state_dim).
        step_cost: Per-candidate cost, shape (N,).
        max_count: Maximum candidates to return.

    Returns:
        Selected trajectories, best-cost-first.
    """
    order = np.argsort(step_cost)
    if max_count < len(order):
        pick_positions = np.unique(np.linspace(0, len(order) - 1, max_count).round().astype(int))
        order = order[pick_positions]
    return step_X[order]


def _select_cloud_indices(step_cost: np.ndarray, best_idx: int, max_samples: int) -> list[int]:
    """Non-best sample indices, best-of-rest first, evenly spaced across the cost-rank spectrum.

    Args:
        step_cost: Per-candidate cost, shape (N,).
        best_idx: Index of the lowest-cost candidate, excluded from the result.
        max_samples: Maximum indices to return.

    Returns:
        Selected indices, best-of-rest-first.
    """
    candidates_by_cost = sorted((k for k in range(step_cost.shape[0]) if k != best_idx), key=lambda k: step_cost[k])
    if max_samples < len(candidates_by_cost):
        pick_positions = np.unique(np.linspace(0, len(candidates_by_cost) - 1, max_samples).round().astype(int))
        candidates_by_cost = [candidates_by_cost[p] for p in pick_positions]
    return candidates_by_cost


def build_cloud_overlay_static(
    viewer, robot, step_X_last_iter: np.ndarray, step_cost_last_iter: np.ndarray, ndof: int,
    num_base_coordinates: int, s_ps: np.ndarray, radii: np.ndarray, ring_sides: int, max_samples: int,
    sample_opacity: float, horizon_step: int, object_half_extent: float,
) -> list[str]:
    """Register fixed-name cloud-sample and best-sample mesh slots, each with a paired object-ghost mesh.

    Must run before the first begin_frame()/end_frame() cycle; opacity is
    fixed per slot rank for the whole video, positions are refreshed per
    step by `update_cloud_overlay_frame`.

    Args:
        viewer: RTX viewer to log into.
        robot: SoRoMoX GVS model.
        step_X_last_iter: Last-iteration candidate trajectories, shape (N, horizon+1, state_dim).
        step_cost_last_iter: Per-candidate cost, shape (N,).
        ndof: Robot internal DOF count.
        num_base_coordinates: Floating-base coordinate count.
        s_ps: Arc-length sample points along the backbone, shape (M,).
        radii: Per-sample cross-section radius, shape (M,).
        ring_sides: Cross-section polygon resolution.
        max_samples: Maximum non-best candidates to draw.
        sample_opacity: Opacity of the best-of-rest sample.
        horizon_step: Horizon step to draw each candidate's shape at.
        object_half_extent: Half the object's footprint size.

    Returns:
        Registered rod slot mesh names, best-of-rest first.
    """
    best_idx = int(np.argmin(step_cost_last_iter))
    cloud_idx = _select_cloud_indices(step_cost_last_iter, best_idx, max_samples)
    n = len(cloud_idx)
    slot_names = [f"cloud_slot_{i}" for i in range(n)]
    for i, k in enumerate(cloud_idx):
        factor = 1.0 if n <= 1 else 1.0 - 0.9 * (i / (n - 1))
        strand_poses = [_rollout_backbone_poses(robot, step_X_last_iter[k, horizon_step], ndof, num_base_coordinates, s_ps)]
        indices = tube_mesh_topology(1, len(s_ps), ring_sides)
        points, normals = tube_mesh_points(strand_poses, radii, ring_sides)
        viewer.log_mesh(
            slot_names[i],
            wp.array(points, dtype=wp.vec3),
            wp.array(indices, dtype=wp.int32),
            normals=wp.array(normals, dtype=wp.vec3),
            color=BEST_ROLLOUT_GHOST_COLOR,
            roughness=1.0,
            metallic=0.0,
            backface_culling=False,
            dynamic=False,
            opacity=sample_opacity * factor,
        )
        object_x, object_y = step_X_last_iter[k, horizon_step, 2 * ndof], step_X_last_iter[k, horizon_step, 2 * ndof + 2]
        object_points, object_indices, object_normals = _object_ghost_mesh(
            float(object_x), float(object_y), object_half_extent, object_half_extent
        )
        viewer.log_mesh(
            f"{slot_names[i]}_object",
            wp.array(object_points, dtype=wp.vec3),
            wp.array(object_indices, dtype=wp.int32),
            normals=wp.array(object_normals, dtype=wp.vec3),
            color=OBJECT_GHOST_COLOR,
            roughness=1.0,
            metallic=0.0,
            backface_culling=False,
            dynamic=False,
            opacity=sample_opacity * factor,
        )
    best_strand_poses = [_rollout_backbone_poses(robot, step_X_last_iter[best_idx, horizon_step], ndof, num_base_coordinates, s_ps)]
    best_indices = tube_mesh_topology(1, len(s_ps), ring_sides)
    best_points, best_normals = tube_mesh_points(best_strand_poses, radii, ring_sides)
    viewer.log_mesh(
        "cloud_best",
        wp.array(best_points, dtype=wp.vec3),
        wp.array(best_indices, dtype=wp.int32),
        normals=wp.array(best_normals, dtype=wp.vec3),
        color=BEST_ROLLOUT_GHOST_COLOR,
        roughness=1.0,
        metallic=0.0,
        dynamic=False,
    )
    best_object_x, best_object_y = step_X_last_iter[best_idx, horizon_step, 2 * ndof], step_X_last_iter[best_idx, horizon_step, 2 * ndof + 2]
    best_object_points, best_object_indices, best_object_normals = _object_ghost_mesh(
        float(best_object_x), float(best_object_y), object_half_extent, object_half_extent
    )
    viewer.log_mesh(
        "cloud_best_object",
        wp.array(best_object_points, dtype=wp.vec3),
        wp.array(best_object_indices, dtype=wp.int32),
        normals=wp.array(best_object_normals, dtype=wp.vec3),
        color=OBJECT_GHOST_COLOR,
        roughness=1.0,
        metallic=0.0,
        dynamic=False,
    )
    return slot_names


def update_cloud_overlay_frame(
    viewer, robot, step_X_last_iter: np.ndarray, step_cost_last_iter: np.ndarray, ndof: int,
    num_base_coordinates: int, s_ps: np.ndarray, radii: np.ndarray, ring_sides: int, horizon_step: int,
    slot_names: list[str], sample_opacity: float, object_half_extent: float,
) -> None:
    """Re-log this step's cloud geometry into the fixed slot names from `build_cloud_overlay_static`.

    Args:
        viewer: RTX viewer to log into.
        robot: SoRoMoX GVS model.
        step_X_last_iter: Last-iteration candidate trajectories, shape (N, horizon+1, state_dim).
        step_cost_last_iter: Per-candidate cost, shape (N,).
        ndof: Robot internal DOF count.
        num_base_coordinates: Floating-base coordinate count.
        s_ps: Arc-length sample points along the backbone, shape (M,).
        radii: Per-sample cross-section radius, shape (M,).
        ring_sides: Cross-section polygon resolution.
        horizon_step: Horizon step to draw each candidate's shape at.
        slot_names: Rod slot mesh names, from `build_cloud_overlay_static`.
        sample_opacity: Opacity of the best-of-rest sample.
        object_half_extent: Half the object's footprint size.
    """
    best_idx = int(np.argmin(step_cost_last_iter))
    cloud_idx = _select_cloud_indices(step_cost_last_iter, best_idx, len(slot_names))
    n = len(slot_names)
    dummy_indices = wp.array(np.zeros(0, dtype=np.int32), dtype=wp.int32)
    for i, (name, k) in enumerate(zip(slot_names, cloud_idx)):
        factor = 1.0 if n <= 1 else 1.0 - 0.9 * (i / (n - 1))
        strand_poses = [_rollout_backbone_poses(robot, step_X_last_iter[k, horizon_step], ndof, num_base_coordinates, s_ps)]
        points, normals = tube_mesh_points(strand_poses, radii, ring_sides)
        viewer.log_mesh(
            name,
            wp.array(points, dtype=wp.vec3),
            dummy_indices,
            normals=wp.array(normals, dtype=wp.vec3),
            color=BEST_ROLLOUT_GHOST_COLOR,
            roughness=1.0,
            metallic=0.0,
            dynamic=False,
            opacity=sample_opacity * factor,
        )
        object_x, object_y = step_X_last_iter[k, horizon_step, 2 * ndof], step_X_last_iter[k, horizon_step, 2 * ndof + 2]
        object_points, _object_indices, object_normals = _object_ghost_mesh(
            float(object_x), float(object_y), object_half_extent, object_half_extent
        )
        viewer.log_mesh(
            f"{name}_object",
            wp.array(object_points, dtype=wp.vec3),
            dummy_indices,
            normals=wp.array(object_normals, dtype=wp.vec3),
            dynamic=False,
        )
    best_strand_poses = [_rollout_backbone_poses(robot, step_X_last_iter[best_idx, horizon_step], ndof, num_base_coordinates, s_ps)]
    best_points, best_normals = tube_mesh_points(best_strand_poses, radii, ring_sides)
    viewer.log_mesh(
        "cloud_best",
        wp.array(best_points, dtype=wp.vec3),
        dummy_indices,
        normals=wp.array(best_normals, dtype=wp.vec3),
        color=BEST_ROLLOUT_GHOST_COLOR,
        roughness=1.0,
        metallic=0.0,
        dynamic=False,
    )
    best_object_x, best_object_y = step_X_last_iter[best_idx, horizon_step, 2 * ndof], step_X_last_iter[best_idx, horizon_step, 2 * ndof + 2]
    best_object_points, _best_object_indices, best_object_normals = _object_ghost_mesh(
        float(best_object_x), float(best_object_y), object_half_extent, object_half_extent
    )
    viewer.log_mesh(
        "cloud_best_object",
        wp.array(best_object_points, dtype=wp.vec3),
        dummy_indices,
        normals=wp.array(best_object_normals, dtype=wp.vec3),
        dynamic=False,
    )


def build_mpc_overlay_static(
    viewer,
    robot,
    step_X_samples: np.ndarray,
    ndof: int,
    num_base_coordinates: int,
    s_ps: np.ndarray,
    radii: np.ndarray,
    ghost_config_count: int,
    ring_sides: int,
    opacity: float,
    min_opacity: float,
) -> list[str]:
    """Register one ghost-tube mesh per ghost config, at their first-step positions.

    Must run before the first begin_frame()/end_frame() cycle; positions are
    refreshed per step by `update_mpc_overlay_frame`.

    Args:
        viewer: RTX viewer to log into.
        robot: SoRoMoX GVS model.
        step_X_samples: Cost-sorted candidate trajectories, shape (N, horizon+1, state_dim).
        ndof: Robot internal DOF count.
        num_base_coordinates: Floating-base coordinate count.
        s_ps: Arc-length sample points along the backbone, shape (M,).
        radii: Per-sample cross-section radius, shape (M,).
        ghost_config_count: Number of ghost configs requested.
        ring_sides: Cross-section polygon resolution.
        opacity: Opacity of the nearest ghost config.
        min_opacity: Opacity of the farthest ghost config.

    Returns:
        Registered ghost mesh names, empty if ghost_config_count is 0.
    """
    strand_poses = _best_rollout_ghost_strand_poses(robot, step_X_samples, ndof, num_base_coordinates, s_ps, ghost_config_count)
    if not strand_poses:
        return []
    mesh_names = [f"mpc_best_rollout_ghost_{k}" for k in range(len(strand_poses))]
    indices = tube_mesh_topology(1, len(s_ps), ring_sides)
    for k, pose in enumerate(strand_poses):
        points, normals = tube_mesh_points([pose], radii, ring_sides)
        viewer.log_mesh(
            mesh_names[k],
            wp.array(points, dtype=wp.vec3),
            wp.array(indices, dtype=wp.int32),
            normals=wp.array(normals, dtype=wp.vec3),
            color=BEST_ROLLOUT_GHOST_COLOR,
            roughness=1.0,
            metallic=0.0,
            backface_culling=False,
            dynamic=False,
            opacity=_ghost_step_opacity(k, opacity, len(strand_poses), min_opacity),
        )
    return mesh_names


def update_mpc_overlay_frame(
    viewer,
    tip_line_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    backbone_line_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    box_line_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    ghost_strands: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    """Re-emit one MPC step's already-computed line/ghost overlay geometry.

    Args:
        viewer: RTX viewer to log into.
        tip_line_data: (starts, ends, colors) for the tip-trajectory overlay, or None if disabled.
        backbone_line_data: (starts, ends, colors) for the backbone overlay, or None if disabled.
        box_line_data: (starts, ends, colors) for the box-trajectory overlay, or None if disabled.
        ghost_strands: Per-ghost-config (points, normals) pairs from `build_mpc_overlay_static`.
    """
    if tip_line_data is not None:
        line_starts, line_ends, line_colors = tip_line_data
        viewer.log_lines(
            "mpc_tip_trajectory_lines",
            wp.array(line_starts, dtype=wp.vec3),
            wp.array(line_ends, dtype=wp.vec3),
            wp.array(line_colors, dtype=wp.vec3),
            width=MPC_ROLLOUT_LINE_WIDTH,
        )
    if backbone_line_data is not None:
        line_starts, line_ends, line_colors = backbone_line_data
        viewer.log_lines(
            "mpc_backbone_lines",
            wp.array(line_starts, dtype=wp.vec3),
            wp.array(line_ends, dtype=wp.vec3),
            wp.array(line_colors, dtype=wp.vec3),
            width=MPC_ROLLOUT_LINE_WIDTH,
        )
    if box_line_data is not None:
        line_starts, line_ends, line_colors = box_line_data
        viewer.log_lines(
            "mpc_box_trajectory_lines",
            wp.array(line_starts, dtype=wp.vec3),
            wp.array(line_ends, dtype=wp.vec3),
            wp.array(line_colors, dtype=wp.vec3),
            width=MPC_ROLLOUT_LINE_WIDTH,
        )

    if not ghost_strands:
        return

    dummy_indices = wp.array(np.zeros(0, dtype=np.int32), dtype=wp.int32)
    for k, (points, normals) in enumerate(ghost_strands):
        viewer.log_mesh(
            f"mpc_best_rollout_ghost_{k}",
            wp.array(points, dtype=wp.vec3),
            dummy_indices,
            normals=wp.array(normals, dtype=wp.vec3),
            dynamic=False,
        )


def build_object_ghost_overlay_static(
    viewer, step_X_samples: np.ndarray, ndof: int, ghost_config_count: int, object_half_extent: float,
    object_z: float, opacity: float, min_opacity: float,
) -> list[str]:
    """Register one solid predicted-object-position mesh per ghost config.

    Args:
        viewer: RTX viewer to log into.
        step_X_samples: Cost-sorted candidate trajectories, shape (N, horizon+1, state_dim).
        ndof: Robot internal DOF count.
        ghost_config_count: Number of ghost configs requested.
        object_half_extent: Half the object's footprint size.
        object_z: World z of the object ghosts.
        opacity: Opacity of the nearest ghost config.
        min_opacity: Opacity of the farthest ghost config.

    Returns:
        Registered mesh names, empty if ghost_config_count is 0.
    """
    centers = _best_rollout_object_ghost_centers(step_X_samples, ndof, ghost_config_count)
    if len(centers) == 0:
        return []
    mesh_names = [f"mpc_best_rollout_object_ghost_{k}" for k in range(len(centers))]
    for k, (cx, cy) in enumerate(centers):
        points, indices, normals = _object_ghost_mesh(float(cx), float(cy), object_z, object_half_extent)
        viewer.log_mesh(
            mesh_names[k],
            wp.array(points, dtype=wp.vec3),
            wp.array(indices, dtype=wp.int32),
            normals=wp.array(normals, dtype=wp.vec3),
            color=OBJECT_GHOST_COLOR,
            roughness=1.0,
            metallic=0.0,
            backface_culling=False,
            dynamic=False,
            opacity=_ghost_step_opacity(k, opacity, len(centers), min_opacity),
        )
    return mesh_names


def update_object_ghost_overlay_frame(viewer, object_ghost_centers: np.ndarray, object_half_extent: float, object_z: float) -> None:
    """Re-emit one MPC step's object-ghost positions.

    Args:
        viewer: RTX viewer to log into.
        object_ghost_centers: Object-center (x, y) per ghost config, from `_best_rollout_object_ghost_centers`.
        object_half_extent: Half the object's footprint size.
        object_z: World z of the object ghosts.
    """
    if len(object_ghost_centers) == 0:
        return
    dummy_indices = wp.array(np.zeros(0, dtype=np.int32), dtype=wp.int32)
    for k, (cx, cy) in enumerate(object_ghost_centers):
        points, _indices, normals = _object_ghost_mesh(float(cx), float(cy), object_z, object_half_extent)
        viewer.log_mesh(
            f"mpc_best_rollout_object_ghost_{k}",
            wp.array(points, dtype=wp.vec3),
            dummy_indices,
            normals=wp.array(normals, dtype=wp.vec3),
            dynamic=False,
        )


def main() -> None:
    """Render a saved run_experiment.py trajectory to an MP4 (and optionally a GIF)."""
    args = parse_args()

    data = np.load(args.run_path, allow_pickle=True)
    if "X_final" not in data.files:
        raise ValueError(f"{args.run_path} has no X_final array (files: {data.files})")
    X_final = np.asarray(data["X_final"])
    if "object_pose" not in data.files:
        raise ValueError(f"{args.run_path} has no object_pose array (files: {data.files})")
    object_pose = np.asarray(data["object_pose"])
    if object_pose.shape != (X_final.shape[0], 7):
        raise ValueError(f"object_pose has shape {object_pose.shape}, expected {(X_final.shape[0], 7)}")
    if args.show_goal_marker:
        if "goal_xy" not in data.files or "reach_tolerance" not in data.files:
            raise ValueError(
                f"{args.run_path} has no goal_xy/reach_tolerance arrays (files: {data.files}) -- "
                "--show-goal-marker needs a run saved by the current iCEM_push_experiment_mpc.py; "
                "rerun the experiment to regenerate the .npz with these logs."
            )
        goal_xy = np.asarray(data["goal_xy"])
        reach_tolerance = np.asarray(data["reach_tolerance"])

    robot_wrapper = FourTendonSoftArm(gravity=[0.0, 0.0, -9.81])
    robot = robot_wrapper.model
    ndof = robot.num_internal_dofs
    num_base_coordinates = robot.num_base_coordinates
    state_dim = 2 * ndof + 4 + num_base_coordinates
    if X_final.shape[1] not in (state_dim, state_dim + 1):
        raise ValueError(
            f"X_final has state_dim={X_final.shape[1]}, expected {state_dim} "
            f"or {state_dim + 1} (object yaw) for this robot"
        )

    run_experiment.OBJECT_SHAPE = args.object_shape

    franka_bodies: list[int] = []

    def add_franka_geometry(builder: newton.ModelBuilder) -> None:
        """Append the decorative arm, record its bodies, and root the scene's jointless ones.

        The push scene carries no joints at all, so SolverMuJoCo converts its
        object and table through the no-articulation path. The arm's URDF adds
        the model's first articulation, which switches that conversion to the
        articulation-aware path, and there every owned body must either belong
        to an articulation or hold a joint to world. Giving the object a free
        joint and the table a fixed one satisfies that without changing the
        render: this model is only ever used for sync_kinematics and direct
        body_q writes, never stepped.
        """
        body_start, body_end, _ = add_franka(builder)
        franka_bodies.extend(range(body_start, body_end))
        builder.add_joint_free(child=_find_label(builder.body_label, "rigid_object"), parent=-1)
        builder.add_joint_fixed(child=_find_label(builder.body_label, "table_surface"), parent=-1)

    model, _, box_bodies, _, coupled_solver, _, _ = (
        run_experiment.build_scene(
            robot,
            args.device,
            world_count=1,
            num_collision_bodies=150,
            tentacle_color=ROD_COLOR,
            object_color=OBJECT_COLOR,
            table_color=(0.34, 0.36, 0.40),
            extra_geometry_fn=add_franka_geometry if args.franka else None,
        )
    )

    franka = None
    if args.franka:
        franka = FrankaDisplay(tuple(args.franka_base), args.device)

    state = model.state()
    q_view, qd_view = gvs_state_views(
        state, world_count=1, coordinate_count=robot.num_coordinates, velocity_count=robot.num_velocities
    )
    box_body = int(box_bodies[0])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = RTXFrameWriter(
        model, args.output.parent / (args.output.stem + "_frames"),
        width=args.width, height=args.height, fps=args.fps, samples_per_pixel=args.samples_per_pixel,
    )

    goal_corners = None
    if args.show_goal_marker:
        goal_corners = build_dynamic_goal_overlay_static(
            writer.viewer, object_pose, goal_xy, reach_tolerance, run_experiment.OBJECT_FOOTPRINT,
            args.goal_tolerance_opacity,
        )

    has_rollouts = False
    s_ps = None
    rollout_X_samples = None
    if args.show_rollouts:
        if "rollout_X_samples" not in data.files:
            raise ValueError(f"--show-rollouts requires 'rollout_X_samples' in --run-path, found only {data.files}")
        has_rollouts = True
        rollout_X_samples = np.asarray(data["rollout_X_samples"])
        if rollout_X_samples.shape[0] != X_final.shape[0] - 1:
            raise ValueError(
                f"rollout_X_samples has {rollout_X_samples.shape[0]} steps, expected "
                f"{X_final.shape[0] - 1} (X_final.shape[0] - 1)"
            )
        s_ps = np.linspace(0.0, float(robot.length), args.backbone_points)
        tip_s = np.array([float(robot.length)])
        ghost_radii = cross_section_radii(robot, s_ps, ndof, num_base_coordinates)
        ghost_mesh_names = build_mpc_overlay_static(
            writer.viewer, robot, rollout_X_samples[0], ndof, num_base_coordinates, s_ps,
            ghost_radii, args.ghost_config_count, args.ghost_ring_sides, args.ghost_opacity,
            args.ghost_min_opacity,
        )
        object_half_extent = 0.5 * run_experiment.OBJECT_FOOTPRINT
        object_ghost_mesh_names = build_object_ghost_overlay_static(
            writer.viewer, rollout_X_samples[0], ndof, args.ghost_config_count,
            object_half_extent, object_half_extent, args.ghost_opacity,
            args.ghost_min_opacity,
        )

        cloud_slot_names = None
        cloud_horizon_step = None
        diagnostics_iter_X_samples = diagnostics_iter_cost = None
        needs_diagnostics = args.show_cloud or args.mpc_line_candidates is not None
        if needs_diagnostics:
            for required in ("diagnostics_iter_X_samples", "diagnostics_iter_cost"):
                if required not in data.files:
                    raise ValueError(
                        f"--show-cloud/--mpc-line-candidates require {required!r} in --run-path, "
                        f"found only {data.files}"
                    )
            diagnostics_iter_X_samples = data["diagnostics_iter_X_samples"]
            diagnostics_iter_cost = data["diagnostics_iter_cost"]
        if args.show_cloud:
            first_step_X_last_iter = np.asarray(diagnostics_iter_X_samples[0, -1])
            first_step_cost_last_iter = np.asarray(diagnostics_iter_cost[0, -1])
            horizon_len = first_step_X_last_iter.shape[1] - 1
            cloud_horizon_step = horizon_len if args.cloud_horizon_step < 0 else args.cloud_horizon_step
            if not (0 <= cloud_horizon_step <= horizon_len):
                raise ValueError(f"--cloud-horizon-step must be in [0, {horizon_len}] (or -1), got {args.cloud_horizon_step}")
            cloud_slot_names = build_cloud_overlay_static(
                writer.viewer, robot, first_step_X_last_iter, first_step_cost_last_iter, ndof,
                num_base_coordinates, s_ps, ghost_radii, args.ghost_ring_sides, args.cloud_max_samples,
                args.cloud_opacity, cloud_horizon_step, object_half_extent,
            )

        cloud_names = []
        if cloud_slot_names:
            cloud_names = (
                list(cloud_slot_names) + [f"{n}_object" for n in cloud_slot_names]
                + ["cloud_best", "cloud_best_object"]
            )
        writer.viewer.shadow_excluded_mesh_names = tuple(
            ghost_mesh_names + object_ghost_mesh_names + cloud_names
        )

    if goal_corners is not None:
        writer.viewer.shadow_excluded_mesh_names = (
            *writer.viewer.shadow_excluded_mesh_names,
            *(f"goal_tolerance_{i}" for i in range(len(goal_corners))),
        )

    body_q = state.body_q.numpy()
    body_qd = state.body_qd.numpy()
    q_full = q_view.numpy()
    qd_full = qd_view.numpy()
    num_base_velocities = robot.num_base_velocities

    def write_frame(
        row: np.ndarray, t: float, rollout_payload, box_pose: np.ndarray, log_overlay: bool = True,
        row_idx: int = 0,
    ) -> None:
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
        body_qd[box_body, 3] = cube_vx
        body_qd[box_body, 4] = cube_vy
        if franka is not None:
            body_q[franka_bodies] = franka.body_poses(base_pose)
        state.body_q.assign(body_q)
        state.body_qd.assign(body_qd)

        coupled_solver.solver("gvs").sync_kinematics(state)

        def extra_log_fn() -> None:
            """Log this frame's MPC/cloud/goal overlays, if enabled."""
            if has_rollouts and log_overlay:
                tip_line_data, backbone_line_data, box_line_data, ghost_strands, object_ghost_centers, row_idx_for_overlay = (
                    rollout_payload
                )
                update_mpc_overlay_frame(writer.viewer, tip_line_data, backbone_line_data, box_line_data, ghost_strands)
                update_object_ghost_overlay_frame(writer.viewer, object_ghost_centers, object_half_extent, object_half_extent)
                if args.show_cloud:
                    step_X_last_iter = np.asarray(diagnostics_iter_X_samples[row_idx_for_overlay, -1])
                    step_cost_last_iter = np.asarray(diagnostics_iter_cost[row_idx_for_overlay, -1])
                    update_cloud_overlay_frame(
                        writer.viewer, robot, step_X_last_iter, step_cost_last_iter, ndof, num_base_coordinates,
                        s_ps, ghost_radii, args.ghost_ring_sides, cloud_horizon_step, cloud_slot_names,
                        args.cloud_opacity, object_half_extent,
                    )
            if args.show_goal_marker and log_overlay:
                update_dynamic_goal_overlay_frame(
                    writer.viewer, goal_xy, reach_tolerance, goal_corners, run_experiment.OBJECT_FOOTPRINT,
                    row_idx,
                )

        writer.write(t, state, extra_log_fn)

    def mpc_line_step_X(row_idx: int) -> np.ndarray:
        """Candidate population to draw tip-trajectory/backbone lines for at this real step.

        Args:
            row_idx: Real MPC step index.

        Returns:
            Candidate trajectories for this step, best-cost-first.
        """
        if args.mpc_line_candidates is None:
            return rollout_X_samples[row_idx]
        step_X_last_iter = np.asarray(diagnostics_iter_X_samples[row_idx, -1])
        step_cost_last_iter = np.asarray(diagnostics_iter_cost[row_idx, -1])
        return _select_best_first_candidates(step_X_last_iter, step_cost_last_iter, args.mpc_line_candidates)

    mpc_box_z = 0.5 * run_experiment.OBJECT_FOOTPRINT

    def compute_mpc_overlay_payload(
        row_idx: int, step_X: np.ndarray,
    ) -> tuple[tuple | None, tuple | None, tuple | None, list[tuple[np.ndarray, np.ndarray]], np.ndarray, int]:
        """Compute one real step's line/ghost overlay data, for `update_mpc_overlay_frame` to re-log each sub-frame.

        Args:
            row_idx: Real MPC step index.
            step_X: Candidate population to draw, from `mpc_line_step_X`.

        Returns:
            (tip_line_data, backbone_line_data, box_line_data, ghost_strands, object_ghost_centers, row_idx).
        """
        tip_line_data = backbone_line_data = box_line_data = None
        if args.show_tip_trajectories:
            segments = _tip_trajectory_segments(robot, step_X, ndof, num_base_coordinates, tip_s)
            tip_line_data = _rank_graded_line_batch(segments)
        if args.show_backbone_lines:
            segments = _candidate_final_backbone_segments(robot, step_X, ndof, num_base_coordinates, s_ps)
            backbone_line_data = _rank_graded_line_batch(segments)
        if args.show_box_trajectories:
            segments = _box_trajectory_segments(step_X, ndof, mpc_box_z)
            box_line_data = _rank_graded_line_batch(segments)

        ghost_strand_poses = _best_rollout_ghost_strand_poses(
            robot, step_X, ndof, num_base_coordinates, s_ps, args.ghost_config_count
        )
        ghost_strands = [tube_mesh_points([pose], ghost_radii, args.ghost_ring_sides) for pose in ghost_strand_poses]
        object_ghost_centers = _best_rollout_object_ghost_centers(step_X, ndof, args.ghost_config_count)
        return tip_line_data, backbone_line_data, box_line_data, ghost_strands, object_ghost_centers, row_idx

    try:
        num_rows = X_final.shape[0] if args.max_steps is None else min(X_final.shape[0], args.max_steps + 1)
        for row_idx in range(num_rows - 1):
            row_start = X_final[row_idx]
            row_end = X_final[row_idx + 1]
            box_pose_start = object_pose[row_idx]
            box_pose_end = object_pose[row_idx + 1]
            if has_rollouts:
                step_rollout_payload = compute_mpc_overlay_payload(row_idx, mpc_line_step_X(row_idx))
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
                rollout_payload = step_rollout_payload if has_rollouts else None
                write_frame(row, t, rollout_payload, box_pose, log_overlay=(sub == 0), row_idx=row_idx)
            print(f"  rendered control step {row_idx + 1}/{num_rows - 1}")
        final_rollout_payload = (
            compute_mpc_overlay_payload(num_rows - 2, mpc_line_step_X(num_rows - 2)) if has_rollouts else None
        )
        write_frame(
            X_final[num_rows - 1], (num_rows - 1) * args.control_dt, final_rollout_payload,
            object_pose[num_rows - 1], row_idx=num_rows - 1,
        )
    finally:
        writer.close()

    if franka is not None:
        print(f"decorative Franka: peak end-effector tracking error {1e3 * franka.maximum_error:.1f} mm")

    if args.max_steps is not None:
        last_frame = writer.frames_dir / f"frame_{writer.frame_index - 1:04d}.png"
        print(f"--max-steps set: skipped MP4/GIF encoding. {writer.frame_index} frames written to "
              f"{writer.frames_dir} -- check e.g. {last_frame}")
        return

    print(f"encoding MP4 from {writer.frame_index} frames...")
    encode_mp4(writer.frames_dir, args.output, args.fps)
    print(f"wrote {args.output}")

    if args.gif:
        gif_path = args.output.with_suffix(".gif")
        encode_gif(writer.frames_dir, gif_path, args.fps)
        print(f"wrote {gif_path}")


if __name__ == "__main__":
    main()
