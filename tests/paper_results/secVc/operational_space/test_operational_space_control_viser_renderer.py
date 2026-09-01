import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

MODULE_DIR = (
    Path(__file__).resolve().parents[4]
    / "paper_results"
    / "secVc_model_based_control"
    / "operational_space_impedance_control"
    / "code"
)
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

RENDERER_PATH = MODULE_DIR / "viser_operational_space_renderer.py"
spec = importlib.util.spec_from_file_location(
    "example_operational_space_viser_renderer",
    RENDERER_PATH,
)
assert spec is not None
assert spec.loader is not None
renderer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = renderer
spec.loader.exec_module(renderer)


def test_render_entrypoint_uses_geometry_neutral_resolution_option(tmp_path):
    captured = {}

    class FakeRenderer:
        def __init__(self, robot, **kwargs):
            captured["robot"] = robot
            captured["init"] = kwargs

        def render_sequence(self, **kwargs):
            captured["render"] = kwargs

        def stop(self):
            captured["stopped"] = True

    robot = object()
    osd = SimpleNamespace(
        n_orientation_dim=4,
        is_planar=False,
        rotation_representation=renderer.RotationRepresentation.QUATERNION,
    )
    trajectory_config = SimpleNamespace(
        tracking="position",
        orientation_primitive="none",
        surface="none",
        period=1.0,
    )
    t = np.linspace(0.0, 1.0, 5)
    pose = np.column_stack(
        (
            np.ones_like(t),
            np.zeros((len(t), 3)),
            0.01 * t,
            np.zeros((len(t), 2)),
        )
    )

    renderer.render_operational_space_tracking(
        robot=robot,
        osd=osd,
        trajectory_config=trajectory_config,
        t_traj=t,
        q_traj=np.zeros((len(t), 1)),
        x_traj_full=pose,
        x_des_traj_full=pose,
        video_output=tmp_path / "tracking.mp4",
        snapshot_paths={},
        open_browser=False,
        renderer_cls=FakeRenderer,
    )

    assert captured["robot"] is robot
    assert captured["init"]["cross_section_resolution"] == 64
    assert "cylinder_sections" not in captured["init"]
    assert captured["stopped"]


def test_paper_robot_color_config_is_opaque_coral_with_dark_base():
    config = renderer.make_paper_robot_color_config(num_points=7)
    colors = np.asarray(config.backbone.point_colors)

    assert colors.shape == (7, 4)
    assert np.allclose(colors[:, :3], renderer.CORAL_RGB)
    assert np.allclose(colors[:, 3], 1.0)
    assert np.allclose(config.base_plate_color, renderer.BASE_RGB)


def test_target_disk_geometry_lies_in_local_yz_plane():
    radius = 0.25
    geometry = renderer.make_target_disk_geometry(radius, num_points=16)

    assert geometry.vertices.shape == (17, 3)
    assert geometry.faces.shape == (16, 3)
    assert geometry.rim_segments.shape == (16, 2, 3)
    assert geometry.cross_segments.shape == (2, 2, 3)
    assert np.allclose(geometry.vertices[:, 0], 0.0)
    assert np.allclose(
        np.linalg.norm(geometry.vertices[1:, 1:], axis=1),
        radius,
    )


def test_sphere_surface_mesh_is_closed_and_has_requested_radius():
    center = np.array([0.1, -0.2, 0.3])
    radius = 0.08

    sphere = renderer.make_sphere_surface_mesh(
        center,
        radius,
        longitude_count=12,
        latitude_count=7,
    )

    assert sphere.vertices.shape == (62, 3)
    assert sphere.faces.shape == (120, 3)
    assert np.all(np.isfinite(sphere.vertices))
    assert np.allclose(np.linalg.norm(sphere.vertices - center, axis=1), radius)
    assert sphere.faces.min() == 0
    assert sphere.faces.max() == len(sphere.vertices) - 1


def test_transformed_disk_segments_follow_desired_tangent_axes():
    geometry = renderer.make_target_disk_geometry(0.2, num_points=16)
    position = np.array([1.0, 2.0, 3.0])
    rotation = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )

    transformed = renderer.transform_segments(
        geometry.cross_segments,
        position,
        rotation,
    )

    assert np.allclose(transformed.mean(axis=1), position)
    directions = transformed[:, 1] - transformed[:, 0]
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    assert np.allclose(np.abs(directions @ rotation[:, 0]), 0.0, atol=1e-7)
    assert np.allclose(np.abs(directions @ rotation[:, 1:]), np.eye(2), atol=1e-7)


@pytest.mark.parametrize(
    ("representation", "orientation"),
    [
        (
            renderer.RotationRepresentation.QUATERNION,
            [0.0, 0.0, 2**-0.5, 2**-0.5],
        ),
        (
            renderer.RotationRepresentation.ROTATION_VECTOR,
            [0.0, 0.0, np.pi / 2.0],
        ),
    ],
)
def test_pose_quaternions_are_reordered_for_viser_wxyz(representation, orientation):
    osd = type(
        "OperationalSpaceDescription",
        (),
        {
            "is_planar": False,
            "n_orientation_dim": len(orientation),
            "rotation_representation": representation,
        },
    )()
    pose = np.asarray([*orientation, 0.0, 0.0, 0.0])[None, :]

    wxyz = renderer._poses_to_quaternions(osd, pose)

    assert np.allclose(wxyz, [[2**-0.5, 0.0, 0.0, 2**-0.5]])
    assert np.allclose(
        renderer._quaternion_to_rotation_matrix(wxyz[0]),
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-12,
    )


def test_reference_beads_use_one_period_and_near_uniform_spacing():
    t = np.linspace(0.0, 8.0, 801)
    phase = 2.0 * np.pi * t / 4.0
    positions = np.column_stack(
        [0.05 * np.sin(phase), 0.03 * np.sin(2.0 * phase), 0.2 + 0.0 * phase]
    )

    beads = renderer.resample_periodic_path_by_arclength(
        t,
        positions,
        period=4.0,
        spacing=0.01,
    )
    distances = np.linalg.norm(np.diff(beads, axis=0), axis=1)

    assert len(beads) > 20
    assert np.median(distances) == pytest.approx(0.01, rel=0.03)
    assert np.max(distances) < 0.011
    assert not np.allclose(beads[0], beads[-1])


def test_camera_uses_elevated_view_and_includes_sphere_bounds():
    reference_positions = np.array(
        [[-0.07, -0.03, 0.16], [0.07, 0.03, 0.20]], dtype=np.float64
    )
    actual_positions = reference_positions + np.array([0.001, -0.001, 0.0])

    sphere = renderer.make_sphere_surface_mesh(
        np.array([0.0, 0.0, 0.12]),
        0.08,
        longitude_count=12,
        latitude_count=7,
    )
    camera = renderer.make_operational_space_camera_config(
        reference_positions,
        actual_positions,
        robot_radius=0.02,
        sphere_surface=sphere,
    )
    direction = np.asarray(camera.position) - np.asarray(camera.look_at)
    direction /= np.linalg.norm(direction)
    expected = np.array(
        [
            np.cos(np.deg2rad(55.0)) * np.cos(np.deg2rad(270.0)),
            np.cos(np.deg2rad(55.0)) * np.sin(np.deg2rad(270.0)),
            np.sin(np.deg2rad(55.0)),
        ]
    )

    assert camera.fov == 60.0
    assert np.all(np.isfinite(camera.position))
    assert np.all(np.isfinite(camera.look_at))
    assert np.allclose(direction, expected)
    assert np.asarray(camera.look_at)[0] == pytest.approx(0.0, abs=1e-12)


def test_canonical_snapshot_times_map_to_expected_frames():
    t = np.linspace(0.0, 10.0, 1001)

    indices = renderer.snapshot_indices_for_times(
        t,
        renderer.DEFAULT_SNAPSHOT_TIMES,
    )

    assert indices == (450, 550, 650, 750)


@pytest.mark.parametrize(
    "times",
    [
        (1.0, 2.0, 3.0),
        (1.0, 1.0, 2.0, 3.0),
        (-1.0, 1.0, 2.0, 3.0),
    ],
)
def test_snapshot_times_must_be_four_distinct_in_range_values(times):
    with pytest.raises(ValueError):
        renderer.snapshot_indices_for_times(np.linspace(0.0, 10.0, 101), times)


def test_common_snapshot_crop_is_square_and_identical(tmp_path):
    paths = []
    boxes = [(35, 20, 90, 95), (45, 15, 100, 90), (30, 25, 85, 100), (40, 18, 95, 98)]
    for index, box in enumerate(boxes):
        path = tmp_path / f"frame_{index}.png"
        image = Image.new("RGB", (160, 120), "white")
        pixels = np.asarray(image).copy()
        left, top, right, bottom = box
        pixels[top:bottom, left:right] = np.array([241, 85, 46], dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        paths.append(path)

    crop_box = renderer.crop_snapshots_to_common_square(paths)

    assert crop_box[2] - crop_box[0] == crop_box[3] - crop_box[1]
    sizes = {Image.open(path).size for path in paths}
    assert len(sizes) == 1
    width, height = sizes.pop()
    assert width == height


def test_snapshot_filename_preserves_png_extension():
    assert renderer.snapshot_filename(4.5) == "snapshot_t04p50s.png"


def test_snapshot_strip_has_exact_publication_page_size(tmp_path):
    paths = []
    coral_u8 = tuple(int(value) for value in np.rint(renderer.CORAL_RGB * 255.0))
    for index in range(4):
        path = tmp_path / f"frame_{index}.png"
        Image.new("RGB", (32, 32), coral_u8).save(path)
        paths.append(path)
    output = tmp_path / "strip.pdf"

    renderer.save_snapshot_strip(
        paths,
        output,
        snapshot_times=(4.5, 5.5, 6.5, 7.5),
    )

    match = re.search(
        rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]",
        output.read_bytes(),
    )
    assert match is not None
    width_points, height_points = (float(value) for value in match.groups())
    assert width_points == pytest.approx(15.2 / 2.54 * 72.0, abs=0.01)
    assert height_points == pytest.approx(4.8 / 2.54 * 72.0, abs=0.01)


def test_snapshot_strip_requires_four_finite_times(tmp_path):
    paths = []
    for index in range(4):
        path = tmp_path / f"frame_{index}.png"
        Image.new("RGB", (32, 32), "white").save(path)
        paths.append(path)

    with pytest.raises(ValueError, match="four finite snapshot times"):
        renderer.save_snapshot_strip(
            paths,
            tmp_path / "strip.pdf",
            snapshot_times=(4.5, 5.5, 6.5),
        )
