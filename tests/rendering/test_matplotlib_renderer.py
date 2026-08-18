from inspect import signature

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from soromox.rendering.actuators import ActuatorVisualLayer
from soromox.rendering.camera_config import CameraConfig
from soromox.rendering.matplotlib_renderer import MatplotlibRenderer
from soromox.systems.components import CrossSectionGeometry
from soromox.utils.geometry import poses


class DummySpatialRobot:
    is_planar = False
    length = jnp.array(1.0)
    segment_length = jnp.array([1.0])

    def __init__(self, base_pose: jnp.ndarray):
        self.base_pose = jnp.asarray(base_pose)
        self.base_transform = poses.quaternion_pose_to_transform(self.base_pose)

    def forward_kinematics_batched(self, q, s_points):
        del q
        transforms = jnp.repeat(
            self.base_transform[None, :, :], s_points.shape[0], axis=0
        )
        offsets = s_points[:, None] * self.base_transform[:3, 0]
        return transforms.at[:, :3, 3].add(offsets)

    def cross_section_geometry(self, q, s):
        del q, s
        return CrossSectionGeometry.CIRCULAR, jnp.array([0.02])


class DummyActuatedSpatialRobot(DummySpatialRobot):
    def actuator_visual_layers(self, q, s_points, *, actuator_inputs=None):
        del q, actuator_inputs
        tendon = jnp.stack(
            [
                s_points,
                jnp.full_like(s_points, 0.1),
                jnp.zeros_like(s_points),
            ],
            axis=1,
        )[None, ...]
        return (
            ActuatorVisualLayer(
                name="sampled_tendon",
                kind="tendon",
                points=tendon,
            ),
        )


class FakeMatplotlib3DAxis:
    def __init__(self):
        self.view = None
        self.projection = None
        self.box_aspect = None

    def set_xlim(self, *args):
        return None

    def set_ylim(self, *args):
        return None

    def set_zlim(self, *args):
        return None

    def set_xlabel(self, *args):
        return None

    def set_ylabel(self, *args):
        return None

    def set_zlabel(self, *args):
        return None

    def set_proj_type(self, projection, *, focal_length):
        self.projection = (projection, float(focal_length))

    def set_box_aspect(self, aspect, *, zoom):
        self.box_aspect = (tuple(aspect), float(zoom))

    def view_init(self, *, elev, azim):
        self.view = (float(elev), float(azim))


class LegacyFakeMatplotlib3DAxis(FakeMatplotlib3DAxis):
    def set_proj_type(self, projection):
        self.projection = projection

    def set_box_aspect(self, aspect):
        self.box_aspect = tuple(aspect)


def test_ground_plane_arguments_follow_color_configuration():
    parameters = list(signature(MatplotlibRenderer).parameters)
    color_config_index = parameters.index("color_config")

    assert parameters[color_config_index + 1 : color_config_index + 3] == [
        "show_ground_plane",
        "ground_plane_size",
    ]


def test_3d_default_camera_uses_base_pose_orientation():
    robot = DummySpatialRobot(jnp.array([0.5, 0.5, -0.5, 0.5, 0.0, 0.0, 0.0]))
    renderer = MatplotlibRenderer(robot)
    axis = FakeMatplotlib3DAxis()

    renderer._setup_axes(
        axis,
        width_m=3.0,
        scene_center=np.zeros(3),
        scene_extent=1.0,
        camera_config=None,
    )

    assert axis.view is not None
    expected_view = np.array(
        [
            np.degrees(np.arctan2(8.0, np.hypot(8.0, -5.0))),
            np.degrees(np.arctan2(-5.0, 8.0)),
        ]
    )
    assert_allclose(np.array(axis.view), expected_view, atol=1e-12)
    assert axis.projection is not None
    assert axis.projection[0] == "persp"
    assert axis.projection[1] == pytest.approx(
        1.0 / np.tan(np.deg2rad(CameraConfig().fov) / 2.0)
    )
    assert axis.box_aspect == ((1.0, 1.0, 1.0), 1.0)


def test_3d_camera_supports_matplotlib_without_focal_length_or_zoom_keywords():
    robot = DummySpatialRobot(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    renderer = MatplotlibRenderer(robot)
    axis = LegacyFakeMatplotlib3DAxis()

    renderer._setup_axes(
        axis,
        width_m=3.0,
        scene_center=np.zeros(3),
        scene_extent=1.0,
        camera_config=CameraConfig(fov=45.0),
    )

    assert axis.projection == "persp"
    assert axis.box_aspect == (1.0, 1.0, 1.0)


def test_3d_camera_uses_shared_field_of_view():
    robot = DummySpatialRobot(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    renderer = MatplotlibRenderer(robot)
    axis = FakeMatplotlib3DAxis()

    renderer._setup_axes(
        axis,
        width_m=3.0,
        scene_center=np.zeros(3),
        scene_extent=1.0,
        camera_config=CameraConfig(fov=45.0),
    )

    assert axis.projection == (
        "persp",
        pytest.approx(1.0 / np.tan(np.deg2rad(45.0) / 2.0)),
    )
    assert axis.box_aspect == ((1.0, 1.0, 1.0), 1.2)


def test_actuator_overlay_changes_rendered_frame():
    robot = DummyActuatedSpatialRobot(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    renderer = MatplotlibRenderer(robot, width=240, height=180, num_points=8)

    without_actuators = renderer.render_frame(jnp.array([]), render_actuators=False)
    with_actuators = renderer.render_frame(jnp.array([]), render_actuators=True)

    assert without_actuators.shape == with_actuators.shape
    assert np.any(without_actuators != with_actuators)


def test_ground_plane_changes_spatial_rendered_frame():
    robot = DummySpatialRobot(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    with_ground = MatplotlibRenderer(
        robot,
        width=240,
        height=180,
        num_points=8,
        show_ground_plane=True,
    ).render_frame(jnp.array([]))
    without_ground = MatplotlibRenderer(
        robot,
        width=240,
        height=180,
        num_points=8,
        show_ground_plane=False,
    ).render_frame(jnp.array([]))

    assert with_ground.shape == without_ground.shape
    assert np.any(with_ground != without_ground)


def test_2d_base_marker_is_perpendicular_to_base_tangent():
    base = np.array([1.0, 2.0])
    tangent = np.array([0.0, 1.0])

    marker = MatplotlibRenderer._base_plate_marker_points(
        base, tangent, marker_diameter=0.4, dim=2
    )

    assert marker.shape == (2, 2)
    assert_allclose(marker.mean(axis=0), base)
    assert_allclose(marker[1] - marker[0], np.array([-0.4, 0.0]))
    assert_allclose(np.dot(marker[1] - marker[0], tangent), 0.0, atol=1e-12)


def test_3d_base_marker_lies_in_plate_face_plane():
    base = np.array([1.0, 2.0, 3.0])
    normal = np.array([1.0, 0.0, 0.0])

    marker = MatplotlibRenderer._base_plate_marker_points(
        base, normal, marker_diameter=0.4, dim=3
    )

    assert marker.shape == (65, 3)
    assert_allclose(marker[0], marker[-1], atol=1e-12)
    assert_allclose((marker - base) @ normal, np.zeros(marker.shape[0]), atol=1e-12)
    assert_allclose(np.linalg.norm(marker - base, axis=1), 0.2, atol=1e-12)
