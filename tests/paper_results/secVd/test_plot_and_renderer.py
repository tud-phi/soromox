"""Plot and Viser-renderer regressions for Section Vd."""

import sys
from pathlib import Path

import jax
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

SECTION_DIR = (
    Path(__file__).resolve().parents[3]
    / "paper_results"
    / "secVd_control_gain_optimization"
)
CODE_DIR = SECTION_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import plot_control_gain_optimization as plotter  # noqa: E402
import render_control_gain_optimization_animations as renderer  # noqa: E402
from secvd_case import build_sec_vd_robot  # noqa: E402
from secvd_results import load_results  # noqa: E402

jax.config.update("jax_enable_x64", True)


def test_plot_loss_accepts_single_start_without_band():
    figure, axis = matplotlib.pyplot.subplots()
    mask = np.ones((5, 1), dtype=bool)
    plotter.plot_loss(axis, np.arange(1.0, 6.0)[:, None], mask, "A")
    assert len(axis.lines) == 1
    assert len(axis.collections) == 0
    matplotlib.pyplot.close(figure)


def test_plot_loss_bands_a_multi_start_run():
    figure, axis = matplotlib.pyplot.subplots()
    loss = np.stack([np.arange(1.0, 6.0), np.arange(3.0, 8.0)], axis=1)
    plotter.plot_loss(axis, loss, np.ones((5, 2), dtype=bool), "A")
    assert len(axis.collections) == 1, "expected a min-max band across starts"
    np.testing.assert_allclose(axis.lines[0].get_ydata(), np.arange(1.0, 6.0))
    matplotlib.pyplot.close(figure)


def test_plot_loss_excludes_frozen_starts_from_the_band():
    """A frozen start must drop out of the band, not appear as a gap."""
    figure, axis = matplotlib.pyplot.subplots()
    loss = np.stack([np.arange(1.0, 6.0), np.full(5, np.nan)], axis=1)
    mask = np.ones((5, 2), dtype=bool)
    mask[:, 1] = False
    plotter.plot_loss(axis, loss, mask, "A")
    np.testing.assert_allclose(axis.lines[0].get_ydata(), np.arange(1.0, 6.0))
    matplotlib.pyplot.close(figure)


def test_nonpositive_losses_are_safe_on_the_log_axis():
    figure, axis = matplotlib.pyplot.subplots()
    loss = np.array([[0.0, 4.0], [-1.0, 2.0], [0.0, 0.0]])
    mask = np.ones_like(loss, dtype=bool)
    mask[1, 0] = False

    plotter.plot_loss(axis, loss, mask, "A")

    plotted = axis.lines[0].get_ydata()
    assert np.all(plotted > 0.0)
    assert plotter.loss_extent(loss, mask) == (2.0, 4.0)
    assert plotter.shared_loss_limits((0.0, 4.0)) is None
    matplotlib.pyplot.close(figure)


def test_an_all_nonpositive_loss_falls_back_to_a_positive_plot_floor():
    figure, axis = matplotlib.pyplot.subplots()
    loss = np.zeros((3, 1))
    mask = np.ones_like(loss, dtype=bool)

    plotter.plot_loss(axis, loss, mask, "A")

    assert np.all(axis.lines[0].get_ydata() > 0.0)
    assert plotter.loss_extent(loss, mask) is None
    matplotlib.pyplot.close(figure)


@pytest.fixture
def stub_archives(monkeypatch):
    """Serve build_figure a minimal pair of archives instead of reading data/."""
    t = np.linspace(0, 1, 4)
    common = {
        "history_finite_mask": np.ones((5, 1), dtype=bool),
        "best_batch": np.asarray(0),
        "t_ts": t,
    }
    # Deliberately different decades: the shared loss limit then covers both and
    # is strictly wider than either panel's own range.
    collocated = {
        **common,
        "history_loss": np.geomspace(2e-3, 1e-3, 5)[:, None],
        "q_ts_init": np.zeros((1, 4, 6)),
        "q_ts_best": np.zeros((1, 4, 6)),
        "q_des_ts": np.zeros((4, 6)),
    }
    synergy_pose = np.zeros((1, 4, 6))
    synergy_pose[..., :3] = 99.0
    synergy_pose[..., 3:6] = [1.0, 2.0, 3.0]
    synergistic = {
        **common,
        "history_loss": np.geomspace(1e-1, 1e-4, 5)[:, None],
        "x_ts_init": synergy_pose,
        "x_ts_best": synergy_pose,
        "x_des_ts": synergy_pose[0],
    }
    monkeypatch.setattr(
        plotter,
        "load_results",
        lambda path, expected_method: (
            collocated if expected_method == "collocated" else synergistic
        ),
    )


def test_plotter_uses_synergistic_position_channels(stub_archives):
    figure = plotter.build_figure(Path("unused"))
    plotted = figure.axes[3].lines[1].get_ydata()
    np.testing.assert_allclose(plotted, 1.0)
    matplotlib.pyplot.close(figure)


def test_the_loss_panels_share_one_axis_by_default(stub_archives):
    figure = plotter.build_figure(Path("unused"))
    assert figure.axes[0].get_ylim() == figure.axes[2].get_ylim()
    matplotlib.pyplot.close(figure)


def test_an_empty_request_autoscales_one_loss_panel_off_the_shared_axis(
    stub_archives,
):
    """The whole point of the empty form: panel A stops paying for panel C."""
    shared = plotter.build_figure(Path("unused"))
    shared_low, shared_high = shared.axes[0].get_ylim()
    shared_c = shared.axes[2].get_ylim()
    matplotlib.pyplot.close(shared)

    figure = plotter.build_figure(Path("unused"), ylim_a=())
    low, high = figure.axes[0].get_ylim()
    assert (low, high) != (shared_low, shared_high)
    assert shared_low < low and high < shared_high, (
        "autoscaling A to its own data must tighten it inside the shared axis"
    )
    assert figure.axes[2].get_ylim() == shared_c, (
        "autoscaling A must leave panel C on the shared limit"
    )
    matplotlib.pyplot.close(figure)


@pytest.mark.parametrize(("panel", "axis"), [("a", 0), ("b", 1), ("c", 2), ("d", 3)])
def test_every_panel_honours_an_explicit_limit(panel, axis, stub_archives):
    figure = plotter.build_figure(Path("unused"), **{f"ylim_{panel}": (2e-4, 8e-2)})
    np.testing.assert_allclose(figure.axes[axis].get_ylim(), (2e-4, 8e-2))
    matplotlib.pyplot.close(figure)


@pytest.mark.parametrize("panel", ["a", "b", "c", "d"])
@pytest.mark.parametrize(
    ("values", "message"),
    [
        (["3", "1"], "needs LOW < HIGH"),
        (["1"], "takes no values"),
        (["1", "2", "3"], "takes no values"),
    ],
)
def test_a_malformed_manual_limit_is_rejected(panel, values, message, monkeypatch):
    argv = ["plot_control_gain_optimization.py", f"--ylim-{panel}", *values]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match=f"--ylim-{panel} {message}"):
        plotter.parse_args()


@pytest.mark.parametrize("panel", ["a", "c"])
def test_log_loss_limits_must_be_positive(panel, monkeypatch):
    argv = ["plot_control_gain_optimization.py", f"--ylim-{panel}", "0", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="positive limits for a log axis"):
        plotter.parse_args()


@pytest.mark.parametrize("panel", ["a", "b", "c", "d"])
def test_an_empty_flag_parses_as_a_request_to_autoscale(panel, monkeypatch):
    argv = ["plot_control_gain_optimization.py", f"--ylim-{panel}"]
    monkeypatch.setattr(sys, "argv", argv)
    args = plotter.parse_args()
    assert getattr(args, f"ylim_{panel}") == ()
    assert all(
        getattr(args, f"ylim_{other}") is None
        for other in plotter.PANELS
        if other != panel
    ), "an empty flag must not disturb the other panels"


def test_resampling_matches_requested_fps():
    t = np.linspace(0, 1, 101)
    q = np.column_stack([t, 2 * t])
    pose = np.column_stack([t] * 6)
    frame_t, frame_q, frame_pose = renderer.resample_trajectory(t, q, pose, fps=10)
    assert len(frame_t) == 11
    np.testing.assert_allclose(frame_q[:, 0], frame_t)
    np.testing.assert_allclose(frame_pose[:, 5], frame_t)


def test_desired_shape_wireframe_is_finite_and_three_dimensional():
    z = np.linspace(0.0, 0.1, 12)
    curve = np.column_stack([0.01 * np.sin(8 * z), np.zeros_like(z), z])
    segments = renderer.make_tube_wire_segments(curve, radius=0.02)
    assert segments.ndim == 3
    assert segments.shape[1:] == (2, 3)
    assert np.all(np.isfinite(segments))


def test_method_selection_and_overwrite(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        renderer,
        "render_method",
        lambda **kwargs: (
            calls.append(kwargs["name"]) or [tmp_path / f"{kwargs['name']}.mp4"]
        ),
    )
    outputs = renderer.render_saved_results(
        data_dir=tmp_path,
        output_dir=tmp_path,
        method="synergistic",
        renderer_factory=object,
    )
    assert calls == ["synergistic"]
    assert outputs == [tmp_path / "synergistic.mp4"]

    (tmp_path / "collocated_tracking.mp4").touch()
    with pytest.raises(FileExistsError, match="--force"):
        renderer.render_saved_results(
            data_dir=tmp_path,
            output_dir=tmp_path,
            method="both",
            renderer_factory=object,
        )
    assert calls == ["synergistic"], "preflight must fail before either render starts"

    (tmp_path / "collocated_tracking.mp4").unlink()
    (tmp_path / "collocated_initial_median_tracking.mp4").touch()
    with pytest.raises(FileExistsError, match="--force"):
        renderer.render_saved_results(
            data_dir=tmp_path,
            output_dir=tmp_path,
            method="collocated",
            trajectory="initial-median",
            renderer_factory=object,
        )


def test_mock_viser_receives_robot_trail_target_and_paths(monkeypatch, tmp_path):
    timesteps = 5
    data = {
        "t_ts": np.linspace(0, 1, timesteps),
        "q_ts_best": np.zeros((1, timesteps, 6)),
        "x_ts_best": np.zeros((1, timesteps, 6)),
        "best_batch": np.asarray(0),
        "x_des_ts": np.zeros((timesteps, 6)),
    }
    monkeypatch.setattr(renderer, "load_results", lambda *_args, **_kwargs: data)
    monkeypatch.setattr(renderer, "build_sec_vd_robot", lambda: (object(), None, 0.1))
    monkeypatch.setattr(renderer, "validate_pose_consistency", lambda *_args: None)
    captured = {}

    class FakeRenderer:
        def __init__(self, robot, **kwargs):
            captured["init"] = (robot, kwargs)

        def render_sequence(self, **kwargs):
            captured["render"] = kwargs

    outputs = renderer.render_method(
        name="synergistic",
        data_dir=tmp_path,
        output_dir=tmp_path,
        fps=4,
        make_gif=False,
        force=False,
        port=8080,
        open_browser=False,
        record_client_timeout=2.0,
        record_frame_timeout=3.0,
        renderer_factory=FakeRenderer,
    )
    assert outputs == [tmp_path / "synergistic_tracking.mp4"]
    assert captured["render"]["q_ts"].shape == (5, 6)
    assert captured["init"][1]["desired_q_ts"] is None
    assert captured["init"][1]["cross_section_resolution"] == 64
    assert "cylinder_sections" not in captured["init"][1]
    backbone_colors = captured["init"][1]["color_config"].backbone.point_colors
    np.testing.assert_allclose(
        backbone_colors[:, 3], renderer.SYNERGISTIC_ROBOT_OPACITY
    )
    assert captured["render"]["dynamic_spheres_positions"].shape == (2, 5, 3)
    np.testing.assert_allclose(
        captured["render"]["dynamic_spheres_radii"],
        [renderer.MARKER_RADIUS, renderer.TARGET_MARKER_RADIUS],
    )
    np.testing.assert_allclose(
        captured["render"]["dynamic_spheres_colors"],
        np.stack([renderer.CURRENT_POSITION_COLOR, renderer.TARGET_COLOR]),
    )
    assert captured["render"]["static_spheres_positions"] is None
    assert captured["render"]["record_path"].endswith("synergistic_tracking.mp4")

    outputs[0].touch()
    with pytest.raises(FileExistsError, match="--force"):
        renderer.render_method(
            name="synergistic",
            data_dir=tmp_path,
            output_dir=tmp_path,
            fps=4,
            make_gif=False,
            force=False,
            port=8080,
            open_browser=False,
            record_client_timeout=2.0,
            record_frame_timeout=3.0,
            renderer_factory=FakeRenderer,
        )


def test_initial_median_renders_the_representative_initial_rollout(
    monkeypatch, tmp_path
):
    timesteps = 5
    q_init = np.stack(
        [np.full((timesteps, 6), value, dtype=float) for value in (1.0, 2.0, 3.0)]
    )
    x_init = np.stack(
        [np.full((timesteps, 6), value, dtype=float) for value in (4.0, 5.0, 6.0)]
    )
    data = {
        "t_ts": np.linspace(0, 1, timesteps),
        "history_loss": np.array([[9.0, 2.0, 5.0]]),
        "history_finite_mask": np.ones((1, 3), dtype=bool),
        "q_ts_init": q_init,
        "x_ts_init": x_init,
        "q_ts_best": np.zeros_like(q_init),
        "x_ts_best": np.zeros_like(x_init),
        "best_batch": np.asarray(0),
        "x_des_ts": np.zeros((timesteps, 6)),
    }
    monkeypatch.setattr(renderer, "load_results", lambda *_args, **_kwargs: data)
    monkeypatch.setattr(renderer, "build_sec_vd_robot", lambda: (object(), None, 0.1))
    monkeypatch.setattr(renderer, "validate_pose_consistency", lambda *_args: None)
    captured = {}

    class FakeRenderer:
        def __init__(self, _robot, **kwargs):
            captured["init"] = kwargs

        def render_sequence(self, **kwargs):
            captured["render"] = kwargs

    outputs = renderer.render_method(
        name="synergistic",
        data_dir=tmp_path,
        output_dir=tmp_path,
        fps=4,
        make_gif=False,
        force=False,
        port=8080,
        open_browser=False,
        record_client_timeout=2.0,
        record_frame_timeout=3.0,
        trajectory="initial-median",
        renderer_factory=FakeRenderer,
    )

    assert outputs == [tmp_path / "synergistic_initial_median_tracking.mp4"]
    np.testing.assert_allclose(captured["render"]["q_ts"], q_init[2])
    assert captured["render"]["robot_name"].endswith("initial median")


@pytest.mark.parametrize(
    ("loss", "expected"),
    [
        ([7.0, 5.0, 9.0, 3.0], 1),
        ([5.0, 5.0, 9.0], 0),
    ],
)
def test_initial_median_ties_prefer_lower_loss_then_batch_index(loss, expected):
    data = {
        "history_loss": np.array([loss]),
        "history_finite_mask": np.ones((1, len(loss)), dtype=bool),
    }
    assert renderer.initial_median_batch(data) == expected


def test_initial_median_requires_a_finite_initial_loss():
    data = {
        "history_loss": np.array([[np.nan, 2.0]]),
        "history_finite_mask": np.array([[False, False]]),
    }
    with pytest.raises(ValueError, match="finite initial loss"):
        renderer.initial_median_batch(data)


def test_synergistic_trail_contains_only_time_varying_reference(monkeypatch, tmp_path):
    timesteps = 5
    reference_pose = np.zeros((timesteps, 6))
    reference_pose[:, 3] = np.linspace(0.0, 0.04, timesteps)
    data = {
        "t_ts": np.linspace(0, 1, timesteps),
        "q_ts_best": np.zeros((1, timesteps, 6)),
        "x_ts_best": np.zeros((1, timesteps, 6)),
        "best_batch": np.asarray(0),
        "x_des_ts": reference_pose,
    }
    monkeypatch.setattr(renderer, "load_results", lambda *_args, **_kwargs: data)
    monkeypatch.setattr(renderer, "build_sec_vd_robot", lambda: (object(), None, 0.1))
    monkeypatch.setattr(renderer, "validate_pose_consistency", lambda *_args: None)
    captured = {}

    class FakeRenderer:
        def __init__(self, _robot, **_kwargs):
            pass

        def render_sequence(self, **kwargs):
            captured.update(kwargs)

    renderer.render_method(
        name="synergistic",
        data_dir=tmp_path,
        output_dir=tmp_path,
        fps=4,
        make_gif=False,
        force=False,
        port=8080,
        open_browser=False,
        record_client_timeout=2.0,
        record_frame_timeout=3.0,
        renderer_factory=FakeRenderer,
    )
    expected_reference = reference_pose[:, 3:6]
    np.testing.assert_allclose(captured["static_spheres_positions"], expected_reference)
    np.testing.assert_allclose(
        captured["static_spheres_colors"],
        np.tile(renderer.TARGET_TRAIL_COLOR, (timesteps, 1)),
    )


def test_collocated_render_receives_configuration_target(monkeypatch, tmp_path):
    timesteps = 5
    data = {
        "t_ts": np.linspace(0, 1, timesteps),
        "q_ts_best": np.zeros((1, timesteps, 6)),
        "q_des_ts": np.ones((timesteps, 6)),
        "x_ts_best": np.zeros((1, timesteps, 6)),
        "best_batch": np.asarray(0),
        "x_des_ts": np.zeros((timesteps, 6)),
    }
    monkeypatch.setattr(renderer, "load_results", lambda *_args, **_kwargs: data)
    monkeypatch.setattr(renderer, "build_sec_vd_robot", lambda: (object(), None, 0.1))
    monkeypatch.setattr(renderer, "validate_pose_consistency", lambda *_args: None)
    captured = {}

    class FakeRenderer:
        def __init__(self, _robot, **kwargs):
            captured["init"] = kwargs

        def render_sequence(self, **kwargs):
            captured["render"] = kwargs

    renderer.render_method(
        name="collocated",
        data_dir=tmp_path,
        output_dir=tmp_path,
        fps=4,
        make_gif=False,
        force=False,
        port=8080,
        open_browser=False,
        record_client_timeout=2.0,
        record_frame_timeout=3.0,
        renderer_factory=FakeRenderer,
    )
    assert captured["init"]["desired_q_ts"].shape == (5, 6)
    np.testing.assert_allclose(
        captured["init"]["color_config"].backbone.point_colors[:, 3], 1.0
    )
    assert captured["render"]["static_spheres_positions"] is None
    assert captured["render"]["dynamic_spheres_positions"] is None


@pytest.mark.parametrize("method", ["collocated", "synergistic"])
@pytest.mark.parametrize("trajectory", renderer.TRAJECTORIES)
def test_committed_pose_matches_forward_kinematics(method, trajectory):
    data = load_results(SECTION_DIR / "data" / method, expected_method=method)
    robot, _routing, total_length = build_sec_vd_robot()
    renderer.validate_pose_consistency(robot, total_length, data, trajectory)
