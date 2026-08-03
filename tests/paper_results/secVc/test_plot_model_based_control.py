import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SECTION_CODE_DIR = (
    REPOSITORY_ROOT
    / "paper_results"
    / "secVc_model_based_control"
    / "code"
)
if str(SECTION_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(SECTION_CODE_DIR))

import plot_model_based_control as composite  # noqa: E402


@pytest.fixture
def composite_figure():
    figure, axes = composite.build_composite_figure()
    figure.canvas.draw()
    yield figure, axes
    plt.close(figure)


def test_composite_has_expected_physical_layout(composite_figure):
    figure, axes = composite_figure

    np.testing.assert_allclose(
        figure.get_size_inches(),
        composite.cm_to_inches(composite.FIGURE_SIZE_CM),
    )
    assert composite.FIGURE_SIZE_CM[0] == pytest.approx(18.1)
    assert composite.FIGURE_SIZE_CM[1] <= 20.0
    assert len(axes.configuration) == 3
    assert len(axes.operational) == 2
    assert len(axes.operational_errors) == 2
    assert len(axes.snapshots) == 4
    assert [
        text.get_text() for text in figure.texts if text.get_text() in tuple("AB")
    ] == ["A", "B"]

    assert axes.operational[0].get_ylabel() == r"Pos. $\mathrm{[m]}$"
    assert axes.operational[1].get_ylabel() == r"Orient. $\mathrm{[rad]}$"
    assert axes.operational_errors[0].get_ylabel() == r"$e_p$ $\mathrm{[mm]}$"
    assert axes.operational_errors[1].get_ylabel() == r"$e_r$ $\mathrm{[deg]}$"
    assert all(len(axis.lines) == 4 for axis in axes.operational_errors)
    assert all(axis.lines[-1].get_gid() == "secondary" for axis in axes.operational_errors)
    assert axes.operational[0].get_position().width == pytest.approx(
        axes.operational_errors[0].get_position().width
    )
    assert axes.operational[0].get_position().x1 < axes.operational_errors[0].get_position().x0

    renderer = figure.canvas.get_renderer()
    assert all(
        axis.get_tightbbox(renderer).x1 <= figure.bbox.x1 + 1.0
        for axis in axes.plot_axes
    )


def test_composite_enforces_shared_style_contract(composite_figure):
    figure, axes = composite_figure
    role_widths = {
        "reference": composite.REFERENCE_LINE_WIDTH,
        "primary": composite.PRIMARY_LINE_WIDTH,
        "secondary": composite.SECONDARY_LINE_WIDTH,
    }

    for axis in axes.plot_axes:
        assert axis.xaxis.label.get_fontsize() == pytest.approx(composite.FONT_SIZE)
        assert axis.yaxis.label.get_fontsize() == pytest.approx(composite.FONT_SIZE)
        assert all(
            tick.get_fontsize() == pytest.approx(composite.SMALL_FONT_SIZE)
            for tick in (*axis.get_xticklabels(), *axis.get_yticklabels())
        )
        assert all(
            spine.get_linewidth() == pytest.approx(composite.SPINE_WIDTH)
            for spine in axis.spines.values()
        )
        assert all(
            line.get_linewidth() == pytest.approx(composite.GRID_WIDTH)
            for line in (*axis.get_xgridlines(), *axis.get_ygridlines())
        )
        for line in axis.lines:
            role = line.get_gid()
            if role in role_widths:
                assert line.get_linewidth() == pytest.approx(role_widths[role])
                marker = line.get_marker()
                if marker not in (None, "None", "", " "):
                    assert line.get_markersize() == pytest.approx(
                        composite.MARKER_SIZE
                    )

    panel_labels = [text for text in figure.texts if text.get_text() in "AB"]
    assert all(
        text.get_fontsize() == pytest.approx(composite.PANEL_LABEL_SIZE)
        for text in panel_labels
    )

    assert all(
        text.get_fontsize() == pytest.approx(composite.SMALL_FONT_SIZE)
        for legend in figure.legends
        for text in legend.get_texts()
    )


def test_snapshot_resolution_and_timestamps_are_publication_ready(composite_figure):
    figure, axes = composite_figure
    renderer = figure.canvas.get_renderer()
    bottom_whitespace_cm = (
        min(axis.texts[0].get_window_extent(renderer).y0 for axis in axes.snapshots)
        / figure.dpi
        * 2.54
    )
    assert 0.10 <= bottom_whitespace_cm <= 0.30

    for axis in axes.snapshots:
        assert len(axis.images) == 1
        assert len(axis.texts) == 1
        assert axis.texts[0].get_fontsize() == pytest.approx(
            composite.SMALL_FONT_SIZE
        )
        image_width = axis.images[0].get_array().shape[1]
        width_inches = axis.get_window_extent(renderer).width / figure.dpi
        assert image_width / width_inches >= 300.0


def test_operational_error_axes_use_saved_geometric_errors(composite_figure):
    _, axes = composite_figure
    run = composite.load_run_npz(composite.DEFAULT_OPERATIONAL_INPUT)
    problem = composite.build_problem(run.trajectory_config)
    _, _, position_error, pose_error = composite.operational_tracking_data(
        run,
        problem,
    )
    assert pose_error is not None

    for component, line in enumerate(axes.operational_errors[0].lines[:3]):
        np.testing.assert_allclose(
            line.get_ydata(),
            1e3 * np.asarray(position_error[:, component]),
        )
    for component, line in enumerate(axes.operational_errors[1].lines[:3]):
        np.testing.assert_allclose(
            line.get_ydata(),
            np.rad2deg(np.asarray(pose_error[:, component])),
        )


def test_legends_labels_and_snapshot_timestamps_do_not_overlap(composite_figure):
    figure, axes = composite_figure
    renderer = figure.canvas.get_renderer()
    configuration_bottom = axes.configuration[-1].get_tightbbox(renderer).y0
    operational_legend = figure.legends[1].get_window_extent(renderer)
    position_label = axes.operational[0].yaxis.label.get_window_extent(renderer)
    orientation_label = axes.operational[1].yaxis.label.get_window_extent(renderer)
    position_error_label = axes.operational_errors[0].yaxis.label.get_window_extent(
        renderer
    )
    orientation_error_label = axes.operational_errors[1].yaxis.label.get_window_extent(
        renderer
    )
    snapshot_bottom = min(
        axis.texts[0].get_window_extent(renderer).y0
        for axis in axes.snapshots
    )
    snapshot_top = max(
        axis.get_window_extent(renderer).y1 for axis in axes.snapshots
    )
    operational_xlabel_bottom = min(
        axis.xaxis.label.get_window_extent(renderer).y0
        for axis in (axes.operational[1], axes.operational_errors[1])
    )

    assert configuration_bottom > operational_legend.y1
    assert position_label.y0 > orientation_label.y1
    assert position_error_label.y0 > orientation_error_label.y1
    assert operational_xlabel_bottom > snapshot_top
    assert snapshot_bottom > figure.bbox.y0
    assert all(
        axis.get_window_extent(renderer).y0
        > axis.texts[0].get_window_extent(renderer).y1
        for axis in axes.snapshots
    )


def test_ylabels_align_within_each_subplot_column(composite_figure):
    figure, axes = composite_figure
    renderer = figure.canvas.get_renderer()

    for group in (
        axes.configuration,
        axes.operational,
        axes.operational_errors,
    ):
        centers = [
            label_box.x0 + 0.5 * label_box.width
            for axis in group
            for label_box in [axis.yaxis.label.get_window_extent(renderer)]
        ]
        assert np.ptp(centers) <= 0.5


def test_save_writes_both_formats_and_requires_force_to_overwrite(
    tmp_path,
    composite_figure,
):
    figure, _ = composite_figure
    output_base = tmp_path / "composite"

    pdf_output, svg_output = composite.save_composite_figure(
        figure,
        output_base,
        force=False,
    )
    assert pdf_output.stat().st_size > 0
    assert svg_output.stat().st_size > 0
    with pytest.raises(FileExistsError, match="--force"):
        composite.save_composite_figure(figure, output_base, force=False)
