"""Tests for Section Vd saved-data CLI modes."""

import sys
import types
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

SECTION_DIR = (
    Path(__file__).resolve().parents[3]
    / "paper_results"
    / "secVd_control_gain_optimization"
)
MODULE_DIR = SECTION_DIR / "code"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import control_gain_optimization_common as common  # noqa: E402
from control_gain_optimization_common import (  # noqa: E402
    parse_args,
    prepare_output_dirs,
    run_saved_postprocessing,
)


def _parse(monkeypatch, *arguments):
    monkeypatch.setattr(sys, "argv", ["section-vd", *arguments])
    return parse_args(
        description="test",
        default_result_dir=Path("results"),
        default_output_dir=Path("outputs"),
    )


def test_saved_modes_are_mutually_exclusive(monkeypatch):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--plot-only", "--render-only")


def test_prepare_output_dirs_does_not_require_force_for_saved_modes(
    monkeypatch, tmp_path
):
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    (result_dir / "optimization_results.npz").touch()
    (result_dir / "animation_data.pkl").touch()

    args = _parse(monkeypatch, "--plot-only")
    args.result_dir = result_dir
    args.output_dir = tmp_path / "outputs"

    resolved_result_dir, resolved_output_dir = prepare_output_dirs(args)

    assert resolved_result_dir == result_dir.resolve()
    assert resolved_output_dir == (tmp_path / "outputs").resolve()


def test_plot_only_dispatches_to_saved_plotter(monkeypatch, tmp_path):
    calls = {}
    fake_module = types.ModuleType("plot_control_gain_optimization")

    def configure_matplotlib():
        calls["configured"] = True

    class FakeFigure:
        def savefig(self, path, dpi):
            calls["savefig"] = (path, dpi)

    def build_figure(data_dir):
        calls["data_dir"] = data_dir
        return FakeFigure()

    fake_module.configure_matplotlib = configure_matplotlib
    fake_module.build_figure = build_figure
    monkeypatch.setitem(sys.modules, "plot_control_gain_optimization", fake_module)
    monkeypatch.setattr(common.plt, "close", lambda _figure: None)

    args = _parse(monkeypatch, "--plot-only", "--no-show", "--force")
    result_dir = tmp_path / "data" / "synergistic"
    output_dir = tmp_path / "outputs"

    assert run_saved_postprocessing(args, result_dir, output_dir)
    assert calls == {
        "configured": True,
        "data_dir": result_dir.parent,
        "savefig": (output_dir / "plot_control_gain_opt.pdf", 300),
    }


def test_render_only_dispatches_saved_renderer(monkeypatch, tmp_path):
    calls = {}
    fake_module = types.ModuleType("render_control_gain_optimization_animations")

    def render_saved_results(**kwargs):
        calls.update(kwargs)
        return [tmp_path / "rendered.mp4"]

    fake_module.render_saved_results = render_saved_results
    monkeypatch.setitem(
        sys.modules,
        "render_control_gain_optimization_animations",
        fake_module,
    )

    args = _parse(monkeypatch, "--render-only", "--no-gif", "--force")
    result_dir = tmp_path / "data" / "synergistic"
    output_dir = tmp_path / "outputs"

    assert run_saved_postprocessing(args, result_dir, output_dir)
    assert calls == {
        "data_dir": result_dir.parent,
        "output_dir": output_dir,
        "fps": 24,
        "make_gif": False,
        "force": True,
    }
