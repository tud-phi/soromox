#!/usr/bin/env python3
"""Capture real renderer output and compose the paper's rendering overview.

Run --capture open3d, --capture simple, and --capture viser before --assemble.
Viser prints a local URL; open it in a browser to supply its WebGL client.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
CASE = Path(__file__).resolve().parents[1]
CAPTURES = CASE / "outputs" / "captures"
OUTPUT = ROOT / "paper_results" / "final_outputs" / "renderers_and_presets"
sys.path.insert(0, str(ROOT / "examples" / "rendering"))
sys.path.insert(0, str(ROOT / "paper_results"))
from equilibrium_scene import SOLUTION, load_scene  # noqa: E402
from paper_style import PAPER_COLORS  # noqa: E402
from tentacle_scene import make_comparison  # noqa: E402

from soromox.rendering import (  # noqa: E402
    BackboneColorConfig,
    CameraConfig,
    MatplotlibRenderer,
    Open3DRenderer,
    OpenCVPlanarRenderer,
    RendererColorConfig,
    ViserRenderer,
)

PRESETS = ("technical", "neutral", "bright", "flat", "clay", "dark")
ROBOT, POSE, EQUILIBRIUM = load_scene()
# Remove empty canvas margins in the two schematic examples. All geometry and
# Matplotlib axis labels remain visible; the original captures stay untouched.
DISPLAY_CROPS = {
    "opencv": (30, 40, 859, 720),
    "matplotlib": (160, 0, 980, 770),
}


class PlanarEquilibriumView:
    """Expose the equilibrium's x-z plane to the unchanged OpenCV backend."""

    is_planar = True
    floating_base = False
    fixed_base_pose = jnp.array([-jnp.pi / 2, 0.0, 0.0])
    base_transform = jnp.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    def __init__(self, robot):
        self.robot = robot

    def __getattr__(self, name):
        return getattr(self.robot, name)

    def forward_kinematics_abscissa_batched(self, q, s):
        poses = self.robot.forward_kinematics_abscissa_batched(q, s)
        return jnp.stack(
            (
                jnp.arctan2(poses[:, 2, 0], poses[:, 0, 0]),
                poses[:, 0, 3],
                poses[:, 2, 3],
            ),
            axis=-1,
        )


class PaperMatplotlibRenderer(MatplotlibRenderer):
    """Keep backend geometry and use sparse, legible axes for a small panel."""

    def _setup_axes(self, ax, *args, **kwargs):
        super()._setup_axes(ax, *args, **kwargs)
        # An oblique view separates the axes that overlap in the frontal view.
        ax.view_init(elev=-20, azim=-68)
        ax.set_xlim(-0.02, 0.25)
        ax.set_ylim(-0.06, 0.06)
        ax.set_zlim(-0.24, 0.025)
        ax.set_xticks([0, 0.2])
        ax.set_yticks([0])
        ax.set_zticks([-0.2, -0.1, 0])
        ax.set_box_aspect((1, 0.45, 1), zoom=1.02)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_formatter(FormatStrFormatter("%g"))
        ax.set_xlabel("$x$ [m]", labelpad=15)
        ax.set_ylabel("$y$ [m]", labelpad=15)
        ax.set_zlabel("$z$ [m]", labelpad=40)
        ax.tick_params(pad=6)


def configuration(preset, *, wide=False):
    """Keep geometry, vertical field of view, and camera fixed across presets."""
    config, _, _ = make_comparison(
        preset,
        count=1,
        width=1400 if wide else 1000,
        height=700 if wide else 820,
        mounting="hanging",
    )
    config.camera = CameraConfig(
        fov=35, position=(0.085, -0.46, -0.25), look_at=(0.085, 0, -0.105)
    )
    if preset != "clay":
        config.colors = RendererColorConfig(
            backbone=BackboneColorConfig(
                segment_palette=[to_rgb(PAPER_COLORS["post_opt_1"])],
            ),
            base_plate_color=(0.2, 0.2, 0.2),
        )
    return config


def save_capture(name, pixels, config, backend, q, **capture_settings):
    """Save the unmodified renderer pixels with reproducible scene metadata."""
    if pixels is None or np.std(pixels[..., :3]) < 1:
        raise RuntimeError(f"Blank capture: {name}")
    CAPTURES.mkdir(parents=True, exist_ok=True)
    path = CAPTURES / f"{name}.png"
    Image.fromarray(pixels).save(path)
    metadata = {
        "backend": backend,
        "version": version(backend),
        "configuration": asdict(config),
        "q": q.tolist(),
        "pose_source": "verified Section Va fitted-model equilibrium under tendon tension and gravity",
        "equilibrium_sha256": hashlib.sha256(SOLUTION.read_bytes()).hexdigest(),
        "tendon_tensions_N": EQUILIBRIUM["tension_N"],
        "capture_settings": capture_settings,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, default=lambda a: np.asarray(a).tolist()) + "\n"
    )
    print(f"Saved {path}", flush=True)


def capture_open3d():
    """Use the modern Open3D renderer for the backend and six preset panels."""
    robot = ROBOT
    for name, preset, wide in [
        ("open3d", "neutral", False),
        *[(f"preset_{p}", p, True) for p in PRESETS],
    ]:
        config = configuration(preset, wide=wide)
        renderer = Open3DRenderer(robot, config=config)
        pixels = renderer.render_frame(POSE, render_actuators=False)
        save_capture(name, pixels, config, "open3d", POSE)


def capture_simple():
    """Capture Matplotlib and the same planar equilibrium through OpenCV."""
    config = configuration("neutral")
    config.geometry.line_width = 7
    # The output occupies about 41 mm on paper: scale the source text accordingly.
    with (
        plt.style.context(ROOT / "paper_results" / "paper.mplstyle"),
        plt.rc_context(
            {
                "font.size": 30,
                "axes.labelsize": 30,
                "xtick.labelsize": 28,
                "ytick.labelsize": 28,
                "axes.labelpad": 12,
            }
        ),
    ):
        renderer = PaperMatplotlibRenderer(ROBOT, config=config)
        pixels = renderer.render_frame(POSE, render_actuators=False)
    save_capture(
        "matplotlib",
        pixels,
        config,
        "matplotlib",
        POSE,
        axes_view={"elevation_deg": -20, "azimuth_deg": -68},
        axes_formatting="PaperMatplotlibRenderer._setup_axes",
    )

    planar = PlanarEquilibriumView(ROBOT)
    config = configuration("neutral")
    config.geometry.line_width = None
    config.scene.ground.height_reference = "world"
    renderer = OpenCVPlanarRenderer(
        planar, config=config, length_scale=1.1, origin_uv=(240, 155)
    )
    q = POSE
    pixels = renderer.render_frame(q, render_actuators=False)[..., ::-1]
    save_capture(
        "opencv",
        pixels,
        config,
        "opencv-python",
        q,
        length_scale=1.1,
        origin_uv=[240, 155],
        projection="world x-z plane of the same GVS equilibrium",
    )


def capture_viser(port):
    """Capture through a connected real Viser browser, preserving its shading."""
    config = configuration("neutral")
    renderer = ViserRenderer(
        ROBOT,
        config=config,
        host="127.0.0.1",
        port=port,
        open_browser=False,
    )
    try:
        print(f"Open {renderer.url} to capture Viser", flush=True)
        deadline = time.monotonic() + 300
        while not renderer.server.get_clients():
            if time.monotonic() > deadline:
                raise RuntimeError("No Viser browser connected within five minutes")
            time.sleep(0.2)
        pixels = renderer.render_frame(POSE, render_actuators=False)
        save_capture("viser", pixels, config, "viser", POSE)
    finally:
        renderer.stop()


def assemble():
    """Compose vector text and captured scenes with compact paper spacing."""
    width, height = 180, 110
    with (
        plt.style.context(ROOT / "paper_results" / "paper.mplstyle"),
        plt.rc_context(
            {
                "font.size": 8,
                "svg.fonttype": "none",
                "pdf.fonttype": 42,
                "savefig.bbox": None,
            }
        ),
    ):
        fig = plt.figure(figsize=(width / 25.4, height / 25.4), facecolor="white")

        def label(x, y, text, **kwargs):
            fig.text(x / width, y / height, text, va="center", **kwargs)

        def panel(x, y, w, h, name):
            ax = fig.add_axes([x / width, y / height, w / width, h / height])
            with Image.open(CAPTURES / f"{name}.png") as source:
                pixels = (
                    source.crop(DISPLAY_CROPS[name])
                    if name in DISPLAY_CROPS
                    else source
                )
                ax.imshow(pixels)
            ax.set_axis_off()
            fig.add_artist(
                Rectangle(
                    (x / width, y / height),
                    w / width,
                    h / height,
                    transform=fig.transFigure,
                    fill=False,
                    edgecolor="#d6d6d6",
                    linewidth=0.35,
                )
            )

        # Put each group letter on its first label line, as in the paper's
        # snapshot figures. The caption supplies the two group descriptions.
        label(0.6, 106.5, "A", fontsize=13)
        for i, (name, title) in enumerate(
            [
                ("opencv", "OpenCV Planar"),
                ("matplotlib", "Matplotlib"),
                ("viser", "Viser"),
                ("open3d", "Open3D"),
            ]
        ):
            x = 5 + i * 43
            label(x + 20.5, 106.5, title, ha="center")
            panel(x, 70, 41, 33.62, name)

        label(0.6, 65.5, "B", fontsize=13)
        for i, preset in enumerate(PRESETS):
            row, col = divmod(i, 3)
            w = (170 - 4) / 3
            x, y = 5 + col * (w + 2), 35 - row * 34
            label(x + w / 2, y + 30.5, preset.capitalize(), ha="center")
            panel(x, y, w, w / 2, f"preset_{preset}")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("pdf", "svg", "png"):
            fig.savefig(OUTPUT.with_suffix(f".{suffix}"), dpi=300, bbox_inches=None)
        svg = OUTPUT.with_suffix(".svg")
        svg.write_text(
            "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
        )
        plt.close(fig)
    manifest = {
        "figure_mm": [width, height],
        "renderer_order": ["OpenCV Planar", "Matplotlib", "Viser", "Open3D"],
        "preset_order": list(PRESETS),
        "preset_backend": "Open3D",
        "display_crops_px": DISPLAY_CROPS,
        "panel_outline": {"color": "#d6d6d6", "width_pt": 0.35},
        "equilibrium_sha256": hashlib.sha256(SOLUTION.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "captures": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(CAPTURES.glob("*.png"))
        },
    }
    OUTPUT.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {OUTPUT}.pdf / .svg / .png", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", choices=("open3d", "simple", "viser"))
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--port", type=int, default=8095)
    args = parser.parse_args()
    if args.capture == "open3d":
        capture_open3d()
    elif args.capture == "simple":
        capture_simple()
    elif args.capture == "viser":
        capture_viser(args.port)
    if args.assemble:
        assemble()
    if not args.capture and not args.assemble:
        parser.error("Choose --capture or --assemble")


if __name__ == "__main__":
    main()
