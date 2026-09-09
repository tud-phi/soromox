"""Render soft tentacle presets in Open3D or a connected Viser browser.

Run ``python examples/rendering/preset_gallery.py --backend open3d`` for native
exports. For Viser, choose ``--backend viser`` and open the printed local URL.
Use ``--preset neutral --interactive`` to inspect one scene after capture, or
``--backend open3d --preset neutral --video-output studio.mp4`` to export motion.
"""

import argparse
import json
import shlex
import sys
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image
from tentacle_scene import PRESETS, make_comparison, make_tentacle

from soromox.rendering import Open3DRenderer, ViserRenderer


def main():
    """Export selected presets, optionally opening a viewer or recording motion.

    Returns:
        None. Writes PNGs and optional Open3D video. With ``--write-manifest``,
        also writes the resolved configuration, version and reproduction command.

    Raises:
        RuntimeError: The Viser connection times out or a capture is empty.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("open3d", "viser"), required=True)
    parser.add_argument("--preset", choices=(*PRESETS, "all"), default="all")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures" / "presets",
    )
    parser.add_argument("--count", type=int, choices=(1, 5), default=5)
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--interactive", action="store_true", help="Inspect one preset after capture"
    )
    parser.add_argument(
        "--video-output", type=Path, help="Export one Open3D preset as a two-second MP4"
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write local configuration/version metadata",
    )
    args = parser.parse_args()
    if args.preset == "all" and (args.interactive or args.video_output):
        parser.error("Select one --preset for interactive viewing or video export")
    if args.video_output and args.backend != "open3d":
        parser.error(
            "--video-output uses Open3D; use ViserRenderer.render_sequence for browser recording"
        )
    if args.width <= 0 or args.height <= 0:
        parser.error("width and height must be positive")
    renderer_type = Open3DRenderer if args.backend == "open3d" else ViserRenderer
    if renderer_type is None:
        parser.error(f"Install the {args.backend} rendering dependency first")
    selected = PRESETS if args.preset == "all" else (args.preset,)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    robot = make_tentacle()
    metadata = []
    server_owner = None
    try:
        if args.backend == "viser":
            server_owner = ViserRenderer(
                robot, host="127.0.0.1", port=args.port, open_browser=False
            )
            print(f"Open {server_owner.url} to capture the presets", flush=True)
            deadline = time.monotonic() + 300
            while not server_owner.server.get_clients():
                if time.monotonic() > deadline:
                    raise RuntimeError("No browser connected within five minutes")
                time.sleep(0.2)
        for preset in selected:
            config, q, offsets = make_comparison(
                preset, count=args.count, width=args.width, height=args.height
            )
            if server_owner is None:
                renderer = renderer_type(robot, config=config)
            else:
                renderer = renderer_type(
                    robot, config=config, auto_start=False, open_browser=False
                )
                renderer._server = server_owner.server
                renderer._clear_scene()
            try:
                pixels = renderer.render_frame(
                    q, base_offsets=offsets, render_actuators=False
                )
                if np.std(pixels[..., :3]) < 1:
                    raise RuntimeError(f"Blank {args.backend} capture for {preset}")
                filename = args.output_dir / f"{args.backend}_{preset}.png"
                Image.fromarray(pixels).save(filename)
                if args.write_manifest:
                    metadata.append(
                        {
                            "preset": preset,
                            "backend": args.backend,
                            "version": version(args.backend),
                            "configuration": asdict(config),
                            "image": filename.name,
                        }
                    )
                print(f"Saved {filename}", flush=True)
                if args.video_output:
                    ts = np.arange(60) / 30.0
                    trajectory = (
                        q[:, None, :]
                        * (1 + 0.15 * np.sin(2 * np.pi * ts))[None, :, None]
                    )
                    renderer.render_sequence(
                        ts,
                        trajectory,
                        base_offsets=offsets,
                        record_path=str(args.video_output),
                    )
                    print(f"Saved {args.video_output}", flush=True)
                if args.interactive:
                    if server_owner is None:
                        renderer.show(q, base_offsets=offsets, render_actuators=False)
                    else:
                        print("Press Ctrl+C to close the Viser viewer", flush=True)
                        # Keep the captured PBR scene and its browser controls available.
                        while True:
                            time.sleep(0.2)
            finally:
                if server_owner is not None:
                    renderer._server = None
    except KeyboardInterrupt:
        pass
    finally:
        if server_owner is not None:
            server_owner.stop()
    if args.write_manifest:
        manifest = {
            "command": shlex.join(
                ["python", "examples/rendering/preset_gallery.py", *sys.argv[1:]]
            ),
            "renders": metadata,
        }
        (args.output_dir / f"{args.backend}_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=lambda x: np.asarray(x).tolist())
            + "\n"
        )


if __name__ == "__main__":
    main()
