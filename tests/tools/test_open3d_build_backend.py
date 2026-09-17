from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

BACKEND_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "open3d" / "build_backend.py"
)
SPEC = importlib.util.spec_from_file_location(
    "soromox_open3d_build_backend", BACKEND_PATH
)
assert SPEC is not None and SPEC.loader is not None
backend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backend)


def test_uv_initialization_patch_is_applied_on_every_source_platform() -> None:
    assert "legacy_mesh_uv_initialization.patch" in {
        patch.name for patch in backend.COMMON_PATCHES
    }


def test_windows_build_uses_msvc_release_and_common_patches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commit = "a" * 40
    calls: list[tuple[str, ...]] = []
    prepared: list[tuple[Path, Path, str, tuple[Path, ...]]] = []

    monkeypatch.setattr(backend.sys, "platform", "win32")
    monkeypatch.setattr(backend.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(backend.shutil, "which", lambda _name: "cmake.exe")
    monkeypatch.setattr(backend, "_pinned_commit", lambda: commit)
    monkeypatch.setattr(backend, "_recipe_fingerprint", lambda _patches: "recipe")
    monkeypatch.setenv("SOROMOX_OPEN3D_CACHE", str(tmp_path))
    monkeypatch.setenv("SOROMOX_CMAKE", "cmake.exe")

    def prepare_source(cache, source, actual_commit, patches):
        prepared.append((cache, source, actual_commit, tuple(patches)))

    def run(*args, cwd=None):
        command = tuple(str(arg) for arg in args)
        calls.append(command)
        if "--build" in command:
            wheel_dir = Path(command[command.index("--build") + 1])
            wheel_dir /= "lib/python_package/pip_package"
            wheel_dir.mkdir(parents=True)
            tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
            (
                wheel_dir / f"open3d-{backend._version()}-{tag}-{tag}-win_amd64.whl"
            ).touch()

    monkeypatch.setattr(backend, "_prepare_source", prepare_source)
    monkeypatch.setattr(backend, "_run", run)

    wheel = backend._build_windows()

    assert wheel.name.endswith("-win_amd64.whl")
    assert prepared[0][2:] == (commit, tuple(backend.COMMON_PATCHES))
    configure, build = calls
    assert configure[:2] == ("cmake.exe", "-S")
    assert (
        configure[configure.index("-G")],
        configure[configure.index("-G") + 1],
    ) == ("-G", "Visual Studio 17 2022")
    assert "-DSTATIC_WINDOWS_RUNTIME=OFF" in configure
    assert "-DOPEN3D_GIT_HASH=aaaaaaa.soromox3" in configure
    assert "--config" in build and build[build.index("--config") + 1] == "Release"
    assert "pip-package" in build

    manifest = json.loads((wheel.parents[3] / "soromox-build.json").read_text())
    assert manifest["platform"] == "windows-x86_64"
