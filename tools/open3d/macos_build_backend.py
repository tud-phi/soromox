"""Build Open3D main on macOS, including its Metal readback fix.

This small PEP 517 adapter exists because Open3D uses CMake to produce its
Python wheel. It does not change the upstream Python package or its metadata.
"""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPOSITORY = "https://github.com/isl-org/Open3D.git"
HERE = Path(__file__).resolve().parent
PATCHES = [HERE / "metal_rgb_readback.patch"]


def _main_commit():
    """Resolve the current upstream main branch through Git.

    Returns:
        Full commit hash advertised by the remote main branch.

    Raises:
        RuntimeError: The remote does not advertise main.
        subprocess.CalledProcessError: Git cannot contact or query the remote.
    """
    remote = subprocess.check_output(
        ["git", "ls-remote", REPOSITORY, "refs/heads/main"], text=True
    ).strip()
    if not remote:
        raise RuntimeError("Could not resolve Open3D main")
    return remote.split()[0]


def _run(*args, cwd=None):
    """Run a build command, forwarding output to the caller's terminal.

    Args:
        *args: Command and arguments, converted to strings without a shell.
        cwd: Working directory, or ``None`` to inherit the current directory.

    Returns:
        None. Waits for the command to finish successfully.

    Raises:
        subprocess.CalledProcessError: The command exits with a nonzero status.
        OSError: The executable or working directory cannot be accessed.
    """
    subprocess.run([str(arg) for arg in args], cwd=cwd, check=True)


def _build():
    """Build or reuse a patched Open3D wheel for the running macOS Python.

    Resolves upstream main, checks the cache manifest, applies the Metal
    readback patch and invokes upstream CMake targets. Cache and source/build
    overrides use the ``SOROMOX_OPEN3D_*`` environment variables documented in
    this directory's README. Shaders are compiled with Apple's Metal toolchain.

    Returns:
        Path to a wheel matching the resolved commit, patch and Python ABI.

    Raises:
        RuntimeError: The platform, source checkout, required build tooling or
            number of resulting wheels is incompatible with this build.
        subprocess.CalledProcessError: Source retrieval, patching or compilation
            fails; command output identifies the failed step.
        OSError: A build executable or required file cannot be accessed.
    """
    if sys.platform != "darwin":
        raise RuntimeError(
            "This adapter is for macOS; use the latest official development wheel elsewhere."
        )
    commit = _main_commit()
    fingerprint = hashlib.sha256(b"".join(p.read_bytes() for p in PATCHES)).hexdigest()[
        :12
    ]
    cache = Path(
        os.environ.get("SOROMOX_OPEN3D_CACHE", Path.home() / ".cache/soromox/open3d")
    )
    cache /= f"{commit[:7]}-{fingerprint}-{sys.implementation.cache_tag}-{platform.machine()}"
    source = Path(os.environ.get("SOROMOX_OPEN3D_SOURCE_DIR", cache / "source"))
    build = Path(os.environ.get("SOROMOX_OPEN3D_BUILD_DIR", cache / "build"))
    wheels = build / "lib/python_package/pip_package"
    manifest = build / "soromox-build.json"
    provenance = {
        "commit": commit,
        "patches": fingerprint,
        "python": sys.implementation.cache_tag,
        "shaders": "compiled",
    }
    pattern = f"open3d-*+{commit[:7]}.soromox1-cp{sys.version_info.major}{sys.version_info.minor}-*.whl"
    existing = list(wheels.glob(pattern))
    if (
        existing
        and manifest.exists()
        and json.loads(manifest.read_text()) == provenance
    ):
        return existing[0]
    # Fail before a long build if Apple's optional compiler is unavailable.
    _run("xcrun", "-sdk", "macosx", "metal", "--version")
    cache.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        _run("git", "init", source)
        _run(
            "git",
            "-C",
            source,
            "fetch",
            "--depth=1",
            REPOSITORY,
            commit,
        )
        _run("git", "-C", source, "checkout", "--detach", "FETCH_HEAD")
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != commit:
        raise RuntimeError(
            f"Source directory is at {actual}, but main is at {commit}. Use a fresh build cache or update the override checkout."
        )
    for patch in PATCHES:
        applied = (
            subprocess.run(
                ["git", "-C", str(source), "apply", "--reverse", "--check", str(patch)],
                capture_output=True,
            ).returncode
            == 0
        )
        if not applied:
            _run("git", "-C", source, "apply", "--check", patch)
            _run("git", "-C", source, "apply", patch)
    # Homebrew CMake avoids stale Python entry-point scripts on PATH.
    brew = shutil.which("brew")
    if not brew:
        raise RuntimeError(
            "Install CMake, Ninja and OpenBLAS; see tools/open3d/README.md."
        )

    def prefix(name):
        """Locate an installed Homebrew formula.

        Args:
            name: Homebrew formula name, such as ``cmake`` or ``openblas``.

        Returns:
            Absolute installation prefix reported by Homebrew.

        Raises:
            subprocess.CalledProcessError: Homebrew cannot resolve the formula.
        """
        return subprocess.check_output([brew, "--prefix", name], text=True).strip()

    cmake = os.environ.get("SOROMOX_CMAKE", str(Path(prefix("cmake")) / "bin/cmake"))
    flags = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DBUILD_GUI=ON",
        "-DBUILD_EXAMPLES=OFF",
        "-DBUILD_WEBRTC=OFF",
        "-DBUILD_JUPYTER_EXTENSION=OFF",
        "-DBUILD_CUDA_MODULE=OFF",
        "-DBUILD_PYTORCH_OPS=OFF",
        "-DBUILD_TENSORFLOW_OPS=OFF",
        "-DBUILD_UNIT_TESTS=OFF",
        "-DBUILD_BENCHMARKS=OFF",
        "-DBUNDLE_OPEN3D_ML=OFF",
        "-DWITH_OPENMP=OFF",
        "-DWITH_STUBGEN=OFF",
        "-DUSE_SYSTEM_BLAS=ON",
        f"-DCMAKE_PREFIX_PATH={prefix('openblas')}",
        f"-DCMAKE_CXX_FLAGS=-I{prefix('openblas')}/include -Wno-error=unused-private-field",
        f"-DOPEN3D_GIT_HASH={commit[:7]}.soromox1",
        f"-DPython3_EXECUTABLE={sys.executable}",
    ]
    _run(cmake, "-S", source, "-B", build, "-G", "Ninja", *flags)
    jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "8")
    # Upstream's Ninja byproduct list omits the macOS architecture subdirectory.
    _run(cmake, "--build", build, "--target", "ext_filament", "--parallel", jobs)
    _run(cmake, "--build", build, "--target", "pip-package", "--parallel", jobs)
    built = list(wheels.glob(pattern))
    if len(built) != 1:
        raise RuntimeError(f"Expected one Open3D wheel in {wheels}, found {built}")
    manifest.write_text(json.dumps(provenance, indent=2) + "\n")
    return built[0]


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Provide the PEP 517 wheel build hook for the macOS dependency.

    Args:
        wheel_directory: Existing destination directory supplied by the frontend.
        config_settings: Optional frontend settings; this backend uses the
            documented environment variables and does not consume this mapping.
        metadata_directory: Previously prepared metadata directory, if supplied.
            The wheel contains the metadata produced by upstream's build.

    Returns:
        Filename of the built wheel copied into ``wheel_directory``.

    Raises:
        RuntimeError: The source build or cache validation fails.
        subprocess.CalledProcessError: An upstream build command fails.
        OSError: Build files or the destination cannot be accessed.
    """
    wheel = _build()
    shutil.copy2(wheel, Path(wheel_directory) / wheel.name)
    return wheel.name


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    """Extract upstream's wheel metadata for PEP 517 dependency resolution.

    The first call builds a wheel; later calls can reuse its validated cache.
    Extracting the wheel's metadata preserves its actual version, dependencies
    and platform information.

    Args:
        metadata_directory: Destination root supplied by the build frontend.
        config_settings: Optional frontend settings; configuration is supplied
            through the documented environment variables instead.

    Returns:
        Name of the extracted ``.dist-info`` directory relative to the root.

    Raises:
        RuntimeError: The source build or cache validation fails.
        subprocess.CalledProcessError: An upstream build command fails.
        OSError: The wheel or destination cannot be accessed.
        zipfile.BadZipFile: The built wheel is not a readable ZIP archive.
    """
    # Use actual upstream metadata, including its platform tag and dependencies.
    # The first resolution builds once; subsequent installs reuse the wheel.
    with zipfile.ZipFile(_build()) as zipped:
        metadata = next(
            name for name in zipped.namelist() if name.endswith(".dist-info/METADATA")
        )
        directory = metadata.split("/")[0]
        for name in zipped.namelist():
            if name.startswith(directory + "/") and not name.endswith("/"):
                target = Path(metadata_directory) / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zipped.read(name))
    return directory
