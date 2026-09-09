"""Build the pinned Open3D main revision with SoRoMoX renderer fixes.

The metadata hook is intentionally platform-neutral: dependency resolution can
inspect platform-specific source mappings while producing a universal lock
without starting a native build.
"""

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/isl-org/Open3D.git"
HERE = Path(__file__).resolve().parent
REVISION_FILE = HERE / "OPEN3D_REVISION"
BASE_VERSION = "0.19.0"
RUNTIME_REQUIREMENTS = (
    "numpy>=1.18.0",
    "dash>=2.6.0",
    "werkzeug>=3.0.0",
    "flask>=3.0.0",
    "nbformat>=5.7.0",
    "configargparse",
)
MACOS_PATCHES = [HERE / "metal_rgb_readback.patch"]
LINUX_PATCHES = [
    HERE / "linux_surfaceless.patch",
    HERE / "linux_distribution_name.patch",
    HERE / "linux_static_curl.patch",
    HERE / "linux_filament_patch_hook.patch",
]
LINUX_FILAMENT_PATCH = HERE / "filament_linux_dual_context.patch"


def _pinned_commit():
    """Read the immutable upstream revision used by builds and metadata.

    Returns:
        Full 40-character commit hash from ``OPEN3D_REVISION``.

    Raises:
        RuntimeError: The revision file is not a full Git commit hash.
    """
    commit = REVISION_FILE.read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError(f"Invalid Open3D revision in {REVISION_FILE}: {commit!r}")
    return commit


def _version():
    """Return the patched wheel version without consulting the platform."""
    return f"{BASE_VERSION}+{_pinned_commit()[:7]}.soromox1"


def _recipe_fingerprint(patches):
    """Hash the adapter recipe and patches used to produce a cached wheel."""
    digest = hashlib.sha256(Path(__file__).read_bytes())
    for patch in patches:
        digest.update(patch.read_bytes())
    return digest.hexdigest()[:12]


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


def _build_macos():
    """Build or reuse a patched Open3D wheel for the running macOS Python.

    Checks the cache manifest, applies the Metal
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
            "The macOS build path can only run on macOS; use the adapter's "
            "platform dispatcher instead."
        )
    commit = _pinned_commit()
    fingerprint = _recipe_fingerprint(MACOS_PATCHES)
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
            f"Source directory is at {actual}, but the pinned revision is {commit}. "
            "Use a fresh build cache or update the override checkout."
        )
    for patch in MACOS_PATCHES:
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


def _build_linux():
    """Build or reuse patched Open3D for the running Linux CPython ABI."""
    if not sys.platform.startswith("linux") or platform.machine() not in (
        "x86_64",
        "AMD64",
    ):
        raise RuntimeError(
            "The SoRoMoX Linux Open3D adapter currently supports Linux x86-64."
        )
    commit = _pinned_commit()
    fingerprint = _recipe_fingerprint([*LINUX_PATCHES, LINUX_FILAMENT_PATCH])
    cache = Path(
        os.environ.get("SOROMOX_OPEN3D_CACHE", Path.home() / ".cache/soromox/open3d")
    )
    cache /= (
        f"linux-{commit[:7]}-{fingerprint}-"
        f"{sys.implementation.cache_tag}-{platform.machine()}"
    )
    source = Path(os.environ.get("SOROMOX_OPEN3D_SOURCE_DIR", cache / "source"))
    build = Path(os.environ.get("SOROMOX_OPEN3D_BUILD_DIR", cache / "build"))
    wheels = build / "lib/python_package/pip_package"
    manifest = build / "soromox-build.json"
    provenance = {
        "commit": commit,
        "patches": fingerprint,
        "python": sys.implementation.cache_tag,
        "platform": "linux-x86_64",
    }
    pattern = (
        f"open3d-{_version()}-cp{sys.version_info.major}{sys.version_info.minor}-*.whl"
    )
    existing = list(wheels.glob(pattern))
    if (
        existing
        and manifest.exists()
        and json.loads(manifest.read_text()) == provenance
    ):
        return existing[0]

    cache.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        _run("git", "init", source)
        _run("git", "-C", source, "fetch", "--depth=1", REPOSITORY, commit)
        _run("git", "-C", source, "checkout", "--detach", "FETCH_HEAD")
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != commit:
        raise RuntimeError(
            f"Source directory is at {actual}, but the pinned revision is {commit}. "
            "Use a fresh build cache or update the override checkout."
        )
    for patch in LINUX_PATCHES:
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

    cmake = os.environ.get("SOROMOX_CMAKE", shutil.which("cmake") or "cmake")
    flags = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=ON",
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
        "-DBUILD_AZURE_KINECT=OFF",
        "-DBUILD_LIBREALSENSE=OFF",
        "-DWITH_STUBGEN=OFF",
        "-DBUILD_FILAMENT_FROM_SOURCE=ON",
        f"-DFILAMENT_PATCH_FILE={LINUX_FILAMENT_PATCH}",
        # Current Assimp/Filament sources omit standard integer/difference
        # declarations exposed transitively by older compiler libraries.
        "-DCMAKE_CXX_FLAGS=-include cstddef -include cstdint "
        "-Wno-invalid-specialization -Wno-nontrivial-memcall",
        f"-DOPEN3D_GIT_HASH={commit[:7]}.soromox1",
        f"-DPython3_EXECUTABLE={sys.executable}",
    ]
    _run(cmake, "-S", source, "-B", build, "-G", "Ninja", *flags)
    jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "8")
    _run(cmake, "--build", build, "--target", "ext_filament", "--parallel", jobs)
    _run(cmake, "--build", build, "--target", "pip-package", "--parallel", jobs)
    built = list(wheels.glob(pattern))
    if len(built) != 1:
        raise RuntimeError(f"Expected one Open3D wheel in {wheels}, found {built}")
    manifest.write_text(json.dumps(provenance, indent=2) + "\n")
    return built[0]


def _build():
    """Dispatch a native Open3D build without affecting metadata preparation."""
    if sys.platform == "darwin":
        return _build_macos()
    if sys.platform.startswith("linux"):
        return _build_linux()
    raise RuntimeError(
        "The SoRoMoX Open3D source adapter supports macOS and Linux x86-64."
    )


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Provide the PEP 517 wheel build hook for the native dependency.

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
    """Write static metadata without invoking a platform-specific build.

    Args:
        metadata_directory: Destination root supplied by the build frontend.
        config_settings: Unused PEP 517 frontend settings.

    Returns:
        Name of the extracted ``.dist-info`` directory relative to the root.

    Raises:
        RuntimeError: The pinned revision is invalid.
        OSError: The metadata destination cannot be written.
    """
    version = _version()
    directory = f"open3d-{version}.dist-info"
    target = Path(metadata_directory) / directory
    target.mkdir(parents=True, exist_ok=True)
    requirements = "".join(
        f"Requires-Dist: {requirement}\n" for requirement in RUNTIME_REQUIREMENTS
    )
    (target / "METADATA").write_text(
        "Metadata-Version: 2.4\n"
        "Name: open3d\n"
        f"Version: {version}\n"
        "Summary: Open3D development build for SoRoMoX rendering\n"
        "Requires-Python: >=3.10\n"
        f"{requirements}"
    )
    return directory
