# Patched Open3D 0.20 source build

SoRoMoX builds the immutable Open3D 0.20.0 release tag on macOS, Linux
x86-64, Linux ARM64 and Windows x86-64. The cross-platform PEP 517 adapter is
[`build_backend.py`](build_backend.py); it applies the temporary renderer fixes
in this directory and caches one native wheel per revision, patch set, Python
ABI, operating system, and CPU architecture. Other platforms use official PyPI
or [Open3D v0.20.0 release](https://github.com/isl-org/Open3D/releases/tag/v0.20.0)
wheels.

`OPEN3D_REVISION` records the exact commit behind the stable tag and makes the
source build reproducible. Stable release upgrades are deliberate changes that
also update `BASE_VERSION`, the lock file, patch applicability and native
validation.

## Ubuntu installation

The Ubuntu source build needs the following packages. This is the exact package
set used for the native validation described below:

```bash
sudo apt-get update
sudo apt-get install -y \
  git build-essential cmake ninja-build glslang-tools \
  xorg-dev libxcb-shm0 libglu1-mesa-dev libssl-dev \
  libc++-dev libc++abi-dev libsdl2-dev libxi-dev libtbb-dev \
  libegl1-mesa-dev libudev-dev libusb-1.0-0-dev \
  mesa-vulkan-drivers autoconf libtool clang gfortran xvfb xauth ffmpeg
```

Then build the rendering environment from the SoRoMoX checkout:

```bash
uv sync --extra rendering
```

The first sync compiles Open3D and can take several minutes and multiple
gigabytes. Later syncs reuse `~/.cache/soromox/open3d` when the upstream revision,
adapter recipe, patches, Python ABI, and architecture are unchanged.

Open3D 0.20 uses Vulkan for modern offscreen and GUI rendering. On machines
without a physical Vulkan device, Mesa's Lavapipe driver provides software
Vulkan and does not need an X display for image or video export:

```bash
env -u DISPLAY \
  uv run --no-sync python examples/rendering/preset_gallery.py --backend open3d --preset neutral \
  --count 1 --output-dir open3d-figures --video-output open3d.mp4
```

Interactive `show()` and trajectory playback need a desktop display. A headless
CI machine can use `xvfb-run`; the integration tests exercise Vulkan for the
modern window and software GLX for the legacy visualizer.

## macOS installation

Install Xcode and its Metal Toolchain, then the native build dependencies:

```bash
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -runFirstLaunch
xcodebuild -downloadComponent metalToolchain
xcrun --kill-cache
xcrun -sdk macosx metal --version
brew install cmake ninja openblas glslang spirv-cross
uv sync --extra rendering
```

The graphical route for the optional compiler is **Xcode → Settings →
Components → Metal Toolchain → Get**. The adapter compiles the C++ library,
Python bindings, Metal shaders, and a Python-specific wheel. The wheel links to
Homebrew OpenBLAS.

## Windows installation

Install Visual Studio 2022 with the Desktop development with C++ workload,
CMake, Git, Python and FFmpeg. Then build the rendering environment from the
SoRoMoX checkout:

```powershell
uv sync --extra rendering
```

The adapter uses the Visual Studio 2022 x64 generator, builds the pinned
Open3D 0.20 source with the shared MSVC runtime, and applies the common renderer
patches before compilation. Windows build products are cached under
`%USERPROFILE%\.cache\soromox\open3d`.

Native Windows compilation and rendering still require CI validation for this
patch revision.

## Other platforms

On other platforms, `uv` and `pip` resolve matching stable Open3D wheels from
PyPI. Install SoRoMoX normally:

```bash
python -m pip install -e ".[rendering]"
```

## Validation

The release-tag pin is `b6c5e196384ad71e75b6e6f9c5da22d046221f1d`.
Patched wheels report `0.20.0+b6c5e19.soromox3`.

Python 3.15.0rc2 was source-built and tested on Apple Silicon (M4 Max).
The frozen rendering/test installation, Metal color-grading check, PNG and
60-frame H.264 export, and modern-window lifecycle all passed. Ubuntu CI also
includes a Python 3.15 x86-64 source-build and native-rendering job.

The native macOS integration checks cover RGB `uint8` readback and color-grading
selection, a 320 × 240 tentacle PNG and 60-frame H.264 MP4, and opening/closing
the real Metal viewer through its event loop.

Run the macOS checks with:

```bash
SOROMOX_RUN_RENDERING_INTEGRATION=1 uv run --no-sync python -m pytest -q \
  tests/rendering/test_open3d_color_integration.py \
  tests/rendering/test_open3d_macos_integration.py
```

Ubuntu x86-64, Ubuntu ARM64 and Windows native builds run in CI. Results for
this revision must pass before merging; macOS checks do not validate the Linux
or Windows patch sets or other Python ABIs.

The Ubuntu integration checks render non-uniform RGB pixels, encode and probe a
three-frame H.264 MP4, open and close a real modern GUI window under Xvfb, and
advance all frames in real legacy interactive playback. They are opt-in and
guarded at module scope so they skip on non-Ubuntu and non-Linux hosts:

```bash
env -u DISPLAY \
  SOROMOX_RUN_RENDERING_INTEGRATION=1 python -m pytest -q \
  tests/rendering/test_open3d_linux_integration.py::test_open3d_headless_frame_and_mp4_export

xvfb-run -a env LIBGL_ALWAYS_SOFTWARE=true \
  SOROMOX_RUN_RENDERING_INTEGRATION=1 python -m pytest -q \
  tests/rendering/test_open3d_linux_integration.py::test_open3d_show_opens_and_closes_real_xvfb_window \
  tests/rendering/test_open3d_linux_integration.py::test_open3d_interactive_sequence_advances_in_real_xvfb_window
```

The modern Xvfb window smoke test disables VSM shadows and SSAO to keep the
software-rendered lifecycle check fast. The headless image/video checks exercise
default lighting and shadows.

The integration tests run modern GUI and legacy visualizer checks in separate
processes because their graphics teardown can interact. Applications should
likewise avoid mixing the two Open3D GUI systems in one process.

## Updating Open3D releases

When adopting a newer stable Open3D release, update `OPEN3D_REVISION` to the
full commit behind that release tag and update `BASE_VERSION` in
`build_backend.py`. Then refresh the lock and build:

```bash
uv lock --upgrade-package open3d --refresh-package open3d
uv sync --extra rendering
```

Re-run the Ubuntu integration tests and the supported-Python source-build matrix
before merging the new pin. If a patch no longer applies, the adapter stops
instead of silently producing a wheel with unreviewed behavior. Remove a patch
only after the stable release contains its fix and the relevant native tests pass
without it.

## Temporary patches and upstreaming

`legacy_mesh_uv_initialization.patch` initializes UV0 for legacy triangle
meshes without triangle UVs. Open3D uses a `TexturedVertex` buffer for this path
and advertises UV0 to Filament, while the buffer comes from `malloc`; without
the patch, those UV bytes are undefined. This can produce heap-dependent
triangles or lighting artifacts with software Vulkan and may affect any
Filament backend; see [Open3D issue #7565](https://github.com/isl-org/Open3D/issues/7565)
and [the proposed upstream fix](https://github.com/isl-org/Open3D/pull/7566).

`neutral_tone_mapping.patch` restores linear, ACES, legacy ACES, Filmic and
Display Range selection through Filament's current `ToneMapper` API; see
[Open3D issue #7557](https://github.com/isl-org/Open3D/issues/7557) and
[the proposed upstream fix](https://github.com/isl-org/Open3D/pull/7567).
SoRoMoX selects Filmic for its default lit rendering: neutral greys
and readable midtones without the legacy ACES highlight tint. The technical
preset selects linear mapping to keep its background white. Uchimura and
Reinhard modes continue to use Filament's default mapper because
Filament 1.76 has no corresponding built-in mapper. Wheels with these fixes use
the `.soromox3` suffix; run `uv sync --extra rendering` to update an older build.
The color-grading regression test exercises native Metal on macOS and software
Vulkan in Ubuntu CI.

The Linux patches serve the following purposes:

- `linux_distribution_name.patch` keeps the source package's distribution name
  as `open3d`.
- `linux_static_curl.patch` groups the bundled curl and BoringSSL archives for
  linking.

Upstream tracking:

| Finding | Upstream discussion |
| --- | --- |
| Modern GUI followed by legacy Visualizer segfault | [Open3D #7553](https://github.com/isl-org/Open3D/issues/7553) |
| Bundled curl/BoringSSL archive grouping | [Open3D #7556](https://github.com/isl-org/Open3D/issues/7556), [Open3D PR #7568](https://github.com/isl-org/Open3D/pull/7568) |

The distribution-name patch supports SoRoMoX's source-build configuration. The
tone-mapping and static-curl linking patches address upstream defects.

## Build configuration and overrides

The native builds enable GUI and CPU geometry operations. CUDA, ML framework
integrations, WebRTC, the Jupyter extension, examples, and upstream unit tests
are disabled. Linux uses Open3D's prebuilt Filament package; macOS builds Metal
shaders locally.

Set `SOROMOX_OPEN3D_CACHE` to relocate the cache and
`CMAKE_BUILD_PARALLEL_LEVEL` to change compilation parallelism (default 8).
`SOROMOX_OPEN3D_SOURCE_DIR` and `SOROMOX_OPEN3D_BUILD_DIR` select existing
source/build directories for incremental development. Override source checkouts
must exactly match `OPEN3D_REVISION`.

The adapter's metadata hook is static and platform-neutral. Dependency solvers
can therefore inspect the local source mapping while resolving a universal lock
without launching a native macOS, Linux or Windows build.
