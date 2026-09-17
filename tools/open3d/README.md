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
  autoconf libtool clang xvfb xauth ffmpeg
```

Then build the rendering environment from the SoRoMoX checkout:

```bash
uv sync --extra rendering
```

The first sync compiles Open3D and Filament and can take several minutes and
multiple gigabytes. Later syncs reuse `~/.cache/soromox/open3d` when the upstream
revision, adapter recipe, patches, Python ABI, and architecture are unchanged.

Offscreen rendering needs an OpenGL/EGL implementation but not an X display.
Select the surfaceless backend before importing Open3D:

```bash
env -u DISPLAY EGL_PLATFORM=surfaceless \
  uv run --no-sync python examples/rendering/preset_gallery.py --backend open3d --preset neutral \
  --count 1 --output-dir open3d-figures --video-output open3d.mp4
```

Interactive `show()` and trajectory playback use GLX and need a desktop display.
Do not set `EGL_PLATFORM=surfaceless` for them. A headless CI machine can use
`xvfb-run`; the integration tests demonstrate the supported setup.

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

Available stable wheel ABIs and architectures are controlled by upstream. Linux
ARM64 uses the local source adapter alongside Linux x86-64 because Open3D 0.20's
uninitialized legacy-mesh UV field is architecture-independent. Selecting the
official ARM64 wheel would omit that fix and the other temporary renderer
patches.

## Validation

The current release-tag pin is
`b6c5e196384ad71e75b6e6f9c5da22d046221f1d`. The preceding version-bump commit,
`d32b4fce639b3cde284184072796480ef9b528d1` (Open3D 0.20.0, Filament 1.76.0),
was source-built and tested on Apple M4 Max, macOS 27.0, Xcode 27.0 (27A266a),
Metal 32023.921, and Python 3.12.11. Patched release-tag wheels report
`0.20.0+b6c5e19.soromox3`; native CI rebuilds and validates the exact pin.

The three native macOS integration checks passed: RGB `uint8` readback and
color-grading selection, a 320 × 240 tentacle PNG and 60-frame H.264 MP4,
and opening/closing the real Metal viewer through its event loop. The exported
PNG was visually inspected. The source build compiled the Metal shaders, and
130 focused renderer/configuration tests passed against the new wheel.

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
env -u DISPLAY EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=true \
  SOROMOX_RUN_RENDERING_INTEGRATION=1 python -m pytest -q \
  tests/rendering/test_open3d_linux_integration.py::test_open3d_surfaceless_frame_and_mp4_export

xvfb-run -a env -u EGL_PLATFORM LIBGL_ALWAYS_SOFTWARE=true \
  SOROMOX_RUN_RENDERING_INTEGRATION=1 python -m pytest -q \
  tests/rendering/test_open3d_linux_integration.py::test_open3d_show_opens_and_closes_real_xvfb_window \
  tests/rendering/test_open3d_linux_integration.py::test_open3d_interactive_sequence_advances_in_real_xvfb_window
```

Xvfb's software GLX exposes Filament feature level 1, so the modern window smoke
test disables VSM shadows and SSAO. The surfaceless image/video checks exercise
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
Display Range selection through Filament's current `ToneMapper` API, and exposes
PBR Neutral; see [Open3D issue #7557](https://github.com/isl-org/Open3D/issues/7557).
SoRoMoX selects Filmic for its default lit rendering: neutral greys
and readable midtones without the legacy ACES highlight tint. The technical
preset selects linear mapping to keep its background white. Upstream's
Uchimura/Reinhard fallback is unchanged. Wheels with these fixes use the
`.soromox3` suffix; run `uv sync --extra rendering` to update an older build.
The color-grading regression test was run with native Metal on Python 3.12;
Ubuntu CI also runs it with surfaceless EGL.

The Linux patches serve the following purposes:

- `linux_distribution_name.patch` keeps the source package's distribution name
  as `open3d`.
- `linux_static_curl.patch` groups the bundled curl and BoringSSL archives for
  linking.
- `linux_surfaceless.patch` selects OpenGL by default for EGL exports and GLX
  viewing. Explicit Vulkan selection remains available.
- `linux_filament_patch_hook.patch` enables EGL support and applies the local
  Filament patch after upstream's Vulkan texture-import patch.
- `filament_linux_dual_context.patch` builds both GLX and EGL-headless platforms,
  selects between them at runtime, uses desktop OpenGL entry points for both,
  and binds the EGL API on worker threads.

Upstream tracking:

| Finding | Upstream discussion |
| --- | --- |
| Modern GUI followed by legacy Visualizer segfault | [Open3D #7553](https://github.com/isl-org/Open3D/issues/7553) |
| Bundled curl/BoringSSL archive grouping | [Open3D #7556](https://github.com/isl-org/Open3D/issues/7556) |
| Desktop OpenGL API binding on EGL worker threads | [Filament #10397](https://github.com/google/filament/issues/10397) |

The distribution-name patch, patch hook, and runtime platform selection support
SoRoMoX's source-build configuration. The tone-mapping, static-curl linking, and
EGL worker-thread fixes address upstream defects.

## Build configuration and overrides

The native builds enable GUI and CPU geometry operations. CUDA, ML framework
integrations, WebRTC, the Jupyter extension, examples, and upstream unit tests
are disabled. Linux builds Filament from source so the GLX/EGL selection fix is
present; macOS builds Metal shaders locally.

Set `SOROMOX_OPEN3D_CACHE` to relocate the cache and
`CMAKE_BUILD_PARALLEL_LEVEL` to change compilation parallelism (default 8).
`SOROMOX_OPEN3D_SOURCE_DIR` and `SOROMOX_OPEN3D_BUILD_DIR` select existing
source/build directories for incremental development. Override source checkouts
must exactly match `OPEN3D_REVISION`.

The adapter's metadata hook is static and platform-neutral. Dependency solvers
can therefore inspect the local source mapping while resolving a universal lock
without launching a native macOS, Linux or Windows build.
