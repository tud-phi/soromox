# Open3D development build

SoRoMoX builds an immutable, checked-in revision of Open3D `main` on macOS and
Linux x86-64. The cross-platform PEP 517 adapter is
[`build_backend.py`](build_backend.py); it applies the temporary renderer fixes
in this directory and caches one native wheel per revision, patch set, Python
ABI, operating system, and CPU architecture. Other platforms use the official
[Open3D main-devel](https://github.com/isl-org/Open3D/releases/tag/main-devel)
wheel channel.

`OPEN3D_REVISION` makes a lock reproducible. The scheduled
`Check Open3D upstream revision` workflow compares that pin with upstream
`main` each day and fails when upstream advances. Maintainers can then update
the pin and rerun the native validation deliberately, keeping each build
immutable without leaving the project on an old development snapshot.

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
xcrun -sdk macosx metal --version
brew install cmake ninja openblas glslang spirv-cross
uv sync --extra rendering
```

The graphical route for the optional compiler is **Xcode → Settings →
Components → Metal Toolchain → Get**. The adapter compiles the C++ library,
Python bindings, Metal shaders, and a Python-specific wheel. The wheel links to
Homebrew OpenBLAS.

## Windows and other platforms

The local adapter currently supports macOS and Linux x86-64. On other platforms,
`uv` resolves a matching official development wheel when one is published. With
`pip`, install that wheel before SoRoMoX:

```bash
python -m pip install --upgrade --pre --only-binary=:all: --no-index \
  --find-links https://github.com/isl-org/Open3D/releases/expanded_assets/main-devel \
  --no-deps open3d
python -m pip install -e ".[rendering]"
```

Available wheel ABIs and architectures are controlled by upstream. On Windows,
also install a current GPU driver, the Microsoft Visual C++ Redistributable, and
FFmpeg on `PATH` for MP4 export. Native Windows rendering has not been validated
for this checkout.

## Validation

The current revision, `1a9eb99`, was source-built on Ubuntu 26.04.1 x86-64 with
GCC 15.2 and tested using Mesa software OpenGL/EGL:

| Python | Source wheel | Native surfaceless render | SoRoMoX PNG + MP4 | Xvfb `show()` + playback |
| --- | --- | --- | --- | --- |
| 3.11.15 | Passed | Passed | Passed | Passed |
| 3.12.13 | Passed | Passed | Passed | Passed |
| 3.13.15 | Passed | Passed | Passed | Passed |
| 3.14.7 | Passed | Passed | Passed | Passed |

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
test disables VSM shadows and SSAO. Full default lighting and shadows passed in
the surfaceless image and video checks.

Open3D can abort during interpreter finalization if its modern GUI and legacy
visualizer are both initialized sequentially in one process. The integration
tests run those two otherwise-successful viewer checks in fresh subprocesses.
Applications should likewise avoid mixing the two Open3D GUI systems in one
process until upstream fixes their teardown interaction.

The same Open3D revision was previously source-built and validated on Apple
Silicon (M4 Max, macOS 26.6.2, Xcode 26.6) with Python 3.12.11, 3.13.15, and
3.14.7. PNG export, MP4 export, compiled Metal shaders, and modern `show()` all
passed.

## Updating Open3D main

Check whether the pin still matches upstream:

```bash
python tools/open3d/update_revision.py
```

When the scheduled check reports a new commit, update deliberately and rebuild:

```bash
python tools/open3d/update_revision.py --update
uv lock --upgrade-package open3d --refresh-package open3d
uv sync --extra rendering
```

Re-run the Ubuntu integration tests and the supported-Python source-build matrix
before merging the new pin. If a patch no longer applies, the adapter stops
instead of silently producing a wheel with unreviewed behavior. This process is
intended to follow Open3D `main` rapidly until the next stable release contains
the required fixes.

## Temporary patches and upstreaming

`metal_rgb_readback.patch` requests RGBA readback on Metal and strips alpha
before returning the normal RGB image. Open3D
[PR #7550](https://github.com/isl-org/Open3D/pull/7550) contains equivalent
handling.

The Linux patches make the source package retain the `open3d` distribution name,
fix current static-curl linking and Filament build integration, avoid creating
the optional Gaussian-splat sharing context in surfaceless mode, and build both
GLX and EGL-headless Filament platforms. The Filament changes also re-bind the
EGL API on worker threads, use a pbuffer-compatible configuration, and avoid a
desktop-GL extension query that can return null.

Upstream tracking:

| Finding | Upstream discussion |
| --- | --- |
| Metal RGB readback | [Validation comment on PR #7550](https://github.com/isl-org/Open3D/pull/7550#issuecomment-5595190113) |
| Modern GUI followed by legacy Visualizer segfault | [Open3D #7553](https://github.com/isl-org/Open3D/issues/7553) |
| Surfaceless rendering initializes splat GLX contexts | [Open3D #7554](https://github.com/isl-org/Open3D/issues/7554) |
| Linux Filament archive byproduct paths | [Open3D #7555](https://github.com/isl-org/Open3D/issues/7555) |
| Bundled curl/BoringSSL archive grouping | [Open3D #7556](https://github.com/isl-org/Open3D/issues/7556) |
| Desktop OpenGL API binding on EGL worker threads | [Filament #10397](https://github.com/google/filament/issues/10397) |

The reports distinguish local patched integration results from upstream builds.
The mixed-GUI segfault was reproduced in a fresh macOS process; a legacy-only
control succeeded. PR #7550 itself has not been built or validated here.
Current Filament main already guards null extension strings and has revised
swapchain selection; the older patch hunks need reassessment when Open3D updates
its embedded Filament. The distribution-name patch, patch hook and runtime
platform-selection policy support the downstream build and are not all
independent upstream defects.

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
without launching a native macOS or Linux build.
