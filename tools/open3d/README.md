# Open3D development build

The macOS dependency builds the current `main` branch of
[Open3D](https://github.com/isl-org/Open3D). Other platforms use the official
`main-devel` wheel channel. `uv.lock` records the version resolved for the checkout.

## Installation on Linux and Windows

From the SoRoMoX checkout, install the rendering extra:

```bash
uv sync --extra rendering
uv run --no-sync python examples/rendering/open3d_studio.py --count 1
```

The repository's `find-links` setting selects official development wheels from
[Open3D main-devel](https://github.com/isl-org/Open3D/releases/tag/main-devel).
No C++ compiler or local readback patch is needed for wheel installations.
Available wheels depend on the Python version and CPU architecture; the channel
currently includes CPython 3.12, 3.13 and 3.14 for Linux x86-64 and Windows x64.
For `pip`, use a virtual environment and select the development wheel explicitly
before installing the rendering extra:

```bash
python -m pip install --upgrade --pre --only-binary=:all: --no-index --find-links https://github.com/isl-org/Open3D/releases/expanded_assets/main-devel --no-deps open3d
python -m pip install -e ".[rendering]"
```

The second command installs Open3D's runtime dependencies as well as SoRoMoX.
Run these commands again to update a `pip` environment. If no wheel matches the
platform, use a supported interpreter/architecture or the source build below.

### Graphics and video requirements

Linux needs a working OpenGL/EGL driver for modern exports. A desktop display
session is needed for `show()` and interactive trajectory playback. On
Debian/Ubuntu, the base runtime libraries and video encoder can be installed with:

```bash
sudo apt install libegl1 libgl1 libgomp1 ffmpeg
```

Use the GPU vendor's driver for hardware rendering. For machines without a GPU,
see Open3D's [CPU rendering instructions](https://www.open3d.org/docs/latest/tutorial/visualization/cpu_rendering.html).
An absent display and an absent graphics driver are separate requirements:
offscreen export does not remove the need for a rendering driver.

On Windows, install a current graphics driver from the GPU vendor and the
[Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).
For MP4 export, install FFmpeg and put its `bin` directory on `PATH`.
`ffmpeg -version` checks that the executable is available on either platform.

### Optional source build on Linux or Windows

A source build is useful when no matching development wheel is available or
when testing upstream changes before wheels are published. The
[upstream build instructions](https://www.open3d.org/docs/latest/compilation.html)
describe supported toolchain versions and additional platform requirements.
Build outside the SoRoMoX checkout and activate the target Python environment
before configuring CMake.

On Ubuntu, install Git, a C++ compiler, CMake (at least 3.24), Ninja and the GLSL
shader compiler, then run upstream's dependency installer:

```bash
sudo apt install git build-essential cmake ninja-build glslang-tools
git clone --depth 1 --branch main https://github.com/isl-org/Open3D.git
cd Open3D
bash util/install_deps_ubuntu.sh
python -m pip install "setuptools>=77" wheel
cmake -S . -B build -G Ninja -DPython3_EXECUTABLE="$(command -v python)" -DBUILD_GUI=ON -DBUILD_CUDA_MODULE=OFF -DBUILD_PYTORCH_OPS=OFF -DBUILD_TENSORFLOW_OPS=OFF -DBUILD_WEBRTC=OFF -DBUILD_JUPYTER_EXTENSION=OFF -DBUNDLE_OPEN3D_ML=OFF -DBUILD_EXAMPLES=OFF -DBUILD_UNIT_TESTS=OFF
cmake --build build --target pip-package --parallel 8
python -m pip install build/lib/python_package/pip_package/open3d-*.whl
```

On Windows, install Git, CMake, Visual Studio 2022 with **Desktop development
with C++** and a Windows SDK. Install the
[Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows) for `glslangValidator`,
and check that `glslangValidator --version` works in the build terminal.
Run the following in PowerShell with the target Python environment activated:

```powershell
git clone --depth 1 --branch main https://github.com/isl-org/Open3D.git
Set-Location Open3D
python -m pip install "setuptools>=77" wheel
$open3dPython = python -c "import sys; print(sys.executable)"
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 "-DPython3_EXECUTABLE=$open3dPython" -DBUILD_GUI=ON -DBUILD_CUDA_MODULE=OFF -DBUILD_PYTORCH_OPS=OFF -DBUILD_TENSORFLOW_OPS=OFF -DBUILD_WEBRTC=OFF -DBUILD_JUPYTER_EXTENSION=OFF -DBUNDLE_OPEN3D_ML=OFF -DBUILD_EXAMPLES=OFF -DBUILD_UNIT_TESTS=OFF
cmake --build build --config Release --target pip-package --parallel 8
Get-ChildItem build/lib/python_package/pip_package/open3d-*.whl | ForEach-Object { python -m pip install $_.FullName }
```

After installing the wheel, return to SoRoMoX, run `python -m pip install -e
".[rendering]"`, and run the studio example with that environment's `python`.
Use `uv run --no-sync` if launching through uv; an ordinary sync restores the
Open3D version selected by `uv.lock`. These source-build recipes follow upstream
instructions; native Linux and Windows builds have not been run on this Mac.

## Installation on macOS

Install Xcode and its required components, then install the Metal Toolchain:

```bash
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -runFirstLaunch
xcodebuild -downloadComponent metalToolchain
xcrun -sdk macosx metal --version
brew install cmake ninja openblas glslang spirv-cross
uv sync --extra rendering
```

The graphical installation path is **Xcode → Settings → Components → Metal
Toolchain → Get**.
If `xcodebuild` reports that a required plugin or framework cannot load, open
Xcode and complete its component installation. Update or reinstall Xcode if it
cannot launch. Administrator authentication takes place on your Mac.
See [Apple's component installation instructions](https://developer.apple.com/documentation/xcode/downloading-and-installing-additional-xcode-components).

The macOS-only `macos_build_backend.py` PEP 517 adapter downloads `main`, compiles the C++ library, Python
bindings and Metal shaders, and produces a Python-specific wheel. The first
build takes several minutes and several gigabytes of disk. The wheel links
against Homebrew OpenBLAS, which must be installed when using it.

## Validation

On Apple Silicon (M4 Max, macOS 26.6.2, Xcode 26.6 with the Metal Toolchain),
source builds at `1a9eb99` with the RGB readback fix passed these native checks:

| Python | Metal shaders | PNG export | MP4 export | Modern `show()` and capture |
| --- | --- | --- | --- | --- |
| 3.12.11 | Compiled locally | Passed | Passed | Passed |
| 3.13.15 | Compiled locally | Passed | Passed | Passed |
| 3.14.7 | Compiled locally | Passed | Passed | Passed |

The window checks open and close two successive views and export an image after
each closure. Video checks encode six frames at 30 FPS. All 149 rendering tests pass under each
Python version; the Python 3.14 suite uses `MPLBACKEND=Agg` for Matplotlib tests.
These results establish
Python compatibility on the tested Mac; they do not validate native Linux or
Windows rendering. The commit identifies the source used for these checks.

## Updating Open3D

To resolve the latest development version and rebuild when needed:

```bash
uv lock --upgrade-package open3d --refresh-package open3d
uv sync --extra rendering
```

A normal sync uses the lockfile and installed/build caches. Resolving a branch
requires a network connection. If upstream changes make the temporary readback
patch inapplicable, the build reports an error so the patch can be reviewed.
The official development release can also remove older wheel assets; refreshing
the dependency selects the current channel contents.

## Temporary Metal readback fix

The current upstream image capture path can abort when it requests RGB8
readback from Metal. `metal_rgb_readback.patch` requests RGBA and strips alpha
before returning the usual RGB image. Patched builds have a version suffix such
as `+<commit>.soromox1`. This suffix identifies the capture fix, independently of
the selected upstream commit.

[Upstream PR #7550](https://github.com/isl-org/Open3D/pull/7550) contains equivalent
handling in the Filament backend. The local patch and build adapter can be
removed when a validated official development wheel includes that fix.

## macOS build configuration

The build enables the GUI and CPU geometry operations; CUDA, ML integrations,
WebRTC and the Jupyter extension are disabled. Builds are cached under
`~/.cache/soromox/open3d`, keyed by the resolved commit, patch, Python and CPU
architecture. A build manifest records that shaders were compiled locally.
Set `SOROMOX_OPEN3D_CACHE` to relocate the cache and
`CMAKE_BUILD_PARALLEL_LEVEL` to change compilation parallelism (default 8).
`SOROMOX_OPEN3D_SOURCE_DIR` and `SOROMOX_OPEN3D_BUILD_DIR` select existing
source/build directories for incremental development. The source checkout must
match current `main` when those overrides are used.

`uv` reads the repository's local source mapping. For `pip` on macOS, install
`tools/open3d` first, then install SoRoMoX's rendering extra. Elsewhere, pass
`--find-links https://github.com/isl-org/Open3D/releases/expanded_assets/main-devel`.
The custom macOS build is not published on PyPI.
