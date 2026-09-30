# Third-party dependency builds

This directory holds source build recipes and patches for dependencies that
need upstream compatibility fixes.

- [`open3d`](open3d/README.md): Open3D renderer patches and platform build recipe.

uv selects the Open3D build recipe through `tool.uv.sources`; pip users install
`./third_party/open3d` before project extras that require it.
