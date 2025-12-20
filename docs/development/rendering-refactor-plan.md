# Plan: Refactor Rendering to Class-Based Architecture with Base Class

## Summary

Create a base class for all renderers and refactor visualization into a class-based architecture with cached forward kinematics.

## Architecture Overview

```
src/soromox/rendering/
├── base.py                      # BaseRobotRenderer (abstract)
├── open3d_renderer.py           # Open3DRenderer (generic, all robots)
├── matplotlib_renderer.py       # MatplotlibRenderer (generic, all robots)
├── animation.py                 # (unchanged utility)
├── planar_hsa/
│   └── opencv_renderer.py       # OpenCVPlanarHSARenderer (HSA-specific)
└── planar_pcs/
    └── opencv_renderer.py       # OpenCVPlanarPCSRenderer (PCS-specific)
```

**Key distinction:**

- **Generic renderers** (Open3D, Matplotlib): Work with ANY continuum soft robot (PlanarPCS, PCS, GVS, etc.)
- **Specialized renderers** (OpenCV): Robot-specific (PlanarHSA, PlanarPCS) with custom drawing logic

## Decisions

- **Rename file**: `open3d_vis_V3.py` → `open3d_renderer.py` (move to rendering root)
- **Add headless mode**: Yes, `render_frame()` method returns image array
- **Backward compatibility**: Not required (breaking change OK)
- **Base class**: Create `BaseRobotRenderer` in `src/soromox/rendering/base.py`

---

## Base Class Design

### Common Functionality Across Renderers

| Feature | OpenCV (HSA) | OpenCV (PCS) | Open3D |
|---------|--------------|--------------|--------|
| Robot system | ✓ | ✓ | ✓ |
| FK via vmap | ✓ | ✓ | ✓ |
| render_frame → image | ✓ | ✓ | (new) |
| render_sequence/animate | ✓ | (via animation.py) | ✓ |
| Coordinate transform | ✓ | ✓ | ✓ |
| num_points | ✓ | ✓ | ✓ |
| Colors config | ✓ | ✓ | ✓ |

### Proposed Base Class

**File**: `src/soromox/rendering/base.py`

```python
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np

class BaseRobotRenderer(ABC):
    """Abstract base class for robot visualization backends."""

    def __init__(
        self,
        robot,
        width: int = 800,
        height: int = 600,
        num_points: int = 50,
        background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        self.robot = robot
        self.width = width
        self.height = height
        self.num_points = num_points
        self.background_color = background_color

        # Cache forward kinematics (batched over arc-length)
        self._batched_fk = jax.vmap(
            robot.forward_kinematics, in_axes=(None, 0)
        )
        self.L_max = float(jnp.sum(robot.L))

    @property
    @abstractmethod
    def is_3d(self) -> bool:
        """Return True for 3D robots, False for 2D/planar robots."""
        pass

    def compute_backbone_curve(self, q: jnp.ndarray) -> jnp.ndarray:
        """Compute backbone points: returns (num_points, 3) or (num_points, 2)."""
        s_ps = jnp.linspace(0.0, self.L_max, self.num_points)
        poses = self._batched_fk(q, s_ps)
        # Extract positions (subclasses implement for 2D vs 3D)
        return self._extract_positions(poses)

    @abstractmethod
    def _extract_positions(self, poses: jnp.ndarray) -> jnp.ndarray:
        """Extract xyz or xy positions from FK poses (backend-specific)."""
        pass

    @abstractmethod
    def render_frame(self, q: jnp.ndarray) -> np.ndarray:
        """Render single configuration, return RGB image array."""
        pass

    def render_sequence(
        self,
        t_list: jnp.ndarray,
        q_list: jnp.ndarray,
        playback_speed: float = 1.0,
        record_path: Optional[str] = None,
    ) -> None:
        """Render animated sequence (interactive or to file)."""
        # Default implementation using render_frame + animation.animate_cv2
        # Subclasses can override for native playback (e.g., Open3D)
        from soromox.rendering.animation import animate_cv2
        animate_cv2(
            rendering_fn=self.render_frame,
            t_ts=np.array(t_list),
            q_ts=np.array(q_list),
            filepath=record_path,
            width=self.width,
            height=self.height,
        )

    def show(self, q: jnp.ndarray) -> None:
        """Display single frame interactively (blocking)."""
        # Default: use matplotlib imshow; subclasses override for native viewers
        import matplotlib.pyplot as plt
        img = self.render_frame(q)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
```

### Methods Summary

| Method | Purpose | Default/Abstract |
|--------|---------|------------------|
| `__init__` | Store robot, cache FK | Provided |
| `is_3d` (property) | True for 3D, False for 2D | **Abstract** |
| `compute_backbone_curve(q)` | Get backbone positions | Provided |
| `_extract_positions(poses)` | Extract xyz/xy from FK | **Abstract** |
| `render_frame(q) → np.ndarray` | Single config → RGB image | **Abstract** |
| `render_sequence(...)` | Video/animation to file | Default: `animate_cv2` |
| `show(q)` | Display frame interactively | Default: matplotlib |
| `animate_matplotlib(...)` | Interactive slider/animation | Provided (uses `is_3d`) |

---

## Matplotlib Animation (Currently Duplicated in Examples)

Multiple example files contain duplicated `animate_robot_matplotlib()` functions with:

- `FuncAnimation` mode (auto-play)
- Slider mode (manual frame scrubbing)
- 2D (planar) and 3D variants

**Recommendation**: Add `animate_matplotlib()` method to `BaseRobotRenderer`:

```python
def animate_matplotlib(
    self,
    t_list: jnp.ndarray,
    q_list: jnp.ndarray,
    interval: int = 50,
    mode: str = "slider",  # "slider" or "animation"
    show: bool = True,
) -> Optional[HTML]:
    """Interactive matplotlib animation with slider or auto-play.

    Args:
        t_list: Time stamps (T,)
        q_list: Configurations (T, DOF)
        interval: Frame interval in ms (for animation mode)
        mode: "slider" for manual scrubbing, "animation" for auto-play
        show: Whether to call plt.show()
    """
    # Uses compute_backbone_curve() and _is_3d property
    ...
```

This consolidates code currently duplicated in:

- `examples/simulation/pcs/simulate_pcs.py`
- `examples/simulation/pcs/simulate_planar_pcs.py`
- `examples/simulation/gvs/simulate_gvs.py`
- ...and 5+ other example files

---

## Current State Analysis

The `open3d_vis_V3.py` (~700 lines) is function-based with:

- **Monolithic `animate_robot_tendons_open3d()` function** (~350 lines) handling all rendering
- **Helper functions** for geometry creation (`_make_sphere`, `_make_polyline_lineset`, etc.)
- **No FK caching** - vmapped FK functions are recreated on every call
- **State managed via closures** (`state = {"idx": 0, "playing": True, ...}`)

## Open3D Renderer (Generic - All Robots)

**File**: `src/soromox/rendering/open3d_renderer.py`

```python
from soromox.rendering.base import BaseRobotRenderer

class Open3DRenderer(BaseRobotRenderer):
    """Open3D visualization for any continuum soft robot (PCS, GVS, etc.)."""

    def __init__(
        self,
        robot,
        width: int = 1280,
        height: int = 800,
        num_points: int = 80,
        seg_colors: Tuple = ((0.1, 0.45, 1.0), (0.0, 0.6, 0.2)),
        sphere_resolution: int = 32,
        tendon_color: Tuple = (0.9, 0.15, 0.15),
        tendon_line_width: float = 2.0,
        ...
    ):
        super().__init__(robot, width, height, num_points, background_color)

        # Open3D-specific: cache tendon FK
        self._batched_fk_tendons = jax.vmap(
            robot.forward_kinematics_tendons, in_axes=(None, 0), out_axes=1
        )

        # Store Open3D-specific config
        self.seg_colors = seg_colors
        self.sphere_resolution = sphere_resolution
        self.tendon_color = tendon_color

    # ---- Required abstract implementations ----
    def _extract_positions(self, poses: jnp.ndarray) -> jnp.ndarray:
        """Extract xyz from SE(3) matrices."""
        return poses[:, :3, 3]  # (N, 3)

    def render_frame(self, q: jnp.ndarray) -> np.ndarray:
        """Render single config headlessly, return RGB array."""
        # Use Open3D offscreen rendering
        ...

    # ---- Override base methods for native Open3D experience ----
    def show(self, q: jnp.ndarray) -> None:
        """Interactive single-frame viewer with camera controls."""
        # Native Open3D window (overrides matplotlib default)
        ...

    def render_sequence(
        self,
        t_list: jnp.ndarray,
        q_list: jnp.ndarray,
        playback_speed: float = 1.0,
        loop: bool = False,
        record_dir: Optional[str] = None,
        ...
    ) -> None:
        """Native Open3D animated playback with keyboard controls."""
        # Override base to use native Open3D animation
        ...

    # ---- Open3D-specific methods ----
    def compute_tendon_curves(self, q: jnp.ndarray) -> jnp.ndarray:
        """Compute tendon paths (Open3D-specific)."""
        s_ps = jnp.linspace(0.0, self.L_max, self.num_points)
        return self._batched_fk_tendons(q, s_ps)
```

### Key Design Decisions

1. **Inherits FK caching from base** - `self._batched_fk` is set up by `BaseRobotRenderer.__init__`
2. **Adds tendon FK caching** - Open3D-specific for tendon visualization
3. **Overrides `show()` and `render_sequence()`** - uses native Open3D windows instead of matplotlib/animate_cv2
4. **Implements `render_frame()` for headless** - uses Open3D offscreen rendering

### Geometry Helpers (Module-Level)

Keep as standalone functions (stateless factories):

```python
def _make_sphere(...) -> o3d.geometry.TriangleMesh: ...
def _make_polyline_lineset(...) -> o3d.geometry.LineSet: ...
def _make_base_plate(...) -> o3d.geometry.TriangleMesh: ...
def _make_target_sphere(...) -> o3d.geometry.TriangleMesh: ...
def _make_obstacle_sphere(...) -> o3d.geometry.TriangleMesh: ...
def _split_counts_by_lengths(...) -> List[int]: ...
def _expand_colors(...) -> List[Tuple]: ...
def _mesh_center(...) -> np.ndarray: ...
```

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/soromox/rendering/base.py` | **Create** - BaseRobotRenderer |
| `src/soromox/rendering/open3d_renderer.py` | **Create** - Open3DRenderer (generic) |
| `src/soromox/rendering/matplotlib_renderer.py` | **Create** - MatplotlibRenderer (generic) |
| `src/soromox/rendering/__init__.py` | **Edit** - export classes |
| `src/soromox/rendering/planar_hsa/open3d_vis_V3.py` | **Delete** |
| `src/soromox/rendering/planar_hsa/opencv_renderer.py` | **Refactor** - OpenCVPlanarHSARenderer |
| `src/soromox/rendering/planar_pcs/opencv_renderer.py` | **Refactor** - OpenCVPlanarPCSRenderer |
| `examples/simulation/pcs/*.py` | **Edit** - use MatplotlibRenderer |
| `examples/simulation/gvs/*.py` | **Edit** - use MatplotlibRenderer |

---

## Implementation Steps

### Phase 1: Base Class

1. Create `src/soromox/rendering/base.py` with `BaseRobotRenderer`:
   - `__init__` with robot, dimensions, num_points, colors
   - Cached `_batched_fk` via vmap
   - Abstract: `is_3d`, `_extract_positions()`, `render_frame()`
   - Default: `show()`, `render_sequence()`

2. Update `src/soromox/rendering/__init__.py` to export all renderer classes

### Phase 2: Generic Matplotlib Renderer

3. Create `src/soromox/rendering/matplotlib_renderer.py`:
   - Class `MatplotlibRenderer(BaseRobotRenderer)` for any robot
   - Implement `is_3d` based on robot FK output dimensionality
   - Implement `render_frame()` returning RGB array from matplotlib figure
   - Implement `animate()` with slider and FuncAnimation modes
   - Consolidates duplicated code from examples

### Phase 3: Generic Open3D Renderer

4. Create `src/soromox/rendering/open3d_renderer.py`:
   - Class `Open3DRenderer(BaseRobotRenderer)` for any robot
   - Optional tendon FK caching (if robot supports `forward_kinematics_tendons`)
   - Implement `is_3d` → True
   - Implement `render_frame()` with Open3D offscreen rendering
   - Override `show()` for native Open3D viewer
   - Override `render_sequence()` for native playback with keyboard controls
   - Keep module-level geometry helpers (`_make_sphere`, etc.)

5. Delete old `src/soromox/rendering/planar_hsa/open3d_vis_V3.py`

### Phase 4: Specialized OpenCV Renderers

6. Refactor `src/soromox/rendering/planar_hsa/opencv_renderer.py`:
   - Create `OpenCVPlanarHSARenderer(BaseRobotRenderer)`
   - Implement `is_3d` → False (planar)
   - Implement `_extract_positions()` for SE(2) → (x, y)
   - Move `draw_robot()` logic into `render_frame()`

7. Refactor `src/soromox/rendering/planar_pcs/opencv_renderer.py`:
   - Create `OpenCVPlanarPCSRenderer(BaseRobotRenderer)`
   - Implement `is_3d` → False (planar)
   - Move `render_planar_pcs()` logic into `render_frame()`

### Phase 5: Update Examples

8. Update example files to use `MatplotlibRenderer`:
   - Remove duplicated `draw_robot_curve` and `animate_robot_matplotlib` functions
   - Create `MatplotlibRenderer(robot)` and call `renderer.animate()`
