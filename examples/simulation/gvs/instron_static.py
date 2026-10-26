"""Sequential force-controlled static equilibrium validation of the N6 module.

Every measured Instron force level is applied to the N6 module as an external
load and the static equilibrium of the reduced GVS model is solved for that
load alone. The equilibrium is verified (force balance, constraints, stability
of the tangent), the resulting displacement is measured, the configuration is
held in Viser for a few seconds, one point is added to the force-displacement
graph, and the next force level follows. No time integration, mass or damping
enters: the equilibrium of the linear lattice law under a given load is unique,
so each level is an independent static problem. The previous level's solution
is used only as the Newton initial guess.

Equilibrium (SoRoMoX convention, inertial-frame tip Jacobian ``J_tip``)::

    K q + G(q) - J_tip(q)^T W(q, F) = 0,   c(q) = 0

``K`` is the characterized linear GVS stiffness of the helix reduction, ``G``
the gravity gradient (zero for the gravity-free N6 statics), ``W`` the external
wrench about the backbone tip in world axes and ``c`` the geometric constraints
of the experiment:

* axial compression (module upright on a fixed base): ``W = [m; f]`` with
  ``f_z = -F`` and the five platen reactions ``[m_x, m_y, m_z, f_x, f_y]`` as
  unknowns, determined by the platen constraint ``c = [p_x, p_y, rot_x,
  rot_y, rot_z] = 0`` of the top ring (translation along the axis only);
* tendon bending (module horizontal, one ring clamped): ``W = [r(q) x T t(q);
  T t(q)]`` with ``t(q)`` the unit chord from the free-ring hole to the
  clamped-ring hole, ``r(q)`` the moment arm from the backbone tip to the hole
  and ``T = max(F, 0)`` (a string cannot push). No constraint at the free end.

Run from the checkout root::

    JAX_PLATFORMS=cpu python examples/simulation/gvs/instron_static.py

Both experiments run by default with a Viser window; ``--no-viser`` runs
headless, ``--hold`` sets the seconds each equilibrium is held on screen.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import webbrowser
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jax import Array

try:
    from examples.simulation.gvs import n6_instron_model as model
except ModuleNotFoundError:  # executed as a script
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from examples.simulation.gvs import n6_instron_model as model
from soromox.utils.lie_algebra import so3

N6ModuleSpec = model.N6ModuleSpec

jax.config.update("jax_enable_x64", True)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
RESULTS_DIR = SCRIPT_DIR / "results" / "instron" / "static"
COMPRESSION_CSV = REPO_ROOT / "data" / "instron" / "Specimen_RawData_1.csv"
BENDING_CSV = REPO_ROOT / "data" / "instron_bending" / "Specimen_RawData_1.csv"

ExperimentName = Literal["compression", "bending"]
#: Default force caps [N] = validity limits of the linear model (documented, not fitted):
#: compression: the top ring reaches the base near 46 N; bending: the free ring
#: collides with the clamped ring near 8.6 N. Pass --max-force to override.
DEFAULT_MAX_FORCE: dict[str, float] = {"compression": 45.0, "bending": 8.0}

__all__ = [
    "BendingExperiment",
    "CompressionExperiment",
    "EquilibriumRecord",
    "ForceSample",
    "InstronBranch",
    "StaticSolver",
    "ViserViewer",
    "branches",
    "build_experiment",
    "detect_cycles",
    "load_branches",
    "read_instron_csv",
    "run_experiment",
    "select_samples",
    "step_forces",
]


# --------------------------------------------------------------------------- #
# Instron CSV export: loading / unloading branches of one crosshead cycle
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InstronBranch:
    """One loading or unloading branch of a displacement-controlled test."""

    source: str
    cycle: int
    branch: str
    rows: tuple[int, int]
    time_s: np.ndarray
    extension_mm: np.ndarray
    load_N: np.ndarray

    @property
    def num_samples(self) -> int:
        return int(self.time_s.size)


def read_instron_csv(path: Path) -> dict[str, Any]:
    """Read ``Time, Extension, Load`` with the Instron units row."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = [h.strip() for h in next(reader)]
        units = [u.strip().strip("()") for u in next(reader)]
        data = np.array([[float(v) for v in row] for row in reader if row])
    columns = {name: data[:, i] for i, name in enumerate(header)}
    unit_of = dict(zip(header, units, strict=True))
    if unit_of["Time"] != "s" or unit_of["Extension"] != "mm" or unit_of["Load"] != "N":
        raise ValueError(f"Unexpected units in {path.name}: {unit_of}")
    return {
        "path": str(path),
        "time_s": columns["Time"],
        "extension_mm": columns["Extension"],
        "load_N": columns["Load"],
        "units": unit_of,
        "sample_period_s": float(np.median(np.diff(columns["Time"]))),
    }


def detect_cycles(extension: np.ndarray, prominence_fraction: float = 0.25) -> list[dict[str, int]]:
    """Trough -> peak -> trough cycles of the crosshead extension."""
    from scipy.signal import find_peaks

    prominence = prominence_fraction * float(np.ptp(extension))
    peaks, _ = find_peaks(extension, prominence=prominence)
    troughs, _ = find_peaks(-extension, prominence=prominence)
    cycles, start = [], 0
    for peak in peaks:
        later = troughs[troughs > peak]
        end = int(later[0]) if later.size else extension.size - 1
        cycles.append({"index": len(cycles) + 1, "start": start, "peak": int(peak), "end": end})
        start = end
    return cycles


def branches(data: dict[str, Any], cycle: int, label: str) -> tuple[InstronBranch, InstronBranch, dict[str, int]]:
    cycles = detect_cycles(data["extension_mm"])
    c = cycles[cycle - 1]
    load_rows = (c["start"], c["peak"])
    unload_rows = (c["peak"], c["end"])

    def make(rows: tuple[int, int], branch: str) -> InstronBranch:
        sl = slice(rows[0], rows[1] + 1)
        return InstronBranch(
            source=label,
            cycle=cycle,
            branch=branch,
            rows=rows,
            time_s=data["time_s"][sl],
            extension_mm=data["extension_mm"][sl],
            load_N=data["load_N"][sl],
        )

    return make(load_rows, "loading"), make(unload_rows, "unloading"), c


@dataclass(frozen=True)
class ForceSample:
    """One applied force level taken from the Instron export."""

    index: int  # position in the schedule
    row: int  # row of the CSV
    branch: str  # loading / unloading
    time_s: float
    force_N: float
    measured_displacement_mm: float


def load_branches(path: Path, cycle: int = 1) -> tuple[InstronBranch, InstronBranch, dict[str, Any]]:
    data = read_instron_csv(path)
    loading, unloading, cyc = branches(data, cycle, path.name)
    return loading, unloading, {"data": data, "cycle": cyc}


def _samples_from_branch(branch: InstronBranch, keep: np.ndarray, start_index: int) -> list[ForceSample]:
    return [
        ForceSample(
            index=start_index + i,
            row=int(branch.rows[0] + k),
            branch=branch.branch,
            time_s=float(branch.time_s[k]),
            force_N=float(branch.load_N[k]),
            measured_displacement_mm=float(branch.extension_mm[k]),
        )
        for i, k in enumerate(keep)
    ]


def select_samples(
    loading: InstronBranch,
    *,
    num_levels: int = 30,
    decimate: int | None = None,
    max_force: float | None = None,
) -> list[ForceSample]:
    """Force levels of the loading branch, in time order.

    With ``decimate`` every n-th measured sample (plus the last one) below the
    force cap is replayed. Otherwise ``num_levels`` representative levels are
    spaced evenly along the measured extension, which spreads the points over
    the whole force-displacement curve, including its flat and steep parts.
    """
    if decimate is not None:
        decimate = max(1, int(decimate))
        keep = np.arange(0, loading.num_samples, decimate)
        if keep[-1] != loading.num_samples - 1:
            keep = np.r_[keep, loading.num_samples - 1]
        if max_force is not None:
            keep = keep[loading.load_N[keep] <= max_force]
        return _samples_from_branch(loading, keep, 0)
    if num_levels < 2:
        raise ValueError("num_levels must be >= 2.")
    x = loading.extension_mm
    candidates = np.arange(loading.num_samples)
    if max_force is not None:
        # Restrict to the part of the branch below the cap BEFORE spacing,
        # so the levels stay dense where the model is valid.
        envelope = np.maximum.accumulate(loading.load_N)
        candidates = candidates[envelope <= max_force]
        if candidates.size < 2:
            raise ValueError(f"max_force = {max_force} N leaves fewer than two samples.")
    xs = x[candidates]
    targets = np.linspace(xs.min(), xs.max(), num_levels)
    keep = sorted({int(candidates[np.argmin(np.abs(xs - t))]) for t in targets})
    return _samples_from_branch(loading, np.asarray(keep), 0)


# --------------------------------------------------------------------------- #
# Experiments: external wrench, constraints, measurement, validity
# --------------------------------------------------------------------------- #
class _Experiment:
    """Load, constraints and measurements of one Instron experiment."""

    name: ExperimentName
    primary_label: str
    axis_labels: tuple[str, str]
    title: str
    num_wrench_unknowns: int = 0  # reaction components solved alongside q
    num_constraints: int = 0

    robot: Any
    spec: N6ModuleSpec

    @property
    def n(self) -> int:
        return int(self.robot.num_coordinates)

    def tip_pose(self, q: Array) -> Array:
        return self.robot.forward_kinematics_tips(q)[-1]

    def wrench(self, q: Array, load: Array, unknowns: Array) -> Array:
        """External wrench ``[m; f]`` about the backbone tip, world axes."""
        raise NotImplementedError

    def constraints(self, q: Array) -> Array:
        return jnp.zeros(0)

    def measure(self, q: np.ndarray) -> dict[str, float]:
        raise NotImplementedError

    def validity(self, q: np.ndarray, measures: dict[str, float]) -> tuple[bool, str]:
        return True, ""

    def string_points(self, q: np.ndarray) -> np.ndarray | None:
        """Polyline of the string for the Viser scene, shape ``(K, 3)``, or None."""
        return None


class CompressionExperiment(_Experiment):
    """Upright N6 between a fixed base and a platen that moves along the axis."""

    name: ExperimentName = "compression"
    primary_label = "compression_m"
    axis_labels = ("Compression displacement (mm)", "Applied compressive force (N)")
    title = "N6 axial compression: static force-controlled equilibria vs Instron"
    num_wrench_unknowns = 5  # [m_x, m_y, m_z, f_x, f_y] platen reactions
    num_constraints = 5  # [p_x, p_y, rot_x, rot_y, rot_z] of the top ring

    def __init__(self, spec: N6ModuleSpec | None = None) -> None:
        self.spec = spec or N6ModuleSpec()
        _, self.helix, self.robot = model.build_upright_n6_module(self.spec)
        q0 = jnp.zeros(self.n)
        pose0 = self.robot.forward_kinematics_tips(q0)[-1]
        self.reference_rotation = pose0[:3, :3]
        self.reference_tip_z = float(pose0[2, 3])

    def wrench(self, q: Array, load: Array, unknowns: Array) -> Array:
        full = jnp.zeros(6, dtype=unknowns.dtype).at[:5].set(unknowns)
        return full.at[5].set(-load)  # positive F = downward (world -z) load

    def constraints(self, q: Array) -> Array:
        pose = self.tip_pose(q)
        relative = self.reference_rotation.T @ pose[:3, :3]
        return jnp.concatenate((pose[:2, 3], so3.vee((relative - relative.T) / 2)))

    def measure(self, q: np.ndarray) -> dict[str, float]:
        pose = np.asarray(self.tip_pose(jnp.asarray(q)))
        tip_z = float(pose[2, 3])
        relative = np.asarray(self.reference_rotation).T @ pose[:3, :3]
        angle = float(np.linalg.norm(np.asarray(so3.log(jnp.asarray(relative)))))
        return {
            "compression_m": self.reference_tip_z - tip_z,
            "tip_z_m": tip_z,
            "lateral_offset_m": float(np.linalg.norm(pose[:2, 3])),
            "tip_rotation_rad": angle,
        }

    def validity(self, q: np.ndarray, measures: dict[str, float]) -> tuple[bool, str]:
        if measures["tip_z_m"] <= 0.0:
            return False, "top ring at or below the base (tip z <= 0): linear model outside its range"
        return True, ""


class BendingExperiment(_Experiment):
    """Horizontal N6, one ring clamped, string from the free ring through the clamped ring."""

    name: ExperimentName = "bending"
    primary_label = "travel_m"
    axis_labels = ("Tendon (string) travel (mm)", "Instron force (N)")
    title = "N6 tendon-driven bending: static force-controlled equilibria vs Instron"

    def __init__(
        self,
        spec: N6ModuleSpec | None = None,
        bending_config: model.TendonBendingConfig | None = None,
    ) -> None:
        self.spec = spec or N6ModuleSpec()
        self.bending_config = bending_config or model.TendonBendingConfig()
        _, self.helix, self.robot = model.build_horizontal_n6_module(self.spec, self.bending_config)
        self.path = model.TendonPath(self.robot, self.spec, self.bending_config)
        self.load = model.TendonLoad(self.path)
        self._monitor = model.RingMonitor(self.path)

    def tension(self, load: Array) -> Array:
        """Anchor tension: the string cannot push (tension-only)."""
        return jnp.maximum(jnp.asarray(load), 0.0)

    def wrench(self, q: Array, load: Array, unknowns: Array) -> Array:
        return self.load.wrench(q, self.tension(load))

    def measure(self, q: np.ndarray) -> dict[str, float]:
        q_j = jnp.asarray(q)
        pose = self.path.free_end_pose(q_j)
        direction = np.asarray(self.load.direction(q_j), float)
        arm = np.asarray(self.load.moment_arm(q_j), float)
        geometry = self._monitor.evaluate(q_j)
        return {
            "travel_m": float(self.path.shortening(q_j)),
            "tendon_length_m": float(self.path.length(q_j)),
            "internal_chord_m": float(geometry["internal_chord"]),
            "ring_dx_m": float(pose["displacement"][0]),
            "ring_dy_m": float(pose["displacement"][1]),
            "ring_dz_m": float(pose["displacement"][2]),
            "ring_displacement_m": float(np.linalg.norm(pose["displacement"])),
            "bending_angle_rad": float(pose["rotation_angle"]),
            "direction_x": float(direction[0]),
            "direction_y": float(direction[1]),
            "direction_z": float(direction[2]),
            "tendon_angle_from_axis_rad": float(math.acos(max(-1.0, min(1.0, -direction[0])))),
            "lever_arm_m": float(np.linalg.norm(np.cross(arm, direction))),
            "ring_collision": float(geometry["ring_collision"]),
        }

    def validity(self, q: np.ndarray, measures: dict[str, float]) -> tuple[bool, str]:
        if measures["ring_collision"] > 0.5:
            return False, "free ring collides with the clamped ring"
        if measures["internal_chord_m"] <= 2e-3:
            return False, "internal chord below 2 mm (rings touching)"
        if measures["bending_angle_rad"] > math.radians(170.0):
            return False, "excessive rotation"
        return True, ""

    def string_points(self, q: np.ndarray) -> np.ndarray:
        q_j = jnp.asarray(q)
        return np.asarray(
            jnp.stack(
                (
                    self.path.free_hole_opening(q_j),
                    self.path.clamped_hole_opening(),
                    self.path.clamped_hole_exit(),
                    self.path.redirect_point(),
                    self.path.grip_point(),
                )
            )
        )


def build_experiment(name: ExperimentName, spec: N6ModuleSpec | None = None) -> _Experiment:
    if name == "compression":
        return CompressionExperiment(spec)
    if name == "bending":
        return BendingExperiment(spec)
    raise ValueError(f"Unknown experiment {name!r}.")


# --------------------------------------------------------------------------- #
# Static equilibrium solver (Newton on q and the reaction unknowns)
# --------------------------------------------------------------------------- #
class StaticSolver:
    """Newton statics of one experiment for a prescribed load.

    Unknowns are the GVS coordinates ``q`` plus the reaction wrench components
    the experiment leaves free. The residual stacks the generalized static
    balance ``K q + G(q) - J_tip(q)^T W = 0`` with the experiment's geometric
    constraints; the system is square because every reaction unknown pairs
    with one constraint.
    """

    def __init__(
        self,
        experiment: _Experiment,
        *,
        rtol: float = 1e-9,
        atol: float = 1e-10,
        max_newton_steps: int = 50,
    ) -> None:
        if experiment.num_wrench_unknowns != experiment.num_constraints:
            raise ValueError("Reaction unknowns and constraints must pair one to one.")
        self.experiment = experiment
        self.robot = experiment.robot
        self.num_q = experiment.n
        self.num_unknowns = self.num_q + experiment.num_wrench_unknowns
        self.solver = optx.Newton(rtol=rtol, atol=atol)
        self.max_newton_steps = max_newton_steps
        self._solve = jax.jit(self._solve_impl)
        self._tangent = jax.jit(self._tangent_impl)

    # -- model pieces ------------------------------------------------------- #
    def generalized_residual(self, q: Array, wrench: Array) -> Array:
        robot = self.robot
        return (
            robot.elastic_force(q)
            + robot.gravitational_force(q)
            - robot.jacobian_tips(q)[-1].T @ wrench
        )

    def tau_ext(self, q: Array, wrench: Array) -> Array:
        return self.robot.jacobian_tips(q)[-1].T @ wrench

    def residual(self, x: Array, load: Array) -> Array:
        q, unknowns = x[: self.num_q], x[self.num_q :]
        wrench = self.experiment.wrench(q, load, unknowns)
        parts = [self.generalized_residual(q, wrench)]
        if self.experiment.num_constraints:
            parts.append(self.experiment.constraints(q))
        return jnp.concatenate(parts)

    # -- jitted kernels ------------------------------------------------------ #
    def _solve_impl(self, x0: Array, load: Array):
        solution = optx.root_find(
            lambda x, args: self.residual(x, args),
            self.solver,
            x0,
            args=load,
            max_steps=self.max_newton_steps,
            throw=False,
        )
        x = solution.value
        return (
            x,
            solution.result == optx.RESULTS.successful,
            solution.stats["num_steps"],
            jnp.linalg.norm(self.residual(x, load)),
        )

    def _tangent_impl(self, q: Array, wrench: Array):
        stiffness = jax.jacfwd(lambda q_: self.generalized_residual(q_, wrench))(q)
        constraint_jacobian = jax.jacfwd(self.experiment.constraints)(q)
        return stiffness, constraint_jacobian

    # -- public API ---------------------------------------------------------- #
    def initial_guess(self, q: Array | None = None) -> np.ndarray:
        x = np.zeros(self.num_unknowns)
        if q is not None:
            x[: self.num_q] = np.asarray(q)
        return x

    def solve(self, x0: np.ndarray, load: float) -> tuple[np.ndarray, bool, int, float]:
        x, ok, steps, norm = self._solve(jnp.asarray(x0), jnp.asarray(load, float))
        return np.asarray(x), bool(ok), int(steps), float(norm)

    def unpack(self, x: np.ndarray, load: float) -> tuple[np.ndarray, np.ndarray]:
        q, unknowns = x[: self.num_q], x[self.num_q :]
        wrench = self.experiment.wrench(jnp.asarray(q), jnp.asarray(load, float), jnp.asarray(unknowns))
        return q, np.asarray(wrench)

    def verify(self, q: np.ndarray, wrench: np.ndarray) -> dict[str, float]:
        """Force balance, constraint residual and tangent stability at ``(q, W)``."""
        q_j, w_j = jnp.asarray(q), jnp.asarray(wrench)
        residual = np.asarray(self.generalized_residual(q_j, w_j))
        tau = np.asarray(self.tau_ext(q_j, w_j))
        constraint = np.asarray(self.experiment.constraints(q_j))
        stiffness, constraint_jacobian = self._tangent(q_j, w_j)
        stiffness = np.asarray(stiffness)
        stiffness = (stiffness + stiffness.T) / 2
        if self.experiment.num_constraints:
            # Stability on the constraint null space: a sign change along the
            # load path marks loss of stability of the followed branch.
            active = np.asarray(constraint_jacobian)
            _, singular, vh = np.linalg.svd(active, full_matrices=True)
            rank = int(np.sum(singular > 1e-12 * max(singular.max(), 1e-300)))
            stiffness = vh[rank:] @ stiffness @ vh[rank:].T
        return {
            "force_residual": float(np.linalg.norm(residual)),
            "force_residual_rel": float(np.linalg.norm(residual) / max(np.linalg.norm(tau), 1e-9)),
            "tau_ext_norm": float(np.linalg.norm(tau)),
            "constraint_residual": float(np.linalg.norm(constraint)) if constraint.size else 0.0,
            "min_tangent_eigenvalue": float(np.linalg.eigvalsh(stiffness).min()) if stiffness.size else math.inf,
        }


# --------------------------------------------------------------------------- #
# Equilibrium records
# --------------------------------------------------------------------------- #
#: Verification tolerances of a recorded equilibrium.
FORCE_RESIDUAL_REL_TOL = 1e-6
CONSTRAINT_TOL = 1e-8


@dataclass
class EquilibriumRecord:
    sample: ForceSample
    converged: bool
    valid: bool
    reason: str
    newton_steps: int
    residual_norm: float
    verification: dict[str, float]
    wrench: np.ndarray
    measures: dict[str, float]
    q: np.ndarray
    wall_time_s: float

    @property
    def primary(self) -> float:
        return float(next(iter(self.measures.values())))

    def to_row(self, primary_label: str) -> dict[str, Any]:
        s = self.sample
        row: dict[str, Any] = {
            "step": s.index,
            "csv_row": s.row,
            "instron_time_s": s.time_s,
            "force_N": s.force_N,
            "measured_displacement_mm": s.measured_displacement_mm,
            "static_displacement_mm": 1e3 * self.measures[primary_label],
            "static_minus_measured_mm": 1e3 * self.measures[primary_label] - s.measured_displacement_mm,
            "converged": self.converged,
            "valid": self.valid,
            "reason": self.reason,
            "newton_steps": self.newton_steps,
            "residual_norm": self.residual_norm,
            **self.verification,
            "tangent_positive_definite": self.verification["min_tangent_eigenvalue"] > 0.0,
            "wall_time_s": self.wall_time_s,
        }
        for key, value in self.measures.items():
            if key == primary_label:
                continue
            if key.endswith("_m"):
                row[key[:-2] + "_mm"] = 1e3 * value
            elif key.endswith("_rad"):
                row[key[:-4] + "_deg"] = math.degrees(value)
            else:
                row[key] = value
        for i, component in enumerate(("m_x", "m_y", "m_z", "f_x", "f_y", "f_z")):
            row[f"wrench_{component}"] = float(self.wrench[i])
        return row


# --------------------------------------------------------------------------- #
# Viser: show each equilibrium, hold it, grow the graph
# --------------------------------------------------------------------------- #
class ViserViewer:
    """Live Viser scene of the module with a force-displacement graph."""

    def __init__(
        self,
        experiment: _Experiment,
        loading: InstronBranch,
        *,
        port: int = 8080,
        open_browser: bool = True,
    ) -> None:
        import plotly.graph_objects as go
        from soromox.rendering import CameraConfig, RendererConfig, SceneConfig
        from soromox.rendering.helicoidal.viser_renderer import (
            HelicoidalModuleVisual,
            HelicoidalViserRenderer,
        )

        self.experiment = experiment
        self.loading = loading
        self._go = go
        spec = experiment.spec
        length = spec.length
        if experiment.name == "compression":
            camera = dict(position=(0.11, -0.11, 0.075), look_at=(0.0, 0.0, 0.045), up=(0.0, 0.0, 1.0))
        else:
            camera = dict(position=(0.4 * length, -0.19, 0.07), look_at=(0.35 * length, 0.0, 0.035), up=(0.0, 0.0, 1.0))
        self.renderer = HelicoidalViserRenderer(
            experiment.robot,
            modules=(
                HelicoidalModuleVisual(
                    0,
                    turns=spec.turns_per_strand,
                    strands_per_family=spec.strands_per_family,
                    phase=spec.phase,
                    family_phase=spec.family_phase,
                    strand_width=spec.strand_width,
                    strand_height=spec.strand_thickness,
                ),
            ),
            config=RendererConfig(camera=CameraConfig(), scene=SceneConfig.studio()),
            port=port,
            open_browser=False,
        )
        # Live mode builds updatable strand meshes on the first push; every
        # later equilibrium only moves their vertices.
        self.controller = self.renderer.start_live_mode()
        q0 = np.zeros(experiment.n)
        self.controller.push_state(q0)
        server = self.renderer.server
        # The library renderer marks the module ends with thin tori at the
        # lattice radius; hide them and draw the rigid N6 end rings instead.
        for handle in getattr(self.renderer._scene_handles, "backbone_points", [[]])[0]:
            if "/ring_" in getattr(handle, "name", ""):
                handle.visible = False
        self._rings = {}
        self._draw_rings(q0)
        self._draw_string(q0)
        # The renderer puts its floor through the base frame; lower it so the
        # base ring stands on it (compression) or the clamped module sits above it (bending).
        drop = spec.end_ring_heights[0] if experiment.name == "compression" else (
            model.TendonBendingConfig().ring_outer_radius + 0.005
        )
        for handle in getattr(self.renderer._scene_handles, "ground_planes", []):
            x, y, z = (float(v) for v in handle.position)
            handle.position = (x, y, z - drop)

        def configure_camera(client) -> None:
            client.camera.position = camera["position"]
            client.camera.look_at = camera["look_at"]
            client.camera.up_direction = camera["up"]

        for client in server.get_clients().values():
            configure_camera(client)
        server.on_client_connect(configure_camera)
        gui = server.gui
        self._readout = gui.add_markdown(self._readout_text(None))
        self._plot = gui.add_plotly(figure=self._figure([]), aspect=1.3)
        print(f"[instron_static] Viser: {self.renderer.url}")
        if open_browser:
            webbrowser.open(self.renderer.url)

    # -- content --------------------------------------------------------------- #
    def _readout_text(self, record: EquilibriumRecord | None) -> str:
        exp = self.experiment
        if record is None:
            return f"**N6 {exp.name}**  \nreference configuration (no load)"
        v = record.verification
        return (
            f"**N6 {exp.name}** step {record.sample.index}  \n"
            f"force **{record.sample.force_N:.3f} N**  \n"
            f"static displacement **{1e3 * record.primary:.3f} mm** (measured {record.sample.measured_displacement_mm:.3f} mm)  \n"
            f"|r| / |tau| {v['force_residual_rel']:.1e}, |c| {v['constraint_residual']:.1e}, "
            f"min tangent eig {v['min_tangent_eigenvalue']:.3g}"
            f"{'' if v['min_tangent_eigenvalue'] > 0 else ' (tangent not positive definite)'}  \n"
            f"{'equilibrium recorded' if record.valid else 'NOT recorded: ' + record.reason}"
        )

    def _figure(self, records: list[EquilibriumRecord]):
        go = self._go
        xl, yl = self.experiment.axis_labels
        label = self.experiment.primary_label
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.loading.extension_mm, y=self.loading.load_N, mode="lines", name="Instron cycle 1 loading", line=dict(color="#52514e", width=2)))
        valid = [r for r in records if r.valid]
        fig.add_trace(go.Scatter(x=[1e3 * r.measures[label] for r in valid], y=[r.sample.force_N for r in valid], mode="lines+markers", name="Static equilibrium", marker=dict(color="#2a78d6", size=8), line=dict(color="#2a78d6", width=1.5)))
        invalid = [r for r in records if not r.valid]
        if invalid:
            fig.add_trace(go.Scatter(x=[1e3 * r.measures[label] for r in invalid], y=[r.sample.force_N for r in invalid], mode="markers", name="Not recorded", marker=dict(color="#eb6834", size=9, symbol="x")))
        fig.update_layout(xaxis_title=xl, yaxis_title=yl, template="plotly_white", margin=dict(l=50, r=10, t=10, b=40), legend=dict(orientation="h", y=-0.25))
        return fig

    def _draw_rings(self, q: np.ndarray) -> None:
        """Rigid end rings (40 mm outer radius, 7.5 mm tall) on the base and tip frames."""
        exp = self.experiment
        t_base, t_tip = exp.spec.end_ring_heights
        outer = model.TendonBendingConfig().ring_outer_radius
        inner = exp.spec.inner_radius
        q_j = jnp.asarray(q)
        frames = {
            "base": (np.asarray(exp.robot.forward_kinematics(q_j, jnp.asarray(0.0))), -t_base, t_base),
            "tip": (np.asarray(exp.tip_pose(q_j)), 0.0, t_tip),
        }
        for name, (pose, x0, height) in frames.items():
            local, faces = _ring_mesh(inner, outer, height)
            vertices = pose[:3, 3] + (local + np.array([x0, 0.0, 0.0])) @ pose[:3, :3].T
            if name in self._rings:
                self._rings[name].vertices = vertices
            else:
                self._rings[name] = self.renderer.server.scene.add_mesh_simple(
                    f"/instron/ring_{name}", vertices, faces, color=(118, 120, 126), side="double"
                )

    def _draw_string(self, q: np.ndarray) -> None:
        points = self.experiment.string_points(q)
        if points is not None:
            segments = np.stack((points[:-1], points[1:]), axis=1)
            self.renderer.server.scene.add_line_segments(
                "/instron/string", segments, colors=(230, 90, 40), thickness=0.0015
            )

    def show(self, record: EquilibriumRecord, records: list[EquilibriumRecord], hold_s: float) -> None:
        self.controller.push_state(record.q)
        self._draw_rings(record.q)
        self._draw_string(record.q)
        self._readout.content = self._readout_text(record)
        self._plot.figure = self._figure(records)
        time.sleep(hold_s)

    def close(self) -> None:
        self.controller.stop()


def _ring_mesh(inner: float, outer: float, height: float, n: int = 72) -> tuple[np.ndarray, np.ndarray]:
    """Closed annular ring, axis along local x from 0 to ``height``: (vertices, faces)."""
    angle = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    circle = np.column_stack((np.cos(angle), np.sin(angle)))
    vertices = np.concatenate(
        [np.column_stack((np.full(n, x), r * circle)) for x in (0.0, height) for r in (outer, inner)]
    )  # blocks: [x0 outer, x0 inner, x1 outer, x1 inner]
    faces = []
    for k in range(n):
        k1 = (k + 1) % n
        o0, i0, o1, i1 = k, n + k, 2 * n + k, 3 * n + k
        o0n, i0n, o1n, i1n = k1, n + k1, 2 * n + k1, 3 * n + k1
        for a, b, c, d in ((o0, o0n, o1n, o1), (i0, i1, i1n, i0n), (o0, i0, i0n, o0n), (o1, o1n, i1n, i1)):
            faces += [(a, b, c), (a, c, d)]
    return vertices, np.asarray(faces)


class ViserRecorder:
    """One PNG per equilibrium: a screenshot of the live Viser scene.

    A headless Chrome (Playwright) connects to the viewer like any browser and
    screenshots the page after each hold, so the frames show exactly the
    Viser scene, readout and graph. ``encode_video`` turns them into mp4/gif.
    """

    def __init__(
        self,
        viewer: ViserViewer,
        frames_dir: Path,
        *,
        max_steps: int | None = None,
        width: int = 1440,
        height: int = 900,
    ) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("recording needs Playwright: pip install playwright") from exc
        self.frames_dir = Path(frames_dir)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps  # last force step to record; frame 0 is the reference
        self.index = 0
        self._playwright = sync_playwright().start()
        args = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
        try:
            self._browser = self._playwright.chromium.launch(headless=True, channel="chrome", args=args)
        except Exception:  # noqa: BLE001 - no system Chrome: fall back to Playwright's Chromium
            self._browser = self._playwright.chromium.launch(headless=True, args=args)
        self.page = self._browser.new_page(viewport={"width": width, "height": height})
        self.page.goto(viewer.renderer.url)
        time.sleep(4.0)  # scene, meshes and GUI arrive over the websocket
        self.save()

    def save(self) -> Path | None:
        if self.max_steps is not None and self.index > self.max_steps + 1:
            return None
        time.sleep(1.0)  # let the client render the latest vertices
        # Drop Viser's "software WebGL" notice so it does not sit on the frame.
        self.page.evaluate("document.querySelectorAll('[class*=Notification]').forEach(e => e.remove())")
        path = self.frames_dir / f"{self.index:03d}.png"
        self.page.screenshot(path=str(path))
        self.index += 1
        return path

    def close(self) -> None:
        self._browser.close()
        self._playwright.stop()


def encode_video(frames_dir: Path, stem: Path, fps: float = 2.0) -> dict[str, Path]:
    """Encode ``frames_dir/NNN.png`` into ``stem.mp4`` and ``stem.gif`` with ffmpeg."""
    import subprocess

    frames = str(frames_dir / "%03d.png")
    mp4, gif = stem.with_suffix(".mp4"), stem.with_suffix(".gif")
    common = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps}", "-i", frames]
    subprocess.run(
        common + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(mp4)],
        check=True,
    )
    subprocess.run(
        common + ["-vf", "scale=960:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse", str(gif)],
        check=True,
    )
    return {"mp4": mp4, "gif": gif}


# --------------------------------------------------------------------------- #
# Force stepping
# --------------------------------------------------------------------------- #
def step_forces(
    experiment: _Experiment,
    samples: list[ForceSample],
    *,
    viewer: ViserViewer | None = None,
    recorder: ViserRecorder | None = None,
    hold_s: float = 2.5,
    max_substeps: int = 6,
    log: Callable[[str], None] | None = print,
) -> tuple[list[EquilibriumRecord], str]:
    """Solve, verify, measure, show and record every force level in order.

    Each level is an independent static problem; the previous solution is only
    the Newton initial guess. If Newton fails from that guess, the load
    increment is halved (up to ``max_substeps`` times) and applied in parts.
    Stepping stops at the first level whose equilibrium is nonphysical or not
    found; the stop reason is returned with the records.
    """
    log = log or (lambda _msg: None)
    solver = StaticSolver(experiment)
    records: list[EquilibriumRecord] = []
    x = solver.initial_guess()
    previous_load = 0.0
    stop_reason = ""
    for sample in samples:
        started = time.time()
        target = sample.force_N
        x_try, ok, steps, norm = solver.solve(x, target)
        if not ok:  # continuation from the previous level in smaller increments
            parts, x_try, ok, steps = 2, x, False, 0
            while parts <= 2**max_substeps and not ok:
                x_sub, ok = x, True
                for k in range(1, parts + 1):
                    load_k = previous_load + (target - previous_load) * k / parts
                    x_sub, ok_k, steps_k, norm = solver.solve(x_sub, load_k)
                    steps += steps_k
                    if not ok_k:
                        ok = False
                        break
                if ok:
                    x_try = x_sub
                parts *= 2
        q, wrench = solver.unpack(x_try, target)
        verification = solver.verify(q, wrench)
        converged = (
            ok
            and verification["force_residual_rel"] <= FORCE_RESIDUAL_REL_TOL
            and verification["constraint_residual"] <= CONSTRAINT_TOL
        )
        measures = experiment.measure(q)
        valid, reason = experiment.validity(q, measures)
        if not converged:
            valid, reason = False, "no verified static equilibrium (Newton failed or residual above tolerance)"
        record = EquilibriumRecord(
            sample=sample,
            converged=converged,
            valid=valid,
            reason=reason,
            newton_steps=steps,
            residual_norm=norm,
            verification=verification,
            wrench=wrench,
            measures=measures,
            q=q,
            wall_time_s=time.time() - started,
        )
        records.append(record)
        log(
            f"  step {sample.index:4d} F = {target:8.4f} N -> {1e3 * record.primary:9.4f} mm "
            f"(measured {sample.measured_displacement_mm:8.3f} mm) Newton {steps:3d} steps, "
            f"|r|/|tau| {verification['force_residual_rel']:.1e}, |c| {verification['constraint_residual']:.1e}, "
            f"min eig {verification['min_tangent_eigenvalue']:.3g} {'ok' if valid else 'INVALID: ' + reason}"
        )
        if viewer is not None:
            viewer.show(record, records, hold_s)
        if recorder is not None:
            recorder.save()
        if not valid:
            stop_reason = f"step {sample.index} at {target:.3f} N: {reason}"
            log("  stop: " + stop_reason)
            break
        x, previous_load = x_try, target
    return records, stop_reason


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _json_default(o: Any):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not serialisable: {type(o)}")


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path


def draw_comparison(ax, experiment: _Experiment, records: list[EquilibriumRecord], loading: InstronBranch) -> None:
    """Measured loading branch and the static equilibria recorded so far on ``ax``."""
    xl, yl = experiment.axis_labels
    label = experiment.primary_label
    ax.plot(loading.extension_mm, loading.load_N, color="#52514e", lw=1.6, label="Instron cycle 1 loading")
    valid = [r for r in records if r.valid]
    invalid = [r for r in records if not r.valid]
    ax.plot([1e3 * r.measures[label] for r in valid], [r.sample.force_N for r in valid], "o-", color="#2a78d6", ms=5, lw=1.2, label="Static equilibrium (force-controlled GVS)")
    if invalid:
        ax.plot([1e3 * r.measures[label] for r in invalid], [r.sample.force_N for r in invalid], "x", color="#eb6834", ms=6, label="Not recorded (unconverged / nonphysical)")
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.grid(True, color="#e6e6e3", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=2, frameon=False, fontsize=8)


def plot_png(experiment: _Experiment, records: list[EquilibriumRecord], loading: InstronBranch, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=150)
    draw_comparison(ax, experiment, records, loading)
    ax.set_title(experiment.title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


@dataclass
class RunSummary:
    experiment: str
    samples: int
    recorded: int
    converged: int
    valid: int
    stopped_early: bool
    stop_reason: str
    runtime_s: float
    max_force_residual_rel: float
    max_constraint_residual: float
    min_tangent_eigenvalue: float
    paths: dict[str, str]


def run_experiment(
    name: ExperimentName,
    *,
    output_dir: Path = RESULTS_DIR,
    num_levels: int = 30,
    decimate: int | None = None,
    max_force: float | None = None,
    cycle: int = 1,
    viser: bool = True,
    hold_s: float = 2.5,
    port: int = 8080,
    open_browser: bool = True,
    record: bool = False,
    record_steps: int | None = 20,
    log: Callable[[str], None] = print,
) -> RunSummary:
    """Run one experiment end to end and write every output file.

    With ``record`` (needs the Viser scene) a headless browser screenshots the
    scene after every hold up to force step ``record_steps`` (None: all); the
    frames are encoded into ``videos/<name>_static.mp4`` and ``.gif`` under
    ``output_dir``. The solve itself always runs every level.
    """
    csv_path = COMPRESSION_CSV if name == "compression" else BENDING_CSV
    loading, unloading, info = load_branches(csv_path, cycle)
    samples = select_samples(loading, num_levels=num_levels, decimate=decimate, max_force=max_force)
    log(f"[{name}] {csv_path.name}: cycle {cycle} loading rows {loading.rows[0]}-{loading.rows[1]}, "
        f"{len(samples)} force levels ({samples[0].force_N:.3f}-{max(s.force_N for s in samples):.3f} N)")
    experiment = build_experiment(name)
    out = Path(output_dir)
    frames_dir = out / "videos" / f"{name}_frames" if record else None
    if record and not viser:
        raise ValueError("recording screenshots the Viser scene: drop --no-viser")
    viewer = ViserViewer(experiment, loading, port=port, open_browser=open_browser) if viser else None
    recorder = ViserRecorder(viewer, frames_dir, max_steps=record_steps) if record else None
    started = time.time()
    try:
        records, stop_reason = step_forces(
            experiment, samples, viewer=viewer, recorder=recorder, hold_s=hold_s, log=log
        )
    finally:
        if recorder is not None:
            recorder.close()
        if viewer is not None:
            viewer.close()
    runtime = time.time() - started
    log(f"[{name}] {len(records)} force steps in {runtime:.1f} s; {sum(r.converged for r in records)} converged, "
        f"{sum(r.valid for r in records)} valid")

    out.mkdir(parents=True, exist_ok=True)
    label = experiment.primary_label
    paths: dict[str, Path] = {}
    if frames_dir is not None:
        paths.update(encode_video(frames_dir, out / "videos" / f"{name}_static"))
        log(f"[{name}] video: {paths['mp4']}")
    paths["equilibrium_csv"] = write_rows(out / f"{name}_equilibrium.csv", [r.to_row(label) for r in records])
    paths["measured_loading_csv"] = write_rows(
        out / f"{name}_instron_cycle{cycle}_loading.csv",
        [{"row": loading.rows[0] + i, "time_s": t, "extension_mm": x, "load_N": f}
         for i, (t, x, f) in enumerate(zip(loading.time_s, loading.extension_mm, loading.load_N, strict=True))],
    )
    paths["png"] = plot_png(experiment, records, loading, out / f"{name}_force_displacement.png")
    valid = [r for r in records if r.valid]
    summary = RunSummary(
        experiment=name,
        samples=len(samples),
        recorded=len(records),
        converged=int(sum(r.converged for r in records)),
        valid=len(valid),
        stopped_early=bool(stop_reason),
        stop_reason=stop_reason,
        runtime_s=runtime,
        max_force_residual_rel=max((r.verification["force_residual_rel"] for r in valid), default=math.nan),
        max_constraint_residual=max((r.verification["constraint_residual"] for r in valid), default=math.nan),
        min_tangent_eigenvalue=min((r.verification["min_tangent_eigenvalue"] for r in valid), default=math.nan),
        paths={k: str(v) for k, v in paths.items()},
    )
    diagnostics = {
        "summary": asdict(summary),
        "spec": json.loads(experiment.spec.to_json()),
        "experiment": {
            "name": name,
            "coordinates": experiment.n,
            "constraints": experiment.num_constraints,
            "solver": "optimistix Newton on K q + G - J_tip^T W = 0 (+ constraints), one independent solve per force level",
            "verification": {"force_residual_rel_tol": FORCE_RESIDUAL_REL_TOL, "constraint_tol": CONSTRAINT_TOL},
            "load_model": "J_tip^T [0;0;0;0;0;-F] with platen reactions (compression) / chord string wrench, tension-only (bending)",
            "bending_config": (
                json.loads(experiment.bending_config.to_json()) if name == "bending" else None
            ),
        },
        "instron": {
            "csv": str(csv_path),
            "cycle": cycle,
            "loading_rows": list(loading.rows),
            "unloading_rows": list(unloading.rows),
            "sample_period_s": info["data"]["sample_period_s"],
            "samples": [asdict(s) for s in samples],
        },
        "records": [r.to_row(label) for r in records],
    }
    paths["diagnostics_json"] = write_json(out / f"{name}_diagnostics.json", diagnostics)
    summary.paths = {k: str(v) for k, v in paths.items()}
    write_json(out / f"{name}_summary.json", asdict(summary))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--experiment", choices=["compression", "bending", "both"], default="both")
    parser.add_argument("--num-levels", type=int, default=30, help="representative force levels spaced along the measured extension")
    parser.add_argument("--decimate", type=int, default=None, help="replay every n-th measured sample instead of --num-levels")
    parser.add_argument("--max-force", type=float, default=None,
                        help="force cap [N]; default: model validity limit (compression 45 N: top ring reaches the base near 46 N; "
                        "bending 8 N: free ring collides with the clamped ring near 8.6 N)")
    parser.add_argument("--cycle", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--hold", type=float, default=2.5, help="seconds each equilibrium is held in Viser")
    parser.add_argument("--no-viser", action="store_true", help="run headless (no Viser scene)")
    parser.add_argument("--no-browser", action="store_true", help="start Viser without opening a browser tab")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--record", action="store_true",
                        help="screenshot the Viser scene after every hold (headless Chrome via Playwright) and encode "
                        "videos/<name>_static.mp4 and .gif under the output directory")
    parser.add_argument("--record-steps", type=int, default=20,
                        help="last force step included in the videos (default 20; the solve still runs every level); "
                        "negative records every step")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    names = ["compression", "bending"] if args.experiment == "both" else [args.experiment]
    for i, name in enumerate(names):
        max_force = DEFAULT_MAX_FORCE[name] if args.max_force is None else args.max_force
        summary = run_experiment(
            name,
            output_dir=args.output_dir,
            num_levels=args.num_levels,
            decimate=args.decimate,
            max_force=max_force,
            cycle=args.cycle,
            viser=not args.no_viser,
            hold_s=args.hold,
            port=args.port + i,
            open_browser=not args.no_browser,
            record=args.record,
            record_steps=None if args.record_steps < 0 else args.record_steps,
        )
        print(json.dumps(asdict(summary), indent=2, default=_json_default))


if __name__ == "__main__":
    main()
