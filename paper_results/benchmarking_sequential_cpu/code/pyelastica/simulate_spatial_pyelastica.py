"""
Minimal PyElastica rod simulation (NO gymnasium, NO custom elastica_env).

What it does:
- Builds a straight Cosserat rod
- Fixes one end (cantilever)
- Adds linear damping
- Optional gravity
- Optional self-contact
- Adds a simple "tendon-like" actuation (4 channels) that bends the rod by applying
  distributed torques about the local material axes.

Run:
    pip install pyelastica numpy
    python New_Code.py

Notes:
- This is NOT your exact Tendon1..4 model (those are from your elastica_env).
  This is a lightweight, Elastica-only approximation.
"""

from __future__ import annotations  # MUST be first (after docstring)

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from elastica import (
    AnalyticalLinearDamper,
    BaseSystemCollection,
    CallBackBaseClass,
    CallBacks,
    Connections,
    Constraints,
    Contact,
    CosseratRod,
    Damping,
    Forcing,
    GravityForces,
    NoForces,
    OneEndFixedBC,
    PositionVerlet,
    RodSelfContact,
)
from elastica.timestepper import extend_stepper_interface

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


# -----------------------------
# System definition
# -----------------------------
class RodSimulator(
    BaseSystemCollection, Constraints, Connections, Forcing, CallBacks, Damping, Contact
):
    pass


# -----------------------------
# Tendon-like actuation (Elastica-only)
# -----------------------------
@dataclass
class TendonActuationParams:
    # 4 activations: [+u bend, -u bend, +v bend, -v bend] in rod material frame
    activation: np.ndarray  # shape (4,)
    # torque gain [N*m] (distributed along the rod)
    torque_gain: float = 0.00
    # spatial weighting along rod (0..1 -> base..tip)
    s_profile: str = "uniform"  # "uniform" | "tip" | "base"


class SimpleTendonLikeActuation(NoForces):
    """
    Apply distributed torques to create bending, using only PyElastica.
    This mimics 4 tendon channels by producing moments about local material axes.
    """

    def __init__(self, params: TendonActuationParams):
        super().__init__()
        self.params = params

    def _weight(self, n_elems: int) -> np.ndarray:
        if self.params.s_profile == "uniform":
            return np.ones(n_elems)
        s = np.linspace(0.0, 1.0, n_elems)
        if self.params.s_profile == "tip":
            return s**2
        if self.params.s_profile == "base":
            return (1.0 - s) ** 2
        raise ValueError(f"Unknown s_profile: {self.params.s_profile}")

    @staticmethod
    def _get_external_torques(system):
        # Version-tolerant accessor
        if hasattr(system, "external_torques"):
            return system.external_torques
        if hasattr(system, "external_torques_collection"):
            return system.external_torques_collection
        raise AttributeError(
            "Could not find external torques array on system (external_torques / external_torques_collection)."
        )

    def apply_torques(self, system, time: float = 0.0):
        # director_collection is shape (3,3,n_elems)
        d = system.director_collection
        n_elems = d.shape[-1]

        a = np.asarray(self.params.activation, dtype=float).reshape(4)
        # net bending torques about local u and v axes:
        #   tau_u = +a0 - a1, tau_v = +a2 - a3
        tau_u = a[0] - a[1]
        tau_v = a[2] - a[3]

        # distribute torques along rod
        w = self._weight(n_elems)
        w = w / (np.sum(w) + 1e-12)

        # Per-element torque magnitude
        tau_elem = self.params.torque_gain * w  # (n_elems,)

        # local axes in world frame
        u_world = d[:, 0, :]  # (3, n_elems)
        v_world = d[:, 1, :]  # (3, n_elems)

        # world torque vector per element
        torque_world = (tau_u * tau_elem)[None, :] * u_world + (tau_v * tau_elem)[
            None, :
        ] * v_world  # (3,n_elems)

        # Apply to external torques
        ext_tau = self._get_external_torques(system)

        # Most common: external torques at nodes: (3, n_elems+1)
        if ext_tau.shape[1] == n_elems + 1:
            ext_tau[:, :-1] += 0.5 * torque_world
            ext_tau[:, 1:] += 0.5 * torque_world
        # Sometimes: external torques per element: (3, n_elems)
        elif ext_tau.shape[1] == n_elems:
            ext_tau += torque_world
        else:
            raise RuntimeError(
                f"Unexpected external torques shape: {ext_tau.shape} (expected (3,{n_elems}) or (3,{n_elems + 1}))"
            )


# -----------------------------
# Callback for logging
# -----------------------------
class RodCallBack(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        super().__init__()
        self.every = int(step_skip)
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every != 0:
            return
        p = self.callback_params
        p["time"].append(float(time))
        p["step"].append(int(current_step))
        p["position"].append(system.position_collection.copy())
        p["velocity"].append(system.velocity_collection.copy())
        p["director"].append(system.director_collection.copy())
        p["omega"].append(system.omega_collection.copy())
        p["com"].append(system.compute_position_center_of_mass())
        p["vcom"].append(system.compute_velocity_center_of_mass())


# -----------------------------
# Main simulation
# -----------------------------
def simulate(
    final_time: float = 3.0,
    dt: float = 1e-4,
    n_elems: int = 12,
    length: float = 0.6,
    radius: float = 0.03,
    density: float = 1000.0,
    youngs_modulus: float = 1e6,
    poisson_ratio: float = 0.5,
    damping_nu: float = 0.048,
    add_gravity: bool = True,
    gravity: np.ndarray = np.array([-9.81 * 1, -9.81 * 1, -9.81 * 0]),
    add_self_contact: bool = False,
    step_skip: int = 20,
):
    sim = RodSimulator()
    stepper = PositionVerlet()

    # Rod geometry
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([0.0, 1.0, 0.0])

    base_radius = np.full(n_elems, radius)
    shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))

    rod = CosseratRod.straight_rod(
        n_elements=n_elems,
        start=start,
        direction=direction,
        normal=normal,
        base_length=length,
        base_radius=base_radius,
        density=density,
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus,
    )
    sim.append(rod)

    # Fix base
    sim.constrain(rod).using(
        OneEndFixedBC,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
    )

    # Damping
    sim.dampen(rod).using(
        AnalyticalLinearDamper,
        damping_constant=damping_nu,
        time_step=dt,
    )

    # Gravity (optional)
    if add_gravity:
        sim.add_forcing_to(rod).using(
            GravityForces, acc_gravity=np.asarray(gravity, dtype=float)
        )

    # Self-contact (optional)
    if add_self_contact:
        sim.detect_contact_between(rod, rod).using(
            RodSelfContact,
            k=1e4,
            nu=10.0,
        )

    # Tendon-like actuation params (4-channel)
    act_params = TendonActuationParams(
        activation=np.array([0.0, 0.0, 0.0, 0.0]),
        torque_gain=0.08 * 0,
        s_profile="tip",
    )

    # Register forcing DIRECTLY (no wrapper!)
    sim.add_forcing_to(rod).using(SimpleTendonLikeActuation, params=act_params)

    # Diagnostics
    log = defaultdict(list)
    sim.collect_diagnostics(rod).using(
        RodCallBack, step_skip=step_skip, callback_params=log
    )

    # Finalize
    sim.finalize()
    do_step, stages_and_updates = extend_stepper_interface(stepper, sim)

    # Time loop
    n_steps = int(np.round(final_time / dt))
    time = 0.0

    for _ in range(n_steps):
        # Example: bend in +u direction for first half, then +v direction
        if time < 0.5 * final_time:
            act_params.activation[:] = np.array([1.0, 0.0, 0.0, 0.0])  # +u bend
        else:
            act_params.activation[:] = np.array([0.0, 0.0, 1.0, 0.0])  # +v bend

        time = do_step(stepper, stages_and_updates, sim, time, dt)

    return log


start = time.perf_counter()

if __name__ == "__main__":
    data = simulate()
    elapsed = time.perf_counter() - start
    print(f"Elapsed time: {elapsed:.6f} seconds")

    # Print final tip position
    final_pos = data["position"][-1]  # shape (3, n_nodes)
    tip = final_pos[:, -1]
    print("Final tip position [m]:", tip)

    time = np.array(data["time"])  # (N,)
    positions = data["position"]  # list of (3, n_nodes)
    tip_pos = np.array([p[:, -1] for p in positions])  # (N, 3)
    x = tip_pos[:, 0]
    y = tip_pos[:, 1]
    z = tip_pos[:, 2]
    plt.figure(figsize=(8, 6))

    plt.subplot(3, 1, 1)
    plt.plot(time, x)
    plt.ylabel("x [m]")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time, y)
    plt.ylabel("y [m]")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time, z)
    plt.ylabel("z [m]")
    plt.xlabel("time [s]")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    df = pd.DataFrame({"time": time, "x": x, "y": y, "z": z})

    # Save to Excel
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_path = DATA_DIR / "end_effector_position_spatial_pyelastica.xlsx"

    df.to_excel(file_path, index=False)
