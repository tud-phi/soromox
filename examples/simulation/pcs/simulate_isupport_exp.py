"""
Simulates an Additive Manufacturing (AM) I-SUPPORT manipulator driven by
RECORDED chamber pressures exported from MATLAB (6 x M matrix sampled on
merged_time), fed as a time-varying open-loop input through the closed-loop
rollout hook.

A first-order continuous low-pass filter is applied to the recorded pressures
inside the ODE. The filtered pressure is carried as an integrated
control_state, so the filter dynamics are solved together with the mechanical
dynamics by the same adaptive solver. The filter is initialized to zero.

    tau * d(u_filt)/dt = u_raw - u_filt      =>      fc = 1 / (2*pi*tau)

System parameters adapted from:
Alessi, Carlo, Egidio Falotico, and Alessandro Lucantonio.
"Ablation study of a dynamic model for a 3d-printed pneumatic soft robotic arm."
IEEE Access 11 (2023): 37840-37853.
https://ieeexplore.ieee.org/abstract/document/10098800
"""

## Disable warning for memory preallocation
import os
from functools import partial

# execute this command here: export XLA_PYTHON_CLIENT_PREALLOCATE=false
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# Test adaptive/implicit integration
from diffrax import Tsit5, ImplicitEuler, Kvaerno5, PIDController

jax.config.update("jax_enable_x64", True)  # double precision
from soromox.rendering import ISupportViserRenderer, MatplotlibRenderer
from soromox.systems import ISupport, ISupportParams, ISupportStructure, SystemState

### Time for Debug ###
# import time

# Figure path
RESULTS_DIR = "amisupport_dataset/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- Load real actuation ----
data = sio.loadmat("amisupport_dataset/2026-06-08-11-59-18_py.mat", squeeze_me=True)

## Extract Data
poses_data = data["poses_data"]  # (7, N, M)
markers = data["markers"]  # (3*num_markers, M)
merged_time = np.asarray(data["merged_time"]).squeeze()
N = int(data["N"])
M = int(data["M"])

# Input samples
act_values = np.asarray(data["actuation_input"], dtype=np.float64)  # (6, M)
assert act_values.shape[0] == 6, act_values.shape
assert act_values.shape[1] == merged_time.shape[0], (
    act_values.shape,
    merged_time.shape,
)

# Air Leaks reading at 3 bar
p_real = [2.33, 2.35, 2.26, 2.09, 2.63, 2.32]

# bar2Pa
PRESSURE_SCALE = 1.0e5
act_values = act_values * PRESSURE_SCALE
NOMINAL_PMAX = 3.0

act_time = merged_time - merged_time[0]  # zero the clock
act_time_j = jnp.asarray(act_time)
act_values_j = jnp.asarray(act_values)  # (6, M)

# =====================================================
# Low-pass filter configuration
# =====================================================
# Cutoff frequency [Hz]. tau = 1 / (2*pi*fc).
LOWPASS_CUTOFF_HZ = 2.8     # [Hz]
LOWPASS_TAU = 1.0 / (2.0 * jnp.pi * LOWPASS_CUTOFF_HZ)

### --- Estimated Params --- ###
# ### Default
# ESTIMATED_E       = 1.6464e+6
# ESTIMATED_ETA     = 1e+4
# ESTIMATED_RHO     = 1.104e+3

## Estimated (Ac = pi * r_in^2)
ESTIMATED_E     = 1.076632416072824e+06
ESTIMATED_ETA   = 0.083795344450652e+06
ESTIMATED_RHO   = 0.006471046068502e+06

ESTIMATED_POI   = 0.5
PLA_DENSITY     = 1210.0

### GEOMETRICAL PARAMS
R_IN            = 6.39e-3   # [m]
R_OUT           = 7.79e-3   # [m]

# =====================================================
# Recorded-input "controller" with a first-order low-pass filter.
# The control_state carries the filtered pressure u_filt (6,); the
# controller returns (u_filt, d(u_filt)/dt) so the solver integrates it.
# =====================================================
class RecordedInput(eqx.Module):
    act_time: jax.Array     # (M,)
    act_values: jax.Array   # (6, M)
    perm: jax.Array         # (6, 6) permutation matrix
    air_leaks: jax.Array    # (6, 6) Scale matrix for Air Leaks
    tau: jax.Array          # () first-order filter time constant [s]

    def __call__(self, state: SystemState):
        """Interpolate the 6 recorded chamber pressures at the live solver time,
        apply the permutation and air-leak scaling, then low-pass filter the result
        with a first-order lag whose state is integrated by the ODE solver.

        Returns:
            u_filt: filtered (6,) pressures actually applied to the robot, ordered
                [seg0_ch0, seg0_ch1, seg0_ch2, seg1_ch0, seg1_ch1, seg1_ch2].
            du_dt: derivative of the filter state, d(u_filt)/dt = (u_raw - u_filt)/tau.
        Permute the rows in `perm` if your ROS node uses a different chamber order.
        """
        t = state.t

        # Raw (unfiltered) command at the live solver time.
        # Improve computational efficiency
        u_raw = jax.vmap(lambda row: jnp.interp(t, self.act_time, row))(self.act_values)
        u_raw = self.air_leaks @ self.perm @ u_raw

        # Filtered pressure is carried in the control_state.
        u_filt = state.control_state

        # First-order low-pass dynamics: tau * du/dt = u_raw - u_filt.
        du_dt = (u_raw - u_filt) / self.tau

        # Apply the filtered pressure; return its derivative for integration.
        return u_filt, du_dt

def permutation_matrix(perm_idx):
    P = jnp.zeros((len(perm_idx), len(perm_idx)))
    P = P.at[jnp.arange(len(perm_idx)), jnp.array(perm_idx)].set(1.0)
    return P

def gain_air_leaks(p_real):
    return jnp.diag(jnp.asarray(p_real)) / NOMINAL_PMAX

# Permutation for matching the experiment setup
perm_idx = [2, 1, 0, 5, 4, 3]

P = permutation_matrix(perm_idx)
air_leaks = gain_air_leaks(p_real)

# Define Input
recorded_input = RecordedInput(
    act_time=act_time_j,
    act_values=act_values_j,
    perm=P,
    air_leaks=air_leaks,
    tau=jnp.asarray(LOWPASS_TAU),
)

if __name__ == "__main__":
    num_pneumatic_segments = 2

    # Elastic modulus and poisson ratio
    E = ESTIMATED_E  # Elastic modulus [Pa]
    poisson_ratio = ESTIMATED_POI
    G = E / (
        2 * (1 + poisson_ratio)
    )  # Shear modulus from elastic modulus and poisson ratio

    # Physical layout: base interface, pneumatic section, middle interface,
    # pneumatic section, tip interface.
    rigid_segment_selector = (True, False, True, False, True)
    physical_segment_lengths = jnp.array([41e-3, 180e-3, 27e-3, 180e-3, 6e-3])
    # Alessi et al. (2023) report a 30 mm circular cross-section radius for
    # the complete arm, including its pneumatic sections and terminal plates.
    physical_segment_radii = 30e-3 * jnp.ones((len(rigid_segment_selector),))
    # The compiled reference model stores 1210 kg/m^3 for every rigid
    # interface and 1104 kg/m^3 for each pneumatic section.
    physical_segment_densities = jnp.array([PLA_DENSITY, 
                                            ESTIMATED_RHO, 
                                            PLA_DENSITY, 
                                            ESTIMATED_RHO, 
                                            PLA_DENSITY])

    # Topology: how many PCS segments approximate each pneumatic segment.
    pcs_segment_counts = (1, 1)
    # Parameters: metric PCS segment lengths, flattened by pneumatic segment.
    # The first two entries sum to 190 mm; the last three sum to 180 mm.
    # Set this to None to divide each pneumatic segment equally according to
    # pcs_segment_counts.
    pcs_segment_lengths = None

    # Material Damping Coefficient for Damping Tensor
    material_damping_coefficient = ESTIMATED_ETA    # \eta [Pa s]

    # Experimental Offset
    theta = -jnp.pi/6.0     # - 30 deg

    params = ISupportParams(
        base_pose=jnp.array([
                jnp.cos(theta / 2), jnp.sin(theta / 2), 0.0, 0.0,   # quaternion [w, x, y, z]
                0.0, 0.0, 0.0,                                      # position [x, y, z]
        ]),
        length=physical_segment_lengths,
        radius=physical_segment_radii,
        density=physical_segment_densities,
        gravity=jnp.array([9.81, 0.0, 0.0]),
        young_modulus=E * jnp.ones((len(rigid_segment_selector),)),
        shear_modulus=G * jnp.ones((len(rigid_segment_selector),)),
        material_damping_coefficient=material_damping_coefficient
        * jnp.ones((len(rigid_segment_selector),)),
        reference_strain=jnp.tile(
            jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            len(rigid_segment_selector),
        ),
        chamber_inner_radius=R_IN * jnp.ones((num_pneumatic_segments,)),
        chamber_outer_radius=R_OUT * jnp.ones((num_pneumatic_segments,)),
        chamber_distance=20 * 1e-3 * jnp.ones((num_pneumatic_segments,)),
        # Explicit [0, 2*pi/3, 4*pi/3] = [0, 120, 240] degrees for each
        # pneumatic segment. Array index is the corresponding pressure channel.
        chamber_azimuth_angles=jnp.tile(
            2.0 * jnp.pi * jnp.arange(3) / 3 + jnp.pi/2,        # From Alessi paper, the phases are [90°, 210°, 330°]
            (num_pneumatic_segments, 1),
        ),
        pcs_segment_lengths=pcs_segment_lengths,
        chamber_effective_pressure_area=(jnp.pi*(R_IN)**2)*jnp.ones((num_pneumatic_segments,)),
    )

    # ======================================================
    # Robot initialization
    # ======================================================
    robot = ISupport(
        params=params,
        structure=ISupportStructure(
            pcs_segment_counts=pcs_segment_counts,
            rigid_segment_selector=rigid_segment_selector,
        ),
    )

    # =====================================================
    # Initial state: undeformed reference strain, base_u = zeros
    # =====================================================
    q0_pneumatic = jnp.repeat(
        jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])[None, :],
        num_pneumatic_segments,
        axis=0,
    )
    q0 = jnp.concatenate(
        [
            jnp.tile(q0_pneumatic[i], pcs_segment_counts[i])
            for i in range(num_pneumatic_segments)
        ]
    )
    qd0 = jnp.zeros_like(q0)

    # Start and End time of the simulation
    t0 = 15.22
    t1 = 60.22

    # Initial filter state: match MATLAB simulate_robot_lp, which starts the
    # filter at the raw (scaled) command evaluated at the window start time.
    u_raw_t0 = jax.vmap(lambda row: jnp.interp(t0, act_time_j, row))(act_values_j)
    u_filt0 = air_leaks @ P @ u_raw_t0

    initial_state = SystemState(
        t=t0,
        y=jnp.concatenate([q0, qd0]),
        u=jnp.zeros((robot.num_actuators,)),  # base_u = 0; all input via controller
        control_state=u_filt0,  # filtered pressures, integrated by the solver
    )

    # =====================================================
    # Single closed-loop call = time-varying open-loop input
    # =====================================================
    solver_dt = 1e-3        # With adaptive-step we can use higher dt
    save_dt = 1e-3          # Clean plot

    # # fixing bug for float accumulation: [t0, t1] issue
    save_ts = jnp.arange(t0, t1 + save_dt, save_dt)
    save_ts = save_ts[save_ts <= t1]

    #################################################################
    # Open-loop control law: implemented as closed_loop_to          #
    # with base_u = 0 and the function that ignores q, qdot, ...    #
    # The filter state (control_state) is integrated alongside y.   #
    #################################################################

    # # ---- Timed rollout ----
    # # JAX is asynchronous: block_until_ready() forces completion before we stop
    # # the timer. The FIRST call also includes JIT compilation time.
    # t_start = time.perf_counter()

    trajectory = robot.rollout_closed_loop_to(
        initial_state=initial_state,
        controller=recorded_input,
        t1=t1,
        solver_dt=solver_dt,
        save_dt=save_dt,
        save_ts=save_ts,
        solver=Tsit5(),                                               # Default but with adaptive-time steps
        # solver=ImplicitEuler(),                                     # Crash :(
        # solver=Kvaerno5(),  # Slow but accurate
        stepsize_controller=PIDController(rtol=1e-5, atol=1e-7),  # adaptive time-steps
        max_steps=None,
    )

    # # Force all device computation to finish (walks the pytree, blocks on each leaf).
    # jax.block_until_ready(trajectory)

    # elapsed = time.perf_counter() - t_start
    # print(f"Rollout (incl. JIT compile) took {elapsed:.4f} s")

    ts = trajectory.t
    q_ts, qd_ts = jnp.split(trajectory.y, 2, axis=1)
    u_ts = trajectory.u  # filtered actuation actually applied at each save step

    # =====================================================
    # Configuration q and velocity qd upon time
    # =====================================================
    # DOF labels per segment: [kappa_x, kappa_y, kappa_z, sigma_x, sigma_y, sigma_z]
    dof_names = ["kx", "ky", "kz", "sx", "sy", "sz"]
    n_dof = q_ts.shape[1]
    n_pcs_total = sum(pcs_segment_counts)
    labels = [
        f"pcs{j}_{dof_names[i]}" for j in range(n_pcs_total) for i in range(6)
    ][:n_dof]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for i in range(n_dof):
        axes[0].plot(ts, q_ts[:, i], label=labels[i])
    axes[0].set_ylabel("q (strain)")
    axes[0].set_title("Configuration q over time")
    axes[0].grid(True)
    axes[0].legend(ncol=4, fontsize=8)

    for i in range(n_dof):
        axes[1].plot(ts, qd_ts[:, i], label=labels[i])
    axes[1].set_ylabel("qd (strain rate)")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_title("Velocity qd over time")
    axes[1].grid(True)
    axes[1].legend(ncol=4, fontsize=8)

    plt.tight_layout()

    # =====================================================
    # End-effector position upon time
    # =====================================================
    forward_kinematics_end_effector = jax.jit(
        partial(
            robot.forward_kinematics,
            s=jnp.sum(robot.L),  # end-effector position
        )
    )
    g_ee_ts = jax.vmap(forward_kinematics_end_effector)(q_ts)

    # =====================================================
    # Overlay recorded Vicon tip position on simulated EE
    # =====================================================
    # Vicon tip = last cross-section; positions are rows 0:3 of the 7-pose
    tip_idx_cs = N - 1
    vicon_tip_pos = np.asarray(poses_data[0:3, tip_idx_cs, :])  # (3, M)
    vicon_time = act_time  # same merged_time clock, zeroed

    print(
        "recording covers t =",
        float(vicon_time.min()),
        "to",
        float(vicon_time.max()),
        "s",
    )
    print("sim window: t0 =", t0, " t1 =", t1)

    # Interpolate each axis onto the simulation save times, skipping NaNs
    ts_np = np.asarray(ts)
    vicon_interp = np.full((3, ts_np.shape[0]), np.nan)
    for ax_i in range(3):
        series = vicon_tip_pos[ax_i, :]
        good = np.isfinite(series)
        if good.sum() >= 2:
            vicon_interp[ax_i, :] = np.interp(
                ts_np,
                vicon_time[good],
                series[good],
                left=np.nan,
                right=np.nan,  # don't extrapolate outside recording
            )

    g_ee_np = np.asarray(g_ee_ts)
    sim_pos = g_ee_np[:, 0:3, 3]  # (T, 3)

    ### Save CSV ####
    import pandas as pd

    df = pd.DataFrame({
        "time": ts_np,
        "sim_x": sim_pos[:, 0],
        "sim_y": sim_pos[:, 1],
        "sim_z": sim_pos[:, 2],
        "vicon_x": vicon_interp[0, :],
        "vicon_y": vicon_interp[1, :],
        "vicon_z": vicon_interp[2, :],
    })
    df.to_csv(os.path.join(RESULTS_DIR, "tip_positions.csv"), index=False)
    print("Tip position saved in the csv file.")
    #################

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axis_names = ["x", "y", "z"]
    for ax_i in range(3):
        axes[ax_i].plot(ts_np, sim_pos[:, ax_i], label="Sim EE", linewidth=2)
        axes[ax_i].plot(
            ts_np, vicon_interp[ax_i, :], "--", label="Vicon tip", linewidth=2
        )
        axes[ax_i].set_ylabel(f"{axis_names[ax_i]} [m]")
        axes[ax_i].grid(True)
        axes[ax_i].legend(loc="upper right")
    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title("Simulated EE vs recorded Vicon tip (per axis)")
    plt.tight_layout()

    # 3D overlay
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(sim_pos[:, 0], sim_pos[:, 1], sim_pos[:, 2], label="Sim EE", linewidth=2)
    ax.plot(
        vicon_interp[0, :],
        vicon_interp[1, :],
        vicon_interp[2, :],
        "--",
        label="Vicon tip",
        linewidth=2,
    )
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.axis("equal")
    ax.legend()
    ax.set_title("Trajectory overlay")
    plt.tight_layout()

    # =====================================================
    # Applied actuation (filtered) vs raw recorded command
    # =====================================================
    # Recompute the raw (unfiltered) command on the save grid for comparison.
    u_raw_ts = jax.vmap(
        lambda t_i: air_leaks
        @ P
        @ jax.vmap(lambda row: jnp.interp(t_i, act_time_j, row))(act_values_j)
    )(ts)
    u_raw_np = np.asarray(u_raw_ts)

    plt.figure()
    for k in range(u_ts.shape[1]):
        line, = plt.plot(ts, u_ts[:, k], label=f"u{k} (filtered)")
        plt.plot(
            ts, u_raw_np[:, k], "--", color=line.get_color(), alpha=0.4
        )
    plt.xlabel("Time [s]")
    plt.ylabel("Applied pressure [Pa]")
    plt.legend(ncol=3)
    plt.grid(True)
    plt.title(f"Applied actuation (solid=filtered, dashed=raw), fc={LOWPASS_CUTOFF_HZ} Hz")
    plt.tight_layout()

    # =====================================================
    # Save ALL figures first (before showing, so nothing is cleared/closed),
    # then display them all at once.
    # =====================================================
    for num in plt.get_fignums():
        fig = plt.figure(num)
        fig.savefig(
            os.path.join(RESULTS_DIR, f"figure_{num}.png"),
            dpi=200,
            bbox_inches="tight",
        )
    print(f"Figures saved in '{RESULTS_DIR}'.")

    # Show all figures at the end (blocks until windows are closed).
    plt.show()

    # # =====================================================
    # # Plot the robot configuration upon time
    # # =====================================================
    # renderer = MatplotlibRenderer(robot, num_points=50)
    # renderer.animate(ts=ts, q_ts=q_ts, interval=100, mode="slider")

    # # =====================================================
    # # Viser web-based visualization
    # # =====================================================
    # if ISupportViserRenderer is None:
    #     print("ISupportViserRenderer is unavailable. Install with `pip install viser`.")
    # else:
    #     viser_renderer = ISupportViserRenderer(
    #         robot,
    #         num_points=50,
    #     )
    #     viser_renderer.render_sequence(
    #         ts=ts,
    #         q_ts=q_ts,
    #         playback_speed=1.0,
    #         autoplay=True,
    #         loop=True,
    #         render_actuators=True,
    #         plot_configurations=True,
    #         robot_name="I-SUPPORT",
    #     )