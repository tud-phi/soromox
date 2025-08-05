import cv2  # importing cv2
from functools import partial
import jax

jax.config.update("jax_enable_x64", True)  # double precision
from diffrax import diffeqsolve, Euler, ODETerm, SaveAt, Tsit5
from jax import Array, vmap
from jax import numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
from pathlib import Path
from typing import Callable, Dict
from IPython.display import HTML
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

import soromox
from soromox import ode_factory
from soromox.systems import planar_pcs_sym

num_segments = 2

# filepath to symbolic expressions
sym_exp_filepath = (
    Path(soromox.__file__).parent
    / "symbolic_expressions"
    / f"planar_pcs_ns-{num_segments}.dill"
)

# set parameters
rho = 1070 * jnp.ones((num_segments,))  # Volumetric density of Dragon Skin 20 [kg/m^3]
params = {
    "th0": jnp.array(0.0),  # initial orientation angle [rad]
    "l": 1e-1 * jnp.ones((num_segments,)),
    "r": 2e-2 * jnp.ones((num_segments,)),
    "rho": rho,
    "g": jnp.array([0.0, -9.81]),
    "E": 2e3 * jnp.ones((num_segments,)),  # Elastic modulus [Pa]
    "G": 1e3 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
}
params["D"] = 1e-3 * jnp.diag(
    (
        jnp.repeat(jnp.array([[1e0, 1e3, 1e3]]), num_segments, axis=0)
        * params["l"][:, None]
    ).flatten()
)

# activate all strains (i.e. bending, shear, and axial)
strain_selector = jnp.ones((3 * num_segments,), dtype=bool)

# define initial configuration
q0 = jnp.repeat(
    jnp.array([5.0 * jnp.pi, 0.1, 0.2])[None, :], num_segments, axis=0
).flatten()
# number of generalized coordinates
n_q = q0.shape[0]

q0 = jnp.zeros(n_q)  # initial configuration

# set simulation parameters
dt = 1e-4  # time step
ts = jnp.arange(0.0, 2, dt)  # time steps
skip_step = 10  # how many time steps to skip in between video frames
video_ts = ts[::skip_step]  # time steps for video

# video settings
video_width, video_height = 700, 700  # img height and width
video_path = Path(__file__).parent / "videos" / f"planar_pcs_ns-{num_segments}.mp4"

def draw_robot(
    batched_forward_kinematics_fn: Callable,
    params: Dict[str, Array],
    q: Array,
    num_points: int = 50,
) -> onp.ndarray:
    # we use for plotting N points along the length of the robot
    s_ps = jnp.linspace(0, jnp.sum(params["l"]), num_points)

    # poses along the robot of shape (3, N)
    chi_ps = batched_forward_kinematics_fn(params, q, s_ps)

    curve = onp.array(chi_ps[:2, :], dtype=onp.float32).T

    return curve

def animate_robot_matplotlib(
    batched_forward_kinematics_fn: Callable,
    params: Dict[str, Array],
    t_list: Array,  # shape (T,)
    q_list: Array,  # shape (T, DOF)
    num_points: int = 50,
    interval: int = 50,
    slider: bool = None,
    animation: bool = None,
    show: bool = True,
):
    if slider is None and animation is None:
        raise ValueError("Either 'slider' or 'animation' must be set to True.")
    if animation and slider:
        raise ValueError(
            "Cannot use both animation and slider at the same time. Choose one."
        )

    width = jnp.linalg.norm(params["l"]) * 3
    height = width

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])  # [left, bottom, width, height]

    if animation:
        (line,) = ax.plot([], [], lw=4, color="blue")
        ax.set_xlim(-width / 2, width / 2)
        ax.set_ylim(0, height)
        title_text = ax.set_title("t = 0.00 s")

        def init():
            line.set_data([], [])
            title_text.set_text("t = 0.00 s")
            return line, title_text

        def update(frame_idx):
            q = q_list[frame_idx]
            t = t_list[frame_idx]
            curve = draw_robot(batched_forward_kinematics_fn, params, q, num_points)
            line.set_data(curve[:, 0], curve[:, 1])
            title_text.set_text(f"t = {t:.2f} s")
            return line, title_text

        ani = FuncAnimation(
            fig,
            update,
            frames=len(q_list),
            init_func=init,
            blit=False,
            interval=interval,
        )

        if show:
            plt.show()
        plt.close(fig)
        return HTML(ani.to_jshtml())

    elif slider:

        def update_plot(frame_idx):
            ax.cla()  # Clear current axes
            ax.set_xlim(-width / 2, width / 2)
            ax.set_ylim(0, height)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_title(f"t = {t_list[frame_idx]:.2f} s")
            q = q_list[frame_idx]
            curve = draw_robot(batched_forward_kinematics_fn, params, q, num_points)
            ax.plot(curve[:, 0], curve[:, 1], lw=4, color="blue")
            fig.canvas.draw_idle()

        # Create slider
        slider = Slider(
            ax=ax_slider,
            label="Frame",
            valmin=0,
            valmax=len(t_list) - 1,
            valinit=0,
            valstep=1,
        )
        slider.on_changed(update_plot)

        update_plot(0)  # Initial plot

        if show:
            plt.show()

        plt.close(fig)
        return HTML(
            "Slider animation not implemented in HTML format. Use matplotlib directly to view the slider."
        )  # Slider cannot be converted to HTML


if __name__ == "__main__":
    strain_basis, forward_kinematics_fn, dynamical_matrices_fn, auxiliary_fns = (
        planar_pcs_sym.factory(sym_exp_filepath, strain_selector)
    )
    # jit the functions
    dynamical_matrices_fn = jax.jit(partial(dynamical_matrices_fn))
    batched_forward_kinematics = jax.vmap(
        forward_kinematics_fn, in_axes=(None, None, 0), out_axes=-1
    )

    x0 = jnp.concatenate([q0, jnp.zeros_like(q0)])  # initial condition
    tau = jnp.zeros_like(q0)  # torques

    ode_fn = ode_factory(dynamical_matrices_fn, params, tau)
    # jit the ODE function
    ode_fn = jax.jit(ode_fn)
    # jit the ODE function
    ode_fn = jax.jit(ode_fn)
    term = ODETerm(ode_fn)

    sol = diffeqsolve(
        term,
        solver=Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=dt,
        y0=x0,
        max_steps=None,
        saveat=SaveAt(ts=video_ts),
    )
    
    # the evolution of the generalized coordinates
    q_ts = sol.ys[:, :n_q]
    # the evolution of the generalized velocities
    q_d_ts = sol.ys[:, n_q:]    

    animate_robot_matplotlib(
        batched_forward_kinematics,
        params,
        video_ts,
        q_ts,
        num_points=50,
        interval=50,
        slider=True,
        animation=False,
        show=True,
    )