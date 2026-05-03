from pathlib import Path

import cv2  # importing cv2
import jax

jax.config.update("jax_enable_x64", True)  # double precision


import jax.numpy as jnp

import soromox
from soromox.parameters.hsa_params import (
    PLANAR_HSA_FPU_CONTROL_PARAMS,
    PLANAR_HSA_FPU_HYSTERESIS_CONTROL_PARAMS,
)
from soromox.rendering.planar_hsa.opencv_renderer import (
    OpenCVPlanarHSARenderer,
)
from soromox.systems import PlanarHSA, PlanarHSAStructure, SystemState

jnp.set_printoptions(
    threshold=jnp.inf,
    linewidth=jnp.inf,
    formatter={"float_kind": lambda x: "0" if x == 0 else f"{x:.2e}"},
)


if __name__ == "__main__":
    num_segments = 1
    num_rods_per_segment = 2

    # filepath to symbolic expressions
    sym_exp_filepath = (
        Path(soromox.__file__).parent
        / "symbolic_expressions"
        / f"planar_hsa_ns-{num_segments}_nrs-{num_rods_per_segment}.dill"
    )

    # activate all strains (i.e. bending, shear, and axial)
    strain_selector = jnp.ones((3 * num_segments,), dtype=bool)
    consider_hysteresis = True

    params = (
        PLANAR_HSA_FPU_HYSTERESIS_CONTROL_PARAMS
        if consider_hysteresis
        else PLANAR_HSA_FPU_CONTROL_PARAMS
    )
    # increase damping for simulation stability
    params = params.replace(
        bending_damping=5 * params.bending_damping,
        shear_damping=5 * params.shear_damping,
        axial_damping=5 * params.axial_damping,
    )

    # ======================================================
    # Robot initialization
    # ======================================================
    robot = PlanarHSA(
        params=params,
        structure=PlanarHSAStructure(
            symbolic_expression_path=str(sym_exp_filepath),
            strain_selector=strain_selector,
            consider_underactuation=True,
            consider_hysteresis=consider_hysteresis,
        ),
    )
    print(
        f"Planar HSA with {num_segments} segments and {num_rods_per_segment} rods per segment initialized."
    )
    renderer = OpenCVPlanarHSARenderer(robot, width=700, height=700)

    # =====================================================
    # Simulation upon time
    # =====================================================
    # Initial configuration
    q0 = jnp.array([jnp.pi, 0.0, 0.0])
    # Initial velocities
    qd0 = jnp.zeros_like(q0)
    # Motor actuation angles
    phi = jnp.array([jnp.pi, jnp.pi / 2])
    # initial hysteresis state
    if consider_hysteresis:
        z0 = jnp.zeros((robot.num_hysteresis,))
    else:
        z0 = jnp.array([])
    y0 = jnp.concatenate([q0, qd0, z0])

    # Displaying the image
    window_name = f"Planar HSA with {num_segments} segments"
    img = renderer.render_frame(q0)

    win = "Planar HSA"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, img)
    key = cv2.waitKey(0) & 0xFF
    if key in (27, ord("q")):
        cv2.destroyWindow(win)

    # Simulation time parameters
    t0 = 0.0
    t1 = 5.0
    solver_dt = 5e-5  # time step
    save_dt = 0.01

    initial_state = SystemState(t=t0, y=y0)
    trajectory = robot.rollout_to(
        initial_state=initial_state,
        u=phi,
        t1=t1,
        solver_dt=solver_dt,
        save_dt=save_dt,
        max_steps=None,
    )
    ts = trajectory.t
    q_ts, qd_ts, _ = jnp.split(
        trajectory.y, [robot.num_dofs, 2 * robot.num_dofs], axis=1
    )

    # create video
    video_path = Path("videos") / "planar_hsa.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)

    renderer.render_sequence(
        ts=ts,
        q_ts=q_ts,
        record_path=str(video_path),
    )
    print(f"Video saved at {video_path}")

    # Playback and display of generated video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Erreur lors de l'ouverture de la vidéo {video_path}")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Animation Planar HSA", frame)
            key = cv2.waitKey(int(1000 / (1 / save_dt)))
            if key in (27, ord("q")):
                break
        cap.release()
        cv2.destroyAllWindows()
