import cv2  # importing cv2
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)  # double precision
from pathlib import Path

import soromox
from soromox.parameters.hsa_params import (
    PARAMS_FPU_CONTROL,
    PARAMS_FPU_HYSTERESIS_CONTROL,
)
from soromox.systems.planar_hsa import PlanarHSA
from soromox.rendering.planar_hsa.opencv_renderer import draw_robot, animate_robot


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
        PARAMS_FPU_HYSTERESIS_CONTROL if consider_hysteresis else PARAMS_FPU_CONTROL
    )
    # increase damping for simulation stability
    params["zetab"] = 5 * params["zetab"]
    params["zetash"] = 5 * params["zetash"]
    params["zetaa"] = 5 * params["zetaa"]

    # ======================================================
    # Robot initialization
    # ======================================================
    robot = PlanarHSA(
        sym_exp_filepath=sym_exp_filepath,
        params=params,
        strain_selector=strain_selector,
        consider_underactuation=True,
        consider_hysteresis=consider_hysteresis,
    )
    print(
        f"Planar HSA with {num_segments} segments and {num_rods_per_segment} rods per segment initialized."
    )

    # =====================================================
    # Simulation upon time
    # =====================================================
    # Initial configuration
    q0 = jnp.array([jnp.pi, 0.0, 0.0])
    # Initial velocities
    qd0 = jnp.zeros_like(q0)
    # Motor actuation angles
    phi = jnp.array([jnp.pi, jnp.pi / 2])

    # Displaying the image
    window_name = f"Planar HSA with {num_segments} segments"
    img = draw_robot(
        robot,
        q=q0,
        show=False,
    )

    win = "Planar HSA"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, img)
    key = cv2.waitKey(0) & 0xFF
    if key in (27, ord("q")):
        cv2.destroyWindow(win)

    # Simulation time parameters
    t0 = 0.0
    t1 = 5.0
    dt = 5e-5  # time step
    save_every_n_steps = 100

    ts, q_ts, qd_ts = robot.resolve_upon_time(
        q0=q0,
        qd0=qd0,
        u0=phi,
        t0=t0,
        t1=t1,
        dt=dt,
        save_every_n_steps=save_every_n_steps,
        max_steps=None,
    )

    # create video
    video_width, video_height = 700, 700  # img height and width
    video_path = Path(__file__).parent / "videos" / "planar_hsa.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)

    animate_robot(
        robot,
        video_path,
        video_ts=ts,
        q_ts=q_ts,
        video_width=video_width,
        video_height=video_height,
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
            key = cv2.waitKey(int(1000 / (1 / dt / skip_step)))
            if key in (27, ord("q")):
                break
        cap.release()
        cv2.destroyAllWindows()
