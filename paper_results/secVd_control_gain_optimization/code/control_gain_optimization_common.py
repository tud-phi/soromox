import argparse
import pickle
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as onp
import scipy.io as sio

RESULT_FILENAMES = ("optimization_results.mat", "animation_data.pkl")


def parse_args(
    *, description: str, default_result_dir: Path, default_output_dir: Path
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=default_result_dir,
        help="Directory for generated MAT and pickle data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory for optional diagnostic figures.",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=100,
        help="Number of optimization iterations (paper-result default: 100).",
    )
    parser.add_argument(
        "--integral-error-saturation-scale",
        type=float,
        default=1e-2,
        help=(
            "Tendon-length error scale in meters for tanh integral-error "
            "saturation. Gamma is its reciprocal (default: 0.01 m, i.e. "
            "gamma = 100 1/m)."
        ),
    )
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--debug-nans",
        action="store_true",
        help=(
            "Enable jax_debug_nans. Slower, and the ODE solve runs inside a "
            "lax.while_loop, so it flags the rollout output rather than the "
            "failing step. The per-iteration finiteness check is always on."
        ),
    )
    return parser.parse_args()


def prepare_output_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.debug_nans:
        jax.config.update("jax_debug_nans", True)
    if args.num_iters < 1:
        raise ValueError("--num-iters must be at least 1")

    result_dir = args.result_dir.resolve()
    output_dir = args.output_dir.resolve()
    for filename in RESULT_FILENAMES:
        output_path = result_dir / filename
        if output_path.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite {output_path}; pass --force")
    return result_dir, output_dir


def finish_figure(args: argparse.Namespace, output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved {output_path}")
    if args.no_show:
        plt.close()
    else:
        plt.show()


def to_np(x):
    return onp.array(x)


def flatten_params_to_dict(params_tree, prefix=""):
    """Flatten a JAX parameter tree into keys suitable for a MAT file."""
    flat_dict = {}
    leaves, _ = jax.tree_util.tree_flatten_with_path(params_tree)
    for path, val in leaves:
        key_str = "_".join(
            str(item.key) if hasattr(item, "key") else str(item) for item in path
        )
        flat_dict[f"{prefix}{key_str}"] = to_np(val)
    return flat_dict


def save_optimization_outputs(
    result_dir: Path,
    *,
    loss_tot,
    time_iter,
    opt_vars_tot,
    t_ts,
    q_ts_init,
    qd_ts_init,
    u_ts_init,
    x_ts_init,
    q_ts_best,
    qd_ts_best,
    u_ts_best,
    x_ts_best,
    num_iters,
    x_des_ts,
    best_idx_opt,
    params,
    active_tendon_routing,
    passive_elements,
    num_segments,
) -> None:
    """Save the common MAT summary and animation pickle for either case."""
    result_dir.mkdir(parents=True, exist_ok=True)

    mat_data = {
        "history_loss": to_np(loss_tot),
        "history_time": to_np(time_iter),
        "t_ts": to_np(t_ts),
        "q_ts_init": to_np(q_ts_init),
        "qd_ts_init": to_np(qd_ts_init),
        "u_ts_init": to_np(u_ts_init),
        "x_ts_init": to_np(x_ts_init),
        "q_ts_best": to_np(q_ts_best),
        "qd_ts_best": to_np(qd_ts_best),
        "u_ts_best": to_np(u_ts_best),
        "x_ts_best": to_np(x_ts_best),
        "num_iters": to_np(num_iters),
        "x_des_ts": to_np(x_des_ts),
        "best_idx_opt": to_np(best_idx_opt),
    }
    stacked_history_tree = jax.tree_util.tree_map(
        lambda *xs: jax.numpy.stack(xs, axis=0), *opt_vars_tot
    )
    mat_data.update(
        flatten_params_to_dict(stacked_history_tree, prefix="history_opt_vars_")
    )

    mat_filename = result_dir / "optimization_results.mat"
    print(f"Saving data to {mat_filename}")
    sio.savemat(mat_filename, mat_data, do_compression=True)

    animation_dict = {
        "params": params,
        "active_tendon_routing": active_tendon_routing,
        "passive_elements": passive_elements,
        "num_segments": num_segments,
        "best_opt_vars": opt_vars_tot[best_idx_opt],
        "t_ts": t_ts,
        "q_ts": q_ts_best,
        "x_des_ts": x_des_ts,
    }
    anim_filename = result_dir / "animation_data.pkl"
    with open(anim_filename, "wb") as f:
        pickle.dump(animation_dict, f)

    print("Save complete.")
