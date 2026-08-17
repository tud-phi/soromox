import argparse
import pickle
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as onp

RESULT_FILENAMES = ("optimization_results.npz", "animation_data.pkl")


def parse_args(
    *, description: str, default_result_dir: Path, default_output_dir: Path
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=default_result_dir,
        help="Directory for generated NumPy archive and pickle data.",
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
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--plot-only",
        action="store_true",
        help=(
            "Skip optimization and recreate the saved Section Vd comparison "
            "plot from the result archives."
        ),
    )
    mode_group.add_argument(
        "--render-only",
        action="store_true",
        help=(
            "Skip optimization and render saved Section Vd trajectories and animations."
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Animation frame rate used by --render-only.",
    )
    parser.add_argument(
        "--gif",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create GIF previews in --render-only mode.",
    )
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
    if not (args.plot_only or args.render_only) and args.num_iters < 1:
        raise ValueError("--num-iters must be at least 1")

    result_dir = args.result_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not (args.plot_only or args.render_only):
        for filename in RESULT_FILENAMES:
            output_path = result_dir / filename
            if output_path.exists() and not args.force:
                raise FileExistsError(
                    f"Refusing to overwrite {output_path}; pass --force"
                )
    return result_dir, output_dir


def run_saved_postprocessing(
    args: argparse.Namespace, result_dir: Path, output_dir: Path
) -> bool:
    """Run a saved-data mode and return whether the generator should exit."""
    if not (args.plot_only or args.render_only):
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = result_dir.parent

    if args.plot_only:
        from plot_control_gain_optimization import build_figure, configure_matplotlib

        output_path = output_dir / "plot_control_gain_opt.pdf"
        if output_path.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite {output_path}; pass --force")

        configure_matplotlib()
        figure = build_figure(data_dir)
        figure.savefig(output_path, dpi=300)
        print(f"Plot saved at: {output_path}")
        if args.no_show:
            plt.close(figure)
        else:
            plt.show()
        return True

    from render_control_gain_optimization_animations import render_saved_results

    outputs = render_saved_results(
        data_dir=data_dir,
        output_dir=output_dir,
        fps=args.fps,
        make_gif=args.gif,
        force=args.force,
    )
    print("Generated:")
    for path in outputs:
        print(f"  {path}")
    return True


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
    """Flatten a JAX parameter tree into keys suitable for a NumPy archive."""
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
    """Save the common NumPy summary and animation pickle for either case."""
    result_dir.mkdir(parents=True, exist_ok=True)

    result_data = {
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
    result_data.update(
        flatten_params_to_dict(stacked_history_tree, prefix="history_opt_vars_")
    )

    result_filename = result_dir / "optimization_results.npz"
    print(f"Saving data to {result_filename}")
    onp.savez_compressed(result_filename, **result_data)

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
