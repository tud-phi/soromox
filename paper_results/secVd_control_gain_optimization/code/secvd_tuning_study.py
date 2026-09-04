"""Tuning study for the Section Vd optimizer and controller saturation.

## Stages

Run one at a time with ``--stage``; each writes its own summary. They are ordered
by what they depend on: the integration step is a property of the plant and the
solver, the saturation scale of the plant and the controller, and only the
learning rate is a property of the optimizer.

``solver-dt``
    How coarse the integration step can be before the gradient is affected.

``saturation``
    Whether the collocated integral-error saturation scale ``e_sat`` reaches the
    optimization at all.

``learning-rate``
    One independent constant-rate optimization per candidate, over the
    learning-rate magnitude (``--alphas``) or, when asked, over the per-family
    split (``--ratios``). Reports the rate to write into ``TUNED_ALPHA`` and the
    generator command that confirms it at the production budget.
"""

from __future__ import annotations

import csv
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from secvd_cli import (  # noqa: E402
    DEVICE_CHOICES,
    configure_optimization_device,
)

configure_optimization_device()

import argparse  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from jax import vmap  # noqa: E402
from secvd_case import (  # noqa: E402
    CASE_HORIZON,
    COMMITTED_E_SAT,
    COMMITTED_LR_RATIO,
    INIT_GAIN_SCALE,
    TUNED_ALPHA,
    TUNED_INIT_SEED,
    TUNED_SOLVER_DT,
    build_evaluator,
    build_optimizer,
    build_sec_vd_case,
)
from secvd_init import (  # noqa: E402
    GAIN_ORDER,
    SECVD_BATCH_SIZE,
    SECVD_INIT_SPREAD,
    sample_initial_gains,
)
from secvd_results import METHODS  # noqa: E402

jax.config.update("jax_enable_x64", True)

CASE_DIR = CODE_DIR.parent
TUNING_DIR = CASE_DIR / "data" / "tuning"

STAGES = ("solver-dt", "saturation", "learning-rate")
SOLVER_DT_GRADIENT_TOLERANCE = 1e-3

# Four decades in six points, so the sweep brackets the answer to within ~4x.
DEFAULT_ALPHAS = (30.0, 120.0, 480.0, 1900.0, 7600.0, 30000.0)


def _family_norms(tree: dict) -> dict:
    """Per-gain-family L2 norm over components, keeping the batch axis."""
    return {name: jnp.linalg.norm(tree[name], axis=-1) for name in GAIN_ORDER}


def windup_metrics(aux: dict) -> dict:
    """Per-start integrator diagnostics from a batched rollout."""
    integral = jnp.abs(aux["integral_error_ts"])
    return {
        "integral_peak": jnp.max(integral, axis=(1, 2)),
        "integral_terminal": jnp.linalg.norm(integral[:, -1, :], axis=-1),
    }


def run_constant_rate(
    *,
    gradient_fn,
    init_gains: dict,
    alpha: float,
    ratio: dict,
    num_iters: int,
    aux_metrics: Callable[[dict], dict] | None = None,
) -> dict:
    """Run one independent optimization at a fixed learning rate.

    Args:
        gradient_fn: ``value_and_grad``-shaped callable over one start.
        init_gains: Batched initial gains, each of shape ``(batch, actuators)``.
        alpha: Learning-rate magnitude.
        ratio: Per-family learning-rate ratio.
        num_iters: Iterations to run, stopping early on a non-finite loss.
        aux_metrics: Optional callable mapping a batched rollout ``aux`` to named
            per-start scalars. Defaults to :func:`windup_metrics`.

    Returns:
        Arrays of shape ``(iterations,)`` or ``(iterations, batch)`` recording
        every diagnostic taken along the run, plus ``final_gains`` of shape
        ``(batch, families, actuators)`` in :data:`GAIN_ORDER` order.
    """
    batch_size = int(init_gains["Kp"].shape[0])
    aux_metrics = windup_metrics if aux_metrics is None else aux_metrics

    optimizer = build_optimizer(alpha, ratio=ratio)
    opt_vars = {"opt_ctr_params": init_gains, "opt_atr_params": {}}
    opt_state = vmap(optimizer.init)(opt_vars)

    record: dict[str, list] = {
        key: []
        for key in ("loss", "grad_norm", "update_norm", "gain_norm", "relative_step")
    }
    record["time_iter"] = []

    for index in range(num_iters):
        started = time.time()
        (loss, aux), grad = vmap(gradient_fn)(opt_vars)
        gains = opt_vars["opt_ctr_params"]
        grad_families = _family_norms(grad["opt_ctr_params"])
        gain_families = _family_norms(gains)

        updates, opt_state = vmap(optimizer.update)(grad, opt_state, opt_vars)
        update_families = _family_norms(updates["opt_ctr_params"])
        opt_vars = optax.apply_updates(opt_vars, updates)
        jax.block_until_ready((loss, opt_vars))

        record["loss"].append(np.asarray(loss))
        for key, value in aux_metrics(aux).items():
            record.setdefault(key, []).append(np.asarray(value))
        for key, families in (
            ("grad_norm", grad_families),
            ("update_norm", update_families),
            ("gain_norm", gain_families),
        ):
            record[key].append(
                np.stack([np.asarray(families[n]) for n in GAIN_ORDER], axis=-1)
            )
        record["relative_step"].append(
            record["update_norm"][-1] / np.maximum(record["gain_norm"][-1], 1e-30)
        )
        record["time_iter"].append(time.time() - started)

        finite = bool(np.isfinite(record["loss"][-1]).all())
        print(
            f"  {index + 1:>3d}/{num_iters}  "
            f"loss = {np.median(record['loss'][-1]):.6e}  "
            f"|g| = {np.median(record['grad_norm'][-1]):.3e}  "
            f"rel step = {np.median(record['relative_step'][-1]):.3e}"
            + ("" if finite else "   [NON-FINITE]"),
            flush=True,
        )
        if not finite:
            print(f"  diverged at iteration {index + 1}; stopping.", flush=True)
            break

    stacked = {key: np.asarray(values) for key, values in record.items()}
    stacked["final_gains"] = np.stack(
        [np.asarray(opt_vars["opt_ctr_params"][n]) for n in GAIN_ORDER], axis=1
    )
    stacked["alpha"] = np.asarray(float(alpha))
    stacked["batch_size"] = np.asarray(batch_size)
    stacked["ratio"] = np.asarray([ratio[n] for n in GAIN_ORDER])
    return stacked


@dataclass(frozen=True)
class Candidate:
    """One point of a sweep: what to run, and what to label the result."""

    key: Any
    problem: Any
    alpha: float
    ratio: dict[str, float]
    aux_metrics: Callable[[dict], dict] | None = None


def sweep(
    candidates: list[Candidate],
    *,
    num_iters: int,
    batch_size: int,
    seed: int,
    label: str,
) -> dict:
    """Run one independent constant-rate optimization per candidate.

    Args:
        candidates: The points to compare. Sweeping ``alpha``, ``ratio`` or a
            problem setting differs only in how these were built.
        num_iters: Iteration budget for each candidate.
        batch_size: Number of starts each candidate optimizes.
        seed: PRNG seed behind the shared initializations.
        label: Prefix for the per-candidate progress banner.

    Returns:
        One result per candidate, keyed by :attr:`Candidate.key`.
    """
    results = {}
    for candidate in candidates:
        print("")
        print(
            f"[{label}] {candidate.key}: alpha = {candidate.alpha:g}, "
            f"ratio = {candidate.ratio}",
            flush=True,
        )
        results[candidate.key] = run_constant_rate(
            gradient_fn=candidate.problem.gradient_fn,
            init_gains=_sample_starts(candidate.problem, batch_size, seed),
            alpha=candidate.alpha,
            ratio=candidate.ratio,
            num_iters=num_iters,
            aux_metrics=candidate.aux_metrics,
        )
    return results


def _best_loss(result: dict) -> float:
    """Lowest median-over-starts loss the run reached, or ``inf`` if it died."""
    loss = np.median(np.asarray(result["loss"]), axis=-1)
    finite = loss[np.isfinite(loss)]
    return float(finite.min()) if finite.size else float("inf")


def _start_spread(result: dict) -> float:
    """Relative spread of the final loss across starts, at its best iteration."""
    loss = np.asarray(result["loss"])
    finite = np.isfinite(loss).all(axis=-1)
    if not finite.any():
        return float("inf")
    row = loss[np.flatnonzero(finite)[-1]]
    return float(row.max() / row.min() - 1.0)


def report_sweep(results: dict, *, axis: str, num_iters: int) -> tuple[Any, bool]:
    """Print each candidate's outcome and explicit the winner.

    Args:
        results: Output of :func:`sweep`.
        axis: Name of the swept quantity, for the table header.
        num_iters: Requested budget, so a short run reads as a divergence.

    Returns:
        ``(winner, separated)``. Only candidates that completed the whole budget
        are ranked: a run that goes non-finite partway still has a finite last
        loss, and ranking on that would let a rate which diverges beat one that
        survives. ``separated`` is ``False`` when the two best survivors differ
        by less than the spread across starts, meaning the budget was too short
        to rank them and both are worth confirming.
    """
    print("")
    print(f"  {axis:>14}  {'best loss':>13}  {'improvement':>11}  {'iters':>6}")
    ranked = []
    for key, result in results.items():
        best = _best_loss(result)
        completed = len(result["loss"])
        if not np.isfinite(best):
            print(f"  {key!s:>14}  {'diverged':>13}  {'':>11}  {completed:>6d}")
            continue
        survived = completed == num_iters
        first = float(np.median(np.asarray(result["loss"]), axis=-1)[0])
        print(
            f"  {key!s:>14}  {best:>13.6e}  {100 * (1 - best / first):>10.2f} %  "
            f"{completed:>6d}" + ("" if survived else f"   [died at {completed}]")
        )
        if survived:
            ranked.append((best, key))
    if not ranked:
        print("")
        print(
            f"  No candidate completed {num_iters} iterations; every rate here "
            "diverges. Extend the sweep downwards."
        )
        return None, False
    ranked.sort()
    winner = ranked[0][1]
    separated = True
    if len(ranked) > 1:
        gap = ranked[1][0] / ranked[0][0] - 1.0
        spread = _start_spread(results[winner])
        separated = gap > spread
        if not separated:
            print("")
            print(
                f"  Top two are within the spread across starts "
                f"({100 * gap:.2f} % apart against {100 * spread:.2f} %): this "
                f"budget does not rank them. Confirm both."
            )
    return winner, separated


def report_step_efficiency(result: dict) -> None:
    """How much of the requested learning rate reaches the parameters."""
    alpha = float(result["alpha"])
    ratio = result["ratio"]
    print("  family   median |update|/lr   spread over the run")
    for position, name in enumerate(GAIN_ORDER):
        lr = alpha * float(ratio[position])
        observed = np.median(result["update_norm"][..., position], axis=-1)
        efficiency = observed / np.maximum(lr, 1e-30)
        finite = efficiency[np.isfinite(efficiency)]
        if finite.size == 0:
            print(f"  {name:>5}   no finite iterations")
            continue
        print(
            f"  {name:>5}   {np.median(finite):16.3e}   "
            f"{finite.min():.2e} .. {finite.max():.2e}"
        )


def save_run(result: dict, out_dir: Path, tag: str) -> Path:
    """Write one candidate's run as an NPZ archive and a per-iteration CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{tag}.npz"
    np.savez_compressed(npz_path, **result)
    csv_path = out_dir / f"{tag}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["iteration", "loss_median"]
            + [
                f"{key}_{name}"
                for key in ("grad", "update", "rel")
                for name in GAIN_ORDER
            ]
        )
        for index in range(len(result["loss"])):
            row = [str(index + 1), f"{np.median(result['loss'][index]):.6e}"]
            for key in ("grad_norm", "update_norm", "relative_step"):
                row += [f"{np.median(result[key][index, :, p]):.6e}" for p in range(3)]
            writer.writerow(row)
    return npz_path


def _sample_starts(problem, batch_size: int, seed: int) -> dict:
    """Draw the same starts the generator uses, so losses stay comparable."""
    scale = INIT_GAIN_SCALE[problem.method]
    centre = {
        name: value * scale[name] for name, value in problem.nominal_gains.items()
    }
    return sample_initial_gains(
        centre,
        batch_size=batch_size,
        seed=seed,
        spread=SECVD_INIT_SPREAD,
    )


def stiffest_start(init_gains: dict) -> int:
    """Return the start whose closed loop is hardest to integrate."""
    return int(np.argmax(np.linalg.norm(np.asarray(init_gains["Kp"]), axis=-1)))


def _gradient_vector(grad: dict) -> np.ndarray:
    return np.concatenate(
        [np.asarray(grad["opt_ctr_params"][n]).reshape(-1) for n in GAIN_ORDER]
    )


def stage_solver_dt(args, case) -> dict:
    """Measure how far the integration step can be raised."""
    reference = build_evaluator(
        args.method, case=case, horizon=args.horizon, e_sat=args.e_sat
    )
    init_gains = _sample_starts(reference, args.batch_size, args.init_seed)
    index = stiffest_start(init_gains)
    gains = {name: value[index] for name, value in init_gains.items()}
    print(
        f"Stiffest of {args.batch_size} starts is index {index} "
        f"(|Kp| = {float(np.linalg.norm(np.asarray(gains['Kp']))):.1f})",
        flush=True,
    )
    header = (
        f"{'solver_dt':>10} {'steps':>8} {'eval':>8} {'loss':>14} "
        f"{'d_loss':>9} {'d_grad':>9}  verdict"
    )
    print(header, flush=True)
    baseline = None
    rows = []
    for solver_dt in sorted(args.solver_dts):
        problem = build_evaluator(
            args.method,
            case=case,
            horizon=args.horizon,
            solver_dt=solver_dt,
            e_sat=args.e_sat,
        )
        opt_vars = {"opt_ctr_params": gains, "opt_atr_params": {}}
        (loss, _aux), grad = problem.gradient_fn(opt_vars)
        float(loss)
        started = time.time()
        (loss, _aux), grad = problem.gradient_fn(opt_vars)
        vector = _gradient_vector(grad)
        elapsed = time.time() - started
        loss = float(loss)
        if baseline is None:
            baseline = (loss, vector)
        d_loss = abs(loss - baseline[0]) / abs(baseline[0])
        d_grad = float(
            np.linalg.norm(vector - baseline[1]) / np.linalg.norm(baseline[1])
        )
        finite = bool(np.isfinite(loss) and np.all(np.isfinite(vector)))
        accepted = bool(finite and d_grad <= SOLVER_DT_GRADIENT_TOLERANCE)
        rows.append((solver_dt, elapsed, loss, d_loss, d_grad, accepted))
        print(
            f"{solver_dt:>10.1e} {int(args.horizon / solver_dt):>8d} "
            f"{elapsed:>7.1f}s {loss:>14.6e} {d_loss:>9.2e} {d_grad:>9.2e}  "
            f"{'ok' if accepted else 'REJECTED'}",
            flush=True,
        )
    accepted = [row for row in rows if row[5]]
    largest = max(row[0] for row in accepted)
    fastest = next(row[1] for row in rows if row[0] == largest)
    print("")
    print(
        f"Largest step within {SOLVER_DT_GRADIENT_TOLERANCE:g} relative gradient "
        f"error: solver_dt = {largest:g}, {rows[0][1] / fastest:.1f}x faster than "
        f"{rows[0][0]:g}"
    )
    print(
        "The controller is queried at every integration step, so this is also "
        "the effective control rate."
    )
    return {
        "stage": np.asarray("solver-dt"),
        "solver_dt": np.asarray([row[0] for row in rows]),
        "eval_seconds": np.asarray([row[1] for row in rows]),
        "loss": np.asarray([row[2] for row in rows]),
        "rel_loss_error": np.asarray([row[3] for row in rows]),
        "rel_gradient_error": np.asarray([row[4] for row in rows]),
        "accepted": np.asarray([row[5] for row in rows]),
        "recommended_solver_dt": np.asarray(largest),
        "start_index": np.asarray(index),
    }


def saturation_window(problem) -> tuple[float, float]:
    """Physically meaningful range for ``e_sat``."""
    (_loss, aux), _grad = problem.gradient_fn(
        {"opt_ctr_params": problem.nominal_gains, "opt_atr_params": {}}
    )
    error = np.abs(np.asarray(problem.control_error_fn(aux)))
    return float(error[-1].max()), float(error[0].max())


def stage_saturation(args, case, alpha: float) -> dict:
    """Sensitivity of the optimization to the saturation scale."""
    probe = build_evaluator(
        args.method, case=case, horizon=args.horizon, solver_dt=args.solver_dt
    )
    steady, step = saturation_window(probe)
    print("")
    print(
        f"Active range of e_sat, from the nominal rollout: [{steady:.3g}, {step:.3g}] m"
    )
    e_sats = args.e_sats
    if e_sats is None:
        e_sats = [float(v) for v in np.geomspace(steady, step, 5)]
        print(f"  Sweeping it: {' '.join(f'{v:.3g}' for v in e_sats)}")
    else:
        outside = [v for v in e_sats if not steady <= v <= step]
        if outside:
            joined = ", ".join(f"{v:g}" for v in outside)
            print(f"  Outside that range: {joined} m.")

    candidates = []
    for e_sat in e_sats:
        problem = build_evaluator(
            args.method,
            case=case,
            horizon=args.horizon,
            solver_dt=args.solver_dt,
            e_sat=e_sat,
        )

        def metrics(aux, e_sat=e_sat, problem=problem):
            error = jnp.abs(vmap(problem.control_error_fn)(aux))
            return windup_metrics(aux) | {
                "saturated_fraction": jnp.mean(error > e_sat, axis=(1, 2))
            }

        candidates.append(
            Candidate(
                key=e_sat,
                problem=problem,
                alpha=alpha,
                ratio=args.ratio,
                aux_metrics=metrics,
            )
        )

    results = sweep(
        candidates,
        num_iters=args.num_iters,
        batch_size=args.batch_size,
        seed=args.init_seed,
        label="saturation",
    )
    summary = []
    for e_sat, result in results.items():
        result["e_sat"] = np.asarray(e_sat)
        loss = np.median(np.asarray(result["loss"]), axis=-1)
        live = np.isfinite(loss)
        last = int(np.flatnonzero(live)[-1]) if live.any() else 0
        summary.append(
            (
                e_sat,
                float(loss[live].min()) if live.any() else float("nan"),
                float(np.median(result["saturated_fraction"][last])),
                float(np.median(result["integral_peak"][live].max(axis=0)))
                if live.any()
                else float("nan"),
                float(np.median(result["integral_terminal"][last])),
                float(np.median(result["grad_norm"][live]))
                if live.any()
                else float("nan"),
                int(live.sum()),
            )
        )

    print("")
    print(f"Median saturation sensitivity over {args.num_iters} iterations.")
    print("'saturated' is the fraction of the rollout with |e| > e_sat")
    print(
        f"  {'e_sat [m]':>10}  {'best loss':>13}  {'saturated':>10}  "
        f"{'peak |I|':>10}  {'final |I|':>10}  {'|grad|':>10}  {'iters':>6}"
    )
    for e_sat, loss, fraction, peak, terminal, grad, live in summary:
        print(
            f"  {e_sat:>10.4g}  {loss:>13.6e}  {100 * fraction:>9.1f}%  "
            f"{peak:>10.3e}  {terminal:>10.3e}  {grad:>10.3e}  {live:>6d}"
        )
    report_saturation_sensitivity(summary, args.num_iters)
    return results


def report_saturation_sensitivity(summary: list, num_iters: int) -> None:
    """Check the sensitivity of the optimization to the saturation scale."""
    usable = [row for row in summary if np.isfinite(row[1])]
    print("")
    if len(usable) < 2:
        print("Too few finite points to judge sensitivity.")
        return

    scales = [row[0] for row in usable]
    losses = [row[1] for row in usable]
    peaks = [row[3] for row in usable]
    grads = [row[5] for row in usable]
    span = max(scales) / min(scales)
    loss_spread = max(losses) / min(losses) - 1.0
    grad_spread = max(grads) / min(grads)
    peak_spread = max(peaks) / min(peaks)
    stalled = [row[0] for row in summary if row[6] < num_iters]

    print(
        f"Across {span:.0f}x in e_sat: loss moves {100 * loss_spread:.3f} %, "
        f"gradient magnitude {grad_spread:.2f}x, integrator peak "
        f"{peak_spread:.1f}x."
    )
    if stalled:
        print(
            "  Runs that did not complete, at e_sat = "
            + ", ".join(f"{value:g}" for value in stalled)
            + " m."
        )
    else:
        print(f"  Every point completed all {num_iters} iterations.")
    settled = loss_spread < 1e-2 and grad_spread < 2.0 and not stalled
    print(
        "  The scale reaches the integrator and not the optimization: keeping "
        f"the committed e_sat = {COMMITTED_E_SAT:g} m."
        if settled
        else "  The scale does reach the optimization; report it with the run."
    )


def _ratio_label(ratio: dict[str, float]) -> str:
    return ",".join(f"{ratio[name]:g}" for name in GAIN_ORDER)


def _confirm_command(method: str, key, result: dict) -> str:
    """The generator run that confirms one candidate at the production budget."""
    generator = CODE_DIR.relative_to(CODE_DIR.parents[2]) / "optimize_control_gains.py"
    slug = f"{key:g}" if isinstance(key, float) else str(key).replace(",", "-")
    ratio = " ".join(f"{value:g}" for value in np.asarray(result["ratio"]))
    return (
        f"  python {generator} --method {method} \\\n"
        f"      --alpha {float(result['alpha']):g} --ratio {ratio} \\\n"
        f"      --num-iters 100 --result-dir /tmp/secvd-confirm-{slug} --force"
    )


def stage_learning_rate(args, case) -> dict:
    """Locate the learning rate, over the magnitude or the per-family split."""
    problem = build_evaluator(
        args.method,
        case=case,
        horizon=args.horizon,
        solver_dt=args.solver_dt,
        e_sat=args.e_sat,
    )
    if args.ratios is not None:
        candidates = [
            Candidate(
                key=_ratio_label(ratio),
                problem=problem,
                alpha=args.alpha,
                ratio=ratio,
            )
            for ratio in args.ratios
        ]
        axis = "ratio"
    else:
        candidates = [
            Candidate(key=alpha, problem=problem, alpha=alpha, ratio=args.ratio)
            for alpha in args.alphas
        ]
        axis = "alpha"

    results = sweep(
        candidates,
        num_iters=args.num_iters,
        batch_size=args.batch_size,
        seed=args.init_seed,
        label="learning-rate",
    )
    winner, separated = report_sweep(results, axis=axis, num_iters=args.num_iters)
    if winner is None:
        return results

    best = results[winner]
    alpha = float(best["alpha"])
    ratio = {name: float(best["ratio"][i]) for i, name in enumerate(GAIN_ORDER)}
    print("")
    print("Step efficiency at the winner, per gain family:")
    report_step_efficiency(best)
    print("")
    print(
        "  (lr_P, lr_I, lr_D) = ("
        + ", ".join(f"{alpha * ratio[name]:g}" for name in GAIN_ORDER)
        + ")"
    )
    if axis == "alpha":
        print(f'  TUNED_ALPHA["{args.method}"] = {alpha}')
    else:
        print(f"  COMMITTED_LR_RATIO = {ratio}")
    print("")
    print(
        f"Confirm it at the production budget -- {args.num_iters} iterations rank "
        "candidates,\nthey do not prove one survives 100:"
    )
    ranked = sorted(
        (_best_loss(r), k)
        for k, r in results.items()
        if len(r["loss"]) == args.num_iters and np.isfinite(_best_loss(r))
    )
    confirm = [winner] if separated else [key for _loss, key in ranked[:2]]
    for key in confirm:
        print(_confirm_command(args.method, key, results[key]))
    return results


def _parse_ratio(text: str) -> dict[str, float]:
    values = [float(part) for part in text.split(",")]
    if len(values) != len(GAIN_ORDER):
        raise argparse.ArgumentTypeError(
            f"a ratio needs {len(GAIN_ORDER)} comma-separated values, got {text!r}"
        )
    return dict(zip(GAIN_ORDER, values, strict=True))


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    parser.add_argument("--stage", choices=STAGES, default="learning-rate")
    parser.add_argument("--method", choices=METHODS, default="collocated")
    parser.add_argument("--horizon", type=float, default=CASE_HORIZON)
    parser.add_argument(
        "--solver-dt",
        type=float,
        default=TUNED_SOLVER_DT,
        help="Integration step for every stage but solver-dt, which sweeps it.",
    )
    parser.add_argument(
        "--solver-dts",
        type=float,
        nargs="+",
        default=[1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
        help="Steps the solver-dt stage compares against the finest.",
    )
    parser.add_argument("--e-sat", type=float, default=COMMITTED_E_SAT)
    parser.add_argument(
        "--e-sats",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Saturation scales in metres. Defaults to five points across the "
            "model-based window the saturation stage measures."
        ),
    )
    parser.add_argument(
        "--ratio",
        type=_parse_ratio,
        default=dict(COMMITTED_LR_RATIO),
        metavar="KP,KI,KD",
        help="Per-family learning-rate split used wherever one is not swept.",
    )
    parser.add_argument(
        "--ratios",
        type=_parse_ratio,
        nargs="+",
        default=None,
        metavar="KP,KI,KD",
        help="Sweep the per-family split at a fixed --alpha.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Fixed magnitude for the stages that do not sweep it; TUNED_ALPHA "
        "by default.",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=list(DEFAULT_ALPHAS),
        help="Magnitudes the learning-rate stage compares.",
    )
    parser.add_argument(
        "--num-iters", type=int, default=10, help="Iterations per candidate."
    )
    parser.add_argument("--batch-size", type=int, default=SECVD_BATCH_SIZE)
    parser.add_argument("--init-seed", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=TUNING_DIR)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--tag", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.alpha is None:
        args.alpha = TUNED_ALPHA[args.method]
    if args.init_seed is None:
        args.init_seed = TUNED_INIT_SEED[args.method]
    tag = args.tag or f"{args.stage}_{args.method}_T{args.horizon:g}"
    case = build_sec_vd_case()
    probe = build_evaluator(args.method, case=case, horizon=args.horizon)
    print(
        f"[{args.stage}] {args.method}, horizon {args.horizon} s, "
        f"batch {args.batch_size}, {args.num_iters} iterations per candidate",
        flush=True,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "solver-dt":
        result = stage_solver_dt(args, case)
        np.savez_compressed(args.out_dir / f"{tag}.npz", **result)
        print(f"Saved {args.out_dir / (tag + '.npz')}")
        return

    if args.stage == "saturation":
        if not probe.supports_saturation:
            print("")
            print(
                f"Skipped: the {args.method} controller does not implement "
                "saturation of the integral error."
            )
            return
        results = stage_saturation(args, case, args.alpha)
    else:
        results = stage_learning_rate(args, case)

    for key, result in results.items():
        suffix = f"{key:g}" if isinstance(key, float) else str(key)
        print(f"Saved {save_run(result, args.out_dir, tag + '_' + suffix)}")


if __name__ == "__main__":
    main()
