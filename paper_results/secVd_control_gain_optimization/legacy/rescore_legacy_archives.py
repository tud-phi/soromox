"""Re-derive the legacy Section Vd objectives from the archived trajectories.

Issue #154's follow-up reported that the losses stored in the legacy MAT archives
are roughly an order of magnitude below what the committed objective scores the
same trajectories at, and asked whether the weights differed, whether the loss
was rescaled before serialization, or whether the trajectories were re-evaluated
against gains other than the ones the loss belongs to.

This script answers all three from the archives alone. It needs no soft-robot
model and no old soromox API -- only the stored trajectories, their reference,
and NumPy -- so it can run in CI as a permanent regression check.

Findings, reproduced by running this file:

* Collocated. The stored loss is recovered **exactly** (<=1e-15 relative) by

      loss = 1e2 * int_0^2.5 e^T W e dt  +  2e2 * int_2.5^5 e^T W e dt
      W    = diag([1, 1, 1, 100, 100, 100])

  ``W`` is not a tuning constant: with the segment length ``L = 0.1 m`` it equals
  ``(1 / L**2) * diag(L**2, L**2, L**2, 1, 1, 1)``, so ``e^T W e = ||e_hat||**2 / L**2``
  for ``e_hat = [L * e_kappa ; e_sigma]``. The legacy collocated objective is
  exactly the strain error non-dimensionalized by segment length, at global
  factor ``1e2`` and an early/late ratio of **1:2**.

* Synergistic. The tracking part uses ``1e3`` and ``5e3`` -- an early/late ratio
  of **1:5** -- and the remainder is a strictly positive additive term, the
  tendon-power penalty ``2.5 * int P_t**2 dt`` carried by the recovered generator
  (``soromox_synergistic.py``). ``P_t`` is not archived, so this script bounds the
  residual rather than reproducing it; ``reproduce_legacy_initialization.py``
  closes that gap by re-running the generator.

So the weights differed between the two controllers, nothing was rescaled at
serialization, and the trajectories are exactly paired with their losses. The
apparent uniform ~10x is a coincidence of a 10x global factor, a 2.5x change in
the late weight, and the strain weighting.

Run with::

    python paper_results/secVd_control_gain_optimization/legacy/rescore_legacy_archives.py
"""

from __future__ import annotations

import io
import subprocess
import sys

import numpy as np
import scipy.io as sio

# PR #149 replaced the legacy MAT archives with the NPZ workflow, so they are no
# longer reachable by path on any branch. Address them by content-addressed blob
# SHA instead, which survives branch movement and deletion.
LEGACY_MAT_BLOBS = {
    "collocated": "6fbaac30c471f4e751fcc8cc73ff0d746110e65a",
    "synergistic": "621f9c76902468bbd324ce5a880b860e4a4b9b6f",
}

SEGMENT_LENGTH = 0.1
# diag(1, 1, 1, 1/L^2, 1/L^2, 1/L^2) * L^2 -- see the module docstring.
COLLOCATED_STRAIN_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) / np.array(
    [1.0, 1.0, 1.0, SEGMENT_LENGTH**2, SEGMENT_LENGTH**2, SEGMENT_LENGTH**2]
)
COLLOCATED_EARLY_WEIGHT = 1e2
COLLOCATED_LATE_WEIGHT = 2e2
SYNERGISTIC_EARLY_WEIGHT = 1e3
SYNERGISTIC_LATE_WEIGHT = 5e3

# The re-derived collocated objective is an algebraic identity, not a fit, so it
# should hold to floating-point noise over a 50001-sample trapezoid.
COLLOCATED_TOLERANCE = 1e-12


def load_legacy_archive(method: str) -> dict:
    """Read one legacy MAT archive out of the Git object store.

    Args:
        method: ``"collocated"`` or ``"synergistic"``.

    Returns:
        The archive's variables, as returned by :func:`scipy.io.loadmat`.
    """
    blob = LEGACY_MAT_BLOBS[method]
    raw = subprocess.check_output(["git", "cat-file", "blob", blob])
    return sio.loadmat(io.BytesIO(raw))


def _split_integrals(error_sq: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """Integrate a per-sample squared error over each half of the horizon.

    The generator split with ``int(len(t) / 2)``, which puts the boundary at
    ``t = 2.5 s`` for both the 50001- and 50000-sample grids.
    """
    split = len(t) // 2
    return (
        float(np.trapezoid(error_sq[:split], t[:split])),
        float(np.trapezoid(error_sq[split:], t[split:])),
    )


def score_collocated(q_ts: np.ndarray, q_des_ts: np.ndarray, t: np.ndarray) -> float:
    """Score one collocated rollout with the recovered legacy objective."""
    error_sq = (((q_des_ts - q_ts) ** 2) * COLLOCATED_STRAIN_WEIGHTS).sum(axis=1)
    early, late = _split_integrals(error_sq, t)
    return COLLOCATED_EARLY_WEIGHT * early + COLLOCATED_LATE_WEIGHT * late


def score_synergistic_tracking(
    x_ts: np.ndarray, x_des: np.ndarray, t: np.ndarray
) -> float:
    """Score the tracking part only; the power penalty is not archived."""
    error_sq = ((x_des - x_ts) ** 2).sum(axis=1)
    early, late = _split_integrals(error_sq, t)
    return SYNERGISTIC_EARLY_WEIGHT * early + SYNERGISTIC_LATE_WEIGHT * late


def improvement_over_initial_median(history_loss: np.ndarray) -> float:
    """The reduction ``docs/research.md`` reports for Section Vd."""
    return float(1.0 - np.min(history_loss[-1]) / np.median(history_loss[0]))


def best_batch(history_loss: np.ndarray) -> int:
    """The start attaining the lowest loss over all iterations."""
    return int(np.argmin(np.min(history_loss, axis=0)))


def check_collocated() -> float:
    """Reproduce every stored collocated loss. Returns the worst relative error."""
    data = load_legacy_archive("collocated")
    t = data["t_ts"][0]
    q_des_ts = data["q_des_ts"]
    history_loss = data["history_loss"]
    worst = 0.0
    print(
        "collocated: 1e2*int_0^2.5 e^T W e + 2e2*int_2.5^5 e^T W e, "
        "W = diag([1,1,1,100,100,100])"
    )
    for label, q_ts_batched, stored in (
        ("init", data["q_ts_init"], history_loss[0]),
        ("best", data["q_ts_best"], history_loss[-1]),
    ):
        for b in range(q_ts_batched.shape[0]):
            predicted = score_collocated(q_ts_batched[b], q_des_ts, t)
            relative = abs(predicted - stored[b]) / stored[b]
            worst = max(worst, relative)
            print(
                f"  {label} b{b}: predicted {predicted:12.6f}  "
                f"stored {stored[b]:12.6f}  rel {relative:.2e}"
            )
    print(f"  worst relative error: {worst:.3e}")
    print(f"  improvement: {100 * improvement_over_initial_median(history_loss):.2f} %")
    print(f"  best_batch:  {best_batch(history_loss)}")
    return worst


def check_synergistic() -> tuple[float, float]:
    """Bound the synergistic residual. Returns its (minimum, maximum)."""
    data = load_legacy_archive("synergistic")
    t = data["t_ts"][0]
    x_des = data["x_des_ts"].ravel()
    history_loss = data["history_loss"]
    residuals = []
    print(
        "\nsynergistic: 1e3*l1 + 5e3*l2 tracking, plus the unarchived "
        "2.5*int P_t^2 power penalty"
    )
    for label, x_ts_batched, stored in (
        ("init", data["x_ts_init"], history_loss[0]),
        ("best", data["x_ts_best"], history_loss[-1]),
    ):
        for b in range(x_ts_batched.shape[0]):
            tracking = score_synergistic_tracking(x_ts_batched[b], x_des, t)
            residual = stored[b] - tracking
            residuals.append(residual)
            print(
                f"  {label} b{b}: tracking {tracking:10.6f}  "
                f"stored {stored[b]:10.6f}  residual {residual:9.6f}"
            )
    low, high = min(residuals), max(residuals)
    print(f"  residual range: [{low:.6f}, {high:.6f}] -- all positive: {low > 0}")
    print(f"  improvement: {100 * improvement_over_initial_median(history_loss):.2f} %")
    print(f"  best_batch:  {best_batch(history_loss)}")
    return low, high


def main() -> int:
    worst = check_collocated()
    low, _high = check_synergistic()
    ok = True
    if worst > COLLOCATED_TOLERANCE:
        print(f"\nFAIL: collocated objective off by {worst:.3e}")
        ok = False
    if low <= 0.0:
        print(
            "\nFAIL: synergistic residual is not strictly positive, so it "
            "cannot be an additive power penalty"
        )
        ok = False
    print("\nOK" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
