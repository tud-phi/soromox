#!/usr/bin/env bash
# Render panel A (one frame per iCEM iteration of MPC step 0) in the section environment.
# Needs an RTX GPU.
#
# Usage: code/render_sample_cloud.sh <run.npz> [horizon-step] [extra render_sample_cloud.py args...]
# Frames go to $OUT_DIR (default outputs/frames, which plot_smpc_summary.py reads).
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NPZ="${1:?usage: $0 <run.npz> [horizon-step] [extra args...]}"
HORIZON_STEP="${2:-10}"
shift $(( $# < 2 ? $# : 2 ))
OUT_DIR="${OUT_DIR:-$CASE_DIR/outputs/frames}"
mkdir -p "$OUT_DIR"

for it in 0 1 2; do
  uv run --project "$CASE_DIR" python "$CASE_DIR/code/render_sample_cloud.py" \
    --npz "$NPZ" --timestep 0 --iteration "$it" --horizon-step "$HORIZON_STEP" "$@" \
    --output "$OUT_DIR/$(basename "$NPZ" .npz)_t0_iter${it}.png"
done
