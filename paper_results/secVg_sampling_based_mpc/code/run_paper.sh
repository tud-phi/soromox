#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$CODE_DIR/../data}"
DEVICE="${DEVICE:-cuda:0}"
SEEDS=(0 1 2)

run() {
  uv run --project "$CODE_DIR/.." python "$CODE_DIR/run_experiment.py" --device "$DEVICE" "$@"
}

declare -A MPPI_TEMP=([200]=30 [400]=25 [800]=20 [1600]=15)
CMP=(--num-mpc-steps 20 --push-goal 0.05 -0.10 --object-shape box)

for s in "${SEEDS[@]}"; do
  out="$RESULTS_DIR/cmp_seeds/seed$s"
  mkdir -p "$out"
  for k in 200 400 800 1600; do
    run "${CMP[@]}" --mppi --mppi-temp "${MPPI_TEMP[$k]}" --k-samples "$k" --noise-beta 0 --seed "$s" \
      --save-path "$out/cmp_mppi_k$k.npz"
  done
  for it in 1 2 3 4; do
    run "${CMP[@]}" --k-samples 200 --num-iters "$it" --seed "$s" \
      --save-path "$out/cmp_icem_k200_it$it.npz"
  done
  for k in 50 100 200 400; do
    run "${CMP[@]}" --k-samples "$k" --num-iters 3 --seed "$s" \
      --save-path "$out/cmp_icem_it3_k$k.npz"
  done
done

for s in "${SEEDS[@]}"; do
  extra=()
  [[ "$s" == "0" ]] && extra=(--save-samples)
  run --stop-after-one-square --object-shape box --k-samples 200 100 50 --num-iters 3 --seed "$s" \
    "${extra[@]}" --save-path "$RESULTS_DIR/square_icem_seed$s.npz"
  run --stop-after-one-square --object-shape box --mppi --mppi-temp 20 --k-samples 800 --noise-beta 0 --seed "$s" \
    --save-path "$RESULTS_DIR/square_mppi_k800_white_seed$s.npz"
done

run --stop-after-one-square --object-shape box --object-mass 0.1 --k-samples 200 100 50 --num-iters 3 --seed 0 \
  --save-path "$RESULTS_DIR/square_icem_box100g.npz"
run --stop-after-one-square --object-shape cylinder --k-samples 200 100 50 --num-iters 3 --seed 0 \
  --save-path "$RESULTS_DIR/square_icem_cylinder.npz"
