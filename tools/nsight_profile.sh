#!/usr/bin/env bash
# Profile the independent CUDA C convolution with Nsight Systems on Linux.
set -euo pipefail

python_bin="${1:-python}"
output_dir="${2:-results/nsight}"
mkdir -p "$output_dir"

nsys profile --force-overwrite true --trace=cuda,nvtx,osrt --sample=none \
  --output="$output_dir/cuda_conv_timeline" \
  "$python_bin" -m experiments.cuda_conv_benchmark

printf 'Saved Nsight Systems report under %s\n' "$output_dir"
