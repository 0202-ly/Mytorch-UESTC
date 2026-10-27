param(
  [string]$Python = "python",
  [string]$Script = "experiments/cuda_conv_benchmark.py",
  [string]$OutputDir = "results/nsight",
  [switch]$IncludeCompute
)

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# Nsight Systems: timeline, CUDA API calls, kernel launches, CPU/GPU overlap.
nsys profile --force-overwrite true --trace=cuda,nvtx,osrt --sample=none `
  --output="$OutputDir/cuda_conv_timeline" $Python $Script

# Nsight Compute hardware counters need permission on some cloud GPUs.
# Keep this optional: the verified coursework result is from Nsight Systems.
if ($IncludeCompute) {
  ncu --set full --target-processes all --export "$OutputDir/cuda_conv_kernel" `
    $Python $Script
}
