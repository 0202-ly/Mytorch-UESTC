"""Custom CUDA convolution benchmark for MyTorch's CuPy tensor backend.

The kernel is intentionally simple (one CUDA thread per output element) so it
can be inspected in Nsight Compute. It is a correctness/performance baseline,
not a replacement for cuDNN.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np


KERNEL = r"""
extern "C" __global__ void conv2d_nchw(const float* x, const float* w,
    const float* b, float* y, int N, int C, int H, int W,
    int K, int R, int S, int OH, int OW) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int total = N * K * OH * OW;
  if (idx >= total) return;
  int ow = idx % OW; idx /= OW;
  int oh = idx % OH; idx /= OH;
  int k = idx % K; idx /= K;
  int n = idx;
  float acc = b[k];
  for (int c = 0; c < C; ++c)
    for (int r = 0; r < R; ++r)
      for (int s = 0; s < S; ++s) {
        int xi = ((n * C + c) * H + oh + r) * W + ow + s;
        int wi = ((k * C + c) * R + r) * S + s;
        acc += x[xi] * w[wi];
      }
  y[((n * K + k) * OH + oh) * OW + ow] = acc;
}
"""


def cpu_conv(x, w, b):
    n, c, h, width = x.shape
    k, _, r, s = w.shape
    y = np.empty((n, k, h - r + 1, width - s + 1), dtype=np.float32)
    for ni in range(n):
        for ki in range(k):
            for oh in range(y.shape[2]):
                for ow in range(y.shape[3]):
                    y[ni, ki, oh, ow] = b[ki] + np.sum(x[ni, :, oh:oh+r, ow:ow+s] * w[ki])
    return y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/cuda_conv_benchmark.csv")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()
    try:
        import cupy as cp
    except ImportError as exc:
        raise SystemExit("需要在 CUDA 环境安装 cupy-cuda12x（或匹配本机 CUDA 的 CuPy）") from exc
    if not cp.cuda.is_available():
        raise SystemExit("当前没有可用 CUDA GPU")
    rng = np.random.default_rng(0)
    x = cp.asarray(rng.normal(size=(4, 3, 32, 32)).astype(np.float32))
    w = cp.asarray(rng.normal(size=(16, 3, 3, 3)).astype(np.float32))
    b = cp.asarray(rng.normal(size=(16,)).astype(np.float32))
    y = cp.empty((4, 16, 30, 30), dtype=cp.float32)
    kernel = cp.RawKernel(KERNEL, "conv2d_nchw")
    threads = 256
    blocks = ((y.size + threads - 1) // threads,)
    for _ in range(args.warmup):
        kernel(blocks, (threads,), (x, w, b, y, 4, 3, 32, 32, 16, 3, 3, 30, 30))
    cp.cuda.Stream.null.synchronize()
    start = cp.cuda.Event(); end = cp.cuda.Event()
    start.record()
    for _ in range(args.repeat):
        kernel(blocks, (threads,), (x, w, b, y, 4, 3, 32, 32, 16, 3, 3, 30, 30))
    end.record(); end.synchronize()
    gpu_ms = cp.cuda.get_elapsed_time(start, end) / args.repeat
    reference = cpu_conv(cp.asnumpy(x), cp.asnumpy(w), cp.asnumpy(b))
    max_error = float(np.max(np.abs(cp.asnumpy(y) - reference)))
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kernel", "gpu_ms", "max_abs_error", "elements"])
        writer.writeheader(); writer.writerow({"kernel": "custom_conv2d_nchw", "gpu_ms": gpu_ms, "max_abs_error": max_error, "elements": y.size})
    print(f"custom_conv2d_nchw: {gpu_ms:.3f} ms, max_abs_error={max_error:.6g}")


if __name__ == "__main__":
    main()
