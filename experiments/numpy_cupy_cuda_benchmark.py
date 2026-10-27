"""Compare identical-input convolutions; GPU times exclude host transfers."""
from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path

import numpy as np


CASES = [(1, 3, 16, 16, 8, 3, 3), (4, 3, 32, 32, 16, 3, 3), (8, 3, 64, 64, 32, 3, 3)]


def conv_numpy(x, w, b):
    n, c, h, width = x.shape
    k, _, r, s = w.shape
    windows = np.lib.stride_tricks.sliding_window_view(x, (r, s), axis=(2, 3))
    # n,c,oh,ow,r,s -> n,oh,ow,c,r,s
    windows = windows.transpose(0, 2, 3, 1, 4, 5)
    return np.einsum("nohcrs,kcrs->nkoh", windows, w, optimize=True) + b[None, :, None, None]


def conv_cupy(cp, x, w, b):
    windows = cp.lib.stride_tricks.sliding_window_view(x, (w.shape[2], w.shape[3]), axis=(2, 3))
    windows = windows.transpose(0, 2, 3, 1, 4, 5)
    return cp.einsum("nohcrs,kcrs->nkoh", windows, w, optimize=True) + b[None, :, None, None]


def timed(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); samples.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(samples))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/numpy_cupy_cuda_benchmark.csv")
    parser.add_argument("--cuda-csv", default="results/cuda_c_benchmark.csv")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--cuda-binary", default="results/cuda_conv_benchmark")
    args = parser.parse_args()
    import cupy as cp
    rows = []
    for ci, shape in enumerate(CASES):
        n, c, h, width, k, r, s = shape
        # Match the values hard-coded in cuda_conv_benchmark.cu so errors are comparable.
        x = np.full((n,c,h,width), 0.01, dtype=np.float32)
        w = np.full((k,c,r,s), 0.02, dtype=np.float32)
        b = np.full((k,), 0.03, dtype=np.float32)
        reference = conv_numpy(x.astype(np.float64), w.astype(np.float64), b.astype(np.float64))
        numpy_out = conv_numpy(x, w, b)
        numpy_ms = timed(lambda: conv_numpy(x,w,b), args.warmup, max(3, min(args.repeats, 20)))
        numpy_diff = np.abs(numpy_out-reference)
        rows.append({"backend":"numpy","case":ci,"n":n,"c":c,"h":h,"w":width,"k":k,"r":r,"s":s,"mean_ms":numpy_ms,"speedup_vs_numpy":1.0,"max_abs_error":float(numpy_diff.max()),"relative_error":float((numpy_diff/np.maximum(np.abs(reference),1e-6)).max())})
        gx, gw, gb = cp.asarray(x), cp.asarray(w), cp.asarray(b)
        cp.cuda.Stream.null.synchronize()
        for _ in range(args.warmup): conv_cupy(cp,gx,gw,gb); cp.cuda.Stream.null.synchronize()
        times=[]
        for _ in range(args.repeats):
            ev0, ev1 = cp.cuda.Event(), cp.cuda.Event(); ev0.record(); gy=conv_cupy(cp,gx,gw,gb); ev1.record(); ev1.synchronize(); times.append(cp.cuda.get_elapsed_time(ev0,ev1))
        cupy_ms=float(np.mean(times)); out=cp.asnumpy(gy); diff=np.abs(out-reference)
        rows.append({"backend":"cupy","case":ci,"n":n,"c":c,"h":h,"w":width,"k":k,"r":r,"s":s,"mean_ms":cupy_ms,"speedup_vs_numpy":numpy_ms/cupy_ms,"max_abs_error":float(diff.max()),"relative_error":float((diff/np.maximum(np.abs(reference),1e-6)).max())})
    cuda_path=Path(args.cuda_csv); cuda_path.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([args.cuda_binary,str(args.repeats)],check=True,stdout=cuda_path.open("w",encoding="utf-8"))
    cuda_rows=list(csv.DictReader(cuda_path.open(encoding="utf-8")))
    for raw in cuda_rows:
        ci=int(raw["case"]); np_row=next(r for r in rows if r["backend"]=="numpy" and r["case"]==ci)
        rows.append({**{k:raw[k] for k in ("backend","case","n","c","h","w","k","r","s")},"mean_ms":float(raw["mean_ms"]),"speedup_vs_numpy":float(np_row["mean_ms"])/float(raw["mean_ms"]),"max_abs_error":float(raw["max_abs_error"]),"relative_error":float(raw["relative_error"])})
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=["backend","case","n","c","h","w","k","r","s","mean_ms","speedup_vs_numpy","max_abs_error","relative_error"]); writer.writeheader(); writer.writerows(rows)
    print(f"saved {out} with {len(rows)} rows")


if __name__ == "__main__": main()
