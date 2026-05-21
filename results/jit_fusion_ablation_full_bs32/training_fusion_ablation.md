# MyTorch Training Fusion / JIT Ablation

Compares MyTorch eager training without fusion, dynamic fused training, and JIT fused training.

- Data root: `.`
- Train list: `splits\temporal_block_gap20\train.txt`
- Val list: `splits\temporal_block_gap20\val.txt`
- Epochs: 5
- Batch size: 8
- Max train batches per epoch: all
- Max eval batches: all
- Device: cuda

| variant | epoch time s | batch p50 ms | batch p90 ms | samples/s | peak GPU MB | CuPy pool MB | val MSE | val MAE | acc@0.10 | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MyTorch no fusion | 234.179 | 233.435 | 253.175 | 33.5814 | 3266.5 | 1559.7 | 0.0273891 | 0.0961858 | 0.6785 | 1 |
| Dynamic fused training | 220.595 | 220.39 | 237.517 | 35.6363 | 3300.5 | 1611.53 | 1.22774 | 1.08125 | 0.0085 | 1.06158 |
| JIT fused training | 222.594 | 223.175 | 239.437 | 35.3125 | 5842.5 | 4094.21 | 0.0223794 | 0.0558498 | 0.9335 | 1.05205 |

Notes:
- `batch p50/p90` are measured inside the true training loop with device synchronization.
- `val MSE/MAE/acc@0.10` are evaluated once after each variant finishes training.
- `JIT fused training` uses `jit.compile_train(..., experimental_conv_bn_fusion=True)`; backward still relies on eager autograd where the framework has no fused backward kernel.
