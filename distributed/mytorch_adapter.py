"""Flatten/load helpers for connecting MyTorch ``Module`` objects to a worker.

The existing Module API already exposes ``parameters()`` and ``state_dict()``.
These helpers keep the distributed transport independent of individual model
layers while preserving the exact parameter ordering on every worker.
"""
from __future__ import annotations

import numpy as np


def parameters(model):
    return list(model.parameters())


def flatten_parameters(model) -> np.ndarray:
    arrays = [np.asarray(_to_cpu(param.data), dtype=np.float32).reshape(-1) for param in parameters(model)]
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float32)


def flatten_gradients(model) -> np.ndarray:
    arrays = []
    for param in parameters(model):
        if param.grad is None:
            arrays.append(np.zeros(param.data.size, dtype=np.float32))
        else:
            arrays.append(np.asarray(_to_cpu(param.grad), dtype=np.float32).reshape(-1))
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float32)


def load_parameters(model, flat: np.ndarray) -> None:
    offset = 0
    for param in parameters(model):
        size = int(param.data.size)
        value = flat[offset:offset + size].reshape(param.data.shape)
        if getattr(param, "device", lambda: "cpu")() == "cuda":
            import cupy as cp
            param.data[...] = cp.asarray(value)
        else:
            param.data[...] = value
        offset += size
    if offset != flat.size:
        raise ValueError(f"parameter payload has {flat.size} values, consumed {offset}")


def _to_cpu(array):
    return array.get() if hasattr(array, "get") else array
