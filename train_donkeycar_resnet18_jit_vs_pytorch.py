import argparse
import csv
import json
import os
import pickle
import queue
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _argv_has(flag: str) -> bool:
    return flag in sys.argv


def _argv_value(flag: str):
    prefix = flag + "="
    for idx, arg in enumerate(sys.argv):
        if arg == flag:
            return sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


def _preconfigure_cupy_compiler():
    accelerators = _argv_value("--cupy-accelerators")
    if accelerators is None:
        # Avoid CuPy CUB reductions triggering NVCC on Windows during ordinary
        # reductions. Users can opt back into CuPy defaults with --cupy-accelerators auto.
        os.environ.setdefault("CUPY_ACCELERATORS", "")
    elif accelerators.lower() in ("auto", "default"):
        pass
    elif accelerators.lower() in ("none", "off", "disabled"):
        os.environ["CUPY_ACCELERATORS"] = ""
    else:
        os.environ["CUPY_ACCELERATORS"] = accelerators

    if _argv_has("--allow-unsupported-nvcc"):
        flag = "-allow-unsupported-compiler"
        existing = os.environ.get("NVCC_PREPEND_FLAGS", "")
        if flag not in existing.split():
            os.environ["NVCC_PREPEND_FLAGS"] = f"{flag} {existing}".strip()


_preconfigure_cupy_compiler()

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import psutil
except ImportError:
    psutil = None

try:
    import cupy as cp
except ImportError:
    cp = None

import mytorch.jit as mytorch_jit
from model.resnet import ResNet18Original
from mytorch.dataloader import Dataloader
from mytorch.loss import MSELoss
from mytorch.modules import Module
from mytorch.optim import Adam
from mytorch.tensor import Tensor


@dataclass
class ExperimentConfig:
    data_root: str
    train_list: str
    val_list: str
    results_dir: str
    backends: List[str]
    order: List[str]
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    output_dim: int
    image_height: int
    image_width: int
    seed: int
    device: str
    max_train_batches: Optional[int]
    max_val_batches: Optional[int]
    loss_log_interval: int
    resource_interval_sec: float
    torch_num_workers: int
    torch_prefetch_factor: int
    torch_persistent_workers: bool
    torch_cudnn_benchmark: bool
    mytorch_loader: str
    mytorch_num_workers: int
    mytorch_prefetch_factor: int
    model_parallel: str
    jit_profile: bool
    jit_dump_graph: bool
    jit_experimental_conv_bn_fusion: bool
    allow_unsupported_nvcc: bool
    cupy_accelerators: Optional[str]


VALID_BACKENDS = ("mytorch_jit", "pytorch")


def set_seed(seed: int):
    np.random.seed(seed)
    if cp is not None:
        try:
            cp.random.seed(seed)
        except Exception:
            pass
    torch = optional_torch()
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def optional_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def sync_cuda_for_backend(backend: str):
    if backend.startswith("mytorch"):
        if cp is not None:
            cp.cuda.Stream.null.synchronize()
    elif backend == "pytorch":
        torch = optional_torch()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()


def clear_backend_memory(backend: str):
    if cp is not None:
        try:
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
    torch = optional_torch()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def to_float(value: Any) -> float:
    if cp is not None and isinstance(value, cp.ndarray):
        value = value.get()
    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0])
    if isinstance(value, np.generic):
        return float(value)
    if hasattr(value, "data") and not isinstance(value, memoryview):
        inner = value.data
        if cp is not None and isinstance(inner, cp.ndarray):
            inner = inner.get()
        if isinstance(inner, np.ndarray):
            return float(inner.reshape(-1)[0])
        if isinstance(inner, np.generic):
            return float(inner)
        value = inner
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            pass
    return float(np.asarray(value).reshape(-1)[0])


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def stat_summary(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "mean": None,
            "std": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


class DonkeyCarNumpyDataset:
    """Shared DonkeyCar reader used by both mytorch and PyTorch adapters."""

    def __init__(
        self,
        data_root: str,
        mode: str,
        image_height: int,
        image_width: int,
        list_path: Optional[str] = None,
    ):
        if cv2 is None and Image is None:
            raise ImportError("opencv-python or Pillow is required for DonkeyCar image loading.")
        self.data_root = data_root
        self.mode = mode.lower()
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        if self.mode not in {"train", "val"}:
            raise ValueError("mode must be 'train' or 'val'")
        if list_path is None:
            list_path = "train.txt" if self.mode == "train" else "val.txt"
        if not os.path.isabs(list_path):
            candidate = os.path.join(data_root, list_path)
            if os.path.exists(candidate):
                list_path = candidate
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"Cannot find DonkeyCar list file: {list_path}")
        self.list_path = list_path
        self.items = []
        with open(list_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                self.items.append((parts[0], float(parts[1])))
        if not self.items:
            raise RuntimeError(f"No samples found in {list_path}")

    def __len__(self):
        return len(self.items)

    def _load_numpy(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        rel_path, steering = self.items[idx]
        image_path = os.path.join(self.data_root, rel_path)
        if cv2 is not None:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Cannot read image: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[0] != self.image_height or img.shape[1] != self.image_width:
                img = cv2.resize(img, (self.image_width, self.image_height), interpolation=cv2.INTER_AREA)
        else:
            with Image.open(image_path) as pil_img:
                pil_img = pil_img.convert("RGB")
                if pil_img.size != (self.image_width, self.image_height):
                    pil_img = pil_img.resize((self.image_width, self.image_height), Image.BILINEAR)
                img = np.asarray(pil_img)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        target = np.array([steering], dtype=np.float32)
        return img, target


class MyTorchDonkeyCarDataset(DonkeyCarNumpyDataset):
    def __getitem__(self, idx: int):
        img, target = self._load_numpy(idx)
        return Tensor(img), Tensor(target)


class MyTorchThreadedDataLoader:
    """
    Threaded producer-consumer loader for mytorch.

    The existing mytorch.dataloader.Dataloader is synchronous: the training loop
    waits while images are decoded and stacked. This loader makes batches the unit
    of work: worker threads consume index batches, load samples, collate tensors,
    and push completed batches into a bounded queue.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        prefetch_factor: int,
        seed: int,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.num_workers = max(1, int(num_workers))
        self.prefetch_factor = max(1, int(prefetch_factor))
        self.seed = int(seed)
        self._epoch = 0
        self._workers = []
        self._index_queue = None
        self._output_queue = None
        self._stop_event = threading.Event()
        self._num_batches = 0
        self._yielded = 0
        self._next_batch_id = 0
        self._pending_outputs = {}
        self._stats = {
            "loader": "mytorch_threaded_producer_consumer",
            "num_workers": self.num_workers,
            "prefetch_factor": self.prefetch_factor,
            "produced_batches": 0,
            "consumed_batches": 0,
        }

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def _collate(self, samples):
        xs, ys = zip(*samples)
        x = Tensor(np.stack([item.data for item in xs], axis=0))
        y = Tensor(np.stack([item.data for item in ys], axis=0))
        return x, y

    def _worker(self):
        while not self._stop_event.is_set():
            try:
                item = self._index_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                return
            batch_id, indices = item
            try:
                samples = [self.dataset[int(idx)] for idx in indices]
                batch = self._collate(samples)
                while not self._stop_event.is_set():
                    try:
                        self._output_queue.put((batch_id, batch, None), timeout=0.1)
                        break
                    except queue.Full:
                        continue
                self._stats["produced_batches"] += 1
            except BaseException as exc:
                while not self._stop_event.is_set():
                    try:
                        self._output_queue.put((batch_id, None, exc), timeout=0.1)
                        break
                    except queue.Full:
                        continue

    def __iter__(self):
        self._shutdown()
        self._epoch += 1
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self._epoch)
            rng.shuffle(indices)
        batches = [
            indices[start:start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
        ]
        self._num_batches = len(batches)
        self._yielded = 0
        self._next_batch_id = 0
        self._pending_outputs = {}
        self._stats["produced_batches"] = 0
        self._stats["consumed_batches"] = 0
        self._index_queue = queue.Queue()
        self._output_queue = queue.Queue(maxsize=self.num_workers * self.prefetch_factor)
        self._stop_event.clear()
        self._workers = [
            threading.Thread(target=self._worker, daemon=True)
            for _ in range(self.num_workers)
        ]
        for worker in self._workers:
            worker.start()
        for batch_id, batch_indices in enumerate(batches):
            self._index_queue.put((batch_id, batch_indices))
        for _ in self._workers:
            self._index_queue.put(None)
        return self

    def __next__(self):
        if self._yielded >= self._num_batches:
            self._shutdown()
            raise StopIteration
        while self._next_batch_id not in self._pending_outputs:
            batch_id, batch, error = self._output_queue.get()
            if error is not None:
                self._shutdown()
                raise error
            self._pending_outputs[batch_id] = batch
        batch = self._pending_outputs.pop(self._next_batch_id)
        self._next_batch_id += 1
        self._yielded += 1
        self._stats["consumed_batches"] += 1
        return batch

    def _shutdown(self):
        self._stop_event.set()
        if self._workers:
            for worker in self._workers:
                worker.join(timeout=1.0)
        self._workers = []

    def get_stats(self):
        out = dict(self._stats)
        if self._output_queue is not None:
            out["queue_size"] = self._output_queue.qsize()
        return out


class TorchDonkeyCarDataset(DonkeyCarNumpyDataset):
    def __getitem__(self, idx: int):
        torch = optional_torch()
        if torch is None:
            raise ImportError("PyTorch is not installed.")
        img, target = self._load_numpy(idx)
        return torch.from_numpy(img), torch.from_numpy(target)


class ResourceMonitor:
    def __init__(self, backend: str, device: str, interval_sec: float = 0.25):
        self.backend = backend
        self.device = device
        self.interval_sec = float(interval_sec)
        self.samples = []
        self._running = False
        self._thread = None
        self._process = psutil.Process(os.getpid()) if psutil is not None else None

    def start(self):
        self.samples = []
        self._running = True
        if self._process is not None:
            try:
                self._process.cpu_percent(interval=None)
                psutil.cpu_percent(interval=None)
            except Exception:
                pass
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self.summary()

    def _loop(self):
        while self._running:
            self.samples.append(resource_snapshot(self.backend, self.device))
            time.sleep(self.interval_sec)

    @staticmethod
    def _avg(values):
        vals = [v for v in values if v is not None]
        return float(np.mean(vals)) if vals else None

    @staticmethod
    def _max(values):
        vals = [v for v in values if v is not None]
        return float(np.max(vals)) if vals else None

    def summary(self) -> Dict[str, Any]:
        if not self.samples:
            return {"sample_count": 0}
        keys = sorted({k for sample in self.samples for k in sample.keys() if k != "timestamp"})
        out = {"sample_count": len(self.samples)}
        for key in keys:
            values = [sample.get(key) for sample in self.samples]
            out[f"avg_{key}"] = self._avg(values)
            out[f"max_{key}"] = self._max(values)
        return out


def resource_snapshot(backend: str, device: str) -> Dict[str, Optional[float]]:
    sample: Dict[str, Optional[float]] = {
        "timestamp": time.time(),
        "process_cpu_percent": None,
        "system_cpu_percent": None,
        "process_rss_mb": None,
        "system_memory_percent": None,
        "gpu_used_mb": None,
        "gpu_total_mb": None,
        "cupy_pool_used_mb": None,
        "cupy_pool_total_mb": None,
        "torch_allocated_mb": None,
        "torch_reserved_mb": None,
    }
    if psutil is not None:
        try:
            process = psutil.Process(os.getpid())
            sample["process_cpu_percent"] = float(process.cpu_percent(interval=None))
            sample["system_cpu_percent"] = float(psutil.cpu_percent(interval=None))
            sample["process_rss_mb"] = process.memory_info().rss / (1024 ** 2)
            sample["system_memory_percent"] = float(psutil.virtual_memory().percent)
        except Exception:
            pass

    if device == "cuda" and cp is not None:
        try:
            free_b, total_b = cp.cuda.runtime.memGetInfo()
            sample["gpu_used_mb"] = (total_b - free_b) / (1024 ** 2)
            sample["gpu_total_mb"] = total_b / (1024 ** 2)
            pool = cp.get_default_memory_pool()
            sample["cupy_pool_used_mb"] = pool.used_bytes() / (1024 ** 2)
            sample["cupy_pool_total_mb"] = pool.total_bytes() / (1024 ** 2)
        except Exception:
            pass

    torch = optional_torch()
    if device == "cuda" and torch is not None and torch.cuda.is_available():
        try:
            sample["torch_allocated_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
            sample["torch_reserved_mb"] = torch.cuda.memory_reserved() / (1024 ** 2)
            for device_idx in range(torch.cuda.device_count()):
                sample[f"torch_allocated_mb_device_{device_idx}"] = (
                    torch.cuda.memory_allocated(device_idx) / (1024 ** 2)
                )
                sample[f"torch_reserved_mb_device_{device_idx}"] = (
                    torch.cuda.memory_reserved(device_idx) / (1024 ** 2)
                )
            if sample["gpu_used_mb"] is None:
                free_b, total_b = torch.cuda.mem_get_info()
                sample["gpu_used_mb"] = (total_b - free_b) / (1024 ** 2)
                sample["gpu_total_mb"] = total_b / (1024 ** 2)
        except Exception:
            pass
    return sample


def iter_named_tensors(module: Module):
    items = []
    seen = set()

    def visit(obj, prefix):
        obj_id = id(obj)
        if obj_id in seen:
            return
        if isinstance(obj, Tensor):
            seen.add(obj_id)
            items.append((prefix, obj))
            return
        if isinstance(obj, Module):
            seen.add(obj_id)
            for name, value in obj.__dict__.items():
                if name.startswith("_"):
                    continue
                visit(value, f"{prefix}.{name}" if prefix else name)
        elif isinstance(obj, (list, tuple)):
            for idx, value in enumerate(obj):
                visit(value, f"{prefix}.{idx}" if prefix else str(idx))
        elif isinstance(obj, dict):
            for key, value in obj.items():
                visit(value, f"{prefix}.{key}" if prefix else str(key))

    visit(module, "")
    return items


def save_mytorch_state(model: Module, path: str):
    state = {}
    for name, tensor in iter_named_tensors(model):
        data = tensor.data
        if hasattr(data, "get"):
            data = data.get()
        state[name] = np.asarray(data)
    payload = {"format": "mytorch_full_state_v1", "state": state}
    with open(path, "wb") as f:
        pickle.dump(payload, f)


class TorchBasicBlock:
    pass


def build_torch_resnet18(output_dim: int, model_parallel_devices=None):
    torch = optional_torch()
    if torch is None:
        raise ImportError("PyTorch is not installed.")
    import torch.nn as nn

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride,
                padding=1, bias=True
            )
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu1 = nn.ReLU(inplace=False)
            self.conv2 = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1,
                padding=1, bias=True
            )
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu2 = nn.ReLU(inplace=False)
            self.downsample = None
            if stride != 1 or in_channels != out_channels:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True),
                    nn.BatchNorm2d(out_channels),
                )

        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu1(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out = out + identity
            out = self.relu2(out)
            return out

    class TorchResNet18(nn.Module):
        def __init__(self, num_classes, parallel_devices=None):
            super().__init__()
            self.parallel_devices = parallel_devices
            self.in_channels = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
            self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
            self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
            self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(512, num_classes)

            if self.parallel_devices is not None:
                dev0, dev1 = self.parallel_devices
                self.conv1.to(dev0)
                self.bn1.to(dev0)
                self.relu.to(dev0)
                self.maxpool.to(dev0)
                self.layer1.to(dev0)
                self.layer2.to(dev0)
                self.layer3.to(dev1)
                self.layer4.to(dev1)
                self.avgpool.to(dev1)
                self.flatten.to(dev1)
                self.fc.to(dev1)

        def _make_layer(self, block, out_channels, blocks, stride):
            layers = [block(self.in_channels, out_channels, stride)]
            self.in_channels = out_channels * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.in_channels, out_channels, 1))
            return nn.Sequential(*layers)

        def forward(self, x):
            if self.parallel_devices is not None:
                dev0, dev1 = self.parallel_devices
                x = x.to(dev0, non_blocking=True)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            if self.parallel_devices is not None:
                x = x.to(dev1, non_blocking=True)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = self.flatten(x)
            return self.fc(x)

    return TorchResNet18(output_dim, parallel_devices=model_parallel_devices)


def conv2d_out(hw: Tuple[int, int], kernel: int, stride: int, padding: int, dilation: int = 1) -> Tuple[int, int]:
    h, w = hw
    kh = kw = kernel
    oh = (h + 2 * padding - dilation * (kh - 1) - 1) // stride + 1
    ow = (w + 2 * padding - dilation * (kw - 1) - 1) // stride + 1
    return int(oh), int(ow)


def pool2d_out(hw: Tuple[int, int], kernel: int, stride: int, padding: int) -> Tuple[int, int]:
    return conv2d_out(hw, kernel=kernel, stride=stride, padding=padding)


def estimate_resnet18_flops(image_height: int, image_width: int, output_dim: int) -> Dict[str, float]:
    """Analytic forward MAC/FLOP estimate per sample for this repo's ResNet18."""
    hw = (int(image_height), int(image_width))
    macs = 0
    elem_ops = 0

    def add_conv(in_c, out_c, kernel, stride=1, padding=0, groups=1):
        nonlocal hw, macs, elem_ops
        hw = conv2d_out(hw, kernel, stride, padding)
        oh, ow = hw
        macs += oh * ow * out_c * (in_c // groups) * kernel * kernel
        elem_ops += oh * ow * out_c  # bias add
        return hw

    def add_bn_relu(channels, relu=True):
        nonlocal elem_ops
        oh, ow = hw
        elem_ops += oh * ow * channels * 4
        if relu:
            elem_ops += oh * ow * channels

    def add_add_relu(channels):
        nonlocal elem_ops
        oh, ow = hw
        elem_ops += oh * ow * channels * 2

    add_conv(3, 64, 7, stride=2, padding=3)
    add_bn_relu(64, relu=True)
    hw = pool2d_out(hw, kernel=3, stride=2, padding=1)

    in_c = 64
    for stage_idx, out_c in enumerate([64, 128, 256, 512]):
        for block_idx in range(2):
            stride = 2 if stage_idx > 0 and block_idx == 0 else 1
            identity_hw = hw
            add_conv(in_c, out_c, 3, stride=stride, padding=1)
            add_bn_relu(out_c, relu=True)
            add_conv(out_c, out_c, 3, stride=1, padding=1)
            add_bn_relu(out_c, relu=False)
            if stride != 1 or in_c != out_c:
                hw_before_downsample = identity_hw
                oh, ow = conv2d_out(hw_before_downsample, 1, stride, 0)
                macs += oh * ow * out_c * in_c
                elem_ops += oh * ow * out_c
                # Downsample BN.
                elem_ops += oh * ow * out_c * 4
            add_add_relu(out_c)
            in_c = out_c

    macs += 512 * int(output_dim)
    elem_ops += int(output_dim)
    forward_flops = 2 * macs + elem_ops
    # Common training estimate: forward + activation gradients + weight gradients.
    train_flops = 3 * forward_flops
    return {
        "forward_macs_per_sample": float(macs),
        "forward_flops_per_sample": float(forward_flops),
        "train_flops_per_sample_est": float(train_flops),
    }


def metric_from_arrays(pred, target) -> Tuple[float, float, int]:
    diff = pred - target
    if cp is not None and isinstance(diff, cp.ndarray):
        sse = float(cp.sum(diff * diff).get())
        sae = float(cp.sum(cp.abs(diff)).get())
        elems = int(diff.size)
        return sse, sae, elems
    arr = np.asarray(diff)
    return float(np.sum(arr * arr)), float(np.sum(np.abs(arr))), int(arr.size)


def summarize_metric_sums(sse: float, sae: float, elems: int) -> Dict[str, Optional[float]]:
    if elems <= 0:
        return {"mse": None, "rmse": None, "mae": None}
    mse = sse / elems
    return {"mse": float(mse), "rmse": float(np.sqrt(mse)), "mae": float(sae / elems)}


def make_mytorch_loaders(config: ExperimentConfig):
    train_ds = MyTorchDonkeyCarDataset(
        config.data_root, "train", config.image_height, config.image_width, config.train_list
    )
    val_ds = MyTorchDonkeyCarDataset(
        config.data_root, "val", config.image_height, config.image_width, config.val_list
    )
    if config.mytorch_loader == "async":
        train_loader = MyTorchThreadedDataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.mytorch_num_workers,
            prefetch_factor=config.mytorch_prefetch_factor,
            seed=config.seed,
        )
        val_loader = MyTorchThreadedDataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.mytorch_num_workers,
            prefetch_factor=config.mytorch_prefetch_factor,
            seed=config.seed + 9973,
        )
    else:
        train_loader = Dataloader(train_ds, batch_size=config.batch_size, shuffle=True)
        val_loader = Dataloader(val_ds, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader, len(train_ds), len(val_ds)


def make_torch_loaders(config: ExperimentConfig):
    torch = optional_torch()
    if torch is None:
        raise ImportError("PyTorch is not installed.")
    train_ds = TorchDonkeyCarDataset(
        config.data_root, "train", config.image_height, config.image_width, config.train_list
    )
    val_ds = TorchDonkeyCarDataset(
        config.data_root, "val", config.image_height, config.image_width, config.val_list
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader_kwargs = {}
    if config.torch_num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.torch_prefetch_factor
        loader_kwargs["persistent_workers"] = bool(config.torch_persistent_workers)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.torch_num_workers,
        pin_memory=(config.device == "cuda"),
        generator=generator,
        **loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.torch_num_workers,
        pin_memory=(config.device == "cuda"),
        **loader_kwargs,
    )
    return train_loader, val_loader, len(train_ds), len(val_ds)


def should_stop_batch(batch_idx: int, max_batches: Optional[int]) -> bool:
    return max_batches is not None and batch_idx > max_batches


def run_mytorch_jit(config: ExperimentConfig, output_dir: str, flops: Dict[str, float]) -> Dict[str, Any]:
    backend = "mytorch_jit"
    clear_backend_memory(backend)
    set_seed(config.seed)
    train_loader, val_loader, train_size, val_size = make_mytorch_loaders(config)

    model = ResNet18Original(num_classes=config.output_dim).train()
    if config.device == "cuda":
        model.cuda()
    runner = mytorch_jit.compile_train(
        model,
        dump_graph=config.jit_dump_graph,
        profile=config.jit_profile,
        experimental_conv_bn_fusion=config.jit_experimental_conv_bn_fusion,
    )
    optimizer = Adam(runner.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = MSELoss(fused=False)

    monitor = ResourceMonitor(backend, config.device, config.resource_interval_sec)
    monitor.start()
    sync_cuda_for_backend(backend)
    wall_t0 = time.perf_counter()

    history = []
    batch_records = []
    total_train_samples = 0
    first_batch_total_ms = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_sse = 0.0
        epoch_sae = 0.0
        epoch_elems = 0
        epoch_samples = 0
        epoch_t0 = time.perf_counter()

        train_iter = iter(train_loader)
        batch_idx = 0
        while True:
            if config.max_train_batches is not None and batch_idx >= config.max_train_batches:
                break
            loader_t0 = time.perf_counter()
            try:
                x, y = next(train_iter)
            except StopIteration:
                break
            loader_fetch_ms = (time.perf_counter() - loader_t0) * 1000.0
            batch_idx += 1
            sync_cuda_for_backend(backend)
            batch_t0 = time.perf_counter()

            transfer_t0 = time.perf_counter()
            if config.device == "cuda":
                x = x.cuda()
                y = y.cuda()
            sync_cuda_for_backend(backend)
            transfer_ms = (time.perf_counter() - transfer_t0) * 1000.0

            zero_t0 = time.perf_counter()
            optimizer.zero_grad()
            zero_ms = (time.perf_counter() - zero_t0) * 1000.0

            forward_t0 = time.perf_counter()
            pred = runner(x)
            loss = criterion(pred, y)
            sync_cuda_for_backend(backend)
            forward_ms = (time.perf_counter() - forward_t0) * 1000.0

            backward_t0 = time.perf_counter()
            loss.backward()
            sync_cuda_for_backend(backend)
            backward_ms = (time.perf_counter() - backward_t0) * 1000.0

            step_t0 = time.perf_counter()
            optimizer.step()
            sync_cuda_for_backend(backend)
            step_ms = (time.perf_counter() - step_t0) * 1000.0
            total_ms = (time.perf_counter() - batch_t0) * 1000.0
            if first_batch_total_ms is None:
                first_batch_total_ms = total_ms

            bs = int(x.shape()[0])
            total_train_samples += bs
            epoch_samples += bs
            if config.loss_log_interval > 0 and (batch_idx == 1 or batch_idx % config.loss_log_interval == 0):
                sse, sae, elems = metric_from_arrays(pred.data, y.data)
                epoch_sse += sse
                epoch_sae += sae
                epoch_elems += elems

            batch_records.append({
                "backend": backend,
                "epoch": epoch,
                "batch": batch_idx,
                "batch_size": bs,
                "loader_fetch_ms": loader_fetch_ms,
                "transfer_ms": transfer_ms,
                "zero_grad_ms": zero_ms,
                "forward_loss_ms": forward_ms,
                "backward_ms": backward_ms,
                "optimizer_step_ms": step_ms,
                "total_batch_ms": total_ms,
                "loss": to_float(loss) if config.loss_log_interval > 0 and batch_idx % config.loss_log_interval == 0 else None,
            })

        if hasattr(train_loader, "_shutdown"):
            train_loader._shutdown()
        train_time_sec = time.perf_counter() - epoch_t0
        val_metrics = validate_mytorch(model, val_loader, criterion, config)
        train_metrics = summarize_metric_sums(epoch_sse, epoch_sae, epoch_elems)
        train_metrics.update({
            "samples": epoch_samples,
            "time_sec": train_time_sec,
            "samples_per_sec": epoch_samples / train_time_sec if train_time_sec > 0 else None,
        })
        history.append({
            "backend": backend,
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        })
        print(
            f"[mytorch_jit] epoch {epoch}/{config.epochs} "
            f"train_time={train_time_sec:.3f}s val_mse={val_metrics['mse']:.6f}"
        )

    sync_cuda_for_backend(backend)
    wall_time_sec = time.perf_counter() - wall_t0
    resources = monitor.stop()
    loader_stats = train_loader.get_stats() if hasattr(train_loader, "get_stats") else None
    save_mytorch_state(model, os.path.join(output_dir, "mytorch_jit_state.pkl"))

    return finalize_backend_result(
        backend=backend,
        train_size=train_size,
        val_size=val_size,
        history=history,
        batch_records=batch_records,
        resources=resources,
        wall_time_sec=wall_time_sec,
        total_train_samples=total_train_samples,
        flops=flops,
        extra={
            "jit_cache_size": len(getattr(runner, "cache", {})),
            "first_batch_total_ms": first_batch_total_ms,
            "data_loader": config.mytorch_loader,
            "data_loader_stats": loader_stats,
            "model_parallel": "not_supported_by_mytorch_current_tensor_device_api",
            "jit_strategy": (
                "experimental_conv_bn_fusion"
                if config.jit_experimental_conv_bn_fusion else
                "default_bn_only_training_jit"
            ),
        },
    )


def validate_mytorch(model, val_loader, criterion, config: ExperimentConfig) -> Dict[str, Any]:
    model.eval()
    backend = "mytorch_jit"
    sse = 0.0
    sae = 0.0
    elems = 0
    samples = 0
    batches = 0
    sync_cuda_for_backend(backend)
    t0 = time.perf_counter()
    val_iter = iter(val_loader)
    batch_idx = 0
    while True:
        if config.max_val_batches is not None and batch_idx >= config.max_val_batches:
            break
        try:
            x, y = next(val_iter)
        except StopIteration:
            break
        batch_idx += 1
        if config.device == "cuda":
            x = x.cuda()
            y = y.cuda()
        pred = model(x)
        _ = criterion(pred, y)
        batch_sse, batch_sae, batch_elems = metric_from_arrays(pred.data, y.data)
        sse += batch_sse
        sae += batch_sae
        elems += batch_elems
        samples += int(x.shape()[0])
        batches += 1
    if hasattr(val_loader, "_shutdown"):
        val_loader._shutdown()
    sync_cuda_for_backend(backend)
    elapsed = time.perf_counter() - t0
    metrics = summarize_metric_sums(sse, sae, elems)
    metrics.update({
        "samples": samples,
        "batches": batches,
        "time_sec": elapsed,
        "samples_per_sec": samples / elapsed if elapsed > 0 else None,
    })
    return metrics


def run_pytorch(config: ExperimentConfig, output_dir: str, flops: Dict[str, float]) -> Dict[str, Any]:
    torch = optional_torch()
    if torch is None:
        raise ImportError("PyTorch is not installed. Install torch or run --backend mytorch_jit.")
    import torch.nn as nn

    backend = "pytorch"
    clear_backend_memory(backend)
    set_seed(config.seed)
    torch.backends.cudnn.benchmark = bool(config.torch_cudnn_benchmark)
    train_loader, val_loader, train_size, val_size = make_torch_loaders(config)
    device = torch.device("cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu")
    parallel_devices = None
    if config.model_parallel == "pytorch":
        if device.type != "cuda" or torch.cuda.device_count() < 2:
            raise RuntimeError(
                "--model-parallel pytorch requires at least two CUDA devices. "
                "Use --model-parallel none for single-device runs."
            )
        parallel_devices = (torch.device("cuda:0"), torch.device("cuda:1"))
        input_device = parallel_devices[0]
        target_device = parallel_devices[1]
        model = build_torch_resnet18(config.output_dim, model_parallel_devices=parallel_devices)
    else:
        input_device = device
        target_device = device
        model = build_torch_resnet18(config.output_dim).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    monitor = ResourceMonitor(backend, config.device, config.resource_interval_sec)
    monitor.start()
    sync_cuda_for_backend(backend)
    wall_t0 = time.perf_counter()

    history = []
    batch_records = []
    total_train_samples = 0
    first_batch_total_ms = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_sse = 0.0
        epoch_sae = 0.0
        epoch_elems = 0
        epoch_samples = 0
        epoch_t0 = time.perf_counter()

        train_iter = iter(train_loader)
        batch_idx = 0
        while True:
            if config.max_train_batches is not None and batch_idx >= config.max_train_batches:
                break
            loader_t0 = time.perf_counter()
            try:
                x, y = next(train_iter)
            except StopIteration:
                break
            loader_fetch_ms = (time.perf_counter() - loader_t0) * 1000.0
            batch_idx += 1
            sync_cuda_for_backend(backend)
            batch_t0 = time.perf_counter()

            transfer_t0 = time.perf_counter()
            x = x.to(input_device, non_blocking=True)
            y = y.to(target_device, non_blocking=True)
            sync_cuda_for_backend(backend)
            transfer_ms = (time.perf_counter() - transfer_t0) * 1000.0

            zero_t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            zero_ms = (time.perf_counter() - zero_t0) * 1000.0

            forward_t0 = time.perf_counter()
            pred = model(x)
            loss = criterion(pred, y)
            sync_cuda_for_backend(backend)
            forward_ms = (time.perf_counter() - forward_t0) * 1000.0

            backward_t0 = time.perf_counter()
            loss.backward()
            sync_cuda_for_backend(backend)
            backward_ms = (time.perf_counter() - backward_t0) * 1000.0

            step_t0 = time.perf_counter()
            optimizer.step()
            sync_cuda_for_backend(backend)
            step_ms = (time.perf_counter() - step_t0) * 1000.0
            total_ms = (time.perf_counter() - batch_t0) * 1000.0
            if first_batch_total_ms is None:
                first_batch_total_ms = total_ms

            bs = int(x.shape[0])
            total_train_samples += bs
            epoch_samples += bs
            if config.loss_log_interval > 0 and (batch_idx == 1 or batch_idx % config.loss_log_interval == 0):
                diff = pred.detach() - y.detach()
                epoch_sse += float(torch.sum(diff * diff).detach().cpu().item())
                epoch_sae += float(torch.sum(torch.abs(diff)).detach().cpu().item())
                epoch_elems += int(diff.numel())

            batch_records.append({
                "backend": backend,
                "epoch": epoch,
                "batch": batch_idx,
                "batch_size": bs,
                "loader_fetch_ms": loader_fetch_ms,
                "transfer_ms": transfer_ms,
                "zero_grad_ms": zero_ms,
                "forward_loss_ms": forward_ms,
                "backward_ms": backward_ms,
                "optimizer_step_ms": step_ms,
                "total_batch_ms": total_ms,
                "loss": float(loss.detach().cpu().item()) if config.loss_log_interval > 0 and batch_idx % config.loss_log_interval == 0 else None,
            })

        train_time_sec = time.perf_counter() - epoch_t0
        val_metrics = validate_pytorch(model, val_loader, criterion, config, input_device, target_device)
        train_metrics = summarize_metric_sums(epoch_sse, epoch_sae, epoch_elems)
        train_metrics.update({
            "samples": epoch_samples,
            "time_sec": train_time_sec,
            "samples_per_sec": epoch_samples / train_time_sec if train_time_sec > 0 else None,
        })
        history.append({
            "backend": backend,
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        })
        print(
            f"[pytorch] epoch {epoch}/{config.epochs} "
            f"train_time={train_time_sec:.3f}s val_mse={val_metrics['mse']:.6f}"
        )

    sync_cuda_for_backend(backend)
    wall_time_sec = time.perf_counter() - wall_t0
    resources = monitor.stop()
    if device.type == "cuda":
        resources["torch_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        resources["torch_peak_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024 ** 2)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
        },
        os.path.join(output_dir, "pytorch_state.pt"),
    )

    return finalize_backend_result(
        backend=backend,
        train_size=train_size,
        val_size=val_size,
        history=history,
        batch_records=batch_records,
        resources=resources,
        wall_time_sec=wall_time_sec,
        total_train_samples=total_train_samples,
        flops=flops,
        extra={
            "first_batch_total_ms": first_batch_total_ms,
            "torch_device": str(device),
            "torch_input_device": str(input_device),
            "torch_target_device": str(target_device),
            "model_parallel": (
                f"pytorch:{parallel_devices[0]}->{parallel_devices[1]}"
                if parallel_devices is not None else "none"
            ),
            "data_loader": {
                "num_workers": config.torch_num_workers,
                "prefetch_factor": config.torch_prefetch_factor if config.torch_num_workers > 0 else None,
                "persistent_workers": bool(config.torch_persistent_workers) if config.torch_num_workers > 0 else False,
            },
            "torch_cudnn_benchmark": bool(config.torch_cudnn_benchmark),
        },
    )


def validate_pytorch(model, val_loader, criterion, config: ExperimentConfig, input_device, target_device) -> Dict[str, Any]:
    torch = optional_torch()
    model.eval()
    sse = 0.0
    sae = 0.0
    elems = 0
    samples = 0
    batches = 0
    sync_cuda_for_backend("pytorch")
    t0 = time.perf_counter()
    with torch.no_grad():
        val_iter = iter(val_loader)
        batch_idx = 0
        while True:
            if config.max_val_batches is not None and batch_idx >= config.max_val_batches:
                break
            try:
                x, y = next(val_iter)
            except StopIteration:
                break
            batch_idx += 1
            x = x.to(input_device, non_blocking=True)
            y = y.to(target_device, non_blocking=True)
            pred = model(x)
            _ = criterion(pred, y)
            diff = pred - y
            sse += float(torch.sum(diff * diff).detach().cpu().item())
            sae += float(torch.sum(torch.abs(diff)).detach().cpu().item())
            elems += int(diff.numel())
            samples += int(x.shape[0])
            batches += 1
    sync_cuda_for_backend("pytorch")
    elapsed = time.perf_counter() - t0
    metrics = summarize_metric_sums(sse, sae, elems)
    metrics.update({
        "samples": samples,
        "batches": batches,
        "time_sec": elapsed,
        "samples_per_sec": samples / elapsed if elapsed > 0 else None,
    })
    return metrics


def finalize_backend_result(
    backend: str,
    train_size: int,
    val_size: int,
    history: List[Dict[str, Any]],
    batch_records: List[Dict[str, Any]],
    resources: Dict[str, Any],
    wall_time_sec: float,
    total_train_samples: int,
    flops: Dict[str, float],
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    total_train_time_sec = sum(float(item["train"].get("time_sec") or 0.0) for item in history)
    total_val_time_sec = sum(float(item["val"].get("time_sec") or 0.0) for item in history)
    total_train_flops = flops["train_flops_per_sample_est"] * total_train_samples
    achieved_tflops = (
        total_train_flops / total_train_time_sec / 1e12
        if total_train_time_sec > 0 else None
    )
    phase_keys = [
        "loader_fetch_ms",
        "transfer_ms",
        "zero_grad_ms",
        "forward_loss_ms",
        "backward_ms",
        "optimizer_step_ms",
        "total_batch_ms",
    ]
    batch_summary = {
        key: stat_summary([float(record[key]) for record in batch_records if record.get(key) is not None])
        for key in phase_keys
    }
    final_val = history[-1]["val"] if history else {}
    result = {
        "backend": backend,
        "train_size": train_size,
        "val_size": val_size,
        "history": history,
        "batch_records": batch_records,
        "batch_summary": batch_summary,
        "resources": resources,
        "wall_time_sec": wall_time_sec,
        "total_train_time_sec": total_train_time_sec,
        "total_val_time_sec": total_val_time_sec,
        "total_train_samples": total_train_samples,
        "samples_per_sec": total_train_samples / total_train_time_sec if total_train_time_sec > 0 else None,
        "end_to_end_samples_per_sec": total_train_samples / wall_time_sec if wall_time_sec > 0 else None,
        "estimated_total_train_flops": total_train_flops,
        "estimated_achieved_tflops": achieved_tflops,
        "flops_model": flops,
        "final_val": final_val,
    }
    result.update(extra)
    return result


def make_comparison(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if "mytorch_jit" not in results or "pytorch" not in results:
        return {}
    jit = results["mytorch_jit"]
    torch_res = results["pytorch"]

    def ratio(a, b):
        if a is None or b in (None, 0):
            return None
        return float(a) / float(b)

    comparison = {
        "train_time_sec": {
            "mytorch_jit": jit.get("total_train_time_sec"),
            "pytorch": torch_res.get("total_train_time_sec"),
            "jit_div_pytorch": ratio(jit.get("total_train_time_sec"), torch_res.get("total_train_time_sec")),
        },
        "wall_time_sec": {
            "mytorch_jit": jit.get("wall_time_sec"),
            "pytorch": torch_res.get("wall_time_sec"),
            "jit_div_pytorch": ratio(jit.get("wall_time_sec"), torch_res.get("wall_time_sec")),
        },
        "samples_per_sec": {
            "mytorch_jit": jit.get("samples_per_sec"),
            "pytorch": torch_res.get("samples_per_sec"),
            "jit_div_pytorch": ratio(jit.get("samples_per_sec"), torch_res.get("samples_per_sec")),
        },
        "achieved_tflops": {
            "mytorch_jit": jit.get("estimated_achieved_tflops"),
            "pytorch": torch_res.get("estimated_achieved_tflops"),
            "jit_div_pytorch": ratio(jit.get("estimated_achieved_tflops"), torch_res.get("estimated_achieved_tflops")),
        },
        "total_batch_ms_p50": {
            "mytorch_jit": jit["batch_summary"]["total_batch_ms"]["p50"],
            "pytorch": torch_res["batch_summary"]["total_batch_ms"]["p50"],
            "jit_div_pytorch": ratio(
                jit["batch_summary"]["total_batch_ms"]["p50"],
                torch_res["batch_summary"]["total_batch_ms"]["p50"],
            ),
        },
        "loader_fetch_ms_p50": {
            "mytorch_jit": jit["batch_summary"]["loader_fetch_ms"]["p50"],
            "pytorch": torch_res["batch_summary"]["loader_fetch_ms"]["p50"],
            "jit_div_pytorch": ratio(
                jit["batch_summary"]["loader_fetch_ms"]["p50"],
                torch_res["batch_summary"]["loader_fetch_ms"]["p50"],
            ),
        },
        "forward_loss_ms_p50": {
            "mytorch_jit": jit["batch_summary"]["forward_loss_ms"]["p50"],
            "pytorch": torch_res["batch_summary"]["forward_loss_ms"]["p50"],
            "jit_div_pytorch": ratio(
                jit["batch_summary"]["forward_loss_ms"]["p50"],
                torch_res["batch_summary"]["forward_loss_ms"]["p50"],
            ),
        },
        "backward_ms_p50": {
            "mytorch_jit": jit["batch_summary"]["backward_ms"]["p50"],
            "pytorch": torch_res["batch_summary"]["backward_ms"]["p50"],
            "jit_div_pytorch": ratio(
                jit["batch_summary"]["backward_ms"]["p50"],
                torch_res["batch_summary"]["backward_ms"]["p50"],
            ),
        },
        "peak_gpu_used_mb": {
            "mytorch_jit": jit.get("resources", {}).get("max_gpu_used_mb"),
            "pytorch": torch_res.get("resources", {}).get("max_gpu_used_mb"),
            "jit_minus_pytorch": (
                None if jit.get("resources", {}).get("max_gpu_used_mb") is None
                or torch_res.get("resources", {}).get("max_gpu_used_mb") is None
                else jit["resources"]["max_gpu_used_mb"] - torch_res["resources"]["max_gpu_used_mb"]
            ),
        },
        "final_val_mse": {
            "mytorch_jit": jit.get("final_val", {}).get("mse"),
            "pytorch": torch_res.get("final_val", {}).get("mse"),
            "jit_minus_pytorch": (
                None if jit.get("final_val", {}).get("mse") is None
                or torch_res.get("final_val", {}).get("mse") is None
                else jit["final_val"]["mse"] - torch_res["final_val"]["mse"]
            ),
        },
    }
    return comparison


def write_json(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=json_default)


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def write_batch_csv(path: str, results: Dict[str, Dict[str, Any]]):
    rows = []
    for backend, result in results.items():
        rows.extend(result.get("batch_records", []))
    if not rows:
        return
    keys = [
        "backend",
        "epoch",
        "batch",
        "batch_size",
        "loader_fetch_ms",
        "transfer_ms",
        "zero_grad_ms",
        "forward_loss_ms",
        "backward_ms",
        "optimizer_step_ms",
        "total_batch_ms",
        "loss",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def write_epoch_csv(path: str, results: Dict[str, Dict[str, Any]]):
    rows = []
    for backend, result in results.items():
        for record in result.get("history", []):
            rows.append({
                "backend": backend,
                "epoch": record["epoch"],
                "train_time_sec": record["train"].get("time_sec"),
                "train_samples_per_sec": record["train"].get("samples_per_sec"),
                "train_mse_sampled": record["train"].get("mse"),
                "val_time_sec": record["val"].get("time_sec"),
                "val_mse": record["val"].get("mse"),
                "val_rmse": record["val"].get("rmse"),
                "val_mae": record["val"].get("mae"),
            })
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_md(path: str, comparison: Dict[str, Any], results: Dict[str, Dict[str, Any]]):
    lines = [
        "# DonkeyCar ResNet18: mytorch JIT vs PyTorch",
        "",
        "All FLOPs are analytic estimates for the same ResNet18 graph. Differences in achieved TFLOP/s come from measured wall time.",
        "",
        "| Metric | mytorch_jit | pytorch | ratio/delta |",
        "|---|---:|---:|---:|",
    ]
    for key, item in comparison.items():
        if not isinstance(item, dict):
            continue
        ratio_key = "jit_div_pytorch" if "jit_div_pytorch" in item else "jit_minus_pytorch"
        lines.append(
            f"| {key} | {fmt(item.get('mytorch_jit'))} | {fmt(item.get('pytorch'))} | {fmt(item.get(ratio_key))} |"
        )
    lines.extend(["", "## Backend Notes", ""])
    for backend, result in results.items():
        lines.append(f"- `{backend}`: wall={fmt(result.get('wall_time_sec'))} sec, "
                     f"samples/s={fmt(result.get('samples_per_sec'))}, "
                     f"achieved TFLOP/s={fmt(result.get('estimated_achieved_tflops'))}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def fmt(value):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.6g}"
    except Exception:
        return str(value)


def parse_backends(args) -> List[str]:
    if args.backend == "both":
        return ["mytorch_jit", "pytorch"]
    if args.backend not in VALID_BACKENDS:
        raise ValueError(f"Unknown backend: {args.backend}")
    return [args.backend]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train DonkeyCar ResNet18 with mytorch JIT and compare against "
            "a same-shape PyTorch ResNet18 baseline."
        )
    )
    parser.add_argument("--data-root", default="./", help="DonkeyCar root containing train.txt and val.txt.")
    parser.add_argument("--train-list", default="train.txt", help="Train list path, absolute or relative to --data-root.")
    parser.add_argument("--val-list", default="val.txt", help="Validation list path, absolute or relative to --data-root.")
    parser.add_argument("--results-dir", default="results/donkeycar_resnet18_jit_vs_pytorch")
    parser.add_argument("--backend", choices=("both",) + VALID_BACKENDS, default="both")
    parser.add_argument(
        "--order",
        default=None,
        help="Comma-separated backend order, e.g. pytorch,mytorch_jit. Default follows --backend.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--output-dim", type=int, default=1)
    parser.add_argument("--image-height", type=int, default=120)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu", action="store_true", help="Force CPU for both backends.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional short-run cap per epoch.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Optional validation cap per epoch.")
    parser.add_argument("--loss-log-interval", type=int, default=20)
    parser.add_argument("--resource-interval-sec", type=float, default=0.25)
    parser.add_argument(
        "--mytorch-loader",
        choices=("sync", "async"),
        default="async",
        help="mytorch data path: sync Dataloader or threaded producer-consumer prefetch loader.",
    )
    parser.add_argument("--mytorch-num-workers", type=int, default=2)
    parser.add_argument("--mytorch-prefetch-factor", type=int, default=4)
    parser.add_argument("--torch-num-workers", type=int, default=0)
    parser.add_argument("--torch-prefetch-factor", type=int, default=2)
    parser.add_argument("--torch-persistent-workers", action="store_true")
    parser.add_argument("--torch-cudnn-benchmark", action="store_true")
    parser.add_argument(
        "--model-parallel",
        choices=("none", "pytorch"),
        default="none",
        help=(
            "Optional model-parallel training. 'pytorch' splits the PyTorch "
            "ResNet18 across cuda:0 and cuda:1. mytorch currently has no "
            "cuda:N tensor device API, so mytorch_jit remains single-device."
        ),
    )
    parser.add_argument("--jit-profile", action="store_true")
    parser.add_argument("--jit-dump-graph", action="store_true")
    parser.add_argument("--jit-experimental-conv-bn-fusion", action="store_true")
    parser.add_argument(
        "--cupy-accelerators",
        default=None,
        help="none/off/disabled disables CuPy accelerators; auto leaves CuPy defaults.",
    )
    parser.add_argument("--allow-unsupported-nvcc", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    backends = parse_backends(args)
    order = backends
    if args.order:
        order = [item.strip() for item in args.order.split(",") if item.strip()]
        invalid = [item for item in order if item not in backends]
        if invalid:
            raise ValueError(f"--order contains backend not selected by --backend: {invalid}")

    torch = optional_torch()
    device = "cpu"
    if not args.cpu:
        if cp is not None:
            device = "cuda"
        elif torch is not None and torch.cuda.is_available():
            device = "cuda"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.results_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    config = ExperimentConfig(
        data_root=args.data_root,
        train_list=args.train_list,
        val_list=args.val_list,
        results_dir=args.results_dir,
        backends=backends,
        order=order,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        output_dim=args.output_dim,
        image_height=args.image_height,
        image_width=args.image_width,
        seed=args.seed,
        device=device,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        loss_log_interval=args.loss_log_interval,
        resource_interval_sec=args.resource_interval_sec,
        torch_num_workers=args.torch_num_workers,
        torch_prefetch_factor=args.torch_prefetch_factor,
        torch_persistent_workers=args.torch_persistent_workers,
        torch_cudnn_benchmark=args.torch_cudnn_benchmark,
        mytorch_loader=args.mytorch_loader,
        mytorch_num_workers=args.mytorch_num_workers,
        mytorch_prefetch_factor=args.mytorch_prefetch_factor,
        model_parallel=args.model_parallel,
        jit_profile=args.jit_profile,
        jit_dump_graph=args.jit_dump_graph,
        jit_experimental_conv_bn_fusion=args.jit_experimental_conv_bn_fusion,
        allow_unsupported_nvcc=args.allow_unsupported_nvcc,
        cupy_accelerators=args.cupy_accelerators,
    )

    flops = estimate_resnet18_flops(args.image_height, args.image_width, args.output_dim)
    write_json(os.path.join(output_dir, "config.json"), asdict(config))
    write_json(os.path.join(output_dir, "flops_estimate.json"), flops)

    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")
    print(f"Backends: {' -> '.join(order)}")
    print("FLOPs estimate per sample:", json.dumps(flops, indent=2))

    results: Dict[str, Dict[str, Any]] = {}
    for backend in order:
        print("\n" + "=" * 88)
        print(f"Running backend: {backend}")
        print("=" * 88)
        if backend == "mytorch_jit":
            results[backend] = run_mytorch_jit(config, output_dir, flops)
        elif backend == "pytorch":
            results[backend] = run_pytorch(config, output_dir, flops)
        write_json(os.path.join(output_dir, f"{backend}_result.json"), results[backend])

    comparison = make_comparison(results)
    summary = {
        "config": asdict(config),
        "output_dir": output_dir,
        "flops_estimate": flops,
        "results": results,
        "comparison": comparison,
    }
    write_json(os.path.join(output_dir, "summary.json"), summary)
    write_batch_csv(os.path.join(output_dir, "batch_records.csv"), results)
    write_epoch_csv(os.path.join(output_dir, "epoch_history.csv"), results)
    write_comparison_md(os.path.join(output_dir, "comparison.md"), comparison, results)

    print("\nSaved results:")
    print(f"  {os.path.join(output_dir, 'summary.json')}")
    print(f"  {os.path.join(output_dir, 'epoch_history.csv')}")
    print(f"  {os.path.join(output_dir, 'batch_records.csv')}")
    print(f"  {os.path.join(output_dir, 'comparison.md')}")
    if comparison:
        print("\nComparison:")
        print(json.dumps(comparison, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
