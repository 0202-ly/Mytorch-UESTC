"""Scheduling operators for MyTorch training tasks.

This module maps framework-level tasks to container command lines and
Kubernetes Job specs. MNIST is kept as a small demo; DonkeyCar is the task used
by the defense project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskOperator(Enum):
    MNIST_TRAIN = "mnist_train"
    DONKEYCAR_TRAIN = "donkeycar_train"


@dataclass
class ResourceSpec:
    cpu_request: str = "1"
    memory_request: str = "2Gi"
    cpu_limit: str = "4"
    memory_limit: str = "8Gi"


@dataclass
class OperatorLaunchSpec:
    operator: TaskOperator
    image: str = "mytorch-donkeycar:latest"
    pull_policy: str = "IfNotPresent"
    argv: List[str] = field(default_factory=list)
    job_name: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    resources: ResourceSpec = field(default_factory=ResourceSpec)
    data_mount_path: str = "/data/donkeycar"
    output_mount_path: str = "/outputs"
    data_pvc: str = "donkeycar-data-pvc"
    output_pvc: str = "donkeycar-results-pvc"
    read_only_data: bool = True


def operator_argv_for_mnist(epochs: int = 5, data_root: str = "/data") -> List[str]:
    return [
        "python",
        "train_MNIST.py",
        "--no-viz",
        "--epochs",
        str(epochs),
        "--data-root",
        data_root,
    ]


def operator_argv_for_donkeycar_train(
    epochs: int = 5,
    batch_size: int = 8,
    data_root: str = "/data/donkeycar",
    train_list: str = "splits/temporal_block_gap20/train.txt",
    val_list: str = "splits/temporal_block_gap20/val.txt",
    results_dir: str = "/outputs/donkeycar_resnet18_mytorch",
    backend: str = "mytorch_jit",
    cpu: bool = True,
    loader: str = "async",
    workers: int = 4,
    prefetch_factor: int = 4,
    max_train_batches: int = 0,
    max_val_batches: int = 0,
) -> List[str]:
    argv = [
        "python",
        "train_donkeycar_resnet18_jit_vs_pytorch.py",
        "--backend",
        backend,
        "--data-root",
        data_root,
        "--train-list",
        train_list,
        "--val-list",
        val_list,
        "--results-dir",
        results_dir,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--mytorch-loader",
        loader,
        "--mytorch-num-workers",
        str(workers),
        "--mytorch-prefetch-factor",
        str(prefetch_factor),
    ]
    if cpu:
        argv.insert(4, "--cpu")
    if max_train_batches:
        argv.extend(["--max-train-batches", str(max_train_batches)])
    if max_val_batches:
        argv.extend(["--max-val-batches", str(max_val_batches)])
    return argv


def build_batch_v1_job(spec: OperatorLaunchSpec) -> Dict[str, Any]:
    name = spec.job_name or f"mytorch-{spec.operator.value.replace('_', '-')}"
    labels = {
        "app": "mytorch-uestc",
        "mytorch/operator": spec.operator.value,
        **spec.labels,
    }
    resources = spec.resources
    container = {
        "name": "trainer",
        "image": spec.image,
        "imagePullPolicy": spec.pull_policy,
        "env": [
            {"name": "MYTORCH_HEADLESS", "value": "1"},
            {"name": "CUPY_ACCELERATORS", "value": ""},
        ],
        "args": spec.argv,
        "resources": {
            "requests": {
                "cpu": resources.cpu_request,
                "memory": resources.memory_request,
            },
            "limits": {
                "cpu": resources.cpu_limit,
                "memory": resources.memory_limit,
            },
        },
        "volumeMounts": [
            {
                "name": "task-data",
                "mountPath": spec.data_mount_path,
                "readOnly": bool(spec.read_only_data),
            },
            {
                "name": "task-results",
                "mountPath": spec.output_mount_path,
            },
        ],
    }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name, "labels": labels},
        "spec": {
            "backoffLimit": 1,
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [container],
                    "volumes": [
                        {
                            "name": "task-data",
                            "persistentVolumeClaim": {"claimName": spec.data_pvc},
                        },
                        {
                            "name": "task-results",
                            "persistentVolumeClaim": {"claimName": spec.output_pvc},
                        },
                    ],
                },
            },
        },
    }


def build_donkeycar_train_job(
    epochs: int = 5,
    batch_size: int = 8,
    image: str = "mytorch-donkeycar:latest",
    job_name: str = "mytorch-donkeycar-train",
) -> Dict[str, Any]:
    spec = OperatorLaunchSpec(
        operator=TaskOperator.DONKEYCAR_TRAIN,
        image=image,
        job_name=job_name,
        argv=operator_argv_for_donkeycar_train(
            epochs=epochs,
            batch_size=batch_size,
        ),
    )
    return build_batch_v1_job(spec)
