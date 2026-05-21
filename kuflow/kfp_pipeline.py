"""Kubeflow Pipeline examples for MyTorch tasks.

The pipeline can be compiled only when `kfp` is installed. Without kfp this file
prints the DonkeyCar operator command line and exits successfully, so it remains
safe to keep in the core repository.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from kuflow.operators import (
    OperatorLaunchSpec,
    TaskOperator,
    operator_argv_for_donkeycar_train,
    operator_argv_for_mnist,
)


def _try_import_kfp():
    try:
        from kfp import compiler, dsl
        from kfp.dsl import ContainerOp, pipeline
    except ImportError:
        print(
            "kfp is not installed. Install it with `pip install kfp` to compile "
            "the Kubeflow pipeline YAML.",
            file=sys.stderr,
        )
        return None
    return compiler, dsl, ContainerOp, pipeline


def build_donkeycar_pipeline(args=None):
    imported = _try_import_kfp()
    if imported is None:
        return None, None
    compiler, _dsl, ContainerOp, pipeline = imported

    @pipeline(
        name="mytorch-donkeycar-train-pipeline",
        description="MyTorch DonkeyCar steering regression training pipeline",
    )
    def donkeycar_train_pipeline(
        epochs: int = 1 if args is None else args.epochs,
        batch_size: int = 8 if args is None else args.batch_size,
        image: str = "mytorch-donkeycar:latest",
    ):
        argv = operator_argv_for_donkeycar_train(
            epochs=int(epochs),
            batch_size=int(batch_size),
            data_root="/data/donkeycar",
            results_dir="/outputs/donkeycar_resnet18_mytorch",
            max_train_batches=0 if args is None else args.max_train_batches,
            max_val_batches=0 if args is None else args.max_val_batches,
        )
        ContainerOp(
            name="donkeycar-train",
            image=image,
            command=argv[:1],
            arguments=argv[1:],
        ).set_memory_request("2Gi").set_cpu_request("1")

    return compiler, donkeycar_train_pipeline


def build_mnist_pipeline():
    imported = _try_import_kfp()
    if imported is None:
        return None, None
    compiler, _dsl, ContainerOp, pipeline = imported

    @pipeline(name="mytorch-mnist-train-pipeline", description="MyTorch MNIST training pipeline")
    def mnist_train_pipeline(epochs: int = 5, image: str = "mytorch-uestc:latest"):
        argv = operator_argv_for_mnist(epochs=int(epochs), data_root="/data")
        ContainerOp(
            name="mnist-train",
            image=image,
            command=argv[:1],
            arguments=argv[1:],
        ).set_memory_request("512Mi").set_cpu_request("500m")

    return compiler, mnist_train_pipeline


def _yaml_scalar(value):
    return json.dumps(value, ensure_ascii=False)


def _yaml_list(values, indent):
    pad = " " * indent
    return "\n".join(f"{pad}- {_yaml_scalar(value)}" for value in values)


def _write_fallback_pipeline_yaml(output, task, image, argv):
    """Write an Argo/Kubeflow-compatible single-step pipeline without kfp.

    This fallback is intentionally small: it gives the defense artifact a real
    DonkeyCar pipeline YAML even when the kfp SDK cannot be installed because of
    network or SSL issues.
    """
    if not output:
        raise ValueError("output path is required")
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    pipeline_name = "mytorch-donkeycar-train-pipeline" if task == "donkeycar" else "mytorch-mnist-train-pipeline"
    step_name = "donkeycar-train" if task == "donkeycar" else "mnist-train"
    description = (
        "MyTorch DonkeyCar steering regression training pipeline"
        if task == "donkeycar"
        else "MyTorch MNIST training pipeline"
    )
    command = argv[:1]
    arguments = argv[1:]
    pipeline_spec = {"name": pipeline_name, "description": description}
    lines = [
        "apiVersion: argoproj.io/v1alpha1",
        "kind: Workflow",
        "metadata:",
        f"  generateName: {pipeline_name}-",
        "  annotations:",
        "    pipelines.kubeflow.org/kfp_sdk_version: manual-fallback",
        f"    pipelines.kubeflow.org/pipeline_spec: {_yaml_scalar(json.dumps(pipeline_spec, ensure_ascii=False))}",
        "spec:",
        f"  entrypoint: {pipeline_name}",
        "  templates:",
        f"    - name: {pipeline_name}",
        "      dag:",
        "        tasks:",
        f"          - name: {step_name}",
        f"            template: {step_name}",
        f"    - name: {step_name}",
        "      container:",
        f"        image: {_yaml_scalar(image)}",
        "        imagePullPolicy: Never",
        "        command:",
        _yaml_list(command, 10),
        "        args:",
        _yaml_list(arguments, 10),
        "        env:",
        "          - name: MYTORCH_HEADLESS",
        "            value: \"1\"",
        "          - name: KMP_DUPLICATE_LIB_OK",
        "            value: \"TRUE\"",
        "          - name: OMP_NUM_THREADS",
        "            value: \"1\"",
        "          - name: MKL_NUM_THREADS",
        "            value: \"1\"",
        "        resources:",
        "          requests:",
        "            cpu: \"500m\"",
        "            memory: 1Gi",
        "          limits:",
        "            cpu: \"2\"",
        "            memory: 4Gi",
    ]
    if task == "donkeycar":
        lines.extend([
            "        volumeMounts:",
            "          - name: donkeycar-data",
            "            mountPath: /data/donkeycar",
            "            readOnly: true",
            "          - name: donkeycar-results",
            "            mountPath: /outputs",
            "  volumes:",
            "    - name: donkeycar-data",
            "      persistentVolumeClaim:",
            "        claimName: donkeycar-data-pvc",
            "    - name: donkeycar-results",
            "      persistentVolumeClaim:",
            "        claimName: donkeycar-results-pvc",
        ])
    with open(output, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Compile MyTorch Kubeflow pipeline examples.")
    parser.add_argument("--task", choices=["donkeycar", "mnist"], default="donkeycar")
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-train-batches", type=int, default=20)
    parser.add_argument("--max-val-batches", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.task == "donkeycar":
        spec = OperatorLaunchSpec(
            operator=TaskOperator.DONKEYCAR_TRAIN,
            argv=operator_argv_for_donkeycar_train(
                epochs=args.epochs,
                batch_size=args.batch_size,
                max_train_batches=args.max_train_batches,
                max_val_batches=args.max_val_batches,
            ),
        )
        print("DonkeyCar OperatorLaunchSpec argv:", spec.argv)
        compiler, pipeline_fn = build_donkeycar_pipeline(args)
        output = args.output or "mytorch_donkeycar_pipeline.yaml"
    else:
        spec = OperatorLaunchSpec(
            operator=TaskOperator.MNIST_TRAIN,
            image="mytorch-uestc:latest",
            data_mount_path="/data",
            argv=operator_argv_for_mnist(epochs=5),
        )
        print("MNIST OperatorLaunchSpec argv:", spec.argv)
        compiler, pipeline_fn = build_mnist_pipeline()
        output = args.output or "mytorch_mnist_pipeline.yaml"

    if compiler is None or pipeline_fn is None:
        image = "mytorch-donkeycar:latest" if args.task == "donkeycar" else "mytorch-uestc:latest"
        _write_fallback_pipeline_yaml(output, args.task, image, spec.argv)
        print(f"kfp is unavailable; wrote fallback Kubeflow/Argo YAML: {output}")
        return 0
    compiler.Compiler().compile(pipeline_fn, package_path=output)
    print(f"Compiled {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
