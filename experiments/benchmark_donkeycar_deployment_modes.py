import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


def run_command(
    name: str,
    cmd: List[str],
    cwd: str,
    timeout: Optional[int],
    extra_env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    start = time.perf_counter()
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        elapsed = time.perf_counter() - start
        return {
            "name": name,
            "status": "ok" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "time_sec": elapsed,
            "command": " ".join(cmd),
            "output_tail": "\n".join((proc.stdout or "").splitlines()[-40:]),
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        return {
            "name": name,
            "status": "timeout",
            "returncode": None,
            "time_sec": elapsed,
            "command": " ".join(cmd),
            "output_tail": "\n".join(((exc.stdout or "") if isinstance(exc.stdout, str) else "").splitlines()[-40:]),
        }


def tool_exists(name: str) -> bool:
    return shutil.which(name) is not None


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(path: str, rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    lines = [
        "# DonkeyCar Deployment Mode Benchmark",
        "",
        "This experiment checks whether the DonkeyCar MyTorch training task can run as a local script, a Docker container, and a Kubernetes Job.",
        "",
        f"- Data root: `{args.data_root}`",
        f"- Train list: `{args.train_list}`",
        f"- Val list: `{args.val_list}`",
        f"- Epochs: {args.epochs}",
        f"- Batch size: {args.batch_size}",
        f"- Max train batches: {args.max_train_batches}",
        f"- Max val batches: {args.max_val_batches}",
        "",
        "| mode | status | time sec | returncode | command |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join([
                row["name"],
                row["status"],
                fmt(row.get("time_sec")),
                fmt(row.get("returncode")),
                "`" + row.get("command", "").replace("|", "\\|") + "`",
            ])
            + " |"
        )
    lines.extend([
        "",
        "Required evidence for the defense report:",
        "- Docker: image build log, `docker run` training log, image size, mounted dataset path, output directory.",
        "- Kubernetes: `kubectl apply` result, pod status, `kubectl logs -f job/mytorch-donkeycar-train`, CPU/memory request/limit.",
        "- Kuflow/Kubeflow: compiled pipeline YAML or UI screenshot, operator argv, resource spec, run log.",
        "- Deployment comparison: local vs Docker wall-clock time, K8s Job start-up overhead, repeatability across at least 3 runs.",
        "",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_results(path: str, rows: List[Dict[str, Any]]) -> None:
    ok_rows = [row for row in rows if row.get("time_sec") is not None and row.get("status") == "ok"]
    if not ok_rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    names = [row["name"] for row in ok_rows]
    times = [float(row["time_sec"]) for row in ok_rows]
    plt.figure(figsize=(8, 4.5))
    plt.bar(names, times, color=["#2563eb", "#238b57", "#d99118"][: len(names)])
    plt.ylabel("Wall-clock seconds")
    plt.title("DonkeyCar Deployment Mode Benchmark")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DonkeyCar local/Docker/K8s deployment modes.")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--train-list", default="splits/temporal_block_gap20/train.txt")
    parser.add_argument("--val-list", default="splits/temporal_block_gap20/val.txt")
    parser.add_argument("--results-dir", default="results/donkeycar_deployment_benchmark")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-train-batches", type=int, default=20)
    parser.add_argument("--max-val-batches", type=int, default=20)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--skip-local", action="store_true")
    parser.add_argument("--skip-docker", action="store_true")
    parser.add_argument("--skip-k8s", action="store_true")
    parser.add_argument("--docker-image", default="mytorch-donkeycar:latest")
    parser.add_argument("--k8s-job-yaml", default="deploy/k8s/donkeycar-training-job.yaml")
    parser.add_argument("--k8s-job-name", default="mytorch-donkeycar-train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    omp_env = {
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MYTORCH_HEADLESS": "1",
    }
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir = os.path.join(args.results_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    rows: List[Dict[str, Any]] = []
    local_results_dir = os.path.join(run_dir, "local_train")
    if not args.skip_local:
        rows.append(run_command(
            "local_bare",
            [
                sys.executable,
                "train_donkeycar_resnet18_jit_vs_pytorch.py",
                "--backend",
                "mytorch_jit",
                "--cpu",
                "--data-root",
                args.data_root,
                "--train-list",
                args.train_list,
                "--val-list",
                args.val_list,
                "--results-dir",
                local_results_dir,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--max-train-batches",
                str(args.max_train_batches),
                "--max-val-batches",
                str(args.max_val_batches),
                "--mytorch-loader",
                "async",
                "--mytorch-num-workers",
                "4",
                "--mytorch-prefetch-factor",
                "4",
            ],
            cwd=repo_root,
            timeout=args.timeout_sec,
            extra_env=omp_env,
        ))

    if not args.skip_docker:
        if tool_exists("docker"):
            build_row = run_command(
                "docker_build",
                ["docker", "build", "-f", "deploy/docker/Dockerfile.donkeycar", "-t", args.docker_image, "."],
                cwd=repo_root,
                timeout=args.timeout_sec,
            )
            rows.append(build_row)
            data_root_abs = os.path.abspath(os.path.join(repo_root, args.data_root))
            output_abs = os.path.abspath(os.path.join(run_dir, "docker_outputs"))
            os.makedirs(output_abs, exist_ok=True)
            if build_row.get("status") == "ok":
                rows.append(run_command(
                    "docker_run",
                    [
                        "docker",
                        "run",
                        "--rm",
                        "-e",
                        "MYTORCH_HEADLESS=1",
                        "-e",
                        "KMP_DUPLICATE_LIB_OK=TRUE",
                        "-e",
                        "OMP_NUM_THREADS=1",
                        "-e",
                        "MKL_NUM_THREADS=1",
                        "-v",
                        f"{data_root_abs}:/data/donkeycar:ro",
                        "-v",
                        f"{output_abs}:/outputs",
                        args.docker_image,
                        "python",
                        "train_donkeycar_resnet18_jit_vs_pytorch.py",
                        "--backend",
                        "mytorch_jit",
                        "--cpu",
                        "--data-root",
                        "/data/donkeycar",
                        "--train-list",
                        args.train_list,
                        "--val-list",
                        args.val_list,
                        "--results-dir",
                        "/outputs/donkeycar_resnet18_mytorch",
                        "--epochs",
                        str(args.epochs),
                        "--batch-size",
                        str(args.batch_size),
                        "--max-train-batches",
                        str(args.max_train_batches),
                        "--max-val-batches",
                        str(args.max_val_batches),
                        "--mytorch-loader",
                        "async",
                        "--mytorch-num-workers",
                        "4",
                        "--mytorch-prefetch-factor",
                        "4",
                    ],
                    cwd=repo_root,
                    timeout=args.timeout_sec,
                ))
            else:
                rows.append({
                    "name": "docker_run",
                    "status": "skipped",
                    "time_sec": None,
                    "returncode": None,
                    "command": "skipped because docker_build failed",
                    "output_tail": "",
                })
        else:
            rows.append({"name": "docker", "status": "skipped", "time_sec": None, "returncode": None, "command": "docker not found"})

    if not args.skip_k8s:
        if tool_exists("kubectl"):
            rows.append(run_command(
                "k8s_delete_old_job",
                ["kubectl", "delete", "job", args.k8s_job_name, "--ignore-not-found=true"],
                cwd=repo_root,
                timeout=120,
            ))
            rows.append(run_command(
                "k8s_apply_job",
                ["kubectl", "apply", "-f", args.k8s_job_yaml],
                cwd=repo_root,
                timeout=args.timeout_sec,
            ))
            rows.append(run_command(
                "k8s_wait_complete",
                ["kubectl", "wait", "--for=condition=complete", f"job/{args.k8s_job_name}", f"--timeout={args.timeout_sec}s"],
                cwd=repo_root,
                timeout=args.timeout_sec + 30,
            ))
            rows.append(run_command(
                "k8s_job_logs",
                ["kubectl", "logs", f"job/{args.k8s_job_name}"],
                cwd=repo_root,
                timeout=300,
            ))
            rows.append(run_command(
                "k8s_job_status",
                ["kubectl", "get", "job", args.k8s_job_name, "-o", "wide"],
                cwd=repo_root,
                timeout=120,
            ))
        else:
            rows.append({"name": "k8s", "status": "skipped", "time_sec": None, "returncode": None, "command": "kubectl not found"})

    write_csv(os.path.join(run_dir, "deployment_modes.csv"), rows)
    write_summary(os.path.join(run_dir, "summary.md"), rows, args)
    plot_results(os.path.join(run_dir, "deployment_walltime.png"), rows)
    with open(os.path.join(run_dir, "raw_results.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"Saved DonkeyCar deployment benchmark to: {run_dir}")
    print(os.path.join(run_dir, "summary.md"))


if __name__ == "__main__":
    main()
