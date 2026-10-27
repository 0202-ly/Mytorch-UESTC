"""Wait for sequential four-mode Ring job, then validate, plot, and archive evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time


MODES = ("ring_socket2", "ring_socket4", "ring_grpc2", "ring_grpc4")


def running(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    if not stat.exists():
        return False
    return stat.read_text().split(") ", 1)[1][0] not in "ZX"


def status(path: Path, phase: str, **details) -> None:
    path.write_text(json.dumps({"phase": phase, "timestamp": time.time(), **details}, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher-pid", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--launcher-log", type=Path, required=True)
    args = parser.parse_args()
    marker = args.run_dir / "finalizer_status.json"
    try:
        status(marker, "waiting", launcher_pid=args.launcher_pid)
        while running(args.launcher_pid):
            time.sleep(15)
        content = args.launcher_log.read_text(encoding="utf-8")
        for mode in MODES:
            if not any(line.startswith(f"DONE {mode} ") for line in content.splitlines()):
                raise RuntimeError(f"{mode} did not complete; inspect launcher log")
        status(marker, "aggregating")
        subprocess.run([sys.executable, "tools/aggregate_ring_transports.py",
                        "--base-run-dir", str(args.base_run_dir),
                        "--transport-run-dir", str(args.run_dir),
                        "--output-dir", str(args.run_dir / "aggregate")], check=True)
        archive = args.run_dir.parent / f"{args.run_dir.name}_results.tgz"
        status(marker, "archiving", archive=str(archive))
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(args.run_dir, arcname=args.run_dir.name)
            handle.add(args.launcher_log, arcname=args.launcher_log.name)
            validation = args.run_dir.parent / "ring_transport_math_validation.log"
            if validation.exists():
                handle.add(validation, arcname=validation.name)
        status(marker, "complete", archive=str(archive))
        print(f"COMPLETE {archive}", flush=True)
    except Exception as exc:
        status(marker, "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
