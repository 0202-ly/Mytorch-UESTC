"""Save one reproducible ResNet18 initialization for all training modes."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from distributed.mytorch_adapter import flatten_parameters
from model.resnet import ResNet18Original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/distributed_initial_parameters.npy"),
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    parameters = flatten_parameters(ResNet18Original(num_classes=1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, parameters)
    print(f"saved {parameters.size} float32 parameters to {args.output}")


if __name__ == "__main__":
    main()
