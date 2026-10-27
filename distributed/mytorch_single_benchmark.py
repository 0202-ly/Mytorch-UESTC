"""Single-process MyTorch DonkeyCar baseline using the complete train split."""
from __future__ import annotations

import argparse
import numpy as np

from model.resnet import ResNet18Original
from mytorch.dataset import AutoDriveDataset
from mytorch.dataloader import Dataloader
from mytorch.loss import MSELoss

from .metrics import EpochMetrics, Stopwatch, write_metrics
from .mytorch_adapter import flatten_gradients, flatten_parameters, load_parameters


def run(args) -> None:
    model = ResNet18Original(num_classes=1).train()
    if args.initial_parameters:
        load_parameters(model, np.asarray(np.load(args.initial_parameters), dtype=np.float32).reshape(-1))
    if args.device == "cuda":
        model.cuda()
    dataset = AutoDriveDataset("train", data_root=args.data_root, list_path=args.train_list)
    loader = Dataloader(dataset, batch_size=args.batch_size, shuffle=True)
    criterion = MSELoss(fused=False)
    rows = []
    for epoch in range(args.epochs):
        total = Stopwatch()
        compute_ms = 0.0
        seen = 0
        correct = 0.0
        for batch_index, (x, y) in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            clock = Stopwatch()
            if args.device == "cuda":
                x = x.cuda()
                y = y.cuda()
            model.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            gradient = flatten_gradients(model)
            parameters = flatten_parameters(model)
            load_parameters(model, parameters - args.learning_rate * gradient)
            compute_ms += clock.elapsed_ms()
            pred = prediction.data.get() if hasattr(prediction.data, "get") else np.asarray(prediction.data)
            target = y.data.get() if hasattr(y.data, "get") else np.asarray(y.data)
            n = int(x.shape()[0])
            seen += n
            correct += float(np.sum(np.abs(pred - target) < args.accuracy_threshold))
        rows.append(EpochMetrics("single-mytorch-resnet18", 1, epoch, total.elapsed_ms(), compute_ms, 0.0, 0, correct / seen if seen else 0.0))
        write_metrics(rows, args.output)
        print(f"epoch={epoch + 1:02d} samples={seen} total_ms={rows[-1].total_ms:.1f} accuracy={rows[-1].accuracy:.4f}", flush=True)
    print(f"output={args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--train-list", default="train.txt")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--accuracy-threshold", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--initial-parameters", default=None)
    parser.add_argument("--output", default="results/single_mytorch.csv")
    run(parser.parse_args())
