"""Small MyTorch MNIST MLP training smoke test (forward, backward, update)."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from dataset.mnist_dataset import MnistDataset
from mytorch import Linear, Module, ReLU, Tensor
from mytorch.loss import MSELoss
from mytorch.optim import SGD


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 128)
        self.relu = ReLU()
        self.fc2 = Linear(128, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--download", action="store_true", help="Download missing MNIST IDX files")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--output", default="results/mnist_mlp_smoke.csv")
    args = parser.parse_args()
    if args.samples < 1 or args.batch_size < 1 or args.epochs < 1:
        parser.error("--samples, --batch-size and --epochs must be positive")

    dataset = MnistDataset(args.data_root, train=True, download=args.download)
    sample_count = min(args.samples, len(dataset))
    model = MLP().train()
    criterion = MSELoss(fused=False)
    optimizer = SGD(model.parameters(), lr=args.learning_rate)
    rows = []

    for epoch in range(args.epochs):
        start = time.perf_counter()
        loss_sum = 0.0
        correct = 0
        for offset in range(0, sample_count, args.batch_size):
            end = min(offset + args.batch_size, sample_count)
            count = end - offset
            images = Tensor(dataset.data[offset:end].reshape(count, -1), requires_grad=False)
            labels = dataset.labels[offset:end]
            targets = np.zeros((count, 10), dtype=np.float32)
            targets[np.arange(count), labels] = 1.0
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, Tensor(targets, requires_grad=False))
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.data) * count
            correct += int(np.sum(np.argmax(logits.data, axis=1) == labels))
        rows.append((epoch + 1, sample_count, (time.perf_counter() - start) * 1000,
                     loss_sum / sample_count, correct / sample_count))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "samples", "total_ms", "proxy_loss", "train_accuracy"])
        writer.writerows(rows)
    print(f"mnist_mlp_ok samples={sample_count} epochs={args.epochs} output={output}")


if __name__ == "__main__":
    main()
