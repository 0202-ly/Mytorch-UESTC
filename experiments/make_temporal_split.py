import argparse
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


FRAME_RE = re.compile(r"(?:^|[/\\])(\d+)_([-+]?\d+(?:\.\d+)?)\.jpg$", re.IGNORECASE)


@dataclass(frozen=True)
class Sample:
    frame_id: int
    rel_path: str
    label: float

    @property
    def line(self) -> str:
        return f"{self.rel_path} {self.label:.4f}"


def parse_sample_line(line: str) -> Sample:
    parts = line.strip().split()
    if len(parts) < 2:
        raise ValueError(f"Bad list line: {line!r}")
    rel_path = parts[0].replace("\\", "/")
    label = float(parts[1])
    match = FRAME_RE.search(rel_path)
    if not match:
        raise ValueError(f"Cannot parse frame id from path: {rel_path}")
    return Sample(frame_id=int(match.group(1)), rel_path=rel_path, label=label)


def read_samples_from_lists(paths: Sequence[Path]) -> List[Sample]:
    by_id: Dict[int, Sample] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                sample = parse_sample_line(line)
                if sample.frame_id in by_id:
                    continue
                by_id[sample.frame_id] = sample
    return sorted(by_id.values(), key=lambda s: s.frame_id)


def read_samples_from_data_dir(data_root: Path) -> List[Sample]:
    samples: List[Sample] = []
    for path in sorted((data_root / "data").glob("*.jpg")):
        rel_path = "./data/" + path.name
        match = FRAME_RE.search(rel_path)
        if not match:
            continue
        samples.append(Sample(frame_id=int(match.group(1)), rel_path=rel_path, label=float(match.group(2))))
    return sorted(samples, key=lambda s: s.frame_id)


def make_blocked_split(
    samples: Sequence[Sample],
    val_ratio: float,
    block_size: int,
    val_block_offset: int,
    purge_gap: int,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    val_every = max(2, round(1.0 / val_ratio))
    blocks = [
        samples[start:start + block_size]
        for start in range(0, len(samples), block_size)
    ]
    val_blocks = {
        block_idx
        for block_idx in range(len(blocks))
        if (block_idx - val_block_offset) % val_every == 0
    }
    val = [sample for block_idx, block in enumerate(blocks) if block_idx in val_blocks for sample in block]
    val_ids = {sample.frame_id for sample in val}

    train: List[Sample] = []
    purged: List[Sample] = []
    for block_idx, block in enumerate(blocks):
        if block_idx in val_blocks:
            continue
        for sample in block:
            near_val = any(sample.frame_id + delta in val_ids for delta in range(-purge_gap, purge_gap + 1))
            if near_val:
                purged.append(sample)
            else:
                train.append(sample)
    return train, val, purged


def make_holdout_split(
    samples: Sequence[Sample],
    val_ratio: float,
    purge_gap: int,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    val_count = int(math.ceil(len(samples) * val_ratio))
    split_at = max(1, len(samples) - val_count)
    train_candidates = list(samples[:split_at])
    val = list(samples[split_at:])
    val_ids = {sample.frame_id for sample in val}
    train: List[Sample] = []
    purged: List[Sample] = []
    for sample in train_candidates:
        near_val = any(sample.frame_id + delta in val_ids for delta in range(-purge_gap, purge_gap + 1))
        if near_val:
            purged.append(sample)
        else:
            train.append(sample)
    return train, val, purged


def neighbor_stats(train: Sequence[Sample], val: Sequence[Sample], windows: Iterable[int]) -> Dict[str, str]:
    train_ids = {sample.frame_id for sample in train}
    stats: Dict[str, str] = {}
    for window in windows:
        count = sum(
            any(sample.frame_id + delta in train_ids for delta in range(-window, window + 1) if delta != 0)
            for sample in val
        )
        stats[f"val_with_train_neighbor_pm_{window}"] = f"{count}/{len(val)} ({count / max(1, len(val)):.2%})"
    return stats


def write_list(path: Path, samples: Sequence[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for sample in samples:
            f.write(sample.line + "\n")


def write_report(path: Path, config: Dict, train: Sequence[Sample], val: Sequence[Sample], purged: Sequence[Sample]) -> None:
    stats = {
        "train_count": len(train),
        "val_count": len(val),
        "purged_count": len(purged),
        "train_minmax": [train[0].frame_id, train[-1].frame_id] if train else None,
        "val_minmax": [val[0].frame_id, val[-1].frame_id] if val else None,
        **neighbor_stats(train, val, [1, 2, 5, 10, 20]),
    }
    lines = [
        "# Temporal DonkeyCar Split",
        "",
        "This split is sorted by frame id and keeps validation frames in contiguous time blocks.",
        "Training frames within the purge gap around validation frames are removed.",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Stats",
        "",
        "```json",
        json.dumps(stats, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def backup_and_activate(data_root: Path, train_path: Path, val_path: Path) -> None:
    for name, src in [("train.txt", train_path), ("val.txt", val_path)]:
        dst = data_root / name
        if dst.exists():
            backup = data_root / f"{name}.random_frame_backup"
            if not backup.exists():
                shutil.copy2(dst, backup)
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-reduced temporal train/val split for DonkeyCar data.")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output-dir", default=os.path.join("splits", "temporal_block_gap20"))
    parser.add_argument("--source-lists", nargs="*", default=["train.txt", "val.txt"])
    parser.add_argument("--strategy", choices=["blocked", "holdout"], default="blocked")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--block-size", type=int, default=500)
    parser.add_argument("--val-block-offset", type=int, default=4)
    parser.add_argument("--purge-gap", type=int, default=20)
    parser.add_argument("--activate", action="store_true", help="Overwrite root train.txt/val.txt after backing them up.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    source_paths = [data_root / path for path in args.source_lists]
    if all(path.exists() for path in source_paths):
        samples = read_samples_from_lists(source_paths)
        source = [str(path) for path in source_paths]
    else:
        samples = read_samples_from_data_dir(data_root)
        source = [str(data_root / "data")]

    if not samples:
        raise RuntimeError("No samples found.")

    if args.strategy == "blocked":
        train, val, purged = make_blocked_split(
            samples,
            val_ratio=args.val_ratio,
            block_size=args.block_size,
            val_block_offset=args.val_block_offset,
            purge_gap=args.purge_gap,
        )
    else:
        train, val, purged = make_holdout_split(samples, args.val_ratio, args.purge_gap)

    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"
    write_list(train_path, train)
    write_list(val_path, val)

    config = vars(args).copy()
    config.update({
        "source": source,
        "total_samples": len(samples),
        "train_path": str(train_path),
        "val_path": str(val_path),
    })
    write_report(output_dir / "split_report.md", config, train, val, purged)

    if args.activate:
        backup_and_activate(data_root, train_path, val_path)

    print(f"Saved temporal split to: {output_dir}")
    print(f"Train list: {train_path} ({len(train)} samples)")
    print(f"Val list: {val_path} ({len(val)} samples)")
    print(f"Purged near-val train frames: {len(purged)}")
    print(f"Report: {output_dir / 'split_report.md'}")
    if args.activate:
        print("Activated split as root train.txt/val.txt; backups end with .random_frame_backup")


if __name__ == "__main__":
    main()
