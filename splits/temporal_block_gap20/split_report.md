# Temporal DonkeyCar Split

This split is sorted by frame id and keeps validation frames in contiguous time blocks.
Training frames within the purge gap around validation frames are removed.

## Config

```json
{
  "data_root": ".",
  "output_dir": "splits\\temporal_block_gap20",
  "source_lists": [
    "train.txt",
    "val.txt"
  ],
  "strategy": "blocked",
  "val_ratio": 0.2,
  "block_size": 500,
  "val_block_offset": 4,
  "purge_gap": 20,
  "activate": true,
  "source": [
    "train.txt",
    "val.txt"
  ],
  "total_samples": 10000,
  "train_path": "splits\\temporal_block_gap20\\train.txt",
  "val_path": "splits\\temporal_block_gap20\\val.txt"
}
```

## Stats

```json
{
  "train_count": 7860,
  "val_count": 2000,
  "purged_count": 140,
  "train_minmax": [
    0,
    9479
  ],
  "val_minmax": [
    2000,
    9999
  ],
  "val_with_train_neighbor_pm_1": "0/2000 (0.00%)",
  "val_with_train_neighbor_pm_2": "0/2000 (0.00%)",
  "val_with_train_neighbor_pm_5": "0/2000 (0.00%)",
  "val_with_train_neighbor_pm_10": "0/2000 (0.00%)",
  "val_with_train_neighbor_pm_20": "0/2000 (0.00%)"
}
```
