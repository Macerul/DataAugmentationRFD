# SyRFDAware — How to Run

`SyRFDAware` is a drop-in replacement for `SyRFD` (from `SyRFD_optimized_dual.py`).
Import and instantiate it the same way; distribution fitting runs automatically at startup.

## Quick start

```python
from syrfd_aware import SyRFDAware

syrfd = SyRFDAware(
    imbalance_dataset_path="imbalanced_datasets/my_dataset.csv",
    rfd_file_path="discovered_rfds/discovered_rfds_processed/RFD2_E0.0_my_dataset_min.txt",
    oversampling=50,          # number of synthetic tuples to generate
    threshold=4,              # RFD threshold
    max_iter=100,             # max repair attempts per tuple
    dist_alpha=0.05,          # significance level for distribution fitting (optional)
)

new_tuples = syrfd.augment_dataset()
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `imbalance_dataset_path` | str | — | Path to the imbalanced CSV dataset |
| `rfd_file_path` | str | — | Path to the discovered RFDs file |
| `oversampling` | int | — | Number of new tuples to generate |
| `threshold` | int | `4` | RFD similarity threshold |
| `max_iter` | int | `100` | Max generation/repair attempts per tuple |
| `selected_rfds` | list\|None | `None` | Restrict to a subset of RFDs (optional) |
| `dist_alpha` | float | `0.05` | Alpha level recorded in the distribution log |

## What happens at startup

1. For every feature, the best-fitting statistical distribution is identified via KS-test across 28 candidate distributions (continuous + discrete).
2. A per-feature generator is configured and used in all subsequent random sampling.
3. A configuration log is saved to:

```
./distribution_analysis/logsyrfd/{dataset}_thr{threshold}_{timestamp}.log
```

## Output

| Path | Content |
|---|---|
| `augmentation_results/{dataset}_new_tuples_{threshold}.csv` | Generated synthetic tuples |
| `distribution_analysis/logsyrfd/` | Distribution fitting log (one file per run) |
| `augmentation_logs/` | Performance log (time, memory, tuple count) |
