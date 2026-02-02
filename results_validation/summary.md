# GLM 4.7 Flash Benchmark: Unsloth vs HuggingFace

**Generated:** 2026-01-31 21:55:48

**Git commit:** 9ab64773

**Model:** unsloth/GLM-4.7-Flash

**Steps per config:** 10

**GPU:** NVIDIA RTX PRO 6000 Blackwell Server Edition (95.0 GB)

## Batch Size Scaling

Fixed sequence length: 2048

![Batch Size Comparison](batch_size/comparison_batch_size.png)

### Results

| dimension   |   value | framework   |   batch_size |   seq_length |   tokens_per_second |   training_vram_gb |   step_time_ms |   peak_vram_gb |   final_loss | status   |
|:------------|--------:|:------------|-------------:|-------------:|--------------------:|-------------------:|---------------:|---------------:|-------------:|:---------|
| batch_size  |       1 | unsloth     |            1 |         2048 |              2898.6 |               0.97 |         1339.2 |          58.76 |       0.5512 | ok       |
| batch_size  |       1 | huggingface |            1 |         2048 |              2694.2 |               3.61 |         1440.8 |          61.38 |       0.6372 | ok       |

## Summary

### Batch Size Sweep

- **Average speedup:** 1.08x faster with Unsloth
- **Average VRAM reduction:** 4.3% less VRAM with Unsloth

