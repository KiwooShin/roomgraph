# Training efficiency and robustness

Treat time to useful validation quality as the optimization target. Record the
complete run cost, preserve the training objective, and check accuracy before
adopting a faster implementation. The original published test benchmark remains
frozen; efficiency experiments use the training and validation splits only.

## Reproduce a matched comparison

```bash
PYTHONPATH=src python scripts/benchmark_training.py \
  --output artifacts/runs/efficiency_v1 --variants eager compiled
PYTHONPATH=src python scripts/compare_training.py \
  --runs eager=artifacts/runs/efficiency_v1/eager \
         compiled=artifacts/runs/efficiency_v1/compiled \
  --output vis/training_efficiency
```

The launcher runs variants sequentially on the same GPU. Each gets an isolated,
initially empty Inductor and Triton cache, so compilation startup is included.
Both recipes retain 384 training images, batch size 16, the same 60-epoch cosine
schedule, and 1,440 optimizer steps. Patience is set to the epoch count to avoid
confounding speed with early stopping. The model architecture, RGB resolution,
loss, augmentations, precision, validation frequency, and checkpoint cadence
are unchanged. `--variants eager loss compiled` also measures compiling only
the loss. Use a fresh output directory for every repetition; `--seed` supports
additional paired trials.

The comparison script checks dataset identity, training work, learning-rate
history, and the reported environment. It loads only `val.npz`, evaluates each
run's best validation-loss checkpoint, and uses a fixed 0.7 threshold. It does
not optimize thresholds, load test images, or rerun test reconstruction.
Overlays use the middle camera in each validation room, selected independently
of prediction quality. Reports, plots, and overlays stay in ignored `vis/`.

Timing fields have different scopes:

- `launcher_wall_seconds`: subprocess startup, Python imports, dataset/model
  initialization, compilation, training, validation, checkpointing, logging,
  and process exit. Use this to judge complete experiment cost.
- `process_seconds`: starts inside `main()`, excluding interpreter/import time.
- `total_seconds`: training session after initialization; includes compilation,
  validation, checkpointing, and logging.
- History `seconds` / `images_per_second`: the training loop only, including
  batching, augmentation, forward/backward, clipping and optimizer updates.
- `peak_allocated_mb`: maximum CUDA tensor allocation, in MiB; not total GPU
  reservation, driver memory, or DGX Spark system RAM.

CUDA work is synchronized at epoch boundaries. First-epoch compilation is never
removed from whole-run timing. Steady throughput is the median of epochs after
the first and is labeled separately. Benchmark fresh uninterrupted runs: resumed
session timing omits the serialization tail of the checkpoint that was restored.

## Implemented choices

The trainer retains BF16 autocast, channels-last convolutions, a fused AdamW
optimizer on CUDA, four CPU compute threads, and compact uint8 arrays in RAM.
It transfers and converts one batch at a time, with photometric augmentation on
the GPU. The optional compiled path wraps training forward and loss functions;
validation and saved model state use the original module, so checkpoints remain
compatible with ordinary inference. Validation stays eager to avoid compiling
an additional inference graph for this small dataset.

`compile_model` and `compile_loss` are independent JSON options and default to
false for the existing recipe. See the measured comparison below before choosing
a configuration. Compilation can change floating-point rounding; matching the
seed and updates does not imply identical final weights. Very short diagnostics
may finish sooner in eager mode. PyTorch's
[compilation tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_compile_full_example.html)
also distinguishes compilation startup from subsequent iteration speed.

Loss totals remain on-device until the epoch ends. A single per-batch finite
check covers both loss and gradient norm before the optimizer changes weights.
This replaces the previous separate loss-check and loss-report synchronizations
while adding gradient protection. Finite checking and validation are retained.

## Interruption and experiment integrity

An epoch-seeded sampler owns its RNG independently of DataLoader worker seeding.
An interrupted run resumes the same sample order at an epoch boundary, including
when persistent workers are recreated. Python, NumPy, CPU and CUDA RNG states,
optimizer, scheduler, configuration, and dataset-index identity are checkpointed.
This is not a claim of bitwise reproducible CUDA arithmetic or mid-batch recovery.

Checkpoint and JSON writes use a temporary file in the destination directory,
flush/fsync, and atomic replacement. On validation improvement, the trainer
commits `last.pt` before publishing `best.pt`; resume can repair an interrupted
best-checkpoint publication from the committed state. An irrecoverable mismatch
is rejected. Unit tests simulate serialization and replacement failures.

The trainer refuses to overwrite a populated run directory and rejects resumed
changes to the training recipe, dataset index, or diagnostic status. Resume must
use the original output directory. Legacy `last.pt` files without configuration
and dataset identity are rejected; their `best.pt` files still work for inference
and evaluation. A completed run exits without rerunning training.

```bash
PYTHONPATH=src python scripts/train_edges.py --config path/to/config.json \
  --run-dir artifacts/runs/my_experiment
PYTHONPATH=src python scripts/train_edges.py --config path/to/config.json \
  --run-dir artifacts/runs/my_experiment \
  --resume artifacts/runs/my_experiment/last.pt
```

`checkpoint_every` defaults to one epoch. Even with a larger interval, every
validation improvement and the final epoch save a resumable checkpoint. A larger
interval trades less disk I/O for more repeated work after interruption.

## Rules for subsequent experiments

1. Start with a small wiring diagnostic, then a representative training/validation
   pilot. Promote a change only after measuring whole-run cost and boundary F1.
2. Keep room/building splits and freeze test evaluation for milestone releases.
   Extend robustness checks to unseen layouts/assets, stronger clutter, mirrors,
   arm occlusion, camera noise, and real captures; this pilot does not establish
   those capabilities.
3. Compare the same objective and data/update budget when claiming a systems
   speedup. Larger batches, smaller images, fewer views, or fewer updates are
   learning changes and need a separate quality-versus-time study.
4. Retain cached labels and render outside the training loop. As data grows,
   benchmark sharded or memory-mapped caches and prefetching before increasing
   worker counts or copying an entire dataset onto the GPU.
5. Consider progressive resolution or teacher/student distillation only when the
   next bottleneck justifies them. Reproject thin-edge targets and camera
   calibration at each resolution; do not erase labels by naive mask resizing.
6. Repeat promising comparisons across seeds and trial order before treating
   small timing or accuracy differences as robust improvements. Use TensorBoard
   for loss, learning rate, throughput, memory, timing, and validation overlays.

## Measured DGX Spark comparison

Measured on NVIDIA GB10 with PyTorch 2.11.0+cu128, 384 × 256 RGB,
batch size 16 and one matched seed (42). Both runs completed 60 epochs /
1,440 updates and selected epoch 59 by validation loss. This is a separate
validation-only experiment; the original test results were not reevaluated.

| Measurement | Eager | Compiled model + loss |
| --- | ---: | ---: |
| Complete subprocess wall time | 246.75 s | 214.90 s |
| Training session including cold compilation | 243.59 s | 210.50 s |
| Steady median throughput (epochs 2–60) | 110.51 images/s | 148.72 images/s |
| Peak CUDA tensor allocation | 2515.23 MiB | 2210.97 MiB |
| Best validation loss | 0.396902 | 0.396956 |
| Validation visible boundary F1 | 0.923564 | 0.922232 |
| Validation amodal boundary F1 | 0.963186 | 0.963018 |
| Session time to validation loss ≤ 0.5 | 159.77 s | 148.41 s |

The compiled run used **12.9% less wall time**, delivered **34.6% higher steady
throughput**, and allocated **12.1% less peak tensor memory**. Its visible F1 was
0.13 percentage points lower and amodal F1 was 0.017 points lower at the fixed
0.7 threshold. One paired seed does not establish statistical quality equivalence.
The target loss was reached at the same scheduled evaluation (epoch 39) in both
runs; the compiled run reached it 11.36 seconds earlier on the session clock.

Use the compiled recipe for subsequent full pilot runs on this environment:

```bash
PYTHONPATH=src python scripts/train_edges.py \
  --config configs/training/headcam_v1_fast.json
```

Keep eager mode for very short wiring checks, and remeasure compilation on a new
GPU, model, resolution, or batch shape. The first training epoch took 6.15 seconds
in eager mode and 31.91 seconds compiled; those times include all first-epoch
work, not only compiler execution. Short screening also tested zero DataLoader
workers and an entire uint8 dataset cached on the GPU. Removing workers did not
help; the GPU cache's small throughput gain did not justify extra memory and a
second data-loading path for this pilot. The released fast recipe retains the
CPU cache, two workers, and the original batch size.

Local artifacts: `vis/training_efficiency/report.html` includes curves, exact
measurements and furnished validation overlays. The full runs, logs, cold caches,
and checkpoints are in `artifacts/runs/efficiency_v1/`. The source repository
contains the benchmark/report commands and configurations, not these large files.
The public project page includes selected charts and sanitized measurements.

Validation completed with 34 unit tests, Ruff lint/format checks, the paired full
training runs, and a completed-run resume check that preserved all checkpoint
and metadata hashes. CUDA forward/backward execution is covered by the actual
training runs; unit tests run on CPU.
