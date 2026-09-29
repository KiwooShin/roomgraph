# Measured head-camera pilot: rooms_v1

One seeded experiment, not a broad generalization benchmark. The test set is
96 images from four held-out parameterized room instances. All splits share the
same rectangular topology and asset recipes. Camera poses and Manhattan axes are
known; renderer depth is excluded from inference.

## Edge detection

| Split / method | Visible F1 | Typed amodal F1 |
| --- | ---: | ---: |
| Training subset (96 images) | 0.927 | 0.998 |
| Validation (96 images) | 0.927 | 0.959 |
| Test (96 images) | 0.916 | 0.958 |
| Canny, test | 0.066 | — |

Metrics use skeletonized predictions and 2-pixel Euclidean tolerance at 384 × 256,
with many-to-one matching allowed. Checkpoint selection uses validation loss;
the decision threshold (0.7) uses validation F1 only. Canny is a
generic image-edge baseline, not a trained architectural detector.

Test visible precision/recall are 0.882 /
0.953; amodal precision/recall are
0.970 / 0.947.
Visible false positives and amodal misses remain. No real-camera evaluation has
been performed.

## Calibrated reconstruction

| Room | Visible F1 | Amodal F1 | Dimension MAE (m) | Edge Chamfer (m) |
| --- | ---: | ---: | ---: | ---: |
| bedroom_05 | 0.910 | 0.965 | 0.0432 | 0.0353 |
| kitchen_05 | 0.927 | 0.944 | 0.0179 | 0.0194 |
| living_05 | 0.913 | 0.938 | 0.0279 | 0.0263 |
| office_05 | 0.918 | 0.982 | 0.0029 | 0.0123 |

Mean dimension MAE is 0.0230 m;
oracle-edge fitting yields 0.0161 m;
a training-median room prior yields 0.3378 m.
The oracle is limited by rasterization and the fitted closed-shell model. The
learned fitter uses eight views per room and reconstructs six planes; it does
not recover openings, furniture, or arbitrary building connectivity.

## Efficiency and artifacts

The model has 12,524,606 parameters with an ImageNet-pretrained
ResNet-18 encoder. Training completed 60 epochs in
247.1 seconds on NVIDIA GB10, selecting epoch 59.
BF16 batch-one GPU forward latency: median 3.29 ms,
p95 3.83 ms (10 warmup and 30 timed iterations).
These timings exclude image I/O, CPU postprocessing, and reconstruction.

Best checkpoint SHA-256:
`bbdad608aa594122563b9e7aebac0c1415a82e9735f29521af660d5ceb21e022`.

Local outputs:

- `artifacts/runs/headcam_v1/`: checkpoints, JSON history, TensorBoard events.
- `artifacts/datasets/headcam_v1/`: validated train/val/test caches and index.
- `vis/perception/report.html`: predictions, measured curves, and 3D comparison.
- `vis/perception/summary.json`: full metrics, per-room results, and run provenance.
- `vis/inference_smoke/`: standalone RGB/calibration inference and OBJ export.
- `vis/headcam_demo/`: rendered robot self-reflection verification.

The humanoid is an original approximate proxy, not an official 1X model. Robot
motions and camera locations are not collision-validated. This work does not
claim autonomous navigation, calibrated uncertainty, or physical robot accuracy.
