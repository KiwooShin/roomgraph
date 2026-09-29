# Head-camera structural perception

This experiment connects the furnished Isaac Sim generator to a supervised edge
model and a calibrated multi-view room-shell fitter. It is a synthetic pilot for
single rectangular rooms. Active navigation and general building reconstruction
are subsequent milestones.

## Data contract and split

`prepare_benchmark.py` generates 24 independent room-instance configurations:
four furnishing families × six variants. Each family contributes four training,
one validation, and one test room. Each room has 24 views, for 384/96/96 images.
No views of the same room cross splits; the exporter also rejects identical
structural geometry across splits. Dimensions, accent color, lighting, seed,
camera directions, and arm reach vary deterministically.

The architectural template retains one rectangular topology, one doorway, and
one window. Furniture assemblies translate with their anchors to accommodate
room dimensions; their meshes are not stretched. Assets and furnishing recipes
are shared across splits. These are held-out parameterized room instances, not
unseen building topologies or real-world homes. Camera samples and robot poses
are not collision-validated walking trajectories.

An original 1.68 m humanoid proxy has a pinhole camera at 1.585 m, forward arms,
a body, head, visor, and hands. The silhouette is inspired by the public height
of [1X NEO](https://www.1x.tech/neo). It is **not an official 1X asset, calibrated
NEO camera, kinematic model, or robot controller**. The body is rendered in 3D,
so arms can occlude edges and the robot can appear in mirrors. The camera is
outside the head surface. Mirrors do not create extra structural-edge labels.

The cache stores RGB and six binary label channels at 384 × 256:
visible structural edges, then full projected wall–wall, wall–floor,
wall–ceiling, doorway, and window edges. Renderer depth is used to create visible
training/evaluation labels only. It is never an input to the network or fitter.
The index retains source manifests, calibration, room IDs, and geometry/config
hashes. Sparse targets are reprojected at the network resolution, not reduced by
an interpolation that could erase one-pixel lines.

## Model and training

EdgeNet uses an ImageNet-pretrained ResNet-18 encoder and a compact skip-connected
decoder with six sigmoid outputs. Pretrained batch-normalization statistics are
frozen; decoder normalization uses GroupNorm. Training uses weighted BCE plus
Dice on one-pixel-dilated targets, photometric augmentation and horizontal flips,
AdamW, cosine learning-rate decay, BF16 automatic mixed precision, and gradient
clipping. Images/labels are cached as uint8 in RAM and converted by batch.

The best checkpoint is selected by validation loss. A single decision threshold
is then selected using the mean of visible and amodal F1 on validation only.
The test split is evaluated after selection. A separate eight-image overfit run
is a wiring diagnostic, never held-out performance. `best.pt`, resumable
`last.pt`, run provenance, JSON history, and TensorBoard events stay local.

## Metrics and baselines

We report precision, recall, and F1 for visible and typed amodal boundaries.
Predictions are skeletonized; matches use symmetric Euclidean distance within
2 pixels at 384 × 256. Matching is many-to-one, not a strict bipartite boundary
benchmark. Counts are pooled for micro scores, with per-channel and per-room
breakdowns. Background pixel accuracy is intentionally omitted.

Canny is a visible-edge baseline, with thresholds selected on validation. The
training-subset score uses every fourth training image, independently of the
held-out test set. GPU inference timing excludes image loading, CPU metrics,
and 3D fitting, and records warmup, batch size, resolution, and precision.

Reconstruction is evaluated with dimension MAE, plane-coordinate MAE, symmetric
sampled 3D edge Chamfer distance, and edge accuracy/completeness within 10 cm.
An oracle-edge reconstruction isolates fitting error. A training-median room
prior provides a non-image baseline. Ground truth is read for scoring only after
fitting has returned.

## Reconstruction assumptions

The fitter consumes predicted edge maps and known metric camera intrinsics/poses.
It estimates six axis-aligned planes from eight views using differential
evolution followed by local refinement of typed edge reprojection distances.
Bounds are derived from camera positions, not the true room dimensions. It
assumes known gravity/Manhattan axes and a single rectangular room.

Outputs include a JSON edge graph and a closed OBJ shell. Door/window pixels are
predicted but their 3D openings are not yet reconstructed. Edge support fractions
and supporting-view counts distinguish supported from inferred segments; they
measure agreement among amodal predictions, not confirmed surface observations.
They are **not calibrated uncertainty or proof of free space**. This
module does not estimate poses, perform SLAM, reconstruct furniture, or establish
collision-free robot motion.

## Run

Use a CUDA-enabled PyTorch build compatible with the host GPU. The DGX Spark
experiment uses the existing host Python/PyTorch environment; Isaac remains in
its separate container. Install the project with learning extras:

```bash
python -m pip install -e '.[learning,dev]'
python scripts/prepare_benchmark.py
./installation/run_python.sh scripts/render_furnished.py \
  --configs artifacts/benchmark-configs/*.json --output vis/benchmark_headcam
python scripts/build_learning_dataset.py vis/benchmark_headcam
python scripts/train_edges.py --overfit
python scripts/evaluate_edges.py --run artifacts/runs/headcam_v1_overfit \
  --output vis/perception_overfit --diagnostic
python scripts/train_edges.py
python scripts/evaluate_edges.py
python scripts/make_perception_report.py
python -m tensorboard.main --logdir artifacts/runs --host 127.0.0.1 --port 6006
```

Resume an interrupted run with `--resume artifacts/runs/headcam_v1/last.pt`.
Use `--run-dir` to start another experiment without overwriting existing outputs.
See [training efficiency](training-efficiency.md) for matched speed comparisons,
optional model/loss compilation, and checkpoint integrity checks.
For inference on another compatible calibrated capture, after evaluation has
written the validation-selected threshold:

```bash
python scripts/infer_room.py path/to/manifest.json --output vis/inferred_room
```

The future active-mapping loop needs pose estimation, persistent multi-room
association, uncertainty calibration, a separate obstacle/free-space map, and
safe next-view planning. A hidden wall inferred by this model is not an observed
surface; the robot should seek confirming observations rather than treating the
prediction as navigable space.
