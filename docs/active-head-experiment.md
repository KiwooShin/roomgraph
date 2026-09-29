# Active head-view reconstruction pilot

The next-view problem is implemented as a small planner around the existing RGB
edge model. A persistent evidence map records what has been looked at and which
parts of a fitted structural hypothesis have image support. The planner chooses
a feasible head target for the next three seconds, executes only the first
movement, receives another RGB image, and plans again.

This is a causal synthetic head-view pilot in one furnished validation office.
It is not a trained exploration policy, SLAM system, 1X controller, collision-free
walking demonstration, or general indoor occupancy reconstruction.

## Camera and body model

The robot base, torso, arms, and feet stay fixed while the proxy head, visor, and
lenses articulate. Optional camera entries contain:

```json
{
  "id": "H12",
  "focal_length_mm": 14,
  "robot_base": {"position": [1.2, 0, 0], "yaw_deg": 90},
  "head": {"yaw_deg": 0, "pitch_deg": 0}
}
```

The body uses +Y forward, +X right, +Z up. Positive yaw turns left; positive pitch
looks up. A rest camera at [0, 0.135, 1.585] m moves with the head about a neck
pivot at [0, 0, 1.43] m. Renderer camera matrices use +X right, +Y down, +Z forward.
The same articulated transform drives rendering, camera calibration, robot
geometry, manifests, and top-down direction markers. Legacy cameras remain valid.

The experiment samples yaw −80/−40/0/40/80° and pitch −30/0/30°. Joint speed,
acceleration, settling time, and limits are explicit configurable **simulation
assumptions**, not official 1X specifications. Camera offsets and optics likewise
belong to our original proxy, not a calibrated NEO model. Dense replay uses
independent simultaneous rest-to-rest triangular/trapezoidal joint profiles.

Head movement changes the optical center slightly because the lens is offset
from the neck pivot. Rotation about a fixed optical center provides no
triangulation baseline. This pilot therefore supplies the same three translated,
metrically calibrated starting views to every policy. It does not demonstrate
metric monocular reconstruction from head rotation alone.

## Evidence and next-view selection

The planner accepts candidate calibration and joint targets, previously acquired
model predictions, and a fitted room-shell hypothesis. It has no interface to
future candidate RGBs, renderer depth, true dimensions, or true furniture.

It keeps two different kinds of state:

- Equal-area directional cells distinguish looked-at directions from unseen
  directions. Looking at an occluder does not confirm the structure behind it.
- Sampled predicted shell edges distinguish unseen, looked-at but unsupported,
  single-camera-center support, and support from centers separated by at least
  0.2 m. Support requires both visible-edge and matching typed-edge predictions
  near the projected point. Amodal predictions alone cannot confirm visibility.

These are interpretable support heuristics, not calibrated uncertainty or proof
of correct 3D geometry. The shell is an initial hypothesis even where no camera
has confirmed it. The map does not establish navigable free space.

The fixed utility combines new directional coverage and unresolved projected
shell samples, discounts repeated inspection, and divides gain by head-motion
time with a small travel penalty. Weights were fixed before scoring this pilot.
A short hypothetical plan does not update evidence or assume future observations
will succeed. Actual evidence updates only when a selected image is acquired.
The initial shell geometry stays fixed during selection and is refitted at the
end; support and directional coverage update at every acquisition.

If head movement cannot obtain new useful evidence, the next stage needs a
translated viewpoint or a new body heading. This prototype can exhaust its head
candidates; it does not execute body navigation. Door/window predictions exist,
but 3D openings and connected-room discovery are still unimplemented.

## Comparison protocol

All policies use the same three translated bootstrap RGBs, initial head view,
15 candidate orientations, motion limits, and 12-second **simulated head-motion
plus settling budget**. We compare the active heuristic, a fixed feasible
serpentine scan, and random selection with seeds 7/17/27/37/47. A three-second
horizon limits each next action. Different travel distances yield different view
counts; the comparison is matched on simulated action time, not frame count.
Candidate RGBs/predictions are revealed only after selection. Shared inference
caching avoids computing the same selected frame repeatedly across policies
without exposing unseen predictions to any policy.

Ground truth is available only to evaluation. Coverage measures the length of
true architectural edges directly visible in at least one acquired view, using
finite positive renderer depth and a 2.5 cm tolerance at projected samples.
Samples are spaced approximately 2.5 cm apart and weighted by represented edge
length. Missing or infinite depth never counts as observed. Multi-view coverage
adds a ≥0.2 m camera-center separation requirement; it is not a triangulation
accuracy guarantee. Time-averaged coverage holds evidence constant between
acquisitions and includes the unused budget tail.

Final geometry scores compare the fitted six-plane shell with reference bounds.
They use RGB predictions and known metric poses for inference; true bounds are
read only for scoring. The existing held-out test split is not used for policy
selection. One development room cannot establish generalization or policy
superiority, even with several random trajectories.

## Reproducibility and speed

This experiment reuses the trained EdgeNet checkpoint and cached observations;
no new network or reinforcement-learning training is required. Head targets are
scored with small NumPy projections on CPU. GPU inference and reconstruction
latency are recorded separately from the simulated motion clock. The final
nonlinear geometry refit is deliberately outside the head-motion budget; this
is not a measured real-time closed-loop robot controller.

During development, repeated runs with automatically chosen BF16 cuDNN kernels
changed a few thresholded pixels and materially changed the three-view fit.
The final protocol fixes deterministic inference kernels and archives every
acquired probability tensor losslessly with its raw-array SHA256. Two independent
process runs then matched all prediction hashes, initial fitted bounds, selected
camera sequences, and final geometry scores exactly on this machine. This does
not promise bitwise equality on different GPUs or software versions.

The runner checks scene, furniture, lighting, camera-grid, body-pose and intrinsic
consistency, records capture/checkpoint hashes, requires a fresh run directory,
and writes completion markers/results atomically. Reports require a completed
experiment. Tests cover missing-depth handling, action budgets, actual head
kinematics, motion-profile velocity/acceleration bounds, support semantics, and
stale capture rejection.

## Run

```bash
PYTHONPATH=src python scripts/prepare_active_head.py
./installation/run_python.sh scripts/render_furnished.py \
  --configs artifacts/active-head-config.json --output vis/active_head/capture
PYTHONPATH=src python scripts/run_active_head.py --output vis/active_head/run_v2
PYTHONPATH=src python scripts/prepare_head_motion.py \
  --results vis/active_head/run_v2/experiment.json
./installation/run_python.sh scripts/render_furnished.py \
  --configs artifacts/active-head-motion.json --output vis/active_head/motion
PYTHONPATH=src python scripts/predict_head_motion.py \
  --experiment vis/active_head/run_v2/experiment.json \
  --manifest vis/active_head/motion/office_04_active_motion/manifest.json
PYTHONPATH=src python scripts/make_active_head_report.py \
  --results vis/active_head/run_v2/experiment.json \
  --motion vis/active_head/motion_predictions/motion.json \
  --output vis/active_head/report
```

For an independent repeat, use a new output directory and compare frozen inputs,
predictions, selected actions, and geometry:

```bash
PYTHONPATH=src python scripts/run_active_head.py --output vis/active_head/replay_check
python scripts/check_active_repeat.py \
  vis/active_head/run_v2/experiment.json vis/active_head/replay_check/experiment.json
```

The report also saves full-color replay composites. This host's FFmpeg build can
make the compact public MP4 without passing through GIF palette quantization:

```bash
ffmpeg -y -framerate 5 -i vis/active_head/report/replay/frame_%03d.webp \
  -c:v libopenh264 -b:v 2600k -pix_fmt yuv420p -movflags +faststart -an \
  vis/active_head/report/active_head.mp4
```

The frozen experiment is configured in
[`active_head_v1.json`](../configs/experiments/active_head_v1.json).
A fresh clone needs the existing furnished benchmark capture and trained
checkpoint first; see [the perception pipeline](perception.md). Choose a new
`--output` for repeated runs so results are not silently overwritten.

The dense 61-frame, 5 fps RGB replay is rendered **after policy evaluation**.
Intermediate images are for display only: they do not enter policy decisions,
map support updates, or benchmark metrics. The synchronized visualization shows
the actual changing RGB/edge overlay and top-down head direction; evidence maps
update at the original decision observations. All captures, checkpoints, raw
predictions, GIFs, and the full local report remain ignored by Git.

## Next experiment

Coverage is a useful first signal, but the next policy should prioritize expected
reduction in geometric uncertainty. Evaluate per-plane/edge uncertainty and
multi-view consistency, account for translation and occlusion, and compare
reconstruction error versus time across held-out development rooms. Then add
3D doorway openings and a separate occupancy/free-space map before connecting
this policy to body navigation. Calibrate hardware camera extrinsics and joint
limits when an actual robot interface becomes available.

Primary background: [Isler et al., ICRA 2016](https://www.ifi.uzh.ch/dam/jcr:7370c0bc-bfdc-4ede-8177-d3ffff7378c3/ICRA16_Isler.pdf)
studies movement-aware next-best-view utility;
[Delmerico et al., Autonomous Robots](https://rpg.ifi.uzh.ch/docs/AURO17_Delmerico.pdf)
evaluates information-gain measures against reconstruction completeness. This
pilot implements a simpler coverage heuristic, not either paper's full method.

## Measured development result

On DGX Spark / GB10, with the frozen existing model and the corrected finite-depth
evaluator, the following results were reproduced in a second independent process:

| Policy | Head images, including initial | Final directly visible edge coverage | Mean coverage over 12 s | Final dimension MAE |
| --- | ---: | ---: | ---: | ---: |
| active | 7 | 56.46% | 51.37% | 3.56 cm |
| raster | 9 | 49.86% | 45.67% | 1.72 cm |
| Random, mean of five seeds | 7–8 | 52.75% | 48.94% | 4.56 cm |

The common three-view initial hypothesis had 4.25 cm dimension MAE. The active
policy improved coverage over the fixed scan, but its final geometry error was
worse. One random trajectory also achieved better time-averaged coverage than
active selection. This establishes a working active-view loop, not a superior
reconstruction policy. The next score should account for geometric uncertainty
and useful translated baselines, not just new coverage.

Median active planning time was 0.34 ms. Deterministic BF16 model forward plus
probability transfer to CPU took 7.74 ms median after the first inference,
excluding image I/O. The active trajectory's final geometry fit took 3.48 s.
All seven policies together took about 30 s of experiment wall time, excluding
rendering; these timings are separate from the 12 s simulated movement budget.
Training was not rerun. The fast evidence loop and slower geometric backend
should be scheduled separately in a future live system.

The final source passed 63 CPU unit tests and Ruff checks. Isaac rendered all
15 candidate orientations and 61 dense replay frames successfully. The independent
replay matched lossless prediction hashes, bootstrap bounds, action sequences,
and geometry scores. Local `vis/active_head/repeatability.json` records that check.
