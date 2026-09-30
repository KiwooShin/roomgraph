# Real edge preprocessing diagnostic

The frozen `headcam_v1` model was evaluated on ARKitScenes development video
42445021 (visit 421380), using the same 240 input timestamps as the RGB-D pilot.
No model weights or threshold were changed. This is a preprocessing diagnostic,
not a new real-world accuracy benchmark or a training run.

Five alternatives use stretch 384×256, letterbox 384×256, portrait 192×256,
portrait 384×512, and VGA-source portrait 384×512. The first four use lowres RGB;
the last uses the same wide camera at 640×480. There are 121 exact timestamp
matches. VGA rays are aligned with a calibration homography before upright
rotation. No high-resolution reference depth enters prediction. All probabilities
are restored to the same upright 192×256 grid before thresholding at 0.7.

The full run took 41.47 seconds, including loading, image processing, inference,
compressed prediction caching and comparison images, excluding download. Peak
allocated tensor memory was 172.15 MiB. Timings in summary.json include forward
and output transfer; methods execute in fixed order with changing shapes, so they
are diagnostics, not an isolated speedup benchmark. Disk I/O and initialization
are included in full-run wall time.

Qualitative review is mixed. In the selected doorway view, larger inputs retain
some top-of-door edges but miss jambs. In the ceiling-corner view, all methods miss
much of the boundary; the portrait method adds a short corner fragment. Greater
resolution does not consistently recover the missing structure. No default was
changed and no fine-tuning was performed.

Eight evenly spaced exact-match frames form an annotation review pack. AI-drafted
polylines and selected regions are local **provisional annotations**, not verified
ground truth. They were drafted by inspecting RGB; no automatic edge detector or
geometry reference generated the labels. Regions include structural boundaries
and negative regions. Endpoints, semantics, and completeness need independent
review. The browser editor permits correction and records reviewer identity.
The evaluator refuses drafts and unreviewed frames. It measures thin visible-edge
precision/recall/F1 at two native pixels on the same paired frames for all methods,
ignoring unreviewed pixels and an interior margin along reviewed-region borders.
Typed/amodal accuracy is not evaluated by this diagnostic scorer.

All frames from this visit remain development data. Do not split these eight
nearby views between training and test. Real fine-tuning needs reviewed labels
from separate training visits and a separate frozen test visit before a claim
of generalization is justified. Synthetic test F1 and depth surface-agreement F1
are not real structural-edge accuracy.

## Reproduce

```bash
python scripts/download_arkitscenes.py --output artifacts/real/arkitscenes_vga \
  --assets vga_wide.zip vga_wide_intrinsics.zip
PYTHONPATH=src python scripts/compare_real_edges.py \
  --output vis/arkitscenes/preprocessing_new
```

Open `vis/arkitscenes/preprocessing_v2/index.html` for the completed comparisons.
Open its `review/index.html`, load `annotations.ai-draft.json` (or the blank
`annotations.json`), correct regions and lines, and export reviewed JSON. The
AI draft was created separately from inference and is not automatically reproduced
by the comparison command. Local image and label files remain Git-ignored.

```bash
PYTHONPATH=src python scripts/evaluate_real_edges.py \
  --run vis/arkitscenes/preprocessing_v2 \
  --annotations /path/to/annotations.reviewed.json \
  --output vis/arkitscenes/preprocessing_v2/reviewed_metrics.json
```

The scripts refuse overwriting existing runs or metric files. Source, checkpoint,
input RGB, VGA calibration, and prediction cache hashes are recorded. The earlier
RGB-D pilot and its frozen source files remain intact.
