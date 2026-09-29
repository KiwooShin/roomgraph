# Dataset and scene design

## Scene layers

Keep structural geometry, furnishings, lighting, and camera trajectories separate.
Structural edges originate from architectural topology; they are not extracted
from RGB gradients. A stable scene identifier links all visual variants of the
same building. All such variants belong to the same train/validation/test split.

Room families will include kitchens, primary and smaller bedrooms, offices,
garages, corridors, and industrial spaces. Furnishing rules must encode support,
clearances, collisions, and room use: chairs near desks, cabinets along walls,
machines with service space, and navigable camera paths. Random scattering alone
does not produce a useful distribution of indoor scenes.

## Assets and scale

Use a versioned asset catalog with source, license, dimensions, units, semantic
class, and available levels of detail. Repeated assets should use USD instancing.
Large scenes should use references/payloads and separate spatial zones. Benchmark
load time, peak memory, and images per minute before scaling dataset generation.
Static captures do not require continuously simulating every object's physics.

The DGX Spark installation proves local feasibility; it does not establish
throughput for a heavily detailed factory. Preserve the ability to generate the
same scene on a larger rendering machine without changing the dataset schema.

## Annotation policy

Baseline structure is the building shell: walls, floors, ceilings, columns,
openings, and explicitly designated architectural partitions. Fixed cabinets and
machines are objects unless an experiment opts into another target definition.
Preserve their semantic identities to support future obstacle-mapping labels.

Distinguish visible edges from hidden projected structure. Keep geometry with no
observation separate from geometry inferred from evidence. Glass, mirrors, and
reflective machinery require explicit annotation policies and separate evaluation
because photometric appearance and opaque-surface visibility may disagree.

## Calibration and evaluation

World coordinates use metres and Z up. Exported camera coordinates use X right,
Y down, and Z forward. Image coordinates start at the top-left boundary, with
pixel centers at (i + 0.5, j + 0.5). Depth means distance along camera Z, not range.

Begin reconstruction with known poses and calibration. Use ground-truth edges
to test triangulation and plane fitting before substituting learned predictions.
Do not feed ground-truth depth into an inference pipeline described as RGB-only.
When poses are estimated from images, report failures and scale ambiguity.

Evaluate boundary precision/recall/F1 with pixel tolerance, corner localization,
3D edge accuracy/completeness, plane orientation, room dimensions, and connectivity.
Separate visible detection from occluded-structure completion. Report results by
room family, clutter, and occlusion rather than only an aggregate score.

## Milestone gates

- Installation: valid headless RGB/depth and independent calibration checks.
- Dataset pilot: valid geometry/placement, no split leakage, reproducible manifests.
- Training: tiny-set overfit, checkpoint reload, held-out evaluation and baselines.
- Reconstruction: oracle-edge baseline before predicted-edge results.
- Furnishing: paired captures, asset validation, mesh-aware visibility and profiling.
