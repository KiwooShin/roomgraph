# First DGX Spark validation

## Environment

- Host: Ubuntu 24.04 ARM64, NVIDIA GB10, driver 580.159.03.
- Renderer: Isaac Sim 5.1.0, pinned source commit and compatibility fixes in README.
- Build/runtime: isolated Ubuntu 24.04 container with GCC 11; persistent user cache.
- NVIDIA driver access and Vulkan enumeration succeeded inside the container.
- Native release build succeeded after the documented compatibility fixes.

## Capture results

Four 640 × 480 calibrated RGB/depth/normal captures of a 6 × 5 × 3 m room with
20 inner-shell structural edges. Each image has a visible-edge mask, full projected
edge mask, and RGB overlay. The scene and camera metadata were exported.

| View | Capture time, seconds | Depth absolute error p95, metres | Valid sampled depth |
|---|---:|---:|---:|
| 000 | 49.32 | 0.00000213 | 100% |
| 001 | 1.36 | 0.00000231 | 100% |
| 002 | 1.39 | 0.00000136 | 100% |
| 003 | 1.40 | 0.00000180 | 100% |

Times include 12 capture steps with four render subframes per step, label
generation, and file export. View 000 includes first-use renderer/shader work.
These measurements are for a simple static scene and are not a throughput
prediction for furnished buildings or factories.

Depth was checked against independent analytic ray/box intersections. Tiny errors
here demonstrate matching units, camera conventions, and depth interpretation;
they do not establish real-world reconstruction accuracy. The contact sheet was
visually inspected and the structural overlays align with the rendered interior.

## Checks and outputs

- Five `unittest` cases pass: projection/basis, ray intersections, openings,
  invalid camera poses, and visible versus occluded edge labels.
- Ruff lint and formatting checks pass; shell scripts pass Bash syntax checking.
- Renderer integration exits successfully with `SMOKE TEST PASSED`.
- Self-contained HTML report includes orbitable structure/camera visualization;
  its rendered layout and graph were inspected in headless Chromium.

Local outputs: `artifacts/smoke/report.html`, `contact_sheet.jpg`, `manifest.json`,
`room.usda`, and per-view files. Outputs are ignored by Git and regenerable.

## Remaining work

This is the installation and empty-room smoke milestone. Parametric multi-room
generation, furnished assets, general mesh visibility, semantic masks, trained
models, and reconstruction are not implemented yet. The renderer uses default
real-time image processing; production data needs explicit quality and
antialiasing settings plus glass/mirror annotation policies.
