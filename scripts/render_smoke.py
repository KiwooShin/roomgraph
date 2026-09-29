"""Render a calibrated room with Isaac Sim; run using installation/run_python.sh."""

import argparse
import json
import time
from pathlib import Path


def main():
    """Create one room, capture four views, and verify rendered metric depth."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/smoke"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Kit must initialize before importing USD and Omniverse extensions.
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
    try:
        import numpy as np
        import omni.replicator.core as rep
        import omni.usd
        from PIL import Image, ImageDraw
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

        from roomgraph.geometry import demo_room, edge_masks, look_at, ray_depth, scene_metadata

        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        boxes, edges = demo_room()
        for box in boxes:
            prim = UsdGeom.Cube.Define(stage, f"/World/Structure/{box.name}")
            prim.CreateSizeAttr(1.0)
            lo, hi = np.array(box.lower), np.array(box.upper)
            prim.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2)))
            prim.AddScaleOp().Set(Gf.Vec3f(*(hi - lo)))
            material = UsdShade.Material.Define(stage, f"/World/Materials/{box.name}")
            shader = UsdShade.Shader.Define(stage, f"/World/Materials/{box.name}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*box.color))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(material)

        dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Sky")
        dome.CreateIntensityAttr(300)
        for index, position in enumerate([(-1.5, -1, 2.7), (1.5, 1, 2.7)]):
            light = UsdLux.SphereLight.Define(stage, f"/World/Lights/Interior_{index}")
            light.AddTranslateOp().Set(Gf.Vec3d(*position))
            light.CreateRadiusAttr(0.15)
            light.CreateIntensityAttr(20000)
            light.CreateColorAttr(Gf.Vec3f(1.0, 0.93, 0.85))

        width, height = args.width, args.height
        focal_pixels = width * 0.75
        intrinsics = np.array(
            [[focal_pixels, 0, width / 2], [0, focal_pixels, height / 2], [0, 0, 1]]
        )
        camera = UsdGeom.Camera.Define(stage, "/World/Camera")
        camera.CreateFocalLengthAttr(18.0)
        camera.CreateHorizontalApertureAttr(24.0)
        camera.CreateVerticalApertureAttr(24.0 * height / width)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100))
        camera.CreateFStopAttr(0)
        camera_transform = camera.AddTransformOp()
        product = rep.create.render_product(str(camera.GetPath()), (width, height))
        annotators = {}
        for name in ["rgb", "distance_to_image_plane", "normals"]:
            annotator = rep.AnnotatorRegistry.get_annotator(name)
            annotator.attach(product)
            annotators[name] = annotator

        views = [
            ((-2, -1.8, 1.5), (0, 2, 1.4)),
            ((-2, 1.5, 1.5), (2, 0, 1.5)),
            ((2, -1.8, 1.6), (-1.3, 2.5, 1.4)),
            ((1.8, 1.8, 1.4), (-1, -2, 1.5)),
        ]
        manifest = scene_metadata(boxes, edges)
        manifest.update({"renderer": "Isaac Sim 5.1.0", "views": []})
        panels, errors = [], []
        for index, (eye, target) in enumerate(views):
            start = time.monotonic()
            pose = look_at(np.array(eye), np.array(target))
            # USD cameras look down -Z with +Y up; Gf matrices use row vectors.
            usd_pose = pose @ np.diag([1, -1, -1, 1])
            camera_transform.Set(Gf.Matrix4d(usd_pose.T.tolist()))
            for _ in range(12):
                rep.orchestrator.step(rt_subframes=4)
            rgb = np.asarray(annotators["rgb"].get_data())[..., :3].copy()
            depth = np.asarray(annotators["distance_to_image_plane"].get_data()).copy()
            normals = np.asarray(annotators["normals"].get_data()).copy()
            if rgb.shape != (height, width, 3) or rgb.std() < 2:
                raise RuntimeError(f"Invalid or blank RGB frame: {rgb.shape}")
            depth = depth.reshape(height, width)
            visible, full = edge_masks(edges, boxes, intrinsics, pose, width, height)

            # Sparse independent ray/solid checks catch pose, unit, and depth errors.
            yy, xx = np.mgrid[20 : height - 20 : 17, 20 : width - 20 : 17]
            pixels = np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5, np.ones(xx.size)])
            directions = (pixels @ np.linalg.inv(intrinsics).T) @ pose[:3, :3].T
            expected = ray_depth(pose[:3, 3], directions, boxes)
            measured = depth[yy.ravel(), xx.ravel()]
            finite = np.isfinite(expected) & np.isfinite(measured) & (measured > 0)
            valid_fraction = float(finite.sum() / max(1, np.isfinite(expected).sum()))
            if valid_fraction < 0.95:
                raise RuntimeError(f"Rendered depth coverage too low: {valid_fraction:.3f}")
            if finite.sum() < 100:
                raise RuntimeError("Insufficient valid rendered depth samples")
            error = np.abs(expected[finite] - measured[finite])
            p95 = float(np.quantile(error, 0.95))
            errors.append(p95)

            stem = f"view_{index:03d}"
            Image.fromarray(rgb).save(args.output / f"{stem}_rgb.png")
            Image.fromarray(visible).save(args.output / f"{stem}_edges_visible.png")
            Image.fromarray(full).save(args.output / f"{stem}_edges_full.png")
            np.savez_compressed(args.output / f"{stem}_geometry.npz", depth=depth, normals=normals)
            overlay = rgb.copy()
            overlay[visible > 0] = [0, 255, 200]
            Image.fromarray(overlay).save(args.output / f"{stem}_overlay.png")
            panel = Image.new("RGB", (2 * width, height + 32), "#101820")
            panel.paste(Image.fromarray(rgb), (0, 32))
            panel.paste(Image.fromarray(overlay), (width, 32))
            ImageDraw.Draw(panel).text(
                (12, 10), f"ROOMGRAPH / {stem}    RGB + structural edges", fill="white"
            )
            panels.append(panel)
            manifest["views"].append(
                {
                    "id": stem,
                    "width": width,
                    "height": height,
                    "intrinsics": intrinsics.tolist(),
                    "camera_to_world": pose.tolist(),
                    "camera_axes": "X right, Y down, Z forward",
                    "pixel_coordinates": "top-left boundary origin; centers at i+0.5,j+0.5",
                    "depth_convention": "optical-axis depth in metres",
                    "depth_p95_absolute_error_m": p95,
                    "depth_checked_pixels": int(finite.sum()),
                    "depth_valid_fraction": valid_fraction,
                    "capture_seconds": time.monotonic() - start,
                    "visible_edge_pixels": int(np.count_nonzero(visible)),
                }
            )
            print(f"{stem}: depth p95 error={p95:.6f} m", flush=True)

        stage.GetRootLayer().Export(str((args.output / "room.usda").resolve()))
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        sheet = Image.new("RGB", (2 * width, len(panels) * (height + 32)))
        for index, panel in enumerate(panels):
            sheet.paste(panel, (0, index * (height + 32)))
        sheet.save(args.output / "contact_sheet.jpg", quality=93)
        if max(errors) > 0.03:
            raise RuntimeError(f"Depth calibration failed: p95 errors={errors}")
        print(f"SMOKE TEST PASSED: {args.output}", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
