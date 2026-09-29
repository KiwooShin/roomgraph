"""Render JSON-configured furnished rooms, perspective views, and camera floor plans."""

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs", nargs="+", type=Path, default=sorted(Path("configs/scenes").glob("*.json"))
    )
    parser.add_argument("--output", type=Path, default=Path("vis/furnished"))
    parser.add_argument("--samples", type=int, help="Override path-tracing samples per pixel")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "PathTracing"})
    try:
        import carb
        import numpy as np
        import omni.replicator.core as rep
        import omni.usd
        from isaacsim.core.utils.semantics import add_update_semantics
        from PIL import Image, ImageDraw, ImageFilter, ImageFont
        from pxr import Gf, UsdGeom, UsdLux

        from roomgraph.furnishings import Part
        from roomgraph.geometry import depth_edge_masks, scaled_room
        from roomgraph.headcam import (
            HEAD_PART_NAMES,
            camera_rig_pose,
            proxy_parts,
            resolve_camera_pose,
            robot_pose,
        )
        from roomgraph.scene_config import compiled_manifest, load_scene
        from roomgraph.usd_furnishing import SceneBuilder
        from roomgraph.visualization import camera_triangle, floorplan_pixel

        settings = carb.settings.get_settings()
        settings.set("/rtx/pathtracing/maxBounces", 6)
        settings.set("/rtx/pathtracing/optixDenoiser/enabled", True)
        settings.set("/rtx/post/aa/op", 0)
        settings.set("/rtx/post/tonemap/filmIso", 100.0)
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        index = []
        for config_path in args.configs:
            config, parts = load_scene(config_path)
            name = config["name"]
            output = args.output / name
            output.mkdir(parents=True, exist_ok=True)
            if stage.GetPrimAtPath("/World"):
                stage.RemovePrim("/World")
            builder = SceneBuilder(stage, "assets/cache")
            samples = args.samples or config["render"]["samples_per_pixel"]
            settings.set("/rtx/pathtracing/spp", min(samples, 32))
            settings.set("/rtx/pathtracing/totalSpp", samples)
            dimensions = np.asarray(config["room"]["dimensions_m"])
            shell_scale = dimensions / [6, 5, 3]
            boxes, edges = scaled_room(dimensions)
            builder.material(
                "wall", (0.78, 0.76, 0.69), texture="white_plaster_02", diffuse_texture=False
            )
            builder.material(
                "accent",
                config["room"]["accent_color"],
                texture="white_plaster_02",
                diffuse_texture=False,
            )
            builder.material("floor", (0.6, 0.4, 0.2), 0.38, texture="wood_floor")
            for box in boxes:
                center = tuple((np.array(box.lower) + box.upper) / 2)
                size = tuple(np.array(box.upper) - box.lower)
                material = "floor" if box.name == "floor" else "wall"
                if box.name.startswith("door_"):
                    material = "accent"
                prim = builder.part(
                    Part(
                        box.name,
                        box.name,
                        "architecture",
                        "rounded_box",
                        center,
                        size,
                        material,
                        radius=0.001,
                    ),
                    root="/World/Architecture",
                )
                add_update_semantics(prim.GetPrim(), "architecture")
            # Interior trim: window reveal, door frame and baseboards.
            trim = [
                ("door_left", (-2.045, 2.46, 1.12), (0.09, 0.09, 2.24)),
                ("door_right", (-0.955, 2.46, 1.12), (0.09, 0.09, 2.24)),
                ("door_top", (-1.5, 2.46, 2.235), (1.18, 0.09, 0.09)),
                ("window_sill", (2.94, 0, 0.985), (0.24, 2.18, 0.075)),
                ("window_top", (2.96, 0, 2.24), (0.10, 2.16, 0.08)),
                ("window_left", (2.96, -1.045, 1.60), (0.10, 0.09, 1.22)),
                ("window_right", (2.96, 1.045, 1.60), (0.10, 0.09, 1.22)),
                ("base_west", (-2.975, 0, 0.055), (0.05, 5, 0.11)),
                ("base_east", (2.975, 0, 0.055), (0.05, 5, 0.11)),
                ("base_south", (0, -2.475, 0.055), (6, 0.05, 0.11)),
            ]
            for trim_name, center, size in trim:
                builder.part(
                    Part(
                        trim_name,
                        trim_name,
                        "trim",
                        "rounded_box",
                        tuple(np.asarray(center) * shell_scale),
                        tuple(np.asarray(size) * shell_scale),
                        "white",
                        radius=0.003,
                    ),
                    root="/World/Trim",
                )
            # Replace selected object recipes with real textured mesh assets.
            grouped = {}
            for part in parts:
                grouped.setdefault(part.object_id, []).append(part)
            imports = []
            for object_id, items in grouped.items():
                category = items[0].category
                asset = None
                override = config["asset_overrides"]
                if category == "plant":
                    asset = override.get("plants")
                    pot = next(p for p in items if p.name.endswith("/pot"))
                    bottom = pot.center[2] - pot.size[2] / 2
                    origin = (pot.center[0], pot.center[1], bottom)
                    extent = max(p.center[2] + p.size[2] / 2 for p in items) - bottom
                    dimension, yaw = "height", 0
                elif category == "sofa":
                    asset = override.get("sofa")
                    base = items[0]
                    origin = (base.center[0], base.center[1], 0)
                    extent, dimension, yaw = 2.40, "width", base.rotation[2]
                elif object_id.startswith("lounge_chair"):
                    asset = override.get("lounge_chairs")
                    origin = (*items[0].center[:2], 0)
                    extent, dimension, yaw = 0.88, "height", items[0].rotation[2] + 180
                elif object_id.startswith("dining_chair"):
                    asset = override.get("dining_chairs")
                    origin = (*items[0].center[:2], 0)
                    extent, dimension, yaw = 0.91, "height", items[0].rotation[2] + 180
                if asset:
                    imports.append(
                        builder.imported(asset, object_id, origin, extent, yaw, dimension)
                    )
                else:
                    for part in items:
                        builder.part(part)
                add_update_semantics(
                    stage.GetPrimAtPath("/World/Furnishings/" + object_id), category
                )

            robot_transform = None
            head_transform = None
            if config.get("headcam", {}).get("enabled", False):
                builder.material(
                    "robot_shell",
                    (0.60, 0.56, 0.48),
                    0.85,
                    texture="denim_fabric",
                    diffuse_texture=False,
                )
                builder.material("robot_dark", (0.025, 0.028, 0.03), 0.5)
                root = UsdGeom.Xform.Define(stage, "/World/Robot")
                robot_transform = root.AddTransformOp()
                head_root = UsdGeom.Xform.Define(stage, "/World/Robot/HeadRig")
                head_transform = head_root.AddTransformOp()
                for part in proxy_parts(config["headcam"].get("arm_reach_m", 0.52)):
                    part_root = (
                        "/World/Robot/HeadRig" if part.name in HEAD_PART_NAMES else "/World/Robot"
                    )
                    builder.part(part, root=part_root)
                add_update_semantics(root.GetPrim(), "robot")

            dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Sky")
            dome.CreateIntensityAttr(config["lighting"]["daylight_intensity"])
            dome.CreateColorAttr(Gf.Vec3f(0.82, 0.90, 1.0))
            # Large invisible area emitters produce soft daylight and ceiling bounce.
            window_light = UsdLux.RectLight.Define(stage, "/World/Lights/Window")
            window_light.AddTranslateOp().Set(Gf.Vec3d(*(np.array([3.35, 0, 1.7]) * shell_scale)))
            window_light.AddRotateYOp().Set(90)
            window_light.CreateWidthAttr(2.0)
            window_light.CreateHeightAttr(1.5)
            window_light.CreateIntensityAttr(config["lighting"]["daylight_intensity"] * 1.6)
            ceiling_light = UsdLux.RectLight.Define(stage, "/World/Lights/Ceiling")
            ceiling_light.AddTranslateOp().Set(Gf.Vec3d(0, 0, float(dimensions[2] - 0.08)))
            ceiling_light.CreateWidthAttr(3.0)
            ceiling_light.CreateHeightAttr(2.0)
            ceiling_light.CreateIntensityAttr(config["lighting"]["ceiling_intensity"])
            ceiling_light.CreateEnableColorTemperatureAttr(True)
            ceiling_light.CreateColorTemperatureAttr(config["lighting"]["color_temperature_kelvin"])

            width, height = config["render"]["width"], config["render"]["height"]
            camera = UsdGeom.Camera.Define(stage, "/World/Camera")
            transform = camera.AddTransformOp()
            camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100))
            camera.CreateFStopAttr(0)
            product = rep.create.render_product(str(camera.GetPath()), (width, height))
            annotators = {}
            for key in ("rgb", "distance_to_image_plane", "normals"):
                annotators[key] = rep.AnnotatorRegistry.get_annotator(key)
                annotators[key].attach(product)
            manifest = compiled_manifest(config, parts)
            manifest.update(
                {
                    "renderer": "Isaac Sim 5.1.0 / PathTracing",
                    "samples_per_pixel": samples,
                    "source_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    "structural_edges": [asdict(e) for e in edges],
                    "imported_objects": imports,
                    "visibility_policy": "Depth test (2.5 cm tolerance); no reflected edges",
                    "views": [],
                }
            )
            for i, view in enumerate(config["cameras"]):
                started = time.monotonic()
                focal = view["focal_length_mm"]
                camera.CreateProjectionAttr("perspective")
                camera.CreateFocalLengthAttr(focal)
                camera.CreateHorizontalApertureAttr(24)
                camera.CreateVerticalApertureAttr(24 * height / width)
                rig = camera_rig_pose(view)
                pose = resolve_camera_pose(view)
                transform.Set(Gf.Matrix4d((pose @ np.diag([1, -1, -1, 1])).T.tolist()))
                if robot_transform is not None:
                    body_pose = rig.base_to_world if rig is not None else robot_pose(pose)
                    head_pose = rig.head_to_base if rig is not None else np.eye(4)
                    robot_transform.Set(Gf.Matrix4d(body_pose.T.tolist()))
                    head_transform.Set(Gf.Matrix4d(head_pose.T.tolist()))
                for _ in range(2):
                    rep.orchestrator.step(rt_subframes=4)
                rgb = np.asarray(annotators["rgb"].get_data())[..., :3].copy()
                depth = (
                    np.asarray(annotators["distance_to_image_plane"].get_data())
                    .reshape(height, width)
                    .copy()
                )
                if rgb.shape != (height, width, 3) or rgb.std() < 3:
                    raise RuntimeError(f"Blank or invalid capture: {name}/{view['id']}")
                if np.count_nonzero(np.isfinite(depth) & (depth > 0)) < width * height * 0.5:
                    raise RuntimeError(f"Invalid depth capture: {name}/{view['id']}")
                intrinsics = np.array(
                    [
                        [width * focal / 24, 0, width / 2],
                        [0, width * focal / 24, height / 2],
                        [0, 0, 1],
                    ]
                )
                visible, full = depth_edge_masks(edges, intrinsics, pose, depth)
                hidden = np.where((full > 0) & (visible == 0), 255, 0).astype(np.uint8)
                stem = f"view_{i:03d}"
                Image.fromarray(rgb).save(output / f"{stem}_rgb.png")
                Image.fromarray(visible).save(output / f"{stem}_visible.png")
                Image.fromarray(hidden).save(output / f"{stem}_hidden.png")
                Image.fromarray(full).save(output / f"{stem}_full.png")
                overlay = rgb.copy()
                mask = np.array(Image.fromarray(visible).filter(ImageFilter.MaxFilter(3))) > 0
                overlay[mask] = [50, 255, 195]
                hidden_mask = np.array(Image.fromarray(hidden).filter(ImageFilter.MaxFilter(3))) > 0
                yy, xx = np.indices(hidden.shape)
                overlay[hidden_mask & (((xx + yy) // 8) % 2 == 0)] = [255, 110, 115]
                Image.fromarray(overlay).save(output / f"{stem}_overlay.png")
                np.savez_compressed(
                    output / f"{stem}_geometry.npz",
                    depth=depth,
                    normals=np.asarray(annotators["normals"].get_data()),
                )
                manifest["views"].append(
                    {
                        **view,
                        "stem": stem,
                        "camera_to_world": pose.tolist(),
                        **(
                            {
                                "robot_base_to_world": rig.base_to_world.tolist(),
                                "head_to_base": rig.head_to_base.tolist(),
                            }
                            if rig is not None
                            else {}
                        ),
                        "intrinsics": intrinsics.tolist(),
                        "capture_seconds": time.monotonic() - started,
                        "visible_pixels": int(np.count_nonzero(visible)),
                        "hidden_pixels": int(np.count_nonzero(hidden)),
                    }
                )
                print(
                    f"{name}/{view['id']}: {manifest['views'][-1]['capture_seconds']:.2f} s",
                    flush=True,
                )

            # Save the complete perspective-ready scene before the roof is hidden.
            # Replace texture paths only in the exported layer for portability on host.
            scene_path = output / "room.usda"
            stage.GetRootLayer().Export(str(scene_path.resolve()))
            exported = scene_path.read_text()
            exported = exported.replace(
                str(builder.asset_root), os.path.relpath(builder.asset_root, output.resolve())
            )
            scene_path.write_text(exported)

            UsdGeom.Imageable(stage.GetPrimAtPath("/World/Architecture/ceiling")).MakeInvisible()
            camera.CreateProjectionAttr("orthographic")
            extent = max(8.7, float(dimensions[0] + 1), float((dimensions[1] + 1) * width / height))
            camera.CreateHorizontalApertureAttr(extent * 10)
            camera.CreateVerticalApertureAttr(extent * 10 * height / width)
            matrix = np.eye(4)
            matrix[:3, 3] = [0, 0, max(8, float(dimensions[2] + 3))]
            transform.Set(Gf.Matrix4d(matrix.T.tolist()))
            for _ in range(2):
                rep.orchestrator.step(rt_subframes=4)
            overhead = Image.fromarray(np.asarray(annotators["rgb"].get_data())[..., :3].copy())
            overhead.save(output / "top_down.png")
            marked = overhead.copy()
            draw = ImageDraw.Draw(marked)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 23)
            except OSError:
                font = ImageFont.load_default(size=23)

            def pixel(point, width=width, height=height, extent=extent):
                return floorplan_pixel(point, width, height, extent)

            triangles = []
            for view in config["cameras"]:
                pose = resolve_camera_pose(view)
                eye, forward = pose[:3, 3], pose[:3, 2].copy()
                if np.linalg.norm(forward[:2]) < 1e-8:
                    # A vertical optical axis has no floor-plane direction.
                    # Display the head's yaw heading in that singular case.
                    forward = np.cross([0, 0, 1], pose[:3, 0])
                origin, direction, vertices = camera_triangle(eye, eye + forward)
                coords = [pixel(v) for v in vertices]
                draw.polygon(coords, fill="#ffbc59", outline="#181b24", width=3)
                label = pixel(origin - direction * 0.37)
                draw.text(
                    label,
                    view["id"],
                    font=font,
                    fill="white",
                    stroke_width=3,
                    stroke_fill="#131a23",
                    anchor="mm",
                )
                triangles.append(
                    {
                        "id": view["id"],
                        "origin_xy": origin.tolist(),
                        "direction_xy": direction.tolist(),
                        "vertices_xy": [v.tolist() for v in vertices],
                        "head_heading_if_vertical": bool(abs(pose[2, 2]) > 1 - 1e-8),
                    }
                )
            marked.save(output / "top_down_cameras.png")
            manifest["top_down"] = {
                "horizontal_extent_m": extent,
                "camera_markers": triangles,
                "ceiling_hidden_for_visualization": True,
            }
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            (output / "scene.json").write_text(json.dumps(config, indent=2) + "\n")
            index.append(
                {"name": name, "title": config["title"], "object_count": manifest["object_count"]}
            )
            (args.output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
            for annotator in annotators.values():
                annotator.detach(product)
            product.destroy()
            print(f"COMPLETE: {name} ({manifest['object_count']} objects)", flush=True)
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
