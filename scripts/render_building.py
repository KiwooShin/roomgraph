"""Warm Isaac Sim camera server for causal exploration of a furnished building.

The requests directory is a simple local IPC boundary. Atomically write a
request_ID.json containing id, robot_base, head and optional focal_length_mm.
The server acquires that view and atomically writes response_ID.json. Creating
STOP cleanly ends the process. No future camera bank is exposed to the policy.
"""

import argparse
import hashlib
import json
import os
import string
import time
from dataclasses import asdict
from pathlib import Path


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/buildings/connected_three_rooms.json")
    )
    parser.add_argument("--output", type=Path, default=Path("vis/multiroom/capture"))
    parser.add_argument("--requests", type=Path, default=Path("vis/multiroom/requests"))
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--resume", action="store_true", help="Resume matching capture provenance")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.requests.mkdir(parents=True, exist_ok=True)
    previous = None
    if (args.output / "manifest.json").exists() and args.resume:
        previous = json.loads((args.output / "manifest.json").read_text())
        if previous["source_config_sha256"] != hashlib.sha256(args.config.read_bytes()).hexdigest():
            raise ValueError("Cannot resume a capture with a different scene config")
    elif (args.output / "manifest.json").exists():
        raise FileExistsError("Use a fresh output directory to preserve capture provenance")
    if not args.resume and (
        list(args.requests.glob("response_*.json")) or (args.requests / "STOP").exists()
    ):
        raise FileExistsError("Use a fresh requests directory")
    if args.resume:
        (args.requests / "STOP").unlink(missing_ok=True)
    if args.max_frames <= 0 or (args.samples is not None and args.samples <= 0):
        raise ValueError("max-frames and samples must be positive")
    atomic_json(args.output / "status.json", {"status": "starting"})

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "PathTracing"})
    try:
        import carb
        import numpy as np
        import omni.replicator.core as rep
        import omni.usd
        from PIL import Image
        from pxr import Gf, UsdGeom, UsdLux

        from roomgraph.building import architectural_parts, load_building
        from roomgraph.headcam import HEAD_PART_NAMES, camera_rig_pose, proxy_parts
        from roomgraph.usd_furnishing import SceneBuilder

        config, boxes, furniture, edges = load_building(args.config)
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        settings = carb.settings.get_settings()
        samples = args.samples or config["render"]["samples_per_pixel"]
        settings.set("/rtx/pathtracing/maxBounces", 6)
        settings.set("/rtx/pathtracing/optixDenoiser/enabled", True)
        settings.set("/rtx/pathtracing/spp", min(samples, 32))
        settings.set("/rtx/pathtracing/totalSpp", samples)
        settings.set("/rtx/post/aa/op", 0)
        settings.set("/rtx/post/tonemap/filmIso", 100.0)
        builder = SceneBuilder(stage, "assets/cache")
        builder.material(
            "wall", (0.79, 0.77, 0.72), texture="white_plaster_02", diffuse_texture=False
        )
        builder.material("floor", (0.60, 0.40, 0.20), 0.38, texture="wood_floor")
        for part in architectural_parts(boxes):
            builder.part(part, root="/World/Architecture")
        groups = {}
        for part in furniture:
            groups.setdefault(part.object_id, []).append(part)
        imports = []
        for object_id, parts in groups.items():
            category = parts[0].category
            asset = config["asset_overrides"].get({"plant": "plants", "sofa": "sofa"}.get(category))
            if asset and category == "plant":
                pot = next(p for p in parts if p.name.endswith("/pot"))
                bottom = pot.center[2] - pot.size[2] / 2
                extent = max(p.center[2] + p.size[2] / 2 for p in parts) - bottom
                imports.append(
                    builder.imported(
                        asset, object_id, (*pot.center[:2], bottom), extent, 0, "height"
                    )
                )
            elif asset and category == "sofa":
                base = parts[0]
                imports.append(
                    builder.imported(
                        asset, object_id, (*base.center[:2], 0), 2.40, base.rotation[2], "width"
                    )
                )
            else:
                for part in parts:
                    builder.part(part)
        builder.material(
            "robot_shell", (0.60, 0.56, 0.48), 0.85, texture="denim_fabric", diffuse_texture=False
        )
        builder.material("robot_dark", (0.025, 0.028, 0.03), 0.5)
        robot = UsdGeom.Xform.Define(stage, "/World/Robot")
        body_transform = robot.AddTransformOp()
        head = UsdGeom.Xform.Define(stage, "/World/Robot/HeadRig")
        head_transform = head.AddTransformOp()
        for part in proxy_parts(config["headcam"].get("arm_reach_m", 0.52)):
            builder.part(
                part,
                root="/World/Robot/HeadRig" if part.name in HEAD_PART_NAMES else "/World/Robot",
            )
        dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Sky")
        dome.CreateIntensityAttr(config["lighting"]["daylight_intensity"])
        dome.CreateColorAttr(Gf.Vec3f(0.82, 0.90, 1.0))
        for region in config["regions"]:
            low, high = np.asarray(region["bounds_xy"], float)
            midpoint = (low + high) / 2
            light = UsdLux.RectLight.Define(stage, "/World/Lights/" + region["id"])
            light.AddTranslateOp().Set(Gf.Vec3d(*midpoint, config["height_m"] - 0.04))
            light.CreateWidthAttr(float(high[0] - low[0] - 0.5))
            light.CreateHeightAttr(float(high[1] - low[1] - 0.5))
            light.CreateIntensityAttr(config["lighting"]["ceiling_intensity"])
            light.CreateEnableColorTemperatureAttr(True)
            light.CreateColorTemperatureAttr(config["lighting"]["color_temperature_kelvin"])

        width, height = config["render"]["width"], config["render"]["height"]
        camera = UsdGeom.Camera.Define(stage, "/World/Camera")
        transform = camera.AddTransformOp()
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100))
        camera.CreateFStopAttr(0)
        top_width, top_height = width * 3, height * 3
        product = rep.create.render_product(str(camera.GetPath()), (top_width, top_height))
        annotators = {
            name: rep.AnnotatorRegistry.get_annotator(name)
            for name in ("rgb", "distance_to_image_plane")
        }
        for annotator in annotators.values():
            annotator.attach(product)

        # A top-down reference is saved separately; it never enters the controller.
        ceiling = UsdGeom.Imageable(stage.GetPrimAtPath("/World/Architecture/ceiling"))
        ceiling.MakeInvisible()
        lintels = [
            UsdGeom.Imageable(stage.GetPrimAtPath("/World/Architecture/" + box.name))
            for box in boxes
            if box.name.endswith("_lintel")
        ]
        for lintel in lintels:
            lintel.MakeInvisible()
        robot.MakeInvisible()
        low, high = np.asarray(config["bounds_xy"], float)
        center = (low + high) / 2
        extent = max(high[0] - low[0] + 0.8, (high[1] - low[1] + 0.8) * width / height)
        camera.CreateProjectionAttr("orthographic")
        camera.CreateHorizontalApertureAttr(float(extent * 10))
        camera.CreateVerticalApertureAttr(float(extent * 10 * height / width))
        top_pose = np.eye(4)
        top_pose[:3, 3] = [*center, 10]
        transform.Set(Gf.Matrix4d(top_pose.T.tolist()))
        for _ in range(2):
            rep.orchestrator.step(rt_subframes=4)
        Image.fromarray(np.asarray(annotators["rgb"].get_data())[..., :3].copy()).save(
            args.output / "top_down.png"
        )
        ceiling.MakeVisible()
        for lintel in lintels:
            lintel.MakeVisible()
        robot.MakeVisible()
        for annotator in annotators.values():
            annotator.detach(product)
        product.destroy()
        product = rep.create.render_product(str(camera.GetPath()), (width, height))
        for annotator in annotators.values():
            annotator.attach(product)
        camera.CreateProjectionAttr("perspective")
        camera.CreateHorizontalApertureAttr(24)
        camera.CreateVerticalApertureAttr(24 * height / width)
        fingerprint = hashlib.sha256(args.config.read_bytes()).hexdigest()
        reference = {
            "role": "renderer_and_evaluator_only",
            "source_config_sha256": fingerprint,
            "config": config,
            "rooms": config["regions"],
            "portals": config["portals"],
            "boxes": [asdict(box) for box in boxes],
            "parts": [asdict(part) for part in furniture],
            "edges": [{**asdict(edge), "region": edge.name.split("/")[0]} for edge in edges],
            "imported_objects": imports,
            "top_down": {
                "center_xy": center.tolist(),
                "horizontal_extent_m": float(extent),
                "width": top_width,
                "height": top_height,
                "ceiling_hidden_for_visualization": True,
                "door_lintels_hidden_for_visualization": True,
            },
        }
        atomic_json(args.output / "reference.json", reference)
        manifest = {
            "schema_version": 1,
            "renderer": "Isaac Sim 5.1.0 / PathTracing",
            "source_config_sha256": fingerprint,
            "sensor": {
                "width": width,
                "height": height,
                "depth_units": "metres",
                "depth_semantics": "distance_to_image_plane; camera-axis Z; finite positive only",
                "poses": "known simulator odometry; CV +X right,+Y down,+Z forward",
            },
            "samples_per_pixel": samples,
            "views": [],
        }
        if previous is not None:
            if previous["samples_per_pixel"] != samples or previous["sensor"] != manifest["sensor"]:
                raise ValueError("Cannot resume with changed renderer sampling or sensor")
            manifest = previous
        atomic_json(args.output / "manifest.json", manifest)
        atomic_json(args.output / "status.json", {"status": "ready", "frames": 0})
        print("BUILDING SERVER READY", flush=True)
        processed = {
            path.name.replace("response_", "request_", 1)
            for path in args.requests.glob("response_*.json")
        }
        while len(manifest["views"]) < args.max_frames and not (args.requests / "STOP").exists():
            pending = [
                path
                for path in sorted(args.requests.glob("request_*.json"))
                if path.name not in processed
            ]
            if not pending:
                time.sleep(0.05)
                continue
            for request_path in pending:
                started = time.monotonic()
                response_path = args.requests / request_path.name.replace(
                    "request_", "response_", 1
                )
                try:
                    request = json.loads(request_path.read_text())
                    identifier = request["id"]
                    if (
                        not isinstance(identifier, str)
                        or not identifier
                        or any(
                            c not in string.ascii_letters + string.digits + "_-" for c in identifier
                        )
                    ):
                        raise ValueError(
                            "Request id must use letters, numbers, hyphen and underscore"
                        )
                    if identifier != request_path.stem.removeprefix("request_"):
                        raise ValueError("Request id and filename differ")
                    rig = camera_rig_pose(request)
                    if rig is None:
                        raise ValueError("Requests require robot_base and head")
                    focal = request.get("focal_length_mm", config["render"]["focal_length_mm"])
                    if not np.isfinite(focal) or focal <= 0:
                        raise ValueError("Focal length must be finite and positive")
                    camera.CreateFocalLengthAttr(float(focal))
                    transform.Set(
                        Gf.Matrix4d((rig.camera_to_world @ np.diag([1, -1, -1, 1])).T.tolist())
                    )
                    body_transform.Set(Gf.Matrix4d(rig.base_to_world.T.tolist()))
                    head_transform.Set(Gf.Matrix4d(rig.head_to_base.T.tolist()))
                    for _ in range(2):
                        rep.orchestrator.step(rt_subframes=4)
                    rgb = np.asarray(annotators["rgb"].get_data())[..., :3].copy()
                    depth = (
                        np.asarray(annotators["distance_to_image_plane"].get_data())
                        .reshape(height, width)
                        .copy()
                    )
                    if rgb.shape != (height, width, 3) or rgb.std() < 3:
                        raise RuntimeError("Invalid RGB capture")
                    if np.count_nonzero(np.isfinite(depth) & (depth > 0)) < width * height * 0.5:
                        raise RuntimeError("Invalid depth capture")
                    stem = "frame_" + identifier
                    Image.fromarray(rgb).save(args.output / f"{stem}_rgb.png")
                    np.savez_compressed(args.output / f"{stem}_geometry.npz", depth=depth)
                    intrinsics = [
                        [width * focal / 24, 0, width / 2],
                        [0, width * focal / 24, height / 2],
                        [0, 0, 1],
                    ]
                    view = {
                        "id": identifier,
                        "stem": stem,
                        "rgb": f"{stem}_rgb.png",
                        "geometry": f"{stem}_geometry.npz",
                        "robot_base": request["robot_base"],
                        "head": request["head"],
                        "focal_length_mm": focal,
                        "camera_to_world": rig.camera_to_world.tolist(),
                        "robot_base_to_world": rig.base_to_world.tolist(),
                        "head_to_base": rig.head_to_base.tolist(),
                        "intrinsics": intrinsics,
                        "capture_seconds": time.monotonic() - started,
                    }
                    manifest["views"].append(view)
                    atomic_json(args.output / "manifest.json", manifest)
                    atomic_json(response_path, {"status": "complete", "view": view})
                    atomic_json(
                        args.output / "status.json",
                        {"status": "ready", "frames": len(manifest["views"])},
                    )
                    print(f"CAPTURE {identifier}: {view['capture_seconds']:.2f} s", flush=True)
                except Exception as error:
                    atomic_json(
                        response_path,
                        {
                            "status": "error",
                            "error": str(error),
                            "error_type": type(error).__name__,
                        },
                    )
                    print(f"REQUEST ERROR {request_path.name}: {error}", flush=True)
                processed.add(request_path.name)
                if len(manifest["views"]) >= args.max_frames:
                    break
        scene_path = args.output / "building.usda"
        stage.GetRootLayer().Export(str(scene_path.resolve()))
        scene_path.write_text(
            scene_path.read_text().replace(
                str(builder.asset_root), os.path.relpath(builder.asset_root, args.output.resolve())
            )
        )
        atomic_json(
            args.output / "status.json", {"status": "complete", "frames": len(manifest["views"])}
        )
        for annotator in annotators.values():
            annotator.detach(product)
        product.destroy()
    except Exception as error:
        atomic_json(args.output / "status.json", {"status": "error", "error": str(error)})
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
