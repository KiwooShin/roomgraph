"""JSON scene input with a restricted recipe registry and explicit conventions."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from roomgraph.furnishings import Furnisher, kitchen
from roomgraph.headcam import camera_rig_pose

RECIPES = {
    "object",
    "add",
    "table",
    "workstation",
    "chair",
    "bed",
    "cabinet",
    "bookshelf",
    "sofa",
    "rug",
    "plant",
    "lamp",
    "mug",
    "mirror",
    "artwork",
    "kitchen_suite",
    "curtains",
}


def load_scene(path: Path):
    """Validate a scene specification and compile its furniture recipes."""
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("Expected scene schema_version 1")
    if config.get("units") != "metres" or config.get("up_axis") != "Z":
        raise ValueError("Scene must use metres and Z up")
    room = config["room"]
    dimensions = np.asarray(room["dimensions_m"], dtype=float)
    if room["template"] not in {"room_6x5_door_window_v1", "scaled_room_v1"}:
        raise ValueError("Unknown room template")
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions < 2):
        raise ValueError("Room dimensions must be three finite values of at least 2 metres")
    if room["template"] == "room_6x5_door_window_v1" and list(dimensions) != [6, 5, 3]:
        raise ValueError("The original room template has fixed dimensions")
    seen = set()
    for camera in config["cameras"]:
        rig = camera_rig_pose(camera)
        if rig is not None:
            if not config.get("headcam", {}).get("enabled", False):
                raise ValueError("Articulated cameras require headcam.enabled")
            # Rig parameters are authoritative. Derived optical positions also
            # support existing floor-plan and dataset consumers of scene.json.
            pose = rig.camera_to_world
            camera["position"] = pose[:3, 3].tolist()
            camera["target"] = (pose[:3, 3] + pose[:3, 2]).tolist()
        eye, target = np.asarray(camera["position"]), np.asarray(camera["target"])
        if eye.shape != (3,) or target.shape != (3,) or not np.isfinite([eye, target]).all():
            raise ValueError("Camera poses must contain three finite coordinates")
        if np.linalg.norm(eye - target) < 1e-6:
            raise ValueError("Camera position and target must differ")
        if not (
            abs(eye[0]) < dimensions[0] / 2
            and abs(eye[1]) < dimensions[1] / 2
            and 0 < eye[2] < dimensions[2]
        ):
            raise ValueError(f"Camera {camera['id']} is outside the room")
        focal = camera["focal_length_mm"]
        if camera["id"] in seen or not np.isfinite(focal) or focal <= 0:
            raise ValueError("Camera IDs must be unique and focal lengths positive")
        seen.add(camera["id"])
    for name in ("width", "height", "samples_per_pixel"):
        if not isinstance(config["render"][name], int) or config["render"][name] <= 0:
            raise ValueError(f"render.{name} must be a positive integer")
    f = Furnisher(config["seed"])
    for item in config["furnishings"]:
        recipe = item["recipe"]
        if recipe not in RECIPES:
            raise ValueError(f"Unknown furnishing recipe: {recipe}")
        if recipe == "kitchen_suite":
            if item["parameters"]:
                raise ValueError("kitchen_suite currently has a fixed cabinet layout")
            kitchen(f)
        else:
            getattr(f, recipe)(**item["parameters"])
    names = [p.name for p in f.parts]
    if len(names) != len(set(names)):
        raise ValueError("Furniture part IDs must be unique")
    for part in f.parts:
        if not np.isfinite([part.center, part.size, part.rotation]).all() or min(part.size) <= 0:
            raise ValueError(f"Invalid furniture geometry: {part.name}")
    if room["template"] == "scaled_room_v1":
        # Translate assembly anchors to the new shell without distorting furniture meshes.
        scale = dimensions[:2] / [6, 5]
        anchors = {}
        for part in f.parts:
            anchors.setdefault(part.object_id, np.asarray(part.center[:2]))
        f.parts = [
            replace(
                p, center=tuple(np.asarray(p.center) + np.r_[anchors[p.object_id] * (scale - 1), 0])
            )
            for p in f.parts
        ]
    return config, f.parts


def compiled_manifest(config, parts):
    """Retain the editable source and procedural geometry before asset overrides."""
    return {
        "scene_config": config,
        "parts_role": "Recipe expansion; imported_objects replace parts with matching object_id",
        "parts": [asdict(p) for p in parts],
        "object_count": len({p.object_id for p in parts}),
        "part_count": len(parts),
    }
