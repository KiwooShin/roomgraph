"""Manifest-connected dataset export with room-level split validation."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from roomgraph.geometry import Edge, depth_edge_masks

CHANNELS = ["visible", "wall_wall", "wall_floor", "wall_ceiling", "door", "window"]


def validate_groups(records):
    owners = {}
    for item in records:
        group, split = item["building_id"], item["split"]
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if group in owners and owners[group] != split:
            raise ValueError(f"Room leakage: {group}")
        owners[group] = split
    return owners


def export_dataset(capture: Path, destination: Path, width=384, height=256):
    destination.mkdir(parents=True, exist_ok=True)
    splits = {key: {"images": [], "targets": [], "records": []} for key in ["train", "val", "test"]}
    records = []
    geometry_owners = {}
    for path in sorted(capture.glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        config = manifest["scene_config"]
        info = config["dataset"]
        building, split = info["building_id"], info["split"]
        geometry_hash = hashlib.sha256(
            json.dumps(manifest["structural_edges"], sort_keys=True).encode()
        ).hexdigest()
        if geometry_hash in geometry_owners and geometry_owners[geometry_hash] != split:
            raise ValueError("Identical room geometry appears across splits")
        geometry_owners[geometry_hash] = split
        edges = [Edge(**edge) for edge in manifest["structural_edges"]]
        for view in manifest["views"]:
            source = path.parent / view["stem"]
            image = cv2.cvtColor(cv2.imread(str(source) + "_rgb.png"), cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            with np.load(str(source) + "_geometry.npz") as geometry:
                depth = cv2.resize(
                    geometry["depth"], (width, height), interpolation=cv2.INTER_NEAREST
                )
            intrinsic = np.asarray(view["intrinsics"]).copy()
            intrinsic[0] *= width / config["render"]["width"]
            intrinsic[1] *= height / config["render"]["height"]
            pose = np.asarray(view["camera_to_world"])
            visible, _ = depth_edge_masks(edges, intrinsic, pose, depth)
            masks = [visible > 0]
            for kind in CHANNELS[1:]:
                _, full = depth_edge_masks(
                    [e for e in edges if e.kind == kind], intrinsic, pose, depth
                )
                masks.append(full > 0)
            target = np.stack(masks).astype(np.uint8)
            record = {
                "id": f"{building}/{view['id']}",
                "building_id": building,
                "split": split,
                "family": info["family"],
                "rgb": str(Path(str(source) + "_rgb.png").resolve()),
                "manifest": str(path.resolve()),
                "intrinsics": intrinsic.tolist(),
                "camera_to_world": pose.tolist(),
                "geometry_sha256": geometry_hash,
                "config_sha256": manifest["source_config_sha256"],
            }
            records.append(record)
            splits[split]["images"].append(image)
            splits[split]["targets"].append(target)
            splits[split]["records"].append(record)
        print(f"Prepared {building}: {len(manifest['views'])} views / {split}", flush=True)
    validate_groups(records)
    if not all(splits[k]["images"] for k in splits):
        raise ValueError("Train, validation, and test data must all be present")
    for split, content in splits.items():
        np.savez_compressed(
            destination / f"{split}.npz",
            images=np.stack(content["images"]),
            targets=np.stack(content["targets"]),
        )
    index = {
        "schema_version": 1,
        "channels": CHANNELS,
        "width": width,
        "height": height,
        "capture": str(capture.resolve()),
        "records": records,
        "splits": {
            k: {"images": len(v["records"]), "rooms": len({r["building_id"] for r in v["records"]})}
            for k, v in splits.items()
        },
    }
    (destination / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index


class EdgeDataset(Dataset):
    """Keep compact uint8 arrays in RAM; cast/augment batches on the GPU."""

    def __init__(self, root, split, limit=None):
        root = Path(root)
        index = json.loads((root / "index.json").read_text())
        validate_groups(index["records"])
        self.records = [r for r in index["records"] if r["split"] == split][:limit]
        with np.load(root / f"{split}.npz") as data:
            self.images = np.ascontiguousarray(data["images"][:limit].transpose(0, 3, 1, 2))
            self.targets = data["targets"][:limit]
        if len(self.images) != len(self.records):
            raise ValueError("Cache and manifest sample counts differ")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return torch.from_numpy(self.images[index]), torch.from_numpy(self.targets[index])
