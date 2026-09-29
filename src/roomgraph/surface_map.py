"""Fuse acquired calibrated RGB-D surfaces without completing unseen geometry."""

from pathlib import Path

import numpy as np

from roomgraph.active_view import _calibration


class SurfaceMap:
    """Voxel means of observed surfaces, with distinct-view support counts."""

    def __init__(self, voxel_size_m=0.06):
        if not np.isfinite(voxel_size_m) or voxel_size_m <= 0:
            raise ValueError("Voxel size must be finite and positive")
        self.voxel_size_m = float(voxel_size_m)
        self._voxels = {}
        self.observations = 0

    def integrate(self, rgb, depth, record, *, stride=3, robot_base_xy=None):
        """Depth is optical-axis distance; RGB and calibration match its resolution."""
        intrinsic, pose = _calibration(record)
        rgb, depth = np.asarray(rgb), np.asarray(depth)
        if depth.ndim != 2 or rgb.shape != (*depth.shape, 3):
            raise ValueError("RGB [H,W,3] and optical depth [H,W] must align")
        if stride < 1 or int(stride) != stride:
            raise ValueError("Stride must be a positive integer")
        yy, xx = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
        z = depth[yy, xx]
        valid = np.isfinite(z) & (z > 0.12) & (z < 12)
        pixels = np.column_stack([xx[valid], yy[valid], np.ones(valid.sum())])
        camera = (pixels @ np.linalg.inv(intrinsic).T) * z[valid, None]
        points = camera @ pose[:3, :3].T + pose[:3, 3]
        colors = rgb[yy[valid], xx[valid]].astype(float)
        keep = (points[:, 2] >= -0.06) & (points[:, 2] <= 4)
        if robot_base_xy is not None:
            # Calibrated proxy envelope, not a renderer segmentation/ground-truth mask.
            keep &= np.linalg.norm(points[:, :2] - robot_base_xy, axis=1) > 0.72
        points, colors = points[keep], colors[keep]
        keys, inverse = np.unique(
            np.floor(points / self.voxel_size_m).astype(np.int32), axis=0, return_inverse=True
        )
        counts = np.bincount(inverse, minlength=len(keys))
        sums = np.zeros((len(keys), 6))
        np.add.at(sums, inverse, np.column_stack([points, colors]))
        means = sums / np.maximum(counts[:, None], 1)
        for key, mean in zip(keys, means, strict=True):
            key = tuple(key)
            if key in self._voxels:
                total, support = self._voxels[key]
                self._voxels[key] = (total + mean, support + 1)
            else:
                self._voxels[key] = (mean, 1)
        self.observations += 1

    def arrays(self):
        if not self._voxels:
            return {"points": np.empty((0, 3)), "colors": np.empty((0, 3)), "support": np.zeros(0)}
        sums, support = zip(*self._voxels.values(), strict=True)
        support = np.asarray(support)
        means = np.asarray(sums) / support[:, None]
        return {"points": means[:, :3], "colors": means[:, 3:], "support": support}


def write_points_ply(path, points, colors=None):
    """Export observed points; no hidden surfaces or ground-truth geometry are inserted."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite points [N,3]")
    colors = np.full(points.shape, 100) if colors is None else np.asarray(colors)
    if colors.shape != points.shape or not np.isfinite(colors).all():
        raise ValueError("Colors must match points")
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with Path(path).open("w") as stream:
        stream.write(header)
        np.savetxt(stream, np.column_stack([points, colors]), fmt="%.4f %.4f %.4f %d %d %d")
