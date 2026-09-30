"""Invertible image preprocessing for real structural-edge evaluation."""

import cv2
import numpy as np


def prepare_image(image, mode):
    """Return network image and content rectangle; sizes are (width, height)."""
    height, width = image.shape[:2]
    if mode == "stretch":
        size = (384, 256)
    elif mode == "portrait":
        size = (192, 256)
    elif mode == "portrait_large":
        size = (384, 512)
    elif mode == "letterbox":
        scale = min(384 / width, 256 / height)
        size = (round(width * scale), round(height * scale))
    else:
        raise ValueError(f"Unknown preprocessing: {mode}")
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    x, y = (384 - size[0]) // 2, (256 - size[1]) // 2
    if mode != "letterbox":
        return resized, (0, 0, *size)
    canvas = np.full((256, 384, 3), (124, 116, 104), dtype=np.uint8)
    canvas[y : y + size[1], x : x + size[0]] = resized
    return canvas, (x, y, *size)


def restore_probabilities(probabilities, rectangle, size):
    """Remove padding before mapping every channel onto the evaluation grid."""
    x, y, width, height = rectangle
    content = probabilities[:, y : y + height, x : x + width]
    if content.shape[1:] != (height, width):
        raise ValueError("Content rectangle extends outside prediction")
    return np.stack([cv2.resize(p, size, interpolation=cv2.INTER_LINEAR) for p in content])


def rasterize_annotation(annotation, size):
    """Rasterize visible polylines plus explicit reviewed evaluation rectangles."""
    width, height = size
    target = np.zeros((height, width), np.uint8)
    mask = np.zeros_like(target)
    for rectangle in annotation["review_regions"]:
        x0, y0, x1, y1 = rectangle
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("Review region outside image")
        mask[y0:y1, x0:x1] = 1
    for line in annotation["polylines"]:
        points = np.asarray(line["points"], dtype=np.int32)
        if len(points) < 2 or points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Polyline requires at least two 2D points")
        if (points < 0).any() or (points[:, 0] >= width).any() or (points[:, 1] >= height).any():
            raise ValueError("Polyline outside image")
        cv2.polylines(target, [points], False, 1, 1)
    return target.astype(bool) & mask.astype(bool), mask.astype(bool)
