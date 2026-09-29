"""World-aligned camera markers for top-down rendered floor plans."""

import numpy as np


def camera_triangle(position, target, length=0.43, half_width=0.15):
    """Return a triangle with its first vertex pointing toward the camera target."""
    origin = np.asarray(position[:2], dtype=float)
    direction = np.asarray(target[:2], dtype=float) - origin
    magnitude = np.linalg.norm(direction)
    if magnitude < 1e-9:
        raise ValueError("Camera needs a nonzero horizontal viewing direction")
    direction /= magnitude
    side = np.array([-direction[1], direction[0]])
    vertices = np.array(
        [
            origin + direction * length,
            origin - direction * 0.15 + side * half_width,
            origin - direction * 0.15 - side * half_width,
        ]
    )
    return origin, direction, vertices


def floorplan_pixel(point, width, height, horizontal_extent):
    """Match the orthographic camera: world +X right, +Y toward image top."""
    return (
        width / 2 + point[0] * width / horizontal_extent,
        height / 2 - point[1] * width / horizontal_extent,
    )
