"""Mesh construction and a deliberately small static glTF 2.0 reader."""

import json
from pathlib import Path

import numpy as np


def rounded_box(size, radius):
    """Build a rounded cuboid with outward normals and planar per-face UVs."""
    half = np.asarray(size) / 2
    radius = max(1e-5, min(radius, float(half.min()) * 0.95))
    positions, normals, uv, indices = [], [], [], []
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3

        def grid(h):
            return np.array(
                [
                    -h,
                    -h + radius * 0.134,
                    -h + radius * 0.5,
                    -h + radius,
                    0,
                    h - radius,
                    h - radius * 0.5,
                    h - radius * 0.134,
                    h,
                ]
            )

        for sign in (-1, 1):
            offset = len(positions)
            gu, gv = grid(half[u]), grid(half[v])
            for y in gv:
                for x in gu:
                    p = np.zeros(3)
                    p[axis], p[u], p[v] = sign * half[axis], x, y
                    inner = np.clip(p, -half + radius, half - radius)
                    n = p - inner
                    n /= np.linalg.norm(n)
                    positions.append(inner + radius * n)
                    normals.append(n)
                    uv.append((x + half[u], y + half[v]))
            for j in range(8):
                for i in range(8):
                    a = offset + j * 9 + i
                    face = [a, a + 1, a + 10, a + 9]
                    if sign < 0:
                        face.reverse()
                    indices.extend(face)
    return np.array(positions), np.array(normals), np.array(uv), np.array(indices)


def load_gltf(path: Path):
    """Read static triangle meshes; preserve material slots and node transforms.

    Selected assets use external buffers. Sparse/normalized accessors and skinning
    are rejected explicitly instead of silently producing incorrect geometry.
    Returned geometry is right-handed Z-up, converted from glTF's Y-up system.
    """
    data = json.loads(path.read_text())
    buffers = [(path.parent / b["uri"]).read_bytes() for b in data["buffers"]]
    dtype = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
    sizes = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}

    def accessor(index):
        a = data["accessors"][index]
        if "sparse" in a or a.get("normalized"):
            raise ValueError("Sparse and normalized glTF accessors are not supported")
        b = data["bufferViews"][a["bufferView"]]
        dt = np.dtype(dtype[a["componentType"]])
        n = sizes[a["type"]]
        offset = a.get("byteOffset", 0) + b.get("byteOffset", 0)
        stride = b.get("byteStride", dt.itemsize * n)
        return np.ndarray(
            (a["count"], n),
            dtype=dt,
            buffer=buffers[b["buffer"]],
            offset=offset,
            strides=(stride, dt.itemsize),
        ).copy()

    result = []
    to_z = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])

    def visit(index, parent):
        node = data["nodes"][index]
        if "skin" in node:
            raise ValueError("Skinned glTF assets are not supported")
        if "matrix" in node:
            local = np.array(node["matrix"]).reshape(4, 4).T
        else:
            x, y, z, w = node.get("rotation", [0, 0, 0, 1])
            rotation = np.array(
                [
                    [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                    [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                    [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
                ]
            )
            local = np.eye(4)
            local[:3, :3] = rotation @ np.diag(node.get("scale", [1, 1, 1]))
            local[:3, 3] = node.get("translation", [0, 0, 0])
        world = parent @ local
        transform = to_z @ world
        if "mesh" in node:
            for primitive in data["meshes"][node["mesh"]]["primitives"]:
                if primitive.get("mode", 4) != 4:
                    raise ValueError("Only glTF triangle meshes are supported")
                attrs = primitive["attributes"]
                positions = accessor(attrs["POSITION"]) @ transform[:3, :3].T + transform[:3, 3]
                normals = accessor(attrs["NORMAL"]) @ np.linalg.inv(transform[:3, :3])
                normals /= np.linalg.norm(normals, axis=1, keepdims=True)
                uv = accessor(attrs["TEXCOORD_0"])
                uv[:, 1] = 1 - uv[:, 1]
                indices = accessor(primitive["indices"]).reshape(-1, 3)
                if np.linalg.det(transform[:3, :3]) < 0:
                    indices = indices[:, ::-1]
                result.append(
                    {
                        "positions": positions,
                        "normals": normals,
                        "uv": uv,
                        "indices": indices.ravel(),
                        "material": primitive["material"],
                    }
                )
        for child in node.get("children", []):
            visit(child, world)

    for root in data["scenes"][data.get("scene", 0)]["nodes"]:
        visit(root, np.eye(4))
    return result, data


def cylinder_mesh(size, segments=48):
    """Smooth cylinder sides with separate flat cap normals and triangle faces."""
    rx, ry, height = np.asarray(size) / 2
    points, normals, uv, indices = [], [], [], []
    for cap, z in enumerate((-height, height)):
        for i in range(segments + 1):
            angle = i * 2 * np.pi / segments
            c, s = np.cos(angle), np.sin(angle)
            points.append((rx * c, ry * s, z))
            normal = np.array([c / rx, s / ry, 0])
            normals.append(normal / np.linalg.norm(normal))
            uv.append((i / segments, cap))
    n = segments + 1
    for i in range(segments):
        indices.extend([i, i + 1, i + n + 1, i, i + n + 1, i + n])
    for cap, z in enumerate((-height, height)):
        center = len(points)
        points.append((0, 0, z))
        normals.append((0, 0, cap * 2 - 1))
        uv.append((0.5, 0.5))
        for i in range(segments):
            a = i * 2 * np.pi / segments
            points.append((rx * np.cos(a), ry * np.sin(a), z))
            normals.append((0, 0, cap * 2 - 1))
            uv.append(((np.cos(a) + 1) / 2, (np.sin(a) + 1) / 2))
        for i in range(segments):
            a, b = center + 1 + i, center + 1 + (i + 1) % segments
            indices.extend([center, a, b] if cap else [center, b, a])
    return np.array(points), np.array(normals), np.array(uv), np.array(indices)
