"""Deterministic, renderer-independent furniture recipes in metres.

All meshes are procedural originals: no external assets, downloads, or textures.
Objects retain semantic identity even when made from several geometric parts.
"""

from dataclasses import dataclass
from math import cos, radians, sin

import numpy as np


@dataclass(frozen=True)
class Material:
    """USD Preview Surface parameters."""

    color: tuple[float, float, float]
    roughness: float = 0.65
    metallic: float = 0.0
    emission: tuple[float, float, float] = (0, 0, 0)


MATERIALS = {
    "oak": Material((0.52, 0.31, 0.14), 0.45),
    "walnut": Material((0.16, 0.073, 0.035), 0.42),
    "ivory": Material((0.84, 0.81, 0.70), 0.72),
    "white": Material((0.88, 0.90, 0.87), 0.45),
    "sage": Material((0.23, 0.38, 0.30), 0.85),
    "teal": Material((0.055, 0.22, 0.25), 0.82),
    "rust": Material((0.54, 0.15, 0.075), 0.8),
    "mustard": Material((0.69, 0.43, 0.10), 0.8),
    "navy": Material((0.04, 0.075, 0.14), 0.75),
    "black": Material((0.018, 0.025, 0.032), 0.4),
    "steel": Material((0.47, 0.51, 0.55), 0.24, 0.85),
    "brass": Material((0.72, 0.48, 0.16), 0.23, 0.8),
    "mirror": Material((0.96, 0.97, 0.98), 0.005, 1.0),
    "screen": Material((0.017, 0.04, 0.065), 0.2, 0.1, (0.05, 0.15, 0.22)),
    "leaf": Material((0.035, 0.19, 0.055), 0.6),
    "leaf_light": Material((0.12, 0.30, 0.07), 0.65),
    "stem": Material((0.10, 0.14, 0.025), 0.85),
    "soil": Material((0.04, 0.018, 0.008), 1.0),
    "linen": Material((0.62, 0.53, 0.38), 0.95),
    "light": Material((0.95, 0.86, 0.61), 0.5, 0.0, (2.5, 2.1, 1.4)),
}


@dataclass(frozen=True)
class Part:
    """A primitive or rounded box, with world-space center and XYZ Euler degrees."""

    name: str
    object_id: str
    category: str
    shape: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    material: str
    rotation: tuple[float, float, float] = (0, 0, 0)
    radius: float = 0.025


class Furnisher:
    """Build object parts in local coordinates with an optional floor-plane rotation."""

    def __init__(self, seed: int):
        self.parts: list[Part] = []
        self.rng = np.random.default_rng(seed)
        self.origin = (0, 0, 0)
        self.yaw = 0
        self.object_id = ""
        self.category = ""

    def object(self, name, category, origin, yaw=0):
        self.object_id, self.category = name, category
        self.origin, self.yaw = origin, yaw

    def add(
        self, name, center, size, material, shape="rounded_box", rotation=(0, 0, 0), radius=0.025
    ):
        angle = radians(self.yaw)
        x, y, z = center
        ox, oy, oz = self.origin
        world = (ox + x * cos(angle) - y * sin(angle), oy + x * sin(angle) + y * cos(angle), oz + z)
        rotation = (rotation[0], rotation[1], rotation[2] + self.yaw)
        self.parts.append(
            Part(
                f"{self.object_id}/{name}",
                self.object_id,
                self.category,
                shape,
                world,
                size,
                material,
                rotation,
                radius,
            )
        )

    def table(self, name, xy, size=(1.5, 0.7, 0.75), material="oak", yaw=0):
        self.object(name, "desk", (*xy, 0), yaw)
        w, d, h = size
        self.add("top", (0, 0, h), (w, d, 0.065), material)
        for i, x in enumerate((-w / 2 + 0.1, w / 2 - 0.1)):
            for j, y in enumerate((-d / 2 + 0.08, d / 2 - 0.08)):
                self.add(f"leg_{i}_{j}", (x, y, h / 2), (0.055, 0.055, h), "black", radius=0.008)

    def workstation(self, name, xy, yaw=0):
        self.table(name, xy, yaw=yaw)
        self.object(name + "_computer", "computer", (*xy, 0), yaw)
        self.add("foot", (0, 0.15, 0.8), (0.32, 0.2, 0.025), "steel")
        self.add("stand", (0, 0.18, 0.97), (0.045, 0.045, 0.32), "black")
        self.add("monitor", (0, 0.18, 1.17), (0.85, 0.06, 0.47), "black")
        self.add("display", (0, 0.144, 1.17), (0.79, 0.009, 0.41), "screen", radius=0.004)
        # Small, physical screen details provide visual texture without external images.
        for i in range(5):
            self.add(
                f"screen_line_{i}",
                (-0.07, 0.137, 1.27 - i * 0.045),
                (0.40 - i * 0.025, 0.003, 0.009),
                "teal",
                radius=0.001,
            )
        self.add("keyboard", (-0.08, -0.17, 0.803), (0.46, 0.15, 0.025), "black", radius=0.012)
        for row in range(4):
            for col in range(12):
                self.add(
                    f"key_{row}_{col}",
                    (-0.28 + col * 0.036, -0.22 + row * 0.033, 0.82),
                    (0.025, 0.021, 0.008),
                    "steel",
                    radius=0.002,
                )
        self.add("mouse", (0.38, -0.16, 0.815), (0.065, 0.105, 0.034), "black", "sphere")
        self.mug(
            name + "_mug",
            (xy[0] - 0.58 * cos(radians(yaw)), xy[1] - 0.58 * sin(radians(yaw)), 0.79),
            "rust",
        )

    def chair(self, name, xy, yaw=0, material="teal"):
        self.object(name, "chair", (*xy, 0), yaw)
        self.add("seat", (0, 0, 0.46), (0.48, 0.46, 0.105), material, radius=0.045)
        self.add(
            "back", (0, -0.22, 0.75), (0.48, 0.09, 0.55), material, rotation=(-7, 0, 0), radius=0.04
        )
        for i, x in enumerate((-0.18, 0.18)):
            for j, y in enumerate((-0.16, 0.16)):
                self.add(f"leg_{i}_{j}", (x, y, 0.22), (0.038, 0.038, 0.44), "oak", radius=0.009)

    def bed(self, xy):
        self.object("bed", "bed", (*xy, 0))
        self.add("frame", (0, 0, 0.25), (1.95, 2.25, 0.32), "oak", radius=0.04)
        self.add("headboard", (0, 1.08, 0.83), (2.08, 0.15, 1.18), "sage", radius=0.07)
        for i in range(9):
            self.add(
                f"headboard_panel_{i}",
                (-0.89 + i * 0.223, 0.992, 0.87),
                (0.212, 0.045, 1.0),
                "sage",
                radius=0.022,
            )
        self.add("mattress", (0, -0.02, 0.51), (1.86, 2.1, 0.27), "white", radius=0.1)
        self.add("duvet", (0, -0.27, 0.67), (1.9, 1.56, 0.14), "ivory", radius=0.065)
        self.add("throw", (0, -0.69, 0.752), (1.94, 0.44, 0.045), "rust", radius=0.02)
        for i in range(13):
            self.add(
                f"throw_rib_{i}",
                (-0.9 + i * 0.15, -0.69, 0.779),
                (0.015, 0.42, 0.002),
                "ivory",
                radius=0.003,
            )
        for i, x in enumerate((-0.48, 0.48)):
            self.add(
                f"pillow_{i}",
                (x, 0.70, 0.72),
                (0.77, 0.44, 0.19),
                "white",
                rotation=(-10, 0, (i * 2 - 1) * 4),
                radius=0.085,
            )
            self.add(
                f"accent_pillow_{i}",
                (x, 0.47, 0.83),
                (0.37, 0.20, 0.29),
                "sage",
                rotation=(-18, 0, 0),
                radius=0.075,
            )

    def cabinet(self, name, xy, size=(1.1, 0.48, 0.85), yaw=0, material="oak"):
        self.object(name, "cabinet", (*xy, 0), yaw)
        w, d, h = size
        self.add("body", (0, 0, h / 2 + 0.06), (w, d, h), material)
        for i in range(3):
            z = 0.07 + (i + 0.5) * h / 3
            self.add(
                f"drawer_{i}",
                (0, -d / 2 - 0.01, z),
                (w - 0.045, 0.04, h / 3 - 0.025),
                material,
                radius=0.01,
            )
            self.add(
                f"pull_{i}", (0, -d / 2 - 0.05, z), (0.19, 0.025, 0.018), "brass", radius=0.008
            )

    def bookshelf(self, name, xy, yaw=0):
        self.object(name, "bookshelf", (*xy, 0), yaw)
        for i, x in enumerate((-0.6, 0.6)):
            self.add(f"side_{i}", (x, 0, 1.0), (0.06, 0.36, 2.0), "oak")
        self.add("back", (0, 0.16, 1.0), (1.2, 0.035, 2), "walnut")
        for i in range(5):
            z = 0.09 + i * 0.46
            self.add(f"shelf_{i}", (0, 0, z), (1.25, 0.39, 0.045), "oak")
            if i < 4:
                for j in range(9):
                    height = float(self.rng.uniform(0.20, 0.36))
                    self.add(
                        f"book_{i}_{j}",
                        (-0.50 + j * 0.115, 0.015, z + 0.025 + height / 2),
                        (0.07, 0.22, height),
                        ["rust", "ivory", "navy", "sage", "mustard"][j % 5],
                        radius=0.003,
                    )

    def sofa(self, name, xy, yaw=0, material="ivory"):
        self.object(name, "sofa", (*xy, 0), yaw)
        self.add("base", (0, 0, 0.28), (2.4, 0.86, 0.30), material, radius=0.10)
        self.add("back", (0, 0.35, 0.69), (2.35, 0.23, 0.67), material, radius=0.095)
        for i, x in enumerate((-1.13, 1.13)):
            self.add(f"arm_{i}", (x, -0.03, 0.53), (0.23, 0.93, 0.5), material, radius=0.09)
        for i, x in enumerate((-0.68, 0, 0.68)):
            self.add(f"seat_{i}", (x, -0.1, 0.49), (0.66, 0.68, 0.18), material, radius=0.07)
            self.add(
                f"back_cushion_{i}",
                (x, 0.18, 0.79),
                (0.64, 0.16, 0.48),
                material,
                rotation=(-8, 0, 0),
                radius=0.065,
            )
        for i, x in enumerate((-0.77, 0.77)):
            self.add(
                f"pillow_{i}",
                (x, -0.10, 0.77),
                (0.4, 0.17, 0.4),
                ("rust", "sage")[i],
                rotation=(-16, 0, i * 18 - 9),
                radius=0.085,
            )
        for i, x in enumerate((-0.95, 0.95)):
            for j, y in enumerate((-0.27, 0.25)):
                self.add(f"foot_{i}_{j}", (x, y, 0.085), (0.07, 0.07, 0.17), "walnut")

    def rug(self, name, xy, size=(2.5, 2.0), material="linen"):
        self.object(name, "rug", (*xy, 0))
        w, d = size
        self.add("woven_base", (0, 0, 0.015), (w, d, 0.023), material, radius=0.01)
        for i, y in enumerate((-d / 2 + 0.12, d / 2 - 0.12)):
            self.add(f"border_{i}", (0, y, 0.028), (w - 0.15, 0.045, 0.003), "ivory", radius=0.001)
        for i in range(24):
            x = -w / 2 + 0.06 + i * (w - 0.12) / 23
            for j, y in enumerate((-d / 2 - 0.035, d / 2 + 0.035)):
                self.add(f"fringe_{i}_{j}", (x, y, 0.013), (0.012, 0.09, 0.012), "ivory")

    def plant(self, name, xyz, scale=1.0):
        self.object(name, "plant", xyz)
        s = scale
        self.add("pot", (0, 0, 0.18 * s), (0.37 * s, 0.37 * s, 0.36 * s), "ivory", "cylinder")
        self.add("rim", (0, 0, 0.35 * s), (0.4 * s, 0.4 * s, 0.045 * s), "ivory", "cylinder")
        self.add("soil", (0, 0, 0.367 * s), (0.335 * s, 0.335 * s, 0.015 * s), "soil", "cylinder")
        self.add("stem", (0, 0, 0.76 * s), (0.025 * s, 0.025 * s, 0.80 * s), "stem", "cylinder")
        for i in range(18):
            angle = i * 137.5
            z = (0.52 + i * 0.036) * s
            reach = (0.27 + 0.05 * sin(i)) * s
            x, y = reach * cos(radians(angle)), reach * sin(radians(angle))
            self.add(
                f"leaf_{i}",
                (x, y, z),
                (0.19 * s, 0.032 * s, 0.52 * s),
                "leaf" if i % 3 else "leaf_light",
                "sphere",
                (0, 60, angle),
            )
            self.add(
                f"branch_{i}",
                (x / 2, y / 2, z - 0.08 * s),
                (0.012 * s, 0.012 * s, 0.33 * s),
                "stem",
                "cylinder",
                (0, 60, angle),
            )

    def lamp(self, name, xyz, floor=False):
        self.object(name, "lamp", xyz)
        h = 1.5 if floor else 0.4
        self.add("base", (0, 0, 0.025), (0.30, 0.30, 0.05), "brass", "cylinder")
        self.add("pole", (0, 0, h / 2), (0.025, 0.025, h), "brass", "cylinder")
        self.add("shade", (0, 0, h), (0.42, 0.42, 0.30), "ivory", "cylinder")
        self.add("diffuser", (0, 0, h - 0.151), (0.38, 0.38, 0.01), "light", "cylinder")

    def mug(self, name, xyz, material="white"):
        self.object(name, "mug", xyz)
        self.add("body", (0, 0, 0.05), (0.08, 0.08, 0.10), material, "cylinder")
        self.add("coffee", (0, 0, 0.101), (0.065, 0.065, 0.002), "walnut", "cylinder")
        self.add("handle", (0.055, 0, 0.05), (0.06, 0.023, 0.064), material, "sphere")

    def mirror(self, name, xy, yaw=0):
        self.object(name, "mirror", (*xy, 0), yaw)
        self.add("frame", (0, 0, 1.48), (0.92, 0.06, 1.60), "brass", radius=0.03)
        self.add("reflector", (0, -0.037, 1.48), (0.82, 0.018, 1.50), "mirror", radius=0.008)

    def artwork(self, name, xy, yaw=0):
        self.object(name, "artwork", (*xy, 0), yaw)
        self.add("frame", (0, 0, 1.85), (0.85, 0.045, 0.66), "oak")
        self.add("canvas", (0, -0.028, 1.85), (0.78, 0.01, 0.59), "ivory", radius=0.005)
        for i, (x, z, color) in enumerate([(-0.16, 1.89, "rust"), (0.13, 1.81, "sage")]):
            self.add(f"shape_{i}", (x, -0.035, z), (0.28, 0.006, 0.36), color, "sphere")

    def curtains(self, name="window_curtains", material="ivory"):
        """Two gathered fabric panels beside the east-wall window."""
        self.object(name, "curtain", (0, 0, 0))
        self.add("rod", (2.79, 0, 2.58), (0.027, 0.027, 2.95), "brass", "cylinder", (90, 0, 0))
        for side in (-1, 1):
            for i in range(6):
                self.add(
                    f"panel_{side + 1}_{i}",
                    (2.80 + 0.025 * cos(i), side * (1.09 + i * 0.068), 1.38),
                    (0.10, 0.10, 2.26),
                    material,
                    radius=0.045,
                )


def kitchen(f: Furnisher):
    """Compact kitchen with an accessible doorway and a separate dining area."""
    for i, x in enumerate((-0.5, 0.35, 1.2, 2.05)):
        f.cabinet(f"base_cabinet_{i}", (x, 2.08), (0.83, 0.70, 0.80), material="sage")
        f.object(f"counter_{i}", "countertop", (x, 2.08, 0))
        f.add("slab", (0, 0, 0.91), (0.85, 0.76, 0.06), "ivory", radius=0.012)
        f.object(f"wall_cabinet_{i}", "cabinet", (x, 2.26, 1.59))
        f.add("body", (0, 0, 0.33), (0.81, 0.36, 0.66), "ivory", radius=0.01)
        for j, offset in enumerate((-0.2, 0.2)):
            f.add(f"door_{j}", (offset, -0.19, 0.33), (0.39, 0.035, 0.63), "ivory", radius=0.008)
            f.add(
                f"pull_{j}",
                (offset * 0.23, -0.225, 0.20),
                (0.018, 0.027, 0.19),
                "brass",
                radius=0.007,
            )
    f.object("backsplash", "backsplash", (0.8, 2.48, 0))
    for i in range(17):
        for j in range(4):
            f.add(
                f"tile_{i}_{j}",
                (-1.65 + i * 0.205, -0.025, 1.0 + j * 0.13),
                (0.199, 0.02, 0.124),
                "white",
                radius=0.002,
            )
    f.object("sink", "sink", (1.15, 1.99, 0))
    f.add("basin", (0, 0, 0.946), (0.55, 0.41, 0.008), "steel", radius=0.04)
    f.add("interior", (0, 0, 0.952), (0.46, 0.33, 0.008), "black", radius=0.035)
    f.add("tap_upright", (0.16, 0.24, 1.11), (0.026, 0.026, 0.36), "steel", "cylinder")
    f.add("tap_spout", (0.16, 0.14, 1.28), (0.028, 0.21, 0.028), "steel")
    f.object("cooker", "cooker", (-0.45, 1.97, 0))
    f.add("hob", (0, 0, 0.951), (0.62, 0.48, 0.02), "black", radius=0.02)
    for i, x in enumerate((-0.16, 0.16)):
        for j, y in enumerate((-0.12, 0.12)):
            f.add(f"ring_{i}_{j}", (x, y, 0.967), (0.17, 0.17, 0.01), "steel", "cylinder")
    f.object("cooking_pot", "cookware", (-0.62, 1.85, 0.98))
    f.add("pot", (0, 0, 0.08), (0.20, 0.20, 0.16), "steel", "cylinder")
    f.add("lid", (0, 0, 0.17), (0.21, 0.21, 0.022), "steel", "cylinder")
    f.add("knob", (0, 0, 0.20), (0.04, 0.04, 0.04), "black", "sphere")
    f.object("fridge", "refrigerator", (-2.49, 1.60, 0), 90)
    f.add("body", (0, 0, 0.98), (0.78, 0.73, 1.96), "steel", radius=0.045)
    for i, (z, h) in enumerate(((0.57, 1.03), (1.52, 0.80))):
        f.add(f"door_{i}", (0, -0.38, z), (0.72, 0.06, h), "white")
        f.add(f"handle_{i}", (-0.26, -0.44, z), (0.028, 0.04, 0.32), "steel")
    f.table("dining_table", (0.30, -0.75), (1.6, 0.85, 0.76))
    for i, (x, y, yaw) in enumerate(
        [(-0.2, -1.65, 0), (0.8, -1.65, 0), (-0.2, 0.10, 180), (0.8, 0.10, 180)]
    ):
        f.chair(f"dining_chair_{i}", (x, y), yaw, "sage")
    for i, x in enumerate((-0.18, 0.80)):
        f.object(f"place_setting_{i}", "tableware", (x, -0.77, 0.80))
        f.add("plate", (0, 0, 0.012), (0.28, 0.28, 0.022), "white", "cylinder")
        f.add("napkin", (0.24, 0, 0.012), (0.12, 0.22, 0.018), "linen", radius=0.003)
        f.mug(f"dining_mug_{i}", (x + 0.1, -0.55, 0.80))
    f.plant("herbs", (2.1, 2.03, 0.95), 0.32)
    f.plant("dining_plant", (2.40, -1.45, 0), 0.80)
    f.artwork("dining_art", (-1.4, -2.43), yaw=180)
