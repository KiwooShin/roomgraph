"""Procedural humanoid proxy for head-camera occlusion and mirror rendering.

The 1.68 m silhouette is inspired by NEO's public height. This is original proxy
geometry, not an official 1X mesh, kinematic model, or calibrated camera model.
"""

from dataclasses import dataclass

import numpy as np

from roomgraph.furnishings import Part

NECK_PIVOT_M = np.array([0.0, 0.0, 1.43])
CAMERA_OFFSET_M = np.array([0.0, 0.135, 0.155])
HEAD_PART_NAMES = frozenset({"head", "visor", "left_lens", "right_lens"})


@dataclass(frozen=True)
class HeadCameraPose:
    """Transforms using column vectors and metres.

    ``head_to_base`` moves the original, body-coordinate head geometry around
    the neck pivot; it is identity in the neutral pose, not a neck-frame pose.
    ``base_to_world`` is independent of head articulation. ``camera_to_world``
    uses computer-vision axes: X right, Y down, Z forward.
    """

    camera_to_world: np.ndarray
    base_to_world: np.ndarray
    head_to_base: np.ndarray


def head_camera_pose(base_position, base_yaw_deg=0, head_yaw_deg=0, head_pitch_deg=0):
    """Articulate the illustrative head rig without moving the body.

    The upright body uses +X right, +Y forward and +Z up. Positive yaw turns
    left about +Z; head yaw is relative to the body. Positive pitch looks up
    about the yawed head's +X axis. The camera rotates *and translates* around
    the neck pivot. Dimensions are proxy assumptions, not official 1X limits
    or calibration; hardware joint limits belong in the experiment settings.
    """
    position = np.asarray(base_position, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("Robot base position must contain three finite coordinates")
    angles = np.asarray([base_yaw_deg, head_yaw_deg, head_pitch_deg], dtype=float)
    if angles.shape != (3,) or not np.isfinite(angles).all():
        raise ValueError("Robot base and head angles must be finite scalar degrees")
    base_yaw, head_yaw, pitch = np.radians(angles)

    def yaw_rotation(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    c, s = np.cos(pitch), np.sin(pitch)
    head_rotation = yaw_rotation(head_yaw) @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    base_to_world = np.eye(4)
    base_to_world[:3, :3] = yaw_rotation(base_yaw)
    base_to_world[:3, 3] = position
    head_to_base = np.eye(4)
    head_to_base[:3, :3] = head_rotation
    head_to_base[:3, 3] = NECK_PIVOT_M - head_rotation @ NECK_PIVOT_M
    neutral_camera = np.eye(4)
    neutral_camera[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    neutral_camera[:3, 3] = NECK_PIVOT_M + CAMERA_OFFSET_M
    return HeadCameraPose(
        base_to_world @ head_to_base @ neutral_camera, base_to_world, head_to_base
    )


def camera_rig_pose(camera):
    """Resolve optional authoritative ``robot_base`` and ``head`` scene fields."""
    if "robot_base" not in camera and "head" not in camera:
        return None
    if "robot_base" not in camera or "head" not in camera:
        raise ValueError("Articulated cameras require both robot_base and head")
    base, head = camera["robot_base"], camera["head"]
    if not isinstance(base, dict) or not isinstance(head, dict) or "position" not in base:
        raise ValueError("robot_base must define position and head must be an object")
    return head_camera_pose(
        base["position"], base.get("yaw_deg", 0), head.get("yaw_deg", 0), head.get("pitch_deg", 0)
    )


def resolve_camera_pose(camera):
    """Return authoritative rig extrinsics, or unchanged legacy look-at extrinsics."""
    from roomgraph.geometry import look_at

    rig = camera_rig_pose(camera)
    return rig.camera_to_world if rig is not None else look_at(camera["position"], camera["target"])


def proxy_parts(reach=0.52):
    parts = []

    def add(name, center, size, material="robot_shell", shape="rounded_box", rotation=(0, 0, 0)):
        parts.append(
            Part(
                name,
                "headcam_robot",
                "robot",
                shape,
                tuple(center),
                tuple(size),
                material,
                tuple(rotation),
                radius=min(min(size) / 3, 0.055),
            )
        )

    def bone(name, a, b, radius, material="robot_shell"):
        a, b = np.asarray(a), np.asarray(b)
        delta = b - a
        length = np.linalg.norm(delta)
        theta = np.degrees(np.arccos(delta[2] / length))
        phi = np.degrees(np.arctan2(delta[1], delta[0]))
        add(name, (a + b) / 2, (radius, radius, length), material, "cylinder", (0, theta, phi))
        add(name + "_cap", b, (radius, radius, radius), material, "sphere")

    add("torso", (0, 0, 1.14), (0.37, 0.23, 0.44))
    add("chest_panel", (0, 0.119, 1.23), (0.24, 0.015, 0.18), "robot_dark")
    add("waist", (0, 0, 0.88), (0.25, 0.20, 0.10), "robot_dark")
    add("pelvis", (0, 0, 0.76), (0.32, 0.24, 0.19))
    add("neck", (0, 0, 1.40), (0.085, 0.085, 0.11), "robot_dark", "cylinder")
    add("head", (0, 0, 1.55), (0.205, 0.195, 0.26))
    add("visor", (0, 0.101, 1.585), (0.174, 0.014, 0.067), "robot_dark")
    for side in (-1, 1):
        prefix = "left" if side < 0 else "right"
        add(
            prefix + "_lens", (side * 0.045, 0.112, 1.585), (0.022, 0.009, 0.022), "black", "sphere"
        )
        bone(prefix + "_thigh", (side * 0.10, 0, 0.73), (side * 0.11, 0.015, 0.42), 0.14)
        add(prefix + "_knee", (side * 0.11, 0.02, 0.39), (0.12, 0.13, 0.11), "robot_dark", "sphere")
        bone(prefix + "_shin", (side * 0.11, 0.015, 0.36), (side * 0.11, 0, 0.11), 0.11)
        add(prefix + "_foot", (side * 0.11, 0.055, 0.052), (0.125, 0.27, 0.105), "robot_dark")
        shoulder = (side * 0.245, 0, 1.31)
        elbow = (side * 0.31, 0.19, 1.04)
        wrist = (side * 0.24, reach, 1.11 + (reach - 0.5) * 0.3)
        add(prefix + "_shoulder", shoulder, (0.15, 0.15, 0.15), shape="sphere")
        bone(prefix + "_upper_arm", shoulder, elbow, 0.115)
        add(prefix + "_elbow", elbow, (0.10, 0.10, 0.10), "robot_dark", "sphere")
        bone(prefix + "_forearm", elbow, wrist, 0.105)
        add(
            prefix + "_palm",
            (wrist[0], wrist[1] + 0.055, wrist[2]),
            (0.09, 0.12, 0.055),
            "robot_dark",
        )
        for finger in range(4):
            add(
                prefix + f"_finger_{finger}",
                (wrist[0] + (finger - 1.5) * 0.019, wrist[1] + 0.13, wrist[2] - 0.008),
                (0.016, 0.065, 0.02),
                "robot_dark",
            )
        add(
            prefix + "_thumb",
            (wrist[0] - side * 0.054, wrist[1] + 0.06, wrist[2] - 0.014),
            (0.025, 0.06, 0.025),
            "robot_dark",
        )
    return parts


def robot_pose(camera_to_world, camera_height=1.585):
    """Place an upright body behind the head camera; metric camera pose is input."""
    forward = np.asarray(camera_to_world[:3, 2]).copy()
    forward[2] = 0
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([right, forward, [0, 0, 1]])
    pose[:3, 3] = camera_to_world[:3, 3] - forward * 0.135 - [0, 0, camera_height]
    return pose
