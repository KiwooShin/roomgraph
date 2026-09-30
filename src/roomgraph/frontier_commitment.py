"""Bounded frontier commitment using only the acquired occupancy map.

The first choice and every release use the frozen baseline's score and filters.
Between these choices an exact goal is retained, with its path recalculated on
the latest footprint-safe map. A nearby current boundary is evidence that the
inspection remains useful; this is spatial association, not known room identity.
Head views, map integration and execution budgets are unchanged.
"""

from collections.abc import Mapping
from numbers import Real

import numpy as np

from roomgraph.mapping import Frontier, OccupancyGrid

POLICY_ID = "frontier_commitment_v2"
DEFAULT_SETTINGS = {
    "goal_tolerance_m": 0.30,
    "frontier_match_radius_m": 0.8,
    "max_commit_steps": 8,
    "minimum_progress_m": 0.1,
    "stagnation_steps": 2,
}


def _settings(overrides):
    if overrides is not None and not isinstance(overrides, Mapping):
        raise ValueError("Commitment settings must be a mapping")
    unknown = set(overrides or ()) - set(DEFAULT_SETTINGS)
    if unknown:
        raise ValueError(f"Unknown commitment settings: {sorted(unknown)}")
    settings = {**DEFAULT_SETTINGS, **(overrides or {})}
    for name in ("goal_tolerance_m", "frontier_match_radius_m", "minimum_progress_m"):
        value = settings[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
        settings[name] = float(value)
    for name in ("max_commit_steps", "stagnation_steps"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        settings[name] = int(value)
    if settings["goal_tolerance_m"] >= settings["frontier_match_radius_m"]:
        raise ValueError("Goal tolerance must be smaller than frontier matching radius")
    return settings


def _baseline_choice(frontiers, visited, attempted):
    """Exactly the baseline score/filter and input-order tie break."""
    choices = []
    for frontier in frontiers:
        goal = np.asarray(frontier.goal_xy)
        if frontier.distance_m < 0.45:
            continue
        if any(np.linalg.norm(goal - point) < 0.5 for point in attempted):
            continue
        revisits = sum(np.linalg.norm(goal - point) < 0.7 for point in visited)
        score = frontier.gain_cells / (1 + frontier.distance_m) / (1 + 4 * revisits)
        choices.append((score, frontier))
    return max(choices, key=lambda item: item[0])[1] if choices else None


def _path_distance(path, position):
    """Remaining physical route length including the actual stance to cell center."""
    return float(
        np.linalg.norm(path[0] - position) + np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    )


class FrontierCommitmentPolicy:
    """Stateful choice adapter with a JSON-serializable deterministic audit trail.

    Call ``observe_grid`` before each sensing sweep or immediately before
    ``choose``. The map object is retained by reference, so the latter choice
    sees that sweep's newly acquired observations. Position is copied. No map
    mutation, ground truth, future frames, room labels or learned parameters are
    used here.

    A commitment is released on arrival, blockage, disappearance of the matched
    frontier, insufficient route progress, an attempted goal, or the fixed cap.
    Release rescoring may select the same goal again; the cap bounds priority,
    rather than banning a location that is still the best baseline candidate.
    """

    def __init__(self, settings=None):
        self.settings = _settings(settings)
        self.decisions = []
        self._grid = None
        self._position = None
        self._goal = None
        self._look = None
        self._steps = 0
        self._stagnation = 0
        self._remaining_distance = None
        self._commitment_id = 0

    @property
    def last_decision(self):
        """Most recent diagnostics, or None before the first choice."""
        return self.decisions[-1] if self.decisions else None

    def observe_grid(self, grid, position):
        """Bind acquired mapping state and known current position only."""
        if not isinstance(grid, OccupancyGrid):
            raise ValueError("Commitment requires an acquired OccupancyGrid")
        position = np.asarray(position, dtype=float)
        if position.shape not in ((2,), (3,)) or not np.isfinite(position).all():
            raise ValueError("Position must be finite XY or XYZ")
        self._grid = grid
        self._position = position[:2].copy()

    def _match(self, frontiers):
        radius = self.settings["frontier_match_radius_m"]
        candidates = []
        for index, frontier in enumerate(frontiers):
            goal_distance = float(np.linalg.norm(np.asarray(frontier.goal_xy) - self._goal))
            look_distance = float(np.linalg.norm(np.asarray(frontier.look_at_xy) - self._look))
            if max(goal_distance, look_distance) <= radius:
                candidates.append((look_distance, goal_distance, index, frontier))
        return min(candidates, key=lambda item: item[:3])[-1] if candidates else None

    def _retained(self, frontiers, attempted):
        """Return safe retained goal, release reason, remaining route and progress."""
        position = self._position
        if np.linalg.norm(position - self._goal) <= self.settings["goal_tolerance_m"]:
            return None, "reached", None, None
        if any(np.linalg.norm(self._goal - point) < 0.5 for point in attempted):
            return None, "attempted", None, None
        path = self._grid.path(position, self._goal)
        if path is None:
            return None, "blocked", None, None
        match = self._match(frontiers)
        if match is None:
            return None, "resolved", None, None
        remaining = _path_distance(path, position)
        progress = self._remaining_distance - remaining
        self._stagnation = (
            self._stagnation + 1 if progress < self.settings["minimum_progress_m"] else 0
        )
        if self._stagnation >= self.settings["stagnation_steps"]:
            return None, "stalled", remaining, progress
        if self._steps >= self.settings["max_commit_steps"]:
            return None, "step_cap", remaining, progress
        self._steps += 1
        self._remaining_distance = remaining
        self._look = np.asarray(match.look_at_xy).copy()
        retained = Frontier(
            tuple(self._goal.tolist()),
            tuple(self._look.tolist()),
            path,
            match.gain_cells,
            remaining,
            match.gain_cells / (1 + remaining),
        )
        return retained, None, remaining, progress

    def choose(self, frontiers, visited, attempted):
        """Select a frontier and append one deterministic, serializable decision."""
        if self._grid is None:
            raise RuntimeError("Call observe_grid before choosing a frontier")
        frontiers, visited, attempted = list(frontiers), list(visited), list(attempted)
        greedy = _baseline_choice(frontiers, visited, attempted)
        previous_goal = None if self._goal is None else self._goal.tolist()
        previous_steps = self._steps
        previous_stagnation = self._stagnation
        selected, reason, remaining, progress = None, None, None, None
        if self._goal is not None:
            selected, reason, remaining, progress = self._retained(frontiers, attempted)
        retained = selected is not None
        if not retained:
            selected = greedy
            self._goal = self._look = self._remaining_distance = None
            self._steps = self._stagnation = 0
            if selected is not None:
                self._goal = np.asarray(selected.goal_xy).copy()
                self._look = np.asarray(selected.look_at_xy).copy()
                self._remaining_distance = _path_distance(selected.path_xy, self._position)
                self._steps = 1
                self._commitment_id += 1
        self.decisions.append(
            {
                "decision": len(self.decisions),
                "action": "retained" if retained else "rescored",
                "release_reason": reason,
                "previous_goal_xy": previous_goal,
                "baseline_goal_xy": None if greedy is None else list(greedy.goal_xy),
                "selected_goal_xy": None if selected is None else list(selected.goal_xy),
                "selected_look_at_xy": None if selected is None else list(selected.look_at_xy),
                "commitment_id": self._commitment_id if selected is not None else None,
                "commit_steps": self._steps,
                "previous_commit_steps": previous_steps,
                "stagnation_steps": self._stagnation,
                "previous_stagnation_steps": previous_stagnation,
                "remaining_distance_m": remaining,
                "route_progress_m": progress,
            }
        )
        return selected
