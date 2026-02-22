"""Data types for route planning."""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Set


@dataclass
class RouteWaypoint:
    """A single waypoint in the planned route."""
    track_id: int
    class_name: str
    order: int
    depth_m: float
    reasoning: str = ""


@dataclass
class PlannedRoute:
    """A planned route through detected objects."""
    waypoints: List[RouteWaypoint] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    avoid_ids: List[int] = field(default_factory=list)
    target_id: Optional[int] = None
    planning_notes: str = ""
    is_valid: bool = True

    def age_seconds(self) -> float:
        """How old is this route plan?"""
        return time.time() - self.timestamp

    def get_waypoint_ids(self) -> List[int]:
        """Get list of track IDs in visit order."""
        return [w.track_id for w in self.waypoints]

    def is_monotonic_depth(self) -> bool:
        """Check if waypoints are in increasing depth order."""
        if len(self.waypoints) < 2:
            return True
        depths = [w.depth_m for w in self.waypoints]
        return all(depths[i] <= depths[i+1] for i in range(len(depths)-1))

    def sort_by_depth(self) -> 'PlannedRoute':
        """Return a new route with waypoints sorted by depth (nearest first)."""
        sorted_waypoints = sorted(self.waypoints, key=lambda w: w.depth_m)
        for i, wp in enumerate(sorted_waypoints):
            wp.order = i + 1
        return PlannedRoute(
            waypoints=sorted_waypoints,
            timestamp=self.timestamp,
            avoid_ids=self.avoid_ids,
            target_id=sorted_waypoints[-1].track_id if sorted_waypoints else None,
            planning_notes=self.planning_notes,
            is_valid=self.is_valid
        )

    def similarity_score(self, other: 'PlannedRoute') -> float:
        """
        Calculate similarity between two routes (0.0 to 1.0).
        Based on overlap of track IDs.
        """
        if not self.waypoints or not other.waypoints:
            return 0.0

        my_ids = set(self.get_waypoint_ids())
        other_ids = set(other.get_waypoint_ids())

        if not my_ids or not other_ids:
            return 0.0

        intersection = len(my_ids & other_ids)
        union = len(my_ids | other_ids)

        return intersection / union if union > 0 else 0.0
