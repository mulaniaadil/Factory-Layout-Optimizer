"""
Distance metrics and flow-weighted material transport objective calculation.
Implements Manhattan distance, Euclidean distance, and Obstacle-Aware Grid-Based Aisle Path Distance.
"""

from __future__ import annotations
import math
from collections import deque
from typing import Tuple, Dict, List, Optional, Any
from dataclasses import dataclass
from src.layout import FactoryLayout


@dataclass
class FlowDetail:
    """Detailed calculation for a single material flow pair."""
    from_entity: str
    to_entity: str
    frequency: float
    distance: float
    flow_distance: float  # frequency * distance
    from_center: Tuple[float, float]
    to_center: Tuple[float, float]


def calculate_manhattan_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Orthogonal rectilinear distance: |x1 - x2| + |y1 - y2|.
    Standard industrial engineering proxy for factory aisles aligned with building axes.
    """
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def calculate_euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Straight-line Euclidean distance: sqrt((x1 - x2)^2 + (y1 - y2)^2)."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


class GridAislePathRouter:
    """
    Breadth-First Search (BFS) shortest path router on a discretized factory grid.
    Accounts for physical machine footprints and structural obstacles.
    """
    def __init__(self, layout: FactoryLayout, cell_size: float = 0.5):
        self.cell_size = cell_size
        self.width = layout.factory_width
        self.height = layout.factory_height
        self.cols = int(math.ceil(self.width / cell_size))
        self.rows = int(math.ceil(self.height / cell_size))
        
        # 0 = free aisle space, 1 = occupied obstacle/machine footprint
        self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self._populate_grid(layout)

    def _coord_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        c = min(max(0, int(x / self.cell_size)), self.cols - 1)
        r = min(max(0, int(y / self.cell_size)), self.rows - 1)
        return (r, c)

    def _grid_to_coord(self, r: int, c: int) -> Tuple[float, float]:
        return ((c + 0.5) * self.cell_size, (r + 0.5) * self.cell_size)

    def _mark_rect(self, bounds: Tuple[float, float, float, float], val: int = 1):
        xmin, ymin, xmax, ymax = bounds
        r_start, c_start = self._coord_to_grid(xmin, ymin)
        r_end, c_end = self._coord_to_grid(xmax, ymax)
        for r in range(r_start, min(r_end + 1, self.rows)):
            for c in range(c_start, min(c_end + 1, self.cols)):
                self.grid[r][c] = val

    def _populate_grid(self, layout: FactoryLayout):
        # Mark structural obstacles
        for obs in layout.obstacles:
            self._mark_rect(obs.bounds, val=1)
        # Mark machines (interior is impassable)
        for m in layout.machines:
            self._mark_rect(m.bounds, val=1)

    def shortest_path_distance(self, start_pos: Tuple[float, float], end_pos: Tuple[float, float]) -> float:
        """Find the true path travel distance along aisles avoiding blocked cells."""
        start_r, start_c = self._coord_to_grid(*start_pos)
        end_r, end_c = self._coord_to_grid(*end_pos)

        if (start_r, start_c) == (end_r, end_c):
            return 0.0

        queue = deque([(start_r, start_c, 0)])
        visited = set([(start_r, start_c)])

        # 4-directional orthogonal movement (North, South, East, West)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            r, c, steps = queue.popleft()
            if (r, c) == (end_r, end_c):
                return steps * self.cell_size

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if (nr, nc) not in visited:
                        # Allow entering target even if occupied (pick/drop point)
                        if self.grid[nr][nc] == 0 or (nr, nc) == (end_r, end_c):
                            visited.add((nr, nc))
                            queue.append((nr, nc, steps + 1))

        # Fallback to Manhattan distance with 1.25x obstacle detour penalty if no free path found
        return calculate_manhattan_distance(start_pos, end_pos) * 1.25


def compute_layout_objective(
    layout: FactoryLayout,
    distance_metric: str = "manhattan"
) -> Tuple[float, List[FlowDetail]]:
    """
    Calculates total flow-distance objective:
    J = sum_{i,j} f(i,j) * d(i,j)

    Parameters:
    - layout: FactoryLayout instance
    - distance_metric: 'manhattan', 'euclidean', or 'aisle_grid'

    Returns:
    - total_objective: sum of flow-weighted distances (metre-moves / shift)
    - flow_details: list of FlowDetail items for inspection and reporting
    """
    total_cost = 0.0
    details: List[FlowDetail] = []
    
    router: Optional[GridAislePathRouter] = None
    if distance_metric == "aisle_grid":
        router = GridAislePathRouter(layout)

    for flow in layout.flows:
        p1 = layout.get_entity_center(flow.from_id)
        p2 = layout.get_entity_center(flow.to_id)

        if p1 is None or p2 is None:
            continue

        if distance_metric == "manhattan":
            dist = calculate_manhattan_distance(p1, p2)
        elif distance_metric == "euclidean":
            dist = calculate_euclidean_distance(p1, p2)
        elif distance_metric == "aisle_grid" and router is not None:
            dist = router.shortest_path_distance(p1, p2)
        else:
            dist = calculate_manhattan_distance(p1, p2)

        weighted_cost = flow.frequency * dist
        total_cost += weighted_cost

        details.append(
            FlowDetail(
                from_entity=flow.from_id,
                to_entity=flow.to_id,
                frequency=flow.frequency,
                distance=dist,
                flow_distance=weighted_cost,
                from_center=p1,
                to_center=p2
            )
        )

    return total_cost, details
