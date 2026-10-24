"""
Constraint verification engine for industrial layout engineering.
Checks boundary, non-overlap, obstacle collision, fixed-machine preservation,
and safety clearance compliance.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from src.layout import FactoryLayout, Machine, Obstacle, Dock


@dataclass
class ConstraintViolation:
    """Detailed record of a constraint violation."""
    type: str  # 'boundary', 'machine_overlap', 'obstacle_collision', 'dock_collision', 'fixed_machine_moved', 'clearance'
    severity: str  # 'critical' (infeasible) or 'warning'
    entity_a: str
    entity_b: Optional[str] = None
    description: str = ""
    amount: float = 0.0  # Violation magnitude (e.g. overlap area or penetration distance)


@dataclass
class FeasibilityReport:
    """Summary and details of layout engineering checks."""
    is_feasible: bool
    total_violations: int
    boundary_violations: int = 0
    machine_overlaps: int = 0
    obstacle_collisions: int = 0
    dock_collisions: int = 0
    fixed_machine_moves: int = 0
    clearance_violations: int = 0
    total_overlap_area: float = 0.0
    violations: List[ConstraintViolation] = field(default_factory=list)

    def summary(self) -> str:
        if self.is_feasible:
            return "FEASIBLE: All geometric, boundary, and obstacle constraints are satisfied."
        lines = [f"INFEASIBLE: {self.total_violations} constraint violation(s) detected:"]
        if self.boundary_violations:
            lines.append(f" - {self.boundary_violations} machine(s) outside factory boundary")
        if self.machine_overlaps:
            lines.append(f" - {self.machine_overlaps} machine-to-machine overlap(s) (area: {self.total_overlap_area:.2f} m²)")
        if self.obstacle_collisions:
            lines.append(f" - {self.obstacle_collisions} collision(s) with structural obstacles")
        if self.dock_collisions:
            lines.append(f" - {self.dock_collisions} obstruction(s) of loading docks")
        if self.fixed_machine_moves:
            lines.append(f" - {self.fixed_machine_moves} fixed equipment position violation(s)")
        if self.clearance_violations:
            lines.append(f" - {self.clearance_violations} safety clearance breach(es)")
        return "\n".join(lines)


def rect_overlap_area(
    r1: Tuple[float, float, float, float],
    r2: Tuple[float, float, float, float],
    eps: float = 1e-5
) -> float:
    """
    Calculate intersection area of two axis-aligned rectangles.
    r = (xmin, ymin, xmax, ymax)
    """
    x_left = max(r1[0], r2[0])
    y_bottom = max(r1[1], r2[1])
    x_right = min(r1[2], r2[2])
    y_top = min(r1[3], r2[3])

    if (x_right - x_left) > eps and (y_top - y_bottom) > eps:
        return (x_right - x_left) * (y_top - y_bottom)
    return 0.0


def check_boundary(m: Machine, factory_width: float, factory_height: float, eps: float = 1e-4) -> Optional[ConstraintViolation]:
    """Verify that a machine footprint lies strictly within the factory perimeter."""
    x_min, y_min, x_max, y_max = m.bounds
    out_x = max(0.0, -x_min) + max(0.0, x_max - factory_width)
    out_y = max(0.0, -y_min) + max(0.0, y_max - factory_height)
    total_out = out_x + out_y

    if total_out > eps:
        return ConstraintViolation(
            type="boundary",
            severity="critical",
            entity_a=m.id,
            description=f"Machine '{m.name}' ({m.id}) extends outside perimeter. "
                        f"Bounds: [{x_min:.2f}, {y_min:.2f}, {x_max:.2f}, {y_max:.2f}], "
                        f"Factory: [0, 0, {factory_width:.1f}, {factory_height:.1f}]",
            amount=total_out
        )
    return None


def validate_layout(
    layout: FactoryLayout,
    original_fixed_positions: Optional[Dict[str, Tuple[float, float]]] = None,
    enforce_safety_clearance: bool = True
) -> FeasibilityReport:
    """
    Exhaustively check all industrial engineering constraints on a layout.
    """
    violations: List[ConstraintViolation] = []
    total_overlap_area = 0.0
    boundary_count = 0
    overlap_count = 0
    obstacle_count = 0
    dock_count = 0
    fixed_count = 0
    clearance_count = 0

    num_machines = len(layout.machines)

    # 1. Boundary checks
    for m in layout.machines:
        b_viol = check_boundary(m, layout.factory_width, layout.factory_height)
        if b_viol:
            violations.append(b_viol)
            boundary_count += 1

    # 2. Machine-to-machine physical overlap and clearance
    for i in range(num_machines):
        m1 = layout.machines[i]
        b1 = m1.bounds
        for j in range(i + 1, num_machines):
            m2 = layout.machines[j]
            b2 = m2.bounds
            
            # Physical footprint overlap
            overlap = rect_overlap_area(b1, b2)
            if overlap > 1e-5:
                total_overlap_area += overlap
                overlap_count += 1
                violations.append(
                    ConstraintViolation(
                        type="machine_overlap",
                        severity="critical",
                        entity_a=m1.id,
                        entity_b=m2.id,
                        description=f"Physical collision between '{m1.name}' and '{m2.name}' (Overlap: {overlap:.2f} m²)",
                        amount=overlap
                    )
                )
            elif enforce_safety_clearance:
                # Check safety clearance breach if no physical overlap
                required_clearance = max(m1.clearance, m2.clearance)
                b1_c = (m1.x - required_clearance, m1.y - required_clearance, m1.x + m1.width + required_clearance, m1.y + m1.height + required_clearance)
                c_overlap = rect_overlap_area(b1_c, b2)
                if c_overlap > 1e-5:
                    clearance_count += 1
                    violations.append(
                        ConstraintViolation(
                            type="clearance",
                            severity="warning",
                            entity_a=m1.id,
                            entity_b=m2.id,
                            description=f"Safety clearance violation ({required_clearance}m buffer) between '{m1.name}' and '{m2.name}'",
                            amount=c_overlap
                        )
                    )

    # 3. Machine vs Obstacle collision
    for m in layout.machines:
        m_bounds = m.bounds
        for obs in layout.obstacles:
            obs_bounds = obs.bounds
            overlap = rect_overlap_area(m_bounds, obs_bounds)
            if overlap > 1e-5:
                obstacle_count += 1
                violations.append(
                    ConstraintViolation(
                        type="obstacle_collision",
                        severity="critical",
                        entity_a=m.id,
                        entity_b=obs.id,
                        description=f"Machine '{m.name}' overlaps structural obstacle '{obs.name}' (Area: {overlap:.2f} m²)",
                        amount=overlap
                    )
                )

    # 4. Machine vs Dock obstruction
    for m in layout.machines:
        m_bounds = m.bounds
        for dock_key, dock in layout.docks.items():
            dock_bounds = dock.bounds
            overlap = rect_overlap_area(m_bounds, dock_bounds)
            if overlap > 1e-5:
                dock_count += 1
                violations.append(
                    ConstraintViolation(
                        type="dock_collision",
                        severity="critical",
                        entity_a=m.id,
                        entity_b=dock_key,
                        description=f"Machine '{m.name}' blocks loading bay '{dock.name}' (Area: {overlap:.2f} m²)",
                        amount=overlap
                    )
                )

    # 5. Fixed machine immobility check
    if original_fixed_positions:
        for m in layout.machines:
            if m.is_fixed and m.id in original_fixed_positions:
                orig_x, orig_y = original_fixed_positions[m.id]
                dx = abs(m.x - orig_x)
                dy = abs(m.y - orig_y)
                if dx > 1e-4 or dy > 1e-4:
                    fixed_count += 1
                    violations.append(
                        ConstraintViolation(
                            type="fixed_machine_moved",
                            severity="critical",
                            entity_a=m.id,
                            description=f"Fixed machine '{m.name}' moved from ({orig_x:.2f}, {orig_y:.2f}) to ({m.x:.2f}, {m.y:.2f})",
                            amount=dx + dy
                        )
                    )

    # A layout is strictly feasible if there are zero critical violations
    critical_violations = [v for v in violations if v.severity == "critical"]
    is_feasible = (len(critical_violations) == 0)

    return FeasibilityReport(
        is_feasible=is_feasible,
        total_violations=len(violations),
        boundary_violations=boundary_count,
        machine_overlaps=overlap_count,
        obstacle_collisions=obstacle_count,
        dock_collisions=dock_count,
        fixed_machine_moves=fixed_count,
        clearance_violations=clearance_count,
        total_overlap_area=total_overlap_area,
        violations=violations
    )


def is_layout_feasible(layout: FactoryLayout, enforce_clearance: bool = False) -> bool:
    """Fast boolean feasibility check for inner optimization loops."""
    # Boundary check
    W = layout.factory_width
    H = layout.factory_height
    for m in layout.machines:
        if m.x < 0 or m.y < 0 or (m.x + m.width) > W or (m.y + m.height) > H:
            return False

    # Obstacle check
    for m in layout.machines:
        mb = m.bounds
        for obs in layout.obstacles:
            if rect_overlap_area(mb, obs.bounds) > 1e-5:
                return False

    # Dock check
    for m in layout.machines:
        mb = m.bounds
        for dock in layout.docks.values():
            if rect_overlap_area(mb, dock.bounds) > 1e-5:
                return False

    # Machine overlap check
    n = len(layout.machines)
    for i in range(n):
        m1 = layout.machines[i]
        b1 = m1.bounds if not enforce_clearance else m1.bounds_with_clearance
        for j in range(i + 1, n):
            m2 = layout.machines[j]
            b2 = m2.bounds if not enforce_clearance else m2.bounds_with_clearance
            if rect_overlap_area(b1, b2) > 1e-5:
                return False

    return True
