"""
Automated unit tests for geometry and industrial constraints.
"""

import pytest
from src.layout import FactoryLayout, Machine, Obstacle, Dock
from src.constraints import (
    check_boundary,
    rect_overlap_area,
    validate_layout,
    is_layout_feasible
)


def test_machine_within_boundary():
    """Verify machine strictly inside boundary passes."""
    m = Machine(id="M1", name="Cutter", category="Cutting machine", x=2.0, y=2.0, width=3.0, height=2.0)
    viol = check_boundary(m, factory_width=30.0, factory_height=20.0)
    assert viol is None


def test_machine_outside_boundary():
    """Verify machine extending outside factory perimeter triggers boundary violation."""
    # Machine extends past x=30.0 (x=29.0, width=3.0 -> xmax=32.0)
    m = Machine(id="M1", name="Cutter", category="Cutting machine", x=29.0, y=5.0, width=3.0, height=2.0)
    viol = check_boundary(m, factory_width=30.0, factory_height=20.0)
    assert viol is not None
    assert viol.type == "boundary"
    assert viol.amount > 0

    # Negative coordinates
    m_neg = Machine(id="M2", name="Drill", category="Drilling machine", x=-1.0, y=5.0, width=2.0, height=2.0)
    viol_neg = check_boundary(m_neg, factory_width=30.0, factory_height=20.0)
    assert viol_neg is not None
    assert viol_neg.amount == 1.0


def test_rect_overlap_area_calculation():
    """Verify exact calculation of rectangle overlap areas."""
    r1 = (0.0, 0.0, 4.0, 4.0)
    r2 = (2.0, 2.0, 6.0, 6.0)
    # Overlap rectangle is [2, 2, 4, 4], width=2, height=2, area=4.0
    area = rect_overlap_area(r1, r2)
    assert pytest.approx(area, rel=1e-4) == 4.0


def test_edge_touching_rectangles():
    """Verify edge-touching rectangles have zero overlap area."""
    r1 = (0.0, 0.0, 4.0, 4.0)
    r2 = (4.0, 0.0, 8.0, 4.0)  # Touches at x=4.0
    area = rect_overlap_area(r1, r2)
    assert area == 0.0


def test_machine_overlapping_detection():
    """Verify that overlapping machine footprints are flagged as infeasible."""
    layout = FactoryLayout(factory_width=30.0, factory_height=20.0)
    m1 = Machine(id="M1", name="Cutter 1", category="Cutting machine", x=5.0, y=5.0, width=3.0, height=2.0)
    m2 = Machine(id="M2", name="Cutter 2", category="Cutting machine", x=6.0, y=5.5, width=3.0, height=2.0)
    layout.machines = [m1, m2]

    report = validate_layout(layout)
    assert not report.is_feasible
    assert report.machine_overlaps >= 1
    assert report.total_overlap_area > 0.0


def test_fixed_machine_immobility():
    """Verify that moving a fixed machine violates constraints."""
    layout = FactoryLayout(factory_width=30.0, factory_height=20.0)
    m_fixed = Machine(id="M_FIXED", name="Raw Buffer", category="Storage area", x=10.0, y=10.0, width=4.0, height=3.0, is_fixed=True)
    layout.machines = [m_fixed]

    # Original position was (5.0, 5.0)
    orig_pos = {"M_FIXED": (5.0, 5.0)}
    report = validate_layout(layout, original_fixed_positions=orig_pos)
    assert not report.is_feasible
    assert report.fixed_machine_moves == 1


def test_obstacle_collision():
    """Verify that machine colliding with structural obstacle is flagged."""
    layout = FactoryLayout(factory_width=30.0, factory_height=20.0)
    obs = Obstacle(id="OBS1", name="Column", x=10.0, y=10.0, width=2.0, height=2.0)
    layout.obstacles = [obs]

    # Place machine directly overlapping column
    m = Machine(id="M1", name="Press", category="Drilling machine", x=9.5, y=9.5, width=2.0, height=2.0)
    layout.machines = [m]

    report = validate_layout(layout)
    assert not report.is_feasible
    assert report.obstacle_collisions == 1
