"""
Automated unit tests for distance calculations and flow-weighted objective function.
"""

import pytest
from src.distance import (
    calculate_manhattan_distance,
    calculate_euclidean_distance,
    compute_layout_objective,
    GridAislePathRouter
)
from src.layout import FactoryLayout, Machine, MaterialFlow, Obstacle


def test_hand_calculated_manhattan_distance():
    """Verify hand-calculated Manhattan distance between two coordinates."""
    p1 = (2.0, 3.0)
    p2 = (8.0, 11.0)
    # |8 - 2| + |11 - 3| = 6 + 8 = 14
    dist = calculate_manhattan_distance(p1, p2)
    assert dist == 14.0


def test_hand_calculated_euclidean_distance():
    """Verify 3-4-5 right triangle Euclidean distance."""
    p1 = (1.0, 1.0)
    p2 = (4.0, 5.0)
    # sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = 5.0
    dist = calculate_euclidean_distance(p1, p2)
    assert pytest.approx(dist, rel=1e-5) == 5.0


def test_hand_calculated_flow_objective():
    """
    Verify hand-calculated flow-weighted objective:
    Machine A center: (2.0, 2.0)
    Machine B center: (6.0, 5.0)  -> dist = |6-2| + |5-2| = 4 + 3 = 7.0
    Machine C center: (10.0, 2.0) -> dist A to C = |10-2| + |2-2| = 8.0
    Flow A->B = 10 moves/shift -> 10 * 7 = 70.0
    Flow B->C = 5 moves/shift  -> dist B to C = |10-6| + |2-5| = 4 + 3 = 7.0 -> 5 * 7 = 35.0
    Flow A->C = 2 moves/shift  -> 2 * 8 = 16.0
    Expected Total J = 70.0 + 35.0 + 16.0 = 121.0
    """
    layout = FactoryLayout(factory_width=20.0, factory_height=15.0)
    # Centers are (x + w/2, y + h/2)
    # Machine A: x=1.0, y=1.0, w=2.0, h=2.0 -> Center=(2.0, 2.0)
    mA = Machine(id="A", name="Machine A", category="Cutting machine", x=1.0, y=1.0, width=2.0, height=2.0)
    # Machine B: x=5.0, y=4.0, w=2.0, h=2.0 -> Center=(6.0, 5.0)
    mB = Machine(id="B", name="Machine B", category="Drilling machine", x=5.0, y=4.0, width=2.0, height=2.0)
    # Machine C: x=9.0, y=1.0, w=2.0, h=2.0 -> Center=(10.0, 2.0)
    mC = Machine(id="C", name="Machine C", category="Assembly workstation", x=9.0, y=1.0, width=2.0, height=2.0)

    layout.machines = [mA, mB, mC]
    layout.flows = [
        MaterialFlow(from_id="A", to_id="B", frequency=10.0),
        MaterialFlow(from_id="B", to_id="C", frequency=5.0),
        MaterialFlow(from_id="A", to_id="C", frequency=2.0),
    ]

    total_cost, details = compute_layout_objective(layout, distance_metric="manhattan")
    assert pytest.approx(total_cost, rel=1e-5) == 121.0
    assert len(details) == 3


def test_zero_flow_pairs():
    """Verify that flow pairs with zero frequency contribute 0 to the objective."""
    layout = FactoryLayout(factory_width=20.0, factory_height=15.0)
    m1 = Machine(id="M1", name="M1", category="Cutting machine", x=1.0, y=1.0, width=2.0, height=2.0)
    m2 = Machine(id="M2", name="M2", category="Cutting machine", x=10.0, y=10.0, width=2.0, height=2.0)
    layout.machines = [m1, m2]
    layout.flows = [MaterialFlow(from_id="M1", to_id="M2", frequency=0.0)]

    total_cost, _ = compute_layout_objective(layout, "manhattan")
    assert total_cost == 0.0


def test_grid_aisle_router_shortest_path():
    """Verify that GridAislePathRouter finds valid path around obstacle."""
    layout = FactoryLayout(factory_width=10.0, factory_height=10.0)
    # Obstacle in middle
    obs = Obstacle(id="O1", name="Pillar", x=4.0, y=2.0, width=2.0, height=6.0)
    layout.obstacles = [obs]

    router = GridAislePathRouter(layout, cell_size=0.5)
    dist = router.shortest_path_distance((2.0, 5.0), (8.0, 5.0))
    # Manhattan distance would be |8-2| + |5-5| = 6.0, but direct path blocked by 2m x 6m wall
    # So true path around obstacle must be > 6.0
    assert dist > 6.0
