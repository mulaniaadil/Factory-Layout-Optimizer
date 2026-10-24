"""
Automated unit tests for layout optimization algorithms (Simulated Annealing and Baseline).
"""

import pytest
from src.layout import FactoryLayout, Machine, MaterialFlow, Dock
from src.optimizer import SimulatedAnnealingOptimizer, RandomSearchOptimizer
from src.constraints import validate_layout


def test_optimizer_improves_suboptimal_layout():
    """
    Test problem where a high-volume machine is placed unnecessarily far from receiving dock.
    Optimizer should move the machine closer to receiving dock, reducing objective cost J.
    """
    layout = FactoryLayout(factory_width=30.0, factory_height=20.0)
    layout.docks["Receiving Dock"] = Dock(name="Receiving Dock", x=0.0, y=8.0, width=2.0, height=4.0)

    # Place Machine A at far right (x=25.0), with massive flow from dock (100 moves)
    mA = Machine(id="M_A", name="Saw", category="Cutting machine", x=25.0, y=8.0, width=3.0, height=2.0, is_fixed=False)
    # Machine B near dock (x=4.0) with small flow (5 moves)
    mB = Machine(id="M_B", name="Drill", category="Drilling machine", x=4.0, y=8.0, width=2.0, height=2.0, is_fixed=False)

    layout.machines = [mA, mB]
    layout.flows = [
        MaterialFlow(from_id="Receiving Dock", to_id="M_A", frequency=100.0),
        MaterialFlow(from_id="Receiving Dock", to_id="M_B", frequency=5.0),
    ]

    optimizer = SimulatedAnnealingOptimizer(
        initial_temperature=300.0,
        cooling_rate=0.95,
        max_iterations=400,
        step_size=2.0,
        random_seed=42
    )

    result = optimizer.optimize(layout)

    assert result.status == "SUCCESS"
    assert result.best_cost < result.initial_cost
    assert result.improvement_percent > 0.0
    assert result.feasibility_report.is_feasible


def test_optimizer_preserves_fixed_machines():
    """Verify that is_fixed=True machines NEVER change coordinates during optimization."""
    layout = FactoryLayout(factory_width=30.0, factory_height=20.0)
    m_fixed = Machine(id="M_FIXED", name="Heavy Press", category="Cutting machine", x=12.0, y=10.0, width=4.0, height=3.0, is_fixed=True)
    m_movable = Machine(id="M_MOVABLE", name="Assembly", category="Assembly workstation", x=22.0, y=5.0, width=3.0, height=2.0, is_fixed=False)

    layout.machines = [m_fixed, m_movable]
    layout.flows = [MaterialFlow(from_id="M_FIXED", to_id="M_MOVABLE", frequency=50.0)]

    optimizer = SimulatedAnnealingOptimizer(
        initial_temperature=200.0,
        cooling_rate=0.96,
        max_iterations=300,
        random_seed=123
    )

    result = optimizer.optimize(layout)
    opt_fixed = result.best_layout.get_machine("M_FIXED")

    assert opt_fixed is not None
    assert opt_fixed.x == 12.0
    assert opt_fixed.y == 10.0


def test_optimizer_reproducibility():
    """Verify that runs with the same random seed produce identical objective values."""
    layout = FactoryLayout.load_default()

    opt1 = SimulatedAnnealingOptimizer(max_iterations=150, random_seed=999)
    res1 = opt1.optimize(layout)

    opt2 = SimulatedAnnealingOptimizer(max_iterations=150, random_seed=999)
    res2 = opt2.optimize(layout)

    assert pytest.approx(res1.best_cost, rel=1e-5) == res2.best_cost


def test_no_overlap_in_returned_layout():
    """Verify that the optimized layout has zero overlapping machine footprints."""
    layout = FactoryLayout.load_default()

    optimizer = SimulatedAnnealingOptimizer(max_iterations=250, random_seed=77)
    result = optimizer.optimize(layout)

    report = validate_layout(result.best_layout)
    assert report.is_feasible
    assert report.machine_overlaps == 0
    assert report.boundary_violations == 0
