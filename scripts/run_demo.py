"""
End-to-End Command-Line Demonstration of the Factory Layout Optimization Engine.
Runs benchmark optimization, checks constraints, and exports results.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
from src.layout import FactoryLayout
from src.constraints import validate_layout
from src.distance import compute_layout_objective
from src.optimizer import SimulatedAnnealingOptimizer, RandomSearchOptimizer
from src.metrics import compare_layouts
from src.visualization import render_comparison_figure, render_convergence_plot
from src.export import (
    export_layout_to_csv,
    export_layout_to_json,
    export_metrics_to_json,
    export_figure_to_png,
    generate_executive_summary_markdown
)
from src.config import OUTPUTS_DIR


def main():
    print("=" * 70)
    print(" FACTORY LAYOUT OPTIMIZER - WORK SYSTEM DESIGN DEMO")
    print("=" * 70)

    # 1. Load benchmark factory
    layout = FactoryLayout.load_default()
    print(f"\n1. Loaded benchmark factory: {layout.factory_width}m x {layout.factory_height}m")
    print(f"   Equipment count: {len(layout.machines)}")
    print(f"   Material flow links: {len(layout.flows)}")
    print(f"   Structural obstacles: {len(layout.obstacles)}")

    # 2. Check initial layout feasibility and objective
    init_feas = validate_layout(layout)
    init_cost, flow_details = compute_layout_objective(layout, "manhattan")
    print(f"\n2. Initial Layout Analysis:")
    print(f"   Feasibility: {'FEASIBLE' if init_feas.is_feasible else 'INFEASIBLE'}")
    print(f"   Flow-Weighted Objective (J): {init_cost:,.2f} m-moves/shift")

    # 3. Run Simulated Annealing optimizer
    print("\n3. Executing Simulated Annealing optimization heuristic...")
    optimizer = SimulatedAnnealingOptimizer(
        initial_temperature=500.0,
        cooling_rate=0.985,
        max_iterations=1200,
        step_size=1.0,
        random_seed=42,
        distance_metric="manhattan"
    )

    result = optimizer.optimize(layout)

    print(f"\n4. Optimization Completed in {result.runtime_seconds:.2f} seconds:")
    print(f"   Status: {result.status}")
    print(f"   Initial Cost: {result.initial_cost:,.2f} m-moves/shift")
    print(f"   Optimized Cost: {result.best_cost:,.2f} m-moves/shift")
    print(f"   Improvement: {result.improvement_percent:+.2f}%")
    print(f"   Final Feasibility: {'FEASIBLE' if result.feasibility_report.is_feasible else 'INFEASIBLE'}")

    # 5. Generate comparison table
    df_comp, summary = compare_layouts(
        original_layout=layout,
        optimized_layout=result.best_layout,
        runtime_seconds=result.runtime_seconds,
        distance_metric="manhattan"
    )
    print("\n5. Quantitative Comparison Table:")
    print(df_comp.to_string(index=False))

    # 6. Export artifacts to outputs/
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUTS_DIR / "optimized_equipment_coordinates.csv"
    json_path = OUTPUTS_DIR / "optimized_factory_config.json"
    metrics_path = OUTPUTS_DIR / "optimization_metrics.json"
    comp_fig_path = OUTPUTS_DIR / "layout_comparison.png"
    conv_fig_path = OUTPUTS_DIR / "sa_convergence.png"
    summary_path = OUTPUTS_DIR / "executive_summary.md"

    export_layout_to_csv(result.best_layout, csv_path)
    export_layout_to_json(result.best_layout, json_path)
    export_metrics_to_json(summary, metrics_path)

    comp_fig = render_comparison_figure(
        layout,
        result.best_layout,
        result.initial_cost,
        result.best_cost,
        result.improvement_percent
    )
    export_figure_to_png(comp_fig, comp_fig_path)

    conv_fig = render_convergence_plot(result.history)
    export_figure_to_png(conv_fig, conv_fig_path)

    md_content = generate_executive_summary_markdown(layout, result.best_layout, summary)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\n6. Exported Deliverables:")
    print(f"   - Coordinates CSV: {csv_path}")
    print(f"   - Layout JSON: {json_path}")
    print(f"   - Metrics JSON: {metrics_path}")
    print(f"   - Comparison Visual: {comp_fig_path}")
    print(f"   - Convergence Visual: {conv_fig_path}")
    print(f"   - Executive Summary: {summary_path}")
    print("\nDemo execution complete!")


if __name__ == "__main__":
    main()
