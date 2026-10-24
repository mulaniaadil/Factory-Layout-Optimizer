"""
Quantitative layout evaluation and comparison metrics.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple
import pandas as pd
from src.layout import FactoryLayout
from src.constraints import validate_layout
from src.distance import compute_layout_objective


def calculate_layout_metrics(
    layout: FactoryLayout,
    distance_metric: str = "manhattan"
) -> Dict[str, Any]:
    """Compute complete quantitative performance metrics for a layout."""
    objective_cost, flow_details = compute_layout_objective(layout, distance_metric)
    feasibility = validate_layout(layout)
    
    total_flow_frequency = sum(f.frequency for f in layout.flows)
    total_unweighted_distance = sum(fd.distance for fd in flow_details)
    avg_distance_per_move = (total_unweighted_distance / len(flow_details)) if flow_details else 0.0

    critical_viols = [v for v in feasibility.violations if v.severity == "critical"]
    warning_viols = [v for v in feasibility.violations if v.severity == "warning"]

    return {
        "is_feasible": feasibility.is_feasible,
        "violations_count": len(critical_viols),
        "warning_count": len(warning_viols),
        "total_violations": feasibility.total_violations,
        "objective_cost": round(objective_cost, 2),
        "total_moves_per_shift": int(total_flow_frequency),
        "total_unweighted_distance_m": round(total_unweighted_distance, 2),
        "avg_distance_per_flow_m": round(avg_distance_per_move, 2),
        "floor_area_m2": round(layout.total_floor_area, 2),
        "equipment_area_m2": round(layout.occupied_equipment_area, 2),
        "obstacle_area_m2": round(layout.occupied_obstacle_area, 2),
        "space_utilization_pct": round(layout.space_utilization_percent, 2),
        "machine_count": len(layout.machines),
        "flow_count": len(layout.flows),
        "distance_metric": distance_metric
    }


def compare_layouts(
    original_layout: FactoryLayout,
    optimized_layout: FactoryLayout,
    runtime_seconds: float = 0.0,
    distance_metric: str = "manhattan"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compare baseline and optimized layouts quantitatively."""
    m_orig = calculate_layout_metrics(original_layout, distance_metric)
    m_opt = calculate_layout_metrics(optimized_layout, distance_metric)

    orig_obj = m_orig["objective_cost"]
    opt_obj = m_opt["objective_cost"]

    if orig_obj > 0 and m_orig["is_feasible"]:
        pct_change = ((orig_obj - opt_obj) / orig_obj) * 100.0
    else:
        pct_change = 0.0

    comparison_dict = {
        "Metric": [
            "Feasibility Status",
            "Flow-Weighted Travel (m-moves/shift)",
            "Total Flow Movements (per shift)",
            "Total Unweighted Distance (m)",
            "Avg Distance per Flow Link (m)",
            "Equipment Footprint Area (m²)",
            "Floor Space Utilization (%)",
            "Critical Violations Count",
            "Optimization Runtime (s)",
            "Improvement vs Original (%)"
        ],
        "Original Layout": [
            "FEASIBLE" if m_orig["is_feasible"] else "INFEASIBLE",
            f"{orig_obj:,.2f}",
            f"{m_orig['total_moves_per_shift']:,}",
            f"{m_orig['total_unweighted_distance_m']:,.2f}",
            f"{m_orig['avg_distance_per_flow_m']:.2f}",
            f"{m_orig['equipment_area_m2']:.2f}",
            f"{m_orig['space_utilization_pct']:.1f}%",
            str(m_orig["violations_count"]),
            "-",
            "-"
        ],
        "Optimized Layout": [
            "FEASIBLE" if m_opt["is_feasible"] else "INFEASIBLE",
            f"{opt_obj:,.2f}",
            f"{m_opt['total_moves_per_shift']:,}",
            f"{m_opt['total_unweighted_distance_m']:,.2f}",
            f"{m_opt['avg_distance_per_flow_m']:.2f}",
            f"{m_opt['equipment_area_m2']:.2f}",
            f"{m_opt['space_utilization_pct']:.1f}%",
            str(m_opt["violations_count"]),
            f"{runtime_seconds:.2f} s",
            f"{pct_change:+.2f}%"
        ]
    }

    df = pd.DataFrame(comparison_dict)
    summary = {
        "original_cost": orig_obj,
        "optimized_cost": opt_obj,
        "percent_improvement": pct_change,
        "runtime_seconds": runtime_seconds,
        "distance_metric": distance_metric,
        "original_feasible": m_orig["is_feasible"],
        "optimized_feasible": m_opt["is_feasible"]
    }

    return df, summary
