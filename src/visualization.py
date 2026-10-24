"""
Engineering visualization module for 2D factory floor plans, material flows,
and before-and-after optimization comparison plots using Matplotlib and Plotly.
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
import numpy as np

from src.layout import FactoryLayout, Machine, Obstacle, Dock
from src.config import CLASS_COLORS
from src.distance import compute_layout_objective


def render_factory_layout_matplotlib(
    layout: FactoryLayout,
    title: str = "Factory Layout",
    show_flows: bool = True,
    show_clearances: bool = True,
    highlight_fixed: bool = True,
    figsize: Tuple[int, int] = (12, 8)
) -> Figure:
    """
    Renders an engineering CAD-quality 2D floor plan of the factory layout.
    Origin at bottom-left (0,0), units in metres.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=120)

    W = layout.factory_width
    H = layout.factory_height

    # 1. Background epoxy floor and perimeter walls
    ax.set_facecolor("#F8F9FA")
    wall = patches.Rectangle((0, 0), W, H, linewidth=3, edgecolor="#2C3E50", facecolor="none", zorder=2)
    ax.add_patch(wall)

    # Grid lines (every 1 metre and 5 metres)
    ax.set_xticks(np.arange(0, W + 1, 5))
    ax.set_yticks(np.arange(0, H + 1, 5))
    ax.grid(True, which="both", color="#E2E8F0", linestyle="--", linewidth=0.7, zorder=1)

    # 2. Render loading docks (Receiving and Dispatch)
    for name, dock in layout.docks.items():
        dock_color = "#3498DB" if "receiv" in name.lower() else "#1ABC9C"
        dock_patch = patches.Rectangle(
            (dock.x, dock.y), dock.width, dock.height,
            linewidth=2, edgecolor="#2980B9", facecolor=dock_color, alpha=0.35,
            hatch="//", zorder=3
        )
        ax.add_patch(dock_patch)
        ax.text(
            dock.x + dock.width / 2.0, dock.y + dock.height / 2.0,
            f"{name}\n({dock.width}x{dock.height}m)",
            color="#1B4F72", fontsize=8, fontweight="bold",
            ha="center", va="center", zorder=4
        )

    # 3. Render structural obstacles
    for obs in layout.obstacles:
        obs_patch = patches.Rectangle(
            (obs.x, obs.y), obs.width, obs.height,
            linewidth=1.5, edgecolor="#34495E", facecolor="#7F8C8D",
            hatch="xx", zorder=3
        )
        ax.add_patch(obs_patch)
        ax.text(
            obs.x + obs.width / 2.0, obs.y + obs.height / 2.0,
            f"{obs.name}",
            color="#2C3E50", fontsize=7, fontweight="bold",
            ha="center", va="center", zorder=4
        )

    # 4. Render material flow vectors (if requested)
    if show_flows and layout.flows:
        max_freq = max((f.frequency for f in layout.flows), default=1.0)
        for flow in layout.flows:
            p1 = layout.get_entity_center(flow.from_id)
            p2 = layout.get_entity_center(flow.to_id)
            if p1 and p2:
                # Proportional thickness and opacity
                norm_w = 0.5 + (flow.frequency / max_freq) * 3.0
                alpha = 0.35 + (flow.frequency / max_freq) * 0.45
                ax.annotate(
                    "",
                    xy=p2, xytext=p1,
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="#E67E22",
                        lw=norm_w,
                        alpha=alpha,
                        mutation_scale=12,
                        shrinkA=12,
                        shrinkB=12
                    ),
                    zorder=5
                )

    # 5. Render machines
    for m in layout.machines:
        col = CLASS_COLORS.get(m.category, "#95A5A6")

        # Safety clearance perimeter (dashed)
        if show_clearances and m.clearance > 0:
            c_patch = patches.Rectangle(
                (m.x - m.clearance, m.y - m.clearance),
                m.width + 2 * m.clearance,
                m.height + 2 * m.clearance,
                linewidth=1, edgecolor="#F39C12", facecolor="none",
                linestyle=":", zorder=6
            )
            ax.add_patch(c_patch)

        # Machine physical body
        edge_style = "-" if not m.is_fixed else "--"
        edge_width = 2.0 if not m.is_fixed else 3.0
        edge_col = "#2C3E50" if not m.is_fixed else "#C0392B"

        m_patch = patches.Rectangle(
            (m.x, m.y), m.width, m.height,
            linewidth=edge_width, edgecolor=edge_col,
            facecolor=col, alpha=0.85, linestyle=edge_style, zorder=7
        )
        ax.add_patch(m_patch)

        # Label inside machine
        fixed_tag = " [FIXED]" if (m.is_fixed and highlight_fixed) else ""
        label_text = f"{m.id}\n{m.name}{fixed_tag}\n{m.width:.1f}x{m.height:.1f}m"
        ax.text(
            m.x + m.width / 2.0, m.y + m.height / 2.0,
            label_text,
            color="#FFFFFF" if col not in ["#2ECC71", "#50E3C2"] else "#1A5276",
            fontsize=7, fontweight="bold",
            ha="center", va="center", zorder=8
        )

    ax.set_xlim(-1, W + 1)
    ax.set_ylim(-1, H + 1)
    ax.set_xlabel("Factory Floor Width (metres)", fontsize=10, fontweight="semibold")
    ax.set_ylabel("Factory Floor Length (metres)", fontsize=10, fontweight="semibold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    return fig


def render_comparison_figure(
    original_layout: FactoryLayout,
    optimized_layout: FactoryLayout,
    orig_cost: float,
    opt_cost: float,
    pct_improvement: float,
    show_flows: bool = True
) -> Figure:
    """Creates a high-impact, side-by-side comparison figure of original vs. optimized layouts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7), dpi=130)

    for ax, layout, title, cost in [
        (ax1, original_layout, f"ORIGINAL LAYOUT\nFlow Cost: {orig_cost:,.1f} m-moves/shift", orig_cost),
        (ax2, optimized_layout, f"OPTIMIZED LAYOUT ({pct_improvement:+.1f}%)\nFlow Cost: {opt_cost:,.1f} m-moves/shift", opt_cost)
    ]:
        W = layout.factory_width
        H = layout.factory_height
        ax.set_facecolor("#F8F9FA")
        wall = patches.Rectangle((0, 0), W, H, linewidth=2.5, edgecolor="#2C3E50", facecolor="none", zorder=2)
        ax.add_patch(wall)
        ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.6, zorder=1)

        # Docks
        for name, dock in layout.docks.items():
            d_patch = patches.Rectangle(
                (dock.x, dock.y), dock.width, dock.height,
                linewidth=1.5, edgecolor="#2980B9", facecolor="#3498DB", alpha=0.35, hatch="//", zorder=3
            )
            ax.add_patch(d_patch)
            ax.text(dock.x + dock.width / 2.0, dock.y + dock.height / 2.0, name, fontsize=7, fontweight="bold", ha="center", va="center", zorder=4)

        # Obstacles
        for obs in layout.obstacles:
            o_patch = patches.Rectangle(
                (obs.x, obs.y), obs.width, obs.height,
                linewidth=1.5, edgecolor="#34495E", facecolor="#7F8C8D", hatch="xx", zorder=3
            )
            ax.add_patch(o_patch)

        # Flows
        if show_flows and layout.flows:
            max_f = max((f.frequency for f in layout.flows), default=1.0)
            for f in layout.flows:
                p1 = layout.get_entity_center(f.from_id)
                p2 = layout.get_entity_center(f.to_id)
                if p1 and p2:
                    ax.annotate(
                        "", xy=p2, xytext=p1,
                        arrowprops=dict(arrowstyle="-|>", color="#E67E22", lw=0.5 + (f.frequency / max_f) * 2.5, alpha=0.4, mutation_scale=10),
                        zorder=5
                    )

        # Machines
        for m in layout.machines:
            col = CLASS_COLORS.get(m.category, "#95A5A6")
            m_patch = patches.Rectangle(
                (m.x, m.y), m.width, m.height,
                linewidth=1.5, edgecolor="#2C3E50" if not m.is_fixed else "#C0392B",
                facecolor=col, alpha=0.85, zorder=6
            )
            ax.add_patch(m_patch)
            ax.text(
                m.x + m.width / 2.0, m.y + m.height / 2.0,
                f"{m.id}\n{m.category[:10]}",
                color="#FFFFFF" if col not in ["#2ECC71", "#50E3C2"] else "#1A5276",
                fontsize=6.5, fontweight="bold", ha="center", va="center", zorder=7
            )

        ax.set_xlim(-1, W + 1)
        ax.set_ylim(-1, H + 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Width (m)", fontsize=9)
        ax.set_ylabel("Length (m)", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    plt.tight_layout()
    return fig


def render_convergence_plot(history: List[Any], title: str = "Simulated Annealing Convergence") -> Figure:
    """Plots objective cost and cooling temperature progression across iterations."""
    fig, ax1 = plt.subplots(figsize=(10, 4.5), dpi=120)

    iterations = [h.iteration for h in history]
    best_costs = [h.best_cost for h in history]
    current_costs = [h.current_cost for h in history]
    temperatures = [h.temperature for h in history]

    ax1.set_xlabel("Iteration", fontweight="semibold")
    ax1.set_ylabel("Flow-Distance Objective (m-moves/shift)", color="#C0392B", fontweight="semibold")
    ax1.plot(iterations, current_costs, label="Current Candidate Cost", color="#E67E22", alpha=0.4, linewidth=1)
    ax1.plot(iterations, best_costs, label="Best Feasible Cost", color="#C0392B", linewidth=2.5)
    ax1.tick_params(axis="y", labelcolor="#C0392B")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Temperature (T)", color="#2980B9", fontweight="semibold")
    ax2.plot(iterations, temperatures, label="Annealing Temperature", color="#2980B9", linestyle=":", linewidth=1.5)
    ax2.tick_params(axis="y", labelcolor="#2980B9")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig
