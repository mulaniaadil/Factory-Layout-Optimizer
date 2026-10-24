"""
AI-Based Factory Layout Optimizer
Web Application built with Streamlit for Work System Design Course Project.
"""

from __future__ import annotations
import json
import io
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Internal project modules
from src.config import (
    CLASS_ID_TO_NAME,
    CLASS_NAME_TO_ID,
    CLASS_COLORS,
    DEFAULT_DIMENSIONS,
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    load_config
)
from src.layout import FactoryLayout, Machine, Obstacle, Dock, MaterialFlow
from src.constraints import validate_layout, is_layout_feasible
from src.distance import compute_layout_objective
from src.calibration import CalibrationScale, reconstruct_layout_from_detections
from src.inference import (
    FactoryDetectorInference,
    generate_synthetic_demo_detections,
    draw_bounding_boxes_on_image
)
from src.optimizer import SimulatedAnnealingOptimizer, RandomSearchOptimizer
from src.metrics import calculate_layout_metrics, compare_layouts
from src.visualization import (
    render_factory_layout_matplotlib,
    render_comparison_figure,
    render_convergence_plot
)
from src.export import (
    export_layout_to_csv,
    export_layout_to_json,
    export_metrics_to_json,
    export_figure_to_png,
    generate_executive_summary_markdown
)
from src.data_generation import generate_sample_factory_image

# ---------------------------------------------------------
# Page Setup & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Factory Layout Optimizer | AI & Work System Design",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Engineering Dashboard CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "layout" not in st.session_state:
    st.session_state.layout = FactoryLayout.load_default()

if "original_layout" not in st.session_state:
    st.session_state.original_layout = st.session_state.layout.clone()

if "optimized_layout" not in st.session_state:
    st.session_state.optimized_layout = None

if "optimization_result" not in st.session_state:
    st.session_state.optimization_result = None

if "detections" not in st.session_state:
    st.session_state.detections = []

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "annotated_image" not in st.session_state:
    st.session_state.annotated_image = None

if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = "Synthetic Demonstration"


# ---------------------------------------------------------
# Sidebar Controls & Navigation
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/factory.png", width=64)
st.sidebar.title("Factory AI Optimizer")
st.sidebar.markdown("**Work System Design & Operations Research**")

page_selection = st.sidebar.radio(
    "Workflow Navigation",
    [
        "1. Project Overview",
        "2. Floor Plan & Detection",
        "3. Configure Factory & Flows",
        "4. Layout Optimization Engine",
        "5. Results & Comparative Analysis"
    ],
    index=0
)

st.sidebar.divider()
st.sidebar.subheader("Global Factory Dimensions")
col_w, col_h = st.sidebar.columns(2)
with col_w:
    new_w = st.number_input("Width (m)", min_value=10.0, max_value=200.0, value=float(st.session_state.layout.factory_width), step=5.0)
with col_h:
    new_h = st.number_input("Height (m)", min_value=10.0, max_value=200.0, value=float(st.session_state.layout.factory_height), step=5.0)

if new_w != st.session_state.layout.factory_width or new_h != st.session_state.layout.factory_height:
    st.session_state.layout.factory_width = new_w
    st.session_state.layout.factory_height = new_h
    st.session_state.original_layout.factory_width = new_w
    st.session_state.original_layout.factory_height = new_h

st.sidebar.markdown(f"**Floor Area**: {st.session_state.layout.total_floor_area:,.1f} m²")
st.sidebar.markdown(f"**Equipment Count**: {len(st.session_state.layout.machines)} units")
st.sidebar.markdown(f"**Active Material Flows**: {len(st.session_state.layout.flows)} links")

if st.sidebar.button("🔄 Reset to Benchmark Layout", use_container_width=True):
    st.session_state.layout = FactoryLayout.load_default()
    st.session_state.original_layout = st.session_state.layout.clone()
    st.session_state.optimized_layout = None
    st.session_state.optimization_result = None
    st.session_state.detections = []
    st.rerun()


# =========================================================
# PAGE 1: PROJECT OVERVIEW
# =========================================================
if page_selection == "1. Project Overview":
    st.markdown('<div class="main-title">Using Computer Vision to Design a Factory with Optimal Placement of Equipment</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">A unified academic framework bridging Deep Learning Object Detection with Industrial Engineering Layout Optimization</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Component A</div>
            <div class="metric-value">Computer Vision</div>
            <p style="color:#64748B; font-size:0.85rem; margin-top:8px;">
            PyTorch Faster R-CNN object detector trained on top-down factory floor imagery to extract equipment classes and bounding boxes.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Component B</div>
            <div class="metric-value">Spatial Calibration</div>
            <p style="color:#64748B; font-size:0.85rem; margin-top:8px;">
            Coordinate transformation from camera pixel space (top-left origin) to real-world metric coordinates (bottom-left origin) with physical footprint mapping.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Component C</div>
            <div class="metric-value">Layout Optimizer</div>
            <p style="color:#64748B; font-size:0.85rem; margin-top:8px;">
            Simulated Annealing heuristic minimizing flow-weighted material transport distance J under strict non-overlap, obstacle, and safety boundary constraints.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### System Architecture & Information Pipeline")
    st.markdown("""
    ```mermaid
    graph LR
        A[Top-Down Floor Plan Image] --> B[PyTorch Faster R-CNN]
        B --> C[Equipment Bounding Boxes & Classes]
        C --> D[Metric Spatial Calibration]
        D --> E[Editable 2D Factory Model]
        F[From-To Material Flow Matrix] --> G[Optimization Engine: Simulated Annealing]
        E --> G
        H[Geometric & Boundary Constraints] --> G
        G --> I[Optimized Feasible Layout]
        I --> J[Comparative Metrics & CAD Export]
    ```
    """)

    st.markdown("### Academic Problem Statement")
    st.info("""
    **Work System Design & Operations Research Context**:
    Traditional facility layout planning relies on manual CRAFT or quadratic assignment heuristics operating on tabular matrices disconnected from physical floor realities. 
    This project introduces an end-to-end cyber-physical pipeline:
    1. **Visual Perception**: Automated floor plan digitization without manual coordinate entry.
    2. **Physical Feasibility**: Strict enforcement of machine boundaries, non-overlap, safety buffers, and structural obstacles.
    3. **Objective**: Minimize total internal logistics transport: $J = \\sum_{i,j} f_{i,j} \\times d_{i,j}$ (m-moves per shift).
    """)

    st.markdown("### Quick Start Workflow")
    b1, b2, b3 = st.columns(3)
    if b1.button("🚀 Start with Synthetic Demonstration", use_container_width=True):
        st.session_state.demo_mode = "Synthetic Demonstration"
        # Prepopulate demo detections
        st.session_state.detections = generate_synthetic_demo_detections((800, 600))
        st.success("Loaded synthetic demonstration floor plan! Switch to Page 2 or Page 3.")
    if b2.button("📐 Configure Benchmark Factory", use_container_width=True):
        st.session_state.layout = FactoryLayout.load_default()
        st.success("Loaded benchmark 30m x 20m layout! Switch to Page 3.")
    if b3.button("⚡ Run Layout Optimization Directly", use_container_width=True):
        st.session_state.layout = FactoryLayout.load_default()
        st.success("Ready! Head to Page 4 to optimize.")


# =========================================================
# PAGE 2: UPLOAD & DETECT EQUIPMENT
# =========================================================
elif page_selection == "2. Floor Plan & Detection":
    st.markdown('<div class="main-title">Floor Plan Inspection & Computer Vision Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Perceive equipment footprints from top-down layout diagrams using PyTorch Faster R-CNN or Synthetic Demonstration</div>', unsafe_allow_html=True)

    mode_choice = st.radio(
        "Perception Mode Selection",
        ["Synthetic Demonstration (Verified CAD Geometry)", "PyTorch Neural Detector (Saved Checkpoint)"],
        horizontal=True
    )

    sample_image_path = DATA_DIR / "sample_factory_floor.png"
    if not sample_image_path.exists():
        generate_sample_factory_image(sample_image_path, seed=101)

    col_up1, col_up2 = st.columns([1, 1])

    with col_up1:
        st.markdown("#### Layout Image Source")
        uploaded_file = st.file_uploader("Upload factory floor plan (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
        
        use_sample = st.checkbox("Use bundled demonstration factory image", value=(uploaded_file is None))

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.session_state.uploaded_image = image
        elif use_sample and sample_image_path.exists():
            image = Image.open(sample_image_path).convert("RGB")
            st.session_state.uploaded_image = image
        else:
            image = None

        conf_threshold = st.slider("Detection Confidence Threshold", min_value=0.1, max_value=0.95, value=0.50, step=0.05)
        apply_footprints = st.checkbox("Enforce standardized machine dimensions during reconstruction", value=True)

    with col_up2:
        st.markdown("#### Input Image Preview")
        if st.session_state.uploaded_image is not None:
            st.image(st.session_state.uploaded_image, caption=f"Input Image ({st.session_state.uploaded_image.width}x{st.session_state.uploaded_image.height} px)", use_container_width=True)
        else:
            st.warning("Please upload a floor plan or check the bundled demonstration image.")

    st.divider()

    # Detection Execution
    if st.button("🔍 Run Equipment Perception", type="primary"):
        if st.session_state.uploaded_image is None:
            st.error("No image available for detection.")
        else:
            img = st.session_state.uploaded_image
            if "Synthetic" in mode_choice:
                # Mode 1: Synthetic Demonstration
                detections = generate_synthetic_demo_detections((img.width, img.height))
                annotated = draw_bounding_boxes_on_image(img, detections)
                st.session_state.detections = detections
                st.session_state.annotated_image = annotated
                st.success(f"Perceived {len(detections)} equipment units via Synthetic Demonstration Mode.")
            else:
                # Mode 2: PyTorch Model Inference
                weights_path = MODELS_DIR / "best_detector.pth"
                if not weights_path.exists():
                    st.error(f"Checkpoint not found at '{weights_path}'. To train a model locally, run 'python scripts/train.py' or train in Google Colab using notebooks/factory_layout_demo.ipynb.")
                    st.info("Switching automatically to Synthetic Demonstration Mode so you can proceed without interruption.")
                    detections = generate_synthetic_demo_detections((img.width, img.height))
                    annotated = draw_bounding_boxes_on_image(img, detections)
                    st.session_state.detections = detections
                    st.session_state.annotated_image = annotated
                else:
                    try:
                        infer = FactoryDetectorInference(weights_path)
                        detections = infer.predict(img, confidence_threshold=conf_threshold)
                        annotated = draw_bounding_boxes_on_image(img, detections)
                        st.session_state.detections = detections
                        st.session_state.annotated_image = annotated
                        st.success(f"PyTorch Faster R-CNN detected {len(detections)} equipment units with confidence >= {conf_threshold}.")
                    except Exception as e:
                        st.error(f"Inference error: {e}")

    # Display Detections & Correction Table
    if st.session_state.annotated_image is not None and st.session_state.detections:
        st.markdown("### Perception Results & Interactive Annotations")
        c_vis1, c_vis2 = st.columns([1, 1])
        with c_vis1:
            st.image(st.session_state.annotated_image, caption="Annotated Bounding Boxes", use_container_width=True)
        with c_vis2:
            st.markdown("#### Detected Equipment Table")
            det_df = pd.DataFrame([
                {
                    "ID": d["detection_id"],
                    "Category": d["class_name"],
                    "Confidence": f"{d['confidence']:.2f}",
                    "BBox [xmin, ymin, xmax, ymax]": str(d["bbox_px"]),
                    "Source": d.get("detection_source", "PyTorch")
                }
                for d in st.session_state.detections
            ])
            st.dataframe(det_df, use_container_width=True)

            if st.button("📥 Apply Detections to Factory Layout", type="secondary"):
                img = st.session_state.uploaded_image
                scale = CalibrationScale(
                    image_width_px=img.width,
                    image_height_px=img.height,
                    factory_width_m=st.session_state.layout.factory_width,
                    factory_height_m=st.session_state.layout.factory_height
                )
                reconstructed = reconstruct_layout_from_detections(
                    st.session_state.detections,
                    scale,
                    use_standard_dimensions=apply_footprints,
                    base_layout=st.session_state.layout
                )
                st.session_state.layout = reconstructed
                st.session_state.original_layout = reconstructed.clone()
                st.success(f"Transferred {len(reconstructed.machines)} equipment units into the factory model! Check Page 3.")


# =========================================================
# PAGE 3: CONFIGURE FACTORY & FLOWS
# =========================================================
elif page_selection == "3. Configure Factory & Flows":
    st.markdown('<div class="main-title">Configure Factory Geometry & Material Flows</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Review, modify, or add equipment dimensions, movable/fixed statuses, dock locations, and material flow frequencies</div>', unsafe_allow_html=True)

    tab_machines, tab_flows, tab_infra = st.tabs(["🏭 Equipment Footprints", "🔄 Material Flow Matrix", "🧱 Obstacles & Docks"])

    # Tab 1: Equipment editor
    with tab_machines:
        st.markdown("#### Equipment Inventory & Physical Footprints")
        st.markdown("Edit machine locations (X, Y in metres from bottom-left origin), footprint sizes, safety clearances, and movable/fixed statuses:")

        m_df = st.session_state.layout.to_dataframe()
        edited_m_df = st.data_editor(
            m_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Fixed": st.column_config.CheckboxColumn("Fixed Position?", help="If checked, machine cannot move during optimization"),
                "Category": st.column_config.SelectboxColumn("Category", options=list(DEFAULT_DIMENSIONS.keys()))
            }
        )

        if st.button("💾 Save Equipment Modifications"):
            updated_machines = []
            for _, row in edited_m_df.iterrows():
                cat = str(row["Category"])
                updated_machines.append(
                    Machine(
                        id=str(row["ID"]),
                        name=str(row["Name"]),
                        category=cat,
                        class_id=CLASS_NAME_TO_ID.get(cat, 1),
                        x=float(row["X (m)"]),
                        y=float(row["Y (m)"]),
                        width=float(row["Width (m)"]),
                        height=float(row["Height (m)"]),
                        is_fixed=bool(row["Fixed"]),
                        clearance=float(row["Clearance (m)"])
                    )
                )
            st.session_state.layout.machines = updated_machines
            st.session_state.original_layout = st.session_state.layout.clone()
            st.success("Updated equipment parameters successfully!")

    # Tab 2: Material Flows
    with tab_flows:
        st.markdown("#### From-To Material Flow Table (moves per shift)")
        st.caption("Flow frequencies represent material movement counts between machine pairs or docks.")

        flow_data = [
            {"From": f.from_id, "To": f.to_id, "Frequency (moves/shift)": f.frequency}
            for f in st.session_state.layout.flows
        ]
        f_df = pd.DataFrame(flow_data)
        edited_f_df = st.data_editor(f_df, num_rows="dynamic", use_container_width=True)

        if st.button("💾 Save Flow Matrix"):
            updated_flows = []
            for _, row in edited_f_df.iterrows():
                updated_flows.append(
                    MaterialFlow(
                        from_id=str(row["From"]),
                        to_id=str(row["To"]),
                        frequency=float(row["Frequency (moves/shift)"])
                    )
                )
            st.session_state.layout.flows = updated_flows
            st.session_state.original_layout = st.session_state.layout.clone()
            st.success("Updated material flow matrix successfully!")

    # Tab 3: Infrastructure (Docks & Obstacles)
    with tab_infra:
        st.markdown("#### Loading Bay Docks")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("**Receiving Dock**")
            rd = st.session_state.layout.docks.get("Receiving Dock", Dock(name="Receiving Dock", x=0, y=8, width=2.5, height=4))
            rd_x = st.number_input("Receiving X (m)", value=float(rd.x), step=0.5)
            rd_y = st.number_input("Receiving Y (m)", value=float(rd.y), step=0.5)
            rd_w = st.number_input("Receiving Width (m)", value=float(rd.width), step=0.5)
            rd_h = st.number_input("Receiving Height (m)", value=float(rd.height), step=0.5)
            st.session_state.layout.docks["Receiving Dock"] = Dock(name="Receiving Dock", x=rd_x, y=rd_y, width=rd_w, height=rd_h, color="#4A90E2")

        with col_d2:
            st.markdown("**Dispatch Dock**")
            dd = st.session_state.layout.docks.get("Dispatch Dock", Dock(name="Dispatch Dock", x=27.5, y=8, width=2.5, height=4))
            dd_x = st.number_input("Dispatch X (m)", value=float(dd.x), step=0.5)
            dd_y = st.number_input("Dispatch Y (m)", value=float(dd.y), step=0.5)
            dd_w = st.number_input("Dispatch Width (m)", value=float(dd.width), step=0.5)
            dd_h = st.number_input("Dispatch Height (m)", value=float(dd.height), step=0.5)
            st.session_state.layout.docks["Dispatch Dock"] = Dock(name="Dispatch Dock", x=dd_x, y=dd_y, width=dd_w, height=dd_h, color="#50E3C2")

        st.markdown("#### Structural Obstacles & Restricted Zones")
        obs_df = pd.DataFrame([
            {"ID": o.id, "Name": o.name, "X (m)": o.x, "Y (m)": o.y, "Width (m)": o.width, "Height (m)": o.height}
            for o in st.session_state.layout.obstacles
        ])
        st.dataframe(obs_df, use_container_width=True)

    st.divider()
    # Floor Plan Live Preview
    st.markdown("### Current 2D Factory Floor Plan Preview")
    show_fl = st.checkbox("Display Material Flow Arrows", value=True)
    show_cl = st.checkbox("Display Safety Clearance Zones", value=True)

    fig = render_factory_layout_matplotlib(
        st.session_state.layout,
        title=f"Configured Layout ({st.session_state.layout.factory_width}m x {st.session_state.layout.factory_height}m)",
        show_flows=show_fl,
        show_clearances=show_cl
    )
    st.pyplot(fig)


# =========================================================
# PAGE 4: OPTIMIZE THE LAYOUT
# =========================================================
elif page_selection == "4. Layout Optimization Engine":
    st.markdown('<div class="main-title">Factory Layout Optimization Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Formulate Quadratic Assignment Objective and Search for Optimal Machine Positions</div>', unsafe_allow_html=True)

    # Initial Layout Diagnostics
    report = validate_layout(st.session_state.layout)
    init_obj, _ = compute_layout_objective(st.session_state.layout, "manhattan")

    st.markdown("### 1. Pre-Optimization Engineering Diagnostics")
    c_d1, c_d2, c_d3, c_d4 = st.columns(4)
    with c_d1:
        st.metric("Initial Feasibility", "FEASIBLE" if report.is_feasible else "INFEASIBLE")
    with c_d2:
        st.metric("Initial Objective (J)", f"{init_obj:,.1f} m-moves")
    with c_d3:
        st.metric("Machine Overlaps", f"{report.machine_overlaps} pairs")
    with c_d4:
        st.metric("Floor Utilization", f"{st.session_state.layout.space_utilization_percent:.1f}%")

    if not report.is_feasible:
        st.warning(f"⚠️ Pre-optimization warning: The current layout has {report.total_violations} constraint violation(s). The optimizer will seek a strictly feasible arrangement.")

    st.divider()

    # Optimization Configuration
    st.markdown("### 2. Algorithm Parameters & Constraints")
    col_a1, col_a2, col_a3 = st.columns(3)

    with col_a1:
        algo_choice = st.selectbox(
            "Optimization Heuristic",
            ["Simulated Annealing (Metaheuristic)", "Random Search (Benchmark Baseline)"]
        )
        dist_metric = st.selectbox(
            "Distance Metric",
            ["manhattan", "euclidean", "aisle_grid"],
            format_func=lambda x: "Manhattan (Rectilinear Aisle Proxy)" if x == "manhattan" else ("Euclidean (Straight Line)" if x == "euclidean" else "Aisle Grid BFS (Obstacle-Aware Path)")
        )

    with col_a2:
        max_iters = st.slider("Maximum Iterations", min_value=100, max_value=3000, value=1200, step=100)
        rnd_seed = st.number_input("Random Seed (Reproducibility)", min_value=1, max_value=99999, value=42)

    with col_a3:
        init_temp = st.number_input("Initial Temperature (T0)", min_value=50.0, max_value=2000.0, value=500.0, step=50.0)
        cooling = st.slider("Cooling Rate (alpha)", min_value=0.90, max_value=0.999, value=0.985, step=0.005)
        enforce_cl = st.checkbox("Enforce Safety Clearance Margins", value=False)

    st.divider()

    # Optimization Execution
    if st.button("⚡ Optimize Factory Layout Now", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(current_iter, total_iters, cur_cost, best_c):
            pct = int((current_iter / total_iters) * 100)
            progress_bar.progress(pct)
            status_text.text(f"Iteration {current_iter}/{total_iters} | Current Cost: {cur_cost:,.1f} | Best Feasible: {best_c:,.1f}")

        start_time = time.time()

        if "Simulated Annealing" in algo_choice:
            optimizer = SimulatedAnnealingOptimizer(
                initial_temperature=init_temp,
                cooling_rate=cooling,
                max_iterations=max_iters,
                step_size=1.0,
                random_seed=int(rnd_seed),
                distance_metric=dist_metric,
                enforce_clearance=enforce_cl
            )
            result = optimizer.optimize(st.session_state.layout, progress_callback=update_progress)
        else:
            optimizer = RandomSearchOptimizer(
                max_iterations=max_iters,
                random_seed=int(rnd_seed),
                distance_metric=dist_metric
            )
            result = optimizer.optimize(st.session_state.layout, progress_callback=update_progress)

        progress_bar.progress(100)
        status_text.text(f"Optimization finished in {result.runtime_seconds:.2f} seconds!")

        st.session_state.optimization_result = result
        st.session_state.optimized_layout = result.best_layout

        if result.status == "SUCCESS":
            st.success(f"🎉 Success! Found feasible arrangement with {result.improvement_percent:+.2f}% improvement in material transport distance.")
        else:
            st.error(f"⚠️ {result.message}")

    # Display Convergence Curve if available
    if st.session_state.optimization_result is not None:
        st.markdown("### 3. Convergence Progression")
        conv_fig = render_convergence_plot(
            st.session_state.optimization_result.history,
            title=f"{algo_choice} Convergence (Seed: {st.session_state.optimization_result.seed})"
        )
        st.pyplot(conv_fig)


# =========================================================
# PAGE 5: RESULTS & COMPARISON
# =========================================================
elif page_selection == "5. Results & Comparative Analysis":
    st.markdown('<div class="main-title">Optimization Results & Quantitative Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Side-by-side engineering evaluation, material flow visual comparison, and downloadable deliverables</div>', unsafe_allow_html=True)

    if st.session_state.optimized_layout is None:
        st.info("No optimized layout found in current session. Run optimization on Page 4 or click below to run benchmark.")
        if st.button("⚡ Run Benchmark Optimization Now", type="primary"):
            opt = SimulatedAnnealingOptimizer(max_iterations=1000, random_seed=42)
            res = opt.optimize(st.session_state.layout)
            st.session_state.optimization_result = res
            st.session_state.optimized_layout = res.best_layout
            st.rerun()
    else:
        orig_layout = st.session_state.original_layout
        opt_layout = st.session_state.optimized_layout
        res = st.session_state.optimization_result

        # Comparison DataFrame
        comp_df, comp_summary = compare_layouts(
            original_layout=orig_layout,
            optimized_layout=opt_layout,
            runtime_seconds=res.runtime_seconds if res else 0.0,
            distance_metric=res.distance_metric if res else "manhattan"
        )

        st.markdown("### 1. Key Performance Indicators")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Flow Travel (Original)", f"{comp_summary['original_cost']:,.1f} m-moves")
        with kpi2:
            st.metric("Flow Travel (Optimized)", f"{comp_summary['optimized_cost']:,.1f} m-moves", delta=f"{comp_summary['percent_improvement']:+.2f}%")
        with kpi3:
            st.metric("Final Feasibility", "FEASIBLE" if comp_summary["optimized_feasible"] else "INFEASIBLE")
        with kpi4:
            st.metric("Runtime", f"{comp_summary['runtime_seconds']:.2f} s")

        st.divider()

        # Side by Side Visual Comparison
        st.markdown("### 2. Visual Layout Comparison")
        toggle_flows = st.checkbox("Show Directed Material Flow Arrows", value=True)

        comp_fig = render_comparison_figure(
            original_layout=orig_layout,
            optimized_layout=opt_layout,
            orig_cost=comp_summary["original_cost"],
            opt_cost=comp_summary["optimized_cost"],
            pct_improvement=comp_summary["percent_improvement"],
            show_flows=toggle_flows
        )
        st.pyplot(comp_fig)

        st.divider()

        # Detailed Table
        st.markdown("### 3. Quantitative Evaluation Matrix")
        st.dataframe(comp_df, use_container_width=True)

        st.divider()

        # Equipment coordinate shifts
        st.markdown("### 4. Equipment Relocation Table")
        reloc_records = []
        for m_orig in orig_layout.machines:
            m_opt = opt_layout.get_machine(m_orig.id)
            if m_opt:
                dx = m_opt.x - m_orig.x
                dy = m_opt.y - m_orig.y
                shift_dist = (dx**2 + dy**2)**0.5
                reloc_records.append({
                    "ID": m_orig.id,
                    "Name": m_orig.name,
                    "Category": m_orig.category,
                    "Original (X, Y)": f"({m_orig.x:.1f}, {m_orig.y:.1f})",
                    "Optimized (X, Y)": f"({m_opt.x:.1f}, {m_opt.y:.1f})",
                    "Displacement (m)": f"{shift_dist:.2f}",
                    "Status": "FIXED" if m_orig.is_fixed else ("MOVED" if shift_dist > 0.05 else "UNCHANGED")
                })
        st.dataframe(pd.DataFrame(reloc_records), use_container_width=True)

        st.divider()

        # Download Deliverables
        st.markdown("### 5. Academic Deliverables & Export Artifacts")
        d_col1, d_col2, d_col3, d_col4 = st.columns(4)

        # 1. Coordinates CSV
        csv_buffer = io.StringIO()
        opt_layout.to_dataframe().to_csv(csv_buffer, index=False)
        d_col1.download_button(
            "📄 Coordinates CSV",
            data=csv_buffer.getvalue(),
            file_name="optimized_equipment_coordinates.csv",
            mime="text/csv",
            use_container_width=True
        )

        # 2. Layout JSON
        layout_json_str = json.dumps(opt_layout.to_dict(), indent=2)
        d_col2.download_button(
            "📐 Layout Config JSON",
            data=layout_json_str,
            file_name="optimized_factory_layout.json",
            mime="application/json",
            use_container_width=True
        )

        # 3. Metrics JSON
        metrics_json_str = json.dumps(comp_summary, indent=2)
        d_col3.download_button(
            "📊 Metrics JSON",
            data=metrics_json_str,
            file_name="layout_evaluation_metrics.json",
            mime="application/json",
            use_container_width=True
        )

        # 4. Executive Summary Markdown
        exec_summary_md = generate_executive_summary_markdown(orig_layout, opt_layout, comp_summary)
        d_col4.download_button(
            "📝 Results Report MD",
            data=exec_summary_md,
            file_name="executive_summary.md",
            mime="text/markdown",
            use_container_width=True
        )

        st.info("""
        **Engineering Assumptions & Limitations**:
        - **Distance Metric**: Manhattan distance serves as an orthogonal aisle proxy. In real factories, aisle turns and forklift congestion affect travel time.
        - **Static Flow Frequency**: Material moves are assumed stationary per shift; dynamic scheduling and batch queuing are not modeled.
        - **Fire & Safety Codes**: While geometric clearance checks are verified, operational egress compliance requires physical industrial inspection.
        """)
