# System Architecture & Technical Specifications

**Project**: AI-Based Factory Layout Optimizer  
**Domain**: Work System Design, Operations Research, Computer Vision  

---

## 1. High-Level Modular Decomposition

The project separates visual perception, spatial reconstruction, combinatorial optimization, user interaction, and data persistence into decoupled layers:

```
+-----------------------------------------------------------------------+
|                         Streamlit Web UI (app.py)                     |
|  [Page 1: Overview] [Page 2: Vision] [Page 3: Config]                 |
|  [Page 4: Optimizer] [Page 5: Comparative Analysis & Downloads]       |
+-----------------------------------+-----------------------------------+
                                    |
+-----------------------------------+-----------------------------------+
|                        Core Subsystems                                |
|                                                                       |
|  +--------------------+  +-------------------+  +-------------------+ |
|  |  Computer Vision   |  |     Spatial       |  |   Optimization    | |
|  |  Faster R-CNN      |  |  Reconstruction   |  |      Engine       | |
|  |  (TorchVision)     |  |  & Calibration    |  |  (Simulated       | |
|  |  - RPN + RoIAlign  |  |  - Pixel to Metre |  |   Annealing &     | |
|  |  - FastRCNNHead    |  |  - Footprint Maps |  |   Random Baseline)| |
|  +--------------------+  +-------------------+  +-------------------+ |
|            |                       |                      |           |
|  +--------------------+  +-------------------+  +-------------------+ |
|  |  Data Generation   |  |   Constraints     |  | Distance Metrics  | |
|  |  - Synthetic Floor |  |  - Non-overlap    |  |  - Manhattan      | |
|  |  - COCO JSON Export|  |  - Perimeter Wall |  |  - Euclidean      | |
|  |  - Pillow / OpenCV |  |  - Pillars & Bays |  |  - Aisle BFS Grid | |
|  +--------------------+  +-------------------+  +-------------------+ |
+-----------------------------------+-----------------------------------+
                                    |
+-----------------------------------+-----------------------------------+
|               Visualization, Evaluation & Artifact Export             |
|  - Matplotlib 2D Floor Plans       - CSV / JSON Coordinate Serializer |
|  - SA Convergence Plots            - Automated Pytest Verification    |
+-----------------------------------------------------------------------+
```

---

## 2. Directory Structure & File Map

```
factory-layout-optimizer/
├── app.py                          # Streamlit multi-page web application
├── requirements.txt                # Pinned dependencies (CPU-compatible)
├── README.md                       # Complete deployment & execution guide
├── .gitignore                      # Git ignore rules for virtualenvs/artifacts
├── config/
│   └── default_config.json         # Benchmark factory parameters, flows, classes
├── data/
│   ├── README.md                   # Dataset specification and class mapping
│   ├── sample_factory_floor.png    # Bundled demonstration floor plan
│   └── synthetic/                  # Generated train, val, test splits
├── models/
│   ├── README.md                   # Checkpoint storage guide & Colab download notes
│   ├── best_detector.pth           # Trained PyTorch Faster R-CNN state dict
│   └── detector_metadata.json      # Training hyperparameters and loss history
├── src/
│   ├── __init__.py                 # Package declaration
│   ├── config.py                   # Global constants and config loader
│   ├── layout.py                   # Machine, Obstacle, Dock, FactoryLayout dataclasses
│   ├── constraints.py              # Boundary, overlap, clearance, fixed-machine checks
│   ├── distance.py                 # Manhattan, Euclidean, and BFS Aisle Grid routing
│   ├── optimizer.py                # Simulated Annealing and Random Search heuristics
│   ├── calibration.py              # Pixel-to-metre and coordinate transformation engine
│   ├── data_generation.py          # Synthetic floor plan generator and COCO exporter
│   ├── dataset.py                  # PyTorch Dataset and collate function
│   ├── train_detector.py           # Training pipeline for Faster R-CNN
│   ├── evaluate_detector.py        # mAP@0.50, precision, recall evaluation engine
│   ├── inference.py                # Trained PyTorch & Synthetic demonstration inference
│   ├── metrics.py                  # Space utilization, flow distance, and comparisons
│   ├── visualization.py            # CAD-quality Matplotlib layout renderers
│   └── export.py                   # CSV, JSON, PNG, and Markdown export tools
├── scripts/
│   ├── generate_dataset.py         # CLI for synthetic data creation
│   ├── train.py                    # CLI for PyTorch detector training
│   ├── evaluate.py                 # CLI for detector evaluation
│   └── run_demo.py                 # End-to-end headless CLI layout optimization
├── tests/
│   ├── test_constraints.py         # Unit tests for boundary, overlap, and pillars
│   ├── test_distance.py            # Hand-calculated tests for distance metrics
│   ├── test_optimizer.py           # Convergence, reproducibility, and fixed checks
│   ├── test_calibration.py        # Coordinate conversion round-trip verification
│   └── test_data_generation.py    # Synthetic image & COCO annotation validation
├── notebooks/
│   └── factory_layout_demo.ipynb   # Standalone Google Colab GPU training notebook
├── reports/
│   ├── project_report.md           # Full academic Work System Design term report
│   ├── experiment_results.md       # Quantitative benchmark experiment data
│   └── limitations.md              # Honest technical assessment & assumptions
└── docs/
    ├── architecture.md             # This document
    ├── methodology.md              # Mathematical formulation and algorithmic theory
    ├── viva_questions.md           # 28 defense questions and answers
    └── user_guide.md               # User manual for all Streamlit pages
```

---

## 3. Data Flow Between Subsystems

1. **Perception**:
   - Input: $W_{px} \times H_{px}$ RGB raster image.
   - Model: Faster R-CNN ($7$ classes including background).
   - Output: List of bounding boxes $[x_{min}, y_{min}, x_{max}, y_{max}]_{px}$, categories, and confidence scores.
2. **Calibration**:
   - Inputs: Pixel bounding boxes, factory metric width $W_m$ and height $H_m$.
   - Transformation: $X_m = x_{min} \times (W_m / W_{px})$, $Y_m = (H_{px} - y_{max}) \times (H_m / H_{px})$.
   - Output: Initialized `FactoryLayout` object with metric dimensions.
3. **Operations Research Optimization**:
   - Inputs: Initial `FactoryLayout`, From-To flow matrix $\{f_{i,j}\}$, geometric constraints.
   - Metaheuristic: Simulated Annealing with continuous coordinate perturbation snapped to $0.5\text{m}$ grid.
   - Evaluator: Total flow cost $J = \sum f_{i,j} d_{i,j}$ and boolean constraint checker.
   - Output: `OptimizationResult` containing `best_layout`, convergence logs, and feasibility report.
4. **Export**:
   - Deliverables: CAD PNG visual, coordinates CSV, config JSON, metrics JSON, executive summary Markdown.
