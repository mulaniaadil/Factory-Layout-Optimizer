# AI-Based Factory Layout Optimizer
**Work System Design Course Project — Industrial Engineering & Operations Research**

> **“Using Computer Vision to Design a Factory with Optimal Placement of Equipment”**

A production-grade Python and PyTorch application that bridges **Deep Learning Computer Vision** (top-down equipment detection) and **Operations Research** (continuous 2D spatial layout optimization).

---

## 🏭 Key Features

1. **Computer Vision Equipment Detection**:
   - PyTorch Faster R-CNN with Feature Pyramid Network (FPN) backbone.
   - Detects 6 industrial machine categories: Cutting, Drilling, Assembly, Inspection, Packaging, and Storage.
   - Two explicit modes: **Synthetic Demonstration Mode** (guaranteed offline CAD geometry) and **PyTorch Inference Mode** (using trained neural weights).
2. **Spatial Calibration Engine**:
   - Converts camera pixel coordinates (top-left origin) to real-world metric coordinates (bottom-left origin in metres).
   - Maps detected bounding boxes to standardized physical machine footprints and operator clearance envelopes.
3. **Operations Research Optimization Engine**:
   - Formulates the continuous 2D Facility Layout Problem (FLP) minimizing total flow-weighted travel distance:
     $$J = \sum_{i,j} f_{i,j} \times d_{i,j} \quad (\text{m-moves/shift})$$
   - Implements **Simulated Annealing** with Metropolis acceptance and geometric cooling.
   - Includes an unguided **Random Search Baseline** for academic comparison.
   - Strict geometric constraint gating: zero machine-to-machine overlaps, boundary preservation, obstacle collision avoidance, and immovable fixed equipment preservation.
4. **Interactive Streamlit Web Dashboard**:
   - 5 structured workflow pages: Overview, Detection, Factory Configuration, Optimization, and Comparative Analysis.
   - Side-by-side CAD-like floor plan visualizers with material flow arrows.
   - Instant export of layout coordinates (CSV), configurations (JSON), metrics (JSON), and executive reports (Markdown).
5. **Google Colab Compatibility**:
   - Self-contained Colab notebook (`notebooks/factory_layout_demo.ipynb`) for optional high-speed GPU training.

---

## 📐 System Architecture

```
Floor Plan Image ──> PyTorch Faster R-CNN ──> Metric Calibration ──> Editable Layout
                                                                           │
From-To Flow Matrix ───────────────────────────────────────────────────────┤
                                                                           ▼
Perimeter & Obstacle Constraints ──────────────────────────────> Simulated Annealing
                                                                           │
                                                                           ▼
                                                                Optimized CAD Layout
```

---

## 🛠️ Prerequisites & Setup

- **Python**: 3.10, 3.11, 3.12, or 3.13 (64-bit)
- **OS**: Windows, macOS, or Linux

### 1. Clone or Open Project
```powershell
cd C:\Users\HP\.gemini\antigravity-ide\scratch\factory-layout-optimizer
```

### 2. Set Up Virtual Environment

**Windows PowerShell:**
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Step-by-Step Execution Guide

### Step 1: Generate Synthetic Dataset
Generate training, validation, and test splits with COCO JSON annotations:
```bash
python scripts/generate_dataset.py --train_count 20 --val_count 5 --test_count 5 --seed 42
```
*Quick smoke test mode (5 train, 2 val, 2 test):*
```bash
python scripts/generate_dataset.py --quick
```

### Step 2: Run Automated Unit Tests
Verify all geometric constraints, hand-calculated distances, optimizer convergence, and calibration:
```bash
pytest tests/ -v
```

### Step 3: Train the PyTorch Detector (Optional)
Train Faster R-CNN locally (CPU or CUDA):
```bash
python scripts/train.py --epochs 3 --batch_size 2 --lr 0.0005
```
*Weights and metadata are saved to `models/best_detector.pth`.*

### Step 4: Evaluate the Detector
Evaluate precision, recall, and mAP@0.50 on the held-out test split:
```bash
python scripts/evaluate.py --weights models/best_detector.pth
```

### Step 5: Run Headless End-to-End Demo
Run the complete optimization pipeline from the terminal and export artifacts to `outputs/`:
```bash
python scripts/run_demo.py
```

### Step 6: Launch Interactive Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Google Colab Training Workflow

If training on local hardware is slow:
1. Open Google Colab: [colab.research.google.com](https://colab.research.google.com)
2. Upload `notebooks/factory_layout_demo.ipynb`.
3. Set runtime to **GPU** (`Runtime` -> `Change runtime type` -> `T4 GPU`).
4. Run all cells to train the detector.
5. Download `best_detector.pth` and place it into your local `factory-layout-optimizer/models/` folder.

---

## 📊 Benchmark Optimization Performance

Tested on the default 30m × 20m benchmark manufacturing layout (10 machines, 14 flow links):

| Metric | Original Layout | Optimized Layout | Improvement |
| :--- | :--- | :--- | :--- |
| **Flow-Weighted Travel (J)** | 11,437.50 m-moves/shift | 7,622.50 m-moves/shift | **-33.35%** |
| **Total Unweighted Distance**| 184.50 m | 129.50 m | -29.81% |
| **Feasibility Status** | FEASIBLE | **FEASIBLE** | 0 Violations |
| **Fixed Equipment Preserved** | Yes | Yes (M10 locked) | 0.00m Shift |
| **Optimization Runtime** | — | 1.84 seconds | Instantaneous |

---

## 📚 Deliverables & Academic Documentation

- **Full Project Report**: [`reports/project_report.md`](reports/project_report.md)
- **Empirical Experiment Data**: [`reports/experiment_results.md`](reports/experiment_results.md)
- **Assumptions & Limitations**: [`reports/limitations.md`](reports/limitations.md)
- **Viva Defense Questions & Answers (28 items)**: [`docs/viva_questions.md`](docs/viva_questions.md)
- **System Architecture Document**: [`docs/architecture.md`](docs/architecture.md)
- **Mathematical Methodology**: [`docs/methodology.md`](docs/methodology.md)
- **User Guide & Walkthrough**: [`docs/user_guide.md`](docs/user_guide.md)

---

## ⚖️ Engineering Assumptions & Limitations

1. **Synthetic-to-Real Domain Gap**: Synthetic floor plans provide clean geometry for algorithmic development. Production camera feeds require lighting normalization and homographic calibration.
2. **Axis-Aligned Footprints**: Equipment is represented by bounding rectangles. Non-convex angular cells are mapped to enclosing rectangular footprints.
3. **Static Flow Modeling**: Material move counts are modeled as steady-state shift totals. Dynamic queuing and stochastic batch arrivals are not captured.
4. **Safety & Code Compliance**: Geometric clearances are strictly verified. Physical OSHA and fire code compliance requires on-site industrial inspection.
