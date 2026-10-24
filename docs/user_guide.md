# Factory Layout Optimizer — User Guide & Manual

This guide provides operational walkthroughs for all 5 pages of the web interface.

---

## Page 1: Project Overview & Quick Start
1. **Title Banner**: Displays the academic research title and system architecture diagram.
2. **Component Summary Cards**: Summarizes Computer Vision, Spatial Calibration, and Optimization.
3. **Quick Action Buttons**:
   - `🚀 Start with Synthetic Demonstration`: Pre-loads 10 calibrated equipment units without training.
   - `📐 Configure Benchmark Factory`: Loads the 30m x 20m benchmark manufacturing layout.
   - `⚡ Run Layout Optimization Directly`: Directly navigates to the optimizer engine.

---

## Page 2: Floor Plan & Computer Vision Perception
1. **Choose Perception Mode**:
   - **Synthetic Demonstration**: Recommended for viva presentations without external weights. Always works reliably.
   - **PyTorch Neural Detector**: Loads `models/best_detector.pth` trained Faster R-CNN.
2. **Image Selection**:
   - Upload any custom floor plan diagram (PNG, JPG), OR
   - Check `Use bundled demonstration factory image` to test the pre-generated sample.
3. **Run Perception**: Click `🔍 Run Equipment Perception`.
4. **Inspect & Transfer**:
   - View annotated image on the left.
   - Review detected categories, bounding boxes, and confidence scores in the table.
   - Click `📥 Apply Detections to Factory Layout` to convert pixel detections into metric equipment objects.

---

## Page 3: Configure Factory & Material Flows
1. **Equipment Footprints Tab**:
   - Edit $X, Y$ positions, widths, heights, categories, and clearance buffers.
   - Toggle the `Fixed` checkbox to designate immovable machines (e.g. heavy presses).
   - Click `💾 Save Equipment Modifications`.
2. **Material Flow Matrix Tab**:
   - Add or modify material move counts between equipment pairs.
   - Click `💾 Save Flow Matrix`.
3. **Obstacles & Docks Tab**:
   - Set coordinates and dimensions for the Receiving Dock and Dispatch Dock.
   - View structural pillars and restricted zones.
4. **Live Preview**: Inspect the 2D CAD floor plan at the bottom of the page.

---

## Page 4: Layout Optimization Engine
1. **Review Pre-Optimization Diagnostics**:
   - Check initial feasibility, baseline objective $J$, and space utilization.
2. **Set Hyperparameters**:
   - **Heuristic**: Select *Simulated Annealing* (recommended) or *Random Search Baseline*.
   - **Distance Metric**: Select *Manhattan (Rectilinear)*, *Euclidean*, or *Aisle Grid BFS*.
   - **Iterations**: Recommended 1000–1500 iterations for desktop execution.
   - **Random Seed**: Enter an integer (default `42`) to reproduce identical runs.
3. **Click `⚡ Optimize Factory Layout Now`**:
   - Watch the live progress bar and iteration status.
   - Review the temperature cooling curve and cost descent plot upon completion.

---

## Page 5: Results & Comparative Analysis
1. **KPI Header**: Review original vs. optimized flow-weighted distance ($J$) and % improvement.
2. **Visual Comparison**: Side-by-side CAD views of the original and optimized layouts with togglable flow vectors.
3. **Quantitative Matrix**: Full table comparing unweighted distance, move counts, space utilization, and feasibility.
4. **Equipment Relocation Table**: Displays each machine's original position, optimized position, and total displacement (metres).
5. **Download Deliverables**:
   - `📄 Coordinates CSV`: Export table of machine coordinates.
   - `📐 Layout Config JSON`: Full JSON specification.
   - `📊 Metrics JSON`: Complete quantitative performance log.
   - `📝 Results Report MD`: Academic executive summary.
