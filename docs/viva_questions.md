# Academic Viva Examination Guide & Defense Questions

**Course**: Work System Design / Operations Research & Industrial Engineering  
**Project Title**: *Using Computer Vision to Design a Factory with Optimal Placement of Equipment*  
**Technology Stack**: PyTorch, TorchVision (Faster R-CNN), OpenCV, Pillow, SciPy, Streamlit, Matplotlib  

---

## 28 Core Viva Questions and Model Answers

### Question 1: Why PyTorch rather than standard heuristics alone?
**Answer**:  
Traditional industrial engineering relies on manual transcription of machine coordinates into spreadsheets or static optimization solvers. PyTorch provides a production-grade deep learning framework enabling end-to-end computer vision. By incorporating PyTorch, the system automates spatial perception directly from top-down architectural floor plans and CAD raster blueprints, closing the cyber-physical gap between photographic or visual records and operations research solvers.

### Question 2: Why object detection rather than image classification or semantic segmentation?
**Answer**:  
Image classification only outputs a single tag for the entire image (e.g., "Factory Floor"), giving zero spatial information. Semantic segmentation classifies every pixel into a class mask, but does not distinguish distinct machine instances (e.g., three separate laser cutters blend into one mask). Object detection outputs both the category ID and the exact individual bounding box `[xmin, ymin, xmax, ymax]` for every machine instance, which is precisely the discrete entity representation required to initialize 2D layout coordinates.

### Question 3: Why Faster R-CNN with a Feature Pyramid Network (FPN) backbone?
**Answer**:  
Faster R-CNN is a two-stage detector consisting of a Region Proposal Network (RPN) followed by RoI Align and Fast R-CNN classification/regression heads. The Feature Pyramid Network (FPN) constructs multi-scale feature maps, which is critical in factory layouts where equipment varies dramatically in physical size—from compact 1.5m inspection stations to 6m raw material storage bays. It offers significantly higher localization accuracy than one-stage detectors on geometric structures.

### Question 4: What is a bounding box, and what coordinate conventions are used?
**Answer**:  
A bounding box is the smallest axis-aligned rectangle enclosing an object. Computer vision frameworks (TorchVision) use pixel coordinates with the origin `(0, 0)` at the **top-left**, where $X$ extends rightwards and $Y$ extends downwards. In contrast, industrial engineering layout drawings use metric coordinates with the origin `(0, 0)` at the **bottom-left**, where $X$ extends rightwards and $Y$ extends upwards. Our calibration module formally performs this affine inversion: $Y_{m} = (H_{px} - y_{px}) \times \text{scale}_y$.

### Question 5: What is Intersection over Union (IoU)?
**Answer**:  
IoU is the standard metric measuring the overlap between two bounding boxes (e.g., predicted box $B_p$ and ground truth box $B_{gt}$):
$$\text{IoU} = \frac{\text{Area}(B_p \cap B_{gt})}{\text{Area}(B_p \cup B_{gt})}$$
An IoU of 1.0 represents a perfect geometric match, while an IoU of 0 indicates disjoint non-overlapping boxes. In object detection evaluation, a detection is considered a True Positive if its IoU with a ground truth box exceeds a set threshold (typically 0.50).

### Question 6: What are Precision, Recall, and Average Precision (AP)?
**Answer**:  
- **Precision**: $\frac{\text{TP}}{\text{TP} + \text{FP}}$ — Measures detection purity: of all predicted machines, what fraction were real equipment?
- **Recall**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$ — Measures detection completeness: of all true equipment on the floor, what fraction did the model find?
- **Average Precision (AP)**: The area under the Precision-Recall curve across varying confidence thresholds for a specific equipment class.
- **mAP@0.50**: The mean of AP scores across all 6 factory equipment classes at an IoU threshold of 0.50.

### Question 7: How does transfer learning benefit this project?
**Answer**:  
Training a deep convolutional backbone like ResNet-50 or MobileNetV3 from scratch requires hundreds of thousands of annotated images. By utilizing pretrained weights initialized on MS COCO, low-level feature extractors (edges, orthogonal lines, metallic textures, geometric corners) are already learned. We replace only the final classification and bounding box regression layer (`FastRCNNPredictor`) with 7 classes (background + 6 equipment categories), enabling rapid fine-tuning on factory floor images.

### Question 8: What are the limitations of synthetic training data?
**Answer**:  
Synthetic data is a development and prototyping aid. While synthetic generation allows controlled variations in floor grid markings, aspect ratios, and machine dimensions, it exhibits a "sim-to-real domain gap": real factory camera feeds suffer from perspective distortion, shadows, worker occlusions, oil spills, and varying ceiling lighting. Synthetic data should never be represented as proof of real-world generalization without domain adaptation.

### Question 9: Why is spatial calibration necessary?
**Answer**:  
Camera pixels have no inherent physical scale. An image measuring $800 \times 600$ pixels might represent a $10\text{m} \times 7.5\text{m}$ workshop or a $100\text{m} \times 75\text{m}$ aircraft hangar. Calibration converts pixels into metres ($\text{scale} = \text{metres} / \text{pixels}$) based on known exterior factory dimensions or a known calibration scale reference bar, ensuring the optimization operates on real spatial footprints.

### Question 10: What is a From-To chart (Material Flow Matrix)?
**Answer**:  
A From-To chart is a classic Work System Design matrix where entry $(i, j)$ represents the quantitative material handling flow $f_{i,j}$ (e.g., unit loads, pallet transfers, or parts per shift) moved from workstation $i$ to workstation $j$. It serves as the weighting matrix in the objective function.

### Question 11: What is the primary objective function $J$ and what are its units?
**Answer**:  
The objective minimizes the total flow-weighted travel distance:
$$J = \sum_{i} \sum_{j} f_{i,j} \times d_{i,j}$$
Units: $f_{i,j}$ has units of movements/shift; $d_{i,j}$ has units of metres. Therefore, $J$ has units of **metre-moves per shift** (flow-metres/shift). Minimizing $J$ directly reduces internal logistics travel time, forklift battery consumption, material handling labor costs, and aisle traffic congestion.

### Question 12: Why use Manhattan distance rather than Euclidean distance?
**Answer**:  
In industrial manufacturing plants, equipment is arranged along orthogonal column grids, and material handling transport (automated guided vehicles, forklifts, overhead cranes, pallet jacks) moves along perpendicular aisles aligned with building axes. Manhattan (rectilinear) distance $d = |x_1 - x_2| + |y_1 - y_2|$ accurately models orthogonal travel, whereas Euclidean distance assumes vehicles can fly straight through physical machines.

### Question 13: What is the Grid-Based Aisle Path Distance?
**Answer**:  
When rectilinear Manhattan distance is insufficient (e.g., when structural concrete walls or massive storage racks block direct orthogonal paths), we discretize the factory floor into a 2D occupancy grid where machine footprints and obstacles are impassable cells. Breadth-First Search (BFS) computes the exact shortest navigable path along accessible aisles.

### Question 14: What is Simulated Annealing and why choose it for layout optimization?
**Answer**:  
The Facility Layout Problem (FLP) is an NP-hard combinatorial optimization problem related to the Quadratic Assignment Problem (QAP). Exact branch-and-bound methods become computationally intractable for more than 12–15 machines. Simulated Annealing is a probabilistic metaheuristic inspired by metallurgy: it explores the solution space by accepting downhill improvements and occasionally accepting uphill moves with probability $P = \exp(-\Delta E / T)$ to escape local minima.

### Question 15: How are machine-to-machine overlaps detected?
**Answer**:  
For any two axis-aligned rectangular footprints $A = [x_{A1}, y_{A1}, x_{A2}, y_{A2}]$ and $B = [x_{B1}, y_{B1}, x_{B2}, y_{B2}]$, the overlap area is computed by:
$$\text{Overlap} = \max(0, \min(x_{A2}, x_{B2}) - \max(x_{A1}, x_{B1})) \times \max(0, \min(y_{A2}, y_{B2}) - \max(y_{A1}, y_{B1}))$$
If $\text{Overlap} > 0$, a physical collision exists. In our engine, strict feasibility gating immediately rejects any candidate layout with $\text{Overlap} > 0$.

### Question 16: How are safety clearances and aisle widths modeled?
**Answer**:  
Each machine defines a safety clearance perimeter margin $c$ (typically $0.5\text{m}$ to $1.5\text{m}$). The machine footprint is expanded by $c$ on all four sides: $[x - c, y - c, x + w + c, y + h + c]$. If the clearance zones intersect, a safety clearance warning is logged, ensuring maintenance access and operator work envelopes are maintained.

### Question 17: How are structural obstacles handled?
**Answer**:  
Factory structures such as building columns, elevator shafts, electrical switchgear rooms, and emergency exit doors are modeled as static, impassable obstacle rectangles. Candidate machine perturbations that intersect obstacle bounds are rejected during feasibility checking.

### Question 18: How are fixed vs movable machines distinguished?
**Answer**:  
In industrial facilities, certain equipment (e.g., heavy concrete foundation drop-hammers, bulk chemical dip tanks, central overhead cranes, raw material receiving buffers) cannot be moved without massive capital expenditure. The model flags these machines with `is_fixed = True`. The optimizer perturbs only movable equipment while maintaining fixed machines at their original coordinates.

### Question 19: Does Simulated Annealing guarantee a global optimum?
**Answer**:  
No. Simulated Annealing is a heuristic, not an exact algorithm. While asymptotic convergence to the global optimum can be mathematically proven under a logarithmic cooling schedule with infinite iterations, practical engineering runs use finite cooling rates ($\alpha \approx 0.985$), which guarantee finding high-quality local optima rather than a proven global optimum.

### Question 20: What happens if the user's initial layout is infeasible?
**Answer**:  
If the original layout has overlapping machines or out-of-boundary equipment, the application issues an engineering diagnostic warning detailing the exact violating machine pairs and boundary exceedances. The optimizer does not accept infeasible solutions; it tracks the best strictly feasible layout discovered during the search. If no feasible layout can be achieved, it honestly reports that constraints are over-constrained.

### Question 21: Why should an optimizer not simply cluster all machines together?
**Answer**:  
While placing all machines adjacent to each other would minimize distance, doing so violates physical non-overlap constraints, blocks forklift travel aisles, eliminates operator egress paths, and breaches fire safety regulations. Strict non-overlap and clearance constraints prevent degenerate clustering.

### Question 22: What is floor space utilization and how is it calculated?
**Answer**:  
Floor space utilization is the ratio of total physical equipment footprint area to total factory gross floor area:
$$\text{Utilization} (\%) = \frac{\sum_{i} \text{width}_i \times \text{height}_i}{\text{Factory Width} \times \text{Factory Height}} \times 100$$
Typical manufacturing plants target 25%–40% space utilization to preserve adequate logistics corridors, buffer zones, and maintenance access.

### Question 23: How do we ensure reproducibility in experiments?
**Answer**:  
Stochastic heuristics and neural networks depend on pseudorandom number generators. By exposing a configurable `random_seed` across Python's `random`, `numpy.random`, and `torch.manual_seed`, identical starting seeds produce bit-identical perturbation trajectories and objective curves.

### Question 24: What is the difference between Synthetic Demonstration mode and PyTorch Inference mode?
**Answer**:  
- **Synthetic Demonstration**: Uses mathematically known CAD geometry and verified ground truth coordinates. It provides a guaranteed 100% reliable demonstration without requiring GPU hardware or prior model training.
- **PyTorch Inference**: Evaluates a trained Faster R-CNN checkpoint file (`best_detector.pth`). It outputs true predicted bounding boxes and probabilistic confidence scores ($0.0 \dots 1.0$).
The application explicitly distinguishes these modes in the UI and never mislabels synthetic ground truth as trained neural predictions.

### Question 25: How would this architecture scale to a real smart manufacturing plant?
**Answer**:  
In a production deployment:
1. Overhead factory CCTV cameras or ceiling-mounted LiDAR scanners stream real-time top-down frames.
2. The PyTorch detector runs continuously at the edge (NVIDIA Jetson) to track movable workbenches and pallet carts.
3. Real-time material moves are captured via RFID tags or MES (Manufacturing Execution System) barcode scans.
4. Optimization runs periodically to recommend layout reconfigurations for high-mix low-volume production batches.

### Question 26: What is the Quadratic Assignment Problem (QAP) formulation?
**Answer**:  
In classical Operations Research, the Quadratic Assignment Problem models assigning $n$ facilities to $n$ fixed locations to minimize $\sum_{i,j} \sum_{k,l} f_{i,j} d_{k,l} x_{i,k} x_{j,l}$, where $x_{i,k} = 1$ if facility $i$ is assigned to location $k$. Our continuous 2D coordinate formulation generalizes this by allowing continuous positions $(x_i, y_i) \in \mathbb{R}^2$ snapped to a grid, removing the artificial constraint of pre-assigned location slots.

### Question 27: What is the Metropolis acceptance criterion?
**Answer**:  
Given energy change $\Delta E = J_{\text{candidate}} - J_{\text{current}}$:
- If $\Delta E < 0$, the candidate improves the objective and is unconditionally accepted.
- If $\Delta E \ge 0$, the candidate worsens the objective but is accepted with probability $P = \exp(-\Delta E / T)$, where $T$ is the current annealing temperature. At high temperatures (early search), large uphill moves are accepted to escape poor configurations; at low temperatures (late search), the algorithm behaves like greedy local descent.

### Question 28: Why is manual verification and correction included in the workflow?
**Answer**:  
No computer vision model is 100% accurate. In real industrial environments, occlusions or lighting artifacts can yield false positives or misclassifications. An automated system that forces an engineer to accept raw neural outputs is dangerous. Providing an editable table where the industrial engineer reviews, corrects, adds, or deletes detections before optimization guarantees human-in-the-loop engineering control.
