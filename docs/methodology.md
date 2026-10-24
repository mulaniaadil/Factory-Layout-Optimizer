# Mathematical Formulation & Algorithmic Methodology

**Academic Topic**: Facility Layout Planning (FLP) / Quadratic Assignment Problem (QAP) Formulation  
**Project**: AI-Based Factory Layout Optimizer  

---

## 1. Problem Formulation

Let the factory floor be bounded by a continuous 2D planar rectangular domain:
$$\Omega = [0, W] \times [0, H] \subset \mathbb{R}^2$$
where $W$ is the factory width (metres) and $H$ is the factory height (metres).

Let $\mathcal{M} = \{M_1, M_2, \dots, M_N\}$ denote the set of $N$ machines to be accommodated on the shop floor. Each machine $M_i$ has:
- Static dimensions: width $w_i \in \mathbb{R}^+$ and height $h_i \in \mathbb{R}^+$.
- Position: bottom-left coordinates $(x_i, y_i) \in \Omega$.
- Machine footprint: $\mathcal{R}_i = [x_i, x_i + w_i] \times [y_i, y_i + h_i]$.
- Safety clearance margin: $c_i \ge 0$.
- Mobility status: $u_i \in \{0, 1\}$, where $u_i = 1$ indicates a fixed machine whose coordinates cannot change, and $u_i = 0$ indicates a movable machine.

Let $\mathcal{O} = \{O_1, O_2, \dots, O_K\}$ denote the set of $K$ fixed structural obstacles (building pillars, electrical transformers, emergency exits), where each obstacle occupies footprint $\mathcal{S}_k = [x_{O_k}, x_{O_k} + w_{O_k}] \times [y_{O_k}, y_{O_k} + h_{O_k}]$.

Let $\mathcal{D} = \{D_{\text{rec}}, D_{\text{disp}}\}$ denote the receiving and dispatch loading docks.

Let $F = [f_{i,j}]_{N \times N}$ be the From-To material flow matrix, where $f_{i,j} \ge 0$ represents the volume of material transfers (pallet loads per shift) between entity $i$ and entity $j$.

---

## 2. Objective Function

The primary objective is to minimize total flow-weighted material handling travel distance $J$:
$$\min_{\{(x_i, y_i)\}} J = \sum_{i=1}^{N} \sum_{j=1}^{N} f_{i,j} \cdot d(C_i, C_j)$$
where $C_i = \left(x_i + \frac{w_i}{2}, y_i + \frac{h_i}{2}\right)$ represents the geometric center of machine $M_i$, and $d(C_i, C_j)$ is the distance metric.

### Distance Metrics
1. **Manhattan (Rectilinear) Distance**:
   $$d_{\text{Manhattan}}(C_i, C_j) = |x_{C_i} - x_{C_j}| + |y_{C_i} - y_{C_j}|$$
   *Justification*: In manufacturing, automated guided vehicles (AGVs) and forklifts navigate strictly orthogonal aisle networks aligned with the facility column grid.

2. **Euclidean Distance**:
   $$d_{\text{Euclidean}}(C_i, C_j) = \sqrt{(x_{C_i} - x_{C_j})^2 + (y_{C_i} - y_{C_j})^2}$$

3. **Obstacle-Aware Grid Aisle Path Distance**:
   Discretizes $\Omega$ into cell matrix $G_{R \times C}$ of resolution $\Delta g = 0.5\text{m}$.
   $$d_{\text{Aisle}}(C_i, C_j) = \text{BFS\_Shortest\_Path}(C_i, C_j, G \setminus (\bigcup \mathcal{R} \cup \bigcup \mathcal{S}))$$

---

## 3. Engineering Constraints

1. **Perimeter Boundary Constraints**:
   $$0 \le x_i \le W - w_i, \quad \forall M_i \in \mathcal{M}$$
   $$0 \le y_i \le H - h_i, \quad \forall M_i \in \mathcal{M}$$

2. **Machine-to-Machine Non-Overlap Constraints**:
   $$\text{Area}(\mathcal{R}_i \cap \mathcal{R}_j) = 0, \quad \forall i \ne j$$
   Expressed algebraically for axis-aligned rectangles:
   $$(x_i + w_i \le x_j) \lor (x_j + w_j \le x_i) \lor (y_i + h_i \le y_j) \lor (y_j + h_j \le y_i)$$

3. **Obstacle Non-Collision Constraints**:
   $$\text{Area}(\mathcal{R}_i \cap \mathcal{S}_k) = 0, \quad \forall M_i \in \mathcal{M}, \forall O_k \in \mathcal{O}$$

4. **Dock Clearance Constraints**:
   $$\text{Area}(\mathcal{R}_i \cap \mathcal{S}_{D}) = 0, \quad \forall M_i \in \mathcal{M}, \forall D \in \mathcal{D}$$

5. **Fixed Machine Immobility Constraints**:
   $$(x_i, y_i) = (x_i^{(0)}, y_i^{(0)}), \quad \text{if } u_i = 1$$

6. **Safety Clearance Buffers**:
   $$\min_{\mathbf{p}_i \in \mathcal{R}_i, \mathbf{p}_j \in \mathcal{R}_j} \|\mathbf{p}_i - \mathbf{p}_j\|_{\infty} \ge \max(c_i, c_j), \quad \forall i \ne j$$

---

## 4. Simulated Annealing Heuristic Formulation

### Neighborhood Generation
At iteration $t$, select a movable machine $M_i \in \mathcal{M}_{\text{movable}}$ uniformly at random.
Generate perturbed candidate coordinates:
$$x_i^{\prime} = \text{clip}\left(\text{round}\left(\frac{x_i + \Delta x}{\Delta g}\right) \Delta g, 0, W - w_i\right)$$
$$y_i^{\prime} = \text{clip}\left(\text{round}\left(\frac{y_i + \Delta y}{\Delta g}\right) \Delta g, 0, H - h_i\right)$$
where $\Delta x, \Delta y \in \{-2\delta, -\delta, 0, \delta, 2\delta\}$ with perturbation step $\delta = 1.0\text{m}$.

### Feasibility Gating
If the candidate layout violates any boundary or overlap constraint, it is immediately discarded:
$$J(\text{candidate}) = \infty \implies \text{Reject}$$

### Metropolis Acceptance Criterion
Compute objective change: $\Delta E = J(\text{candidate}) - J(\text{current})$.
- If $\Delta E < 0$ (improvement): Candidate is accepted.
- If $\Delta E \ge 0$ (uphill move): Candidate is accepted with probability:
  $$P(\text{accept}) = \exp\left(-\frac{\Delta E}{T_t}\right)$$

### Geometric Cooling Schedule
$$T_{t+1} = \max(T_{\min}, \alpha \cdot T_t)$$
where initial temperature $T_0 = 500.0$, cooling coefficient $\alpha = 0.985$, and $T_{\min} = 0.1$.
