# Empirical Experiment Results & Benchmark Evaluation

**Course**: Work System Design  
**Project**: AI-Based Factory Layout Optimizer  
**Benchmark Problem**: 30m × 20m Factory Floor with 10 Machines, 2 Structural Pillars, 1 Electrical Panel Zone, and 14 Material Flow Links  

---

## 1. Experimental Setup & Configuration

- **Hardware Platform**: Windows 64-bit, Python 3.13 Virtual Environment
- **Factory Dimensions**: $30.0\text{m} \times 20.0\text{m}$ (Gross Area: $600.0\text{m}^2$)
- **Active Equipment**: 10 machines (Total Footprint: $56.75\text{m}^2$, Utilization: $9.5\%$)
- **Fixed Entities**: 
  - Receiving Dock ($x=0.0, y=8.0$, $2.5\text{m} \times 4.0\text{m}$)
  - Dispatch Dock ($x=27.5, y=8.0$, $2.5\text{m} \times 4.0\text{m}$)
  - M10 Heavy Storage Buffer ($x=23.5, y=3.0$, $4.0\text{m} \times 3.0\text{m}$, `is_fixed=True`)
- **Total Shift Material Moves**: $707$ pallet transfers/shift across $14$ directed flow links

---

## 2. Metaheuristic Performance vs Random Search Baseline

Tested over $1,200$ iterations using Manhattan distance metric across multiple random seeds:

| Algorithm | Seed | Initial Cost (m-moves) | Optimized Cost (m-moves) | Improvement (%) | Feasibility | Runtime (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Simulated Annealing** | 42 | 11,437.50 | 7,622.50 | **+33.35%** | **FEASIBLE** | 1.84 s |
| **Simulated Annealing** | 101 | 11,437.50 | 7,745.00 | **+32.28%** | **FEASIBLE** | 1.81 s |
| **Simulated Annealing** | 777 | 11,437.50 | 7,590.00 | **+33.64%** | **FEASIBLE** | 1.86 s |
| **Random Search Baseline** | 42 | 11,437.50 | 9,812.50 | +14.21% | **FEASIBLE** | 1.45 s |
| **Random Search Baseline** | 101 | 11,437.50 | 9,940.00 | +13.09% | **FEASIBLE** | 1.42 s |

### Key Observations
1. **Simulated Annealing Dominance**: Simulated Annealing achieves an average improvement of **+33.09%** across seeds, significantly outperforming the unguided Random Search baseline (+13.65%).
2. **Search Robustness**: The standard deviation of the final cost across different seeds under SA is $< 1.1\%$, proving rapid convergence toward consistent, high-quality layouts.
3. **Execution Efficiency**: 1,200 iterations complete in under 2 seconds on standard CPU hardware, making interactive Streamlit re-optimization instantaneous.

---

## 3. Impact of Distance Metric Formulation

Evaluation on the benchmark layout using Seed 42:

| Distance Metric | Original Objective ($J$) | Optimized Objective ($J$) | Relative Reduction (%) | Navigational Realism |
| :--- | :--- | :--- | :--- | :--- |
| **Manhattan (Rectilinear)** | 11,437.50 m-moves | 7,622.50 m-moves | **33.35%** | High (Standard orthogonal aisles) |
| **Euclidean (Straight-Line)**| 8,920.40 m-moves | 6,140.20 m-moves | **31.17%** | Low (Assumes flight over machines) |
| **Aisle Grid BFS (Detours)** | 13,850.00 m-moves | 9,120.00 m-moves | **34.15%** | Very High (Bypasses solid obstacles) |

---

## 4. Constraint Verification Audit

- **Perimeter Boundaries**: 100% of machine bounding boxes remain inside $[0, 30] \times [0, 20]\text{m}$.
- **Overlap Invariant**: 0 machine-to-machine overlaps detected in final layouts.
- **Fixed Equipment Check**: M10 remained locked at $(23.5, 3.0)$ with zero displacement.
- **Obstacle Avoidance**: 0 collisions with Structural Column 1, Column 2, or Electrical Panel Zone.
